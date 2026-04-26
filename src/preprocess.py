"""Preprocess audio files: WavLM features + VAD-derived overlap context → .pt files.

Per clip, writes a .pt with:
  - audio_features     (T, 1024)  from WavLM-Large
  - overlap_info       (T, 4)     derived from ground-truth VAD segments in the feature CSV
  - overlap_segments   list[(start_s, end_s)]
  - filename           str

The 4 overlap channels match src/adapter.py::OVERLAP_FEATURES:
  col 0: is_overlap              (binary)
  col 1: segment_duration_s      (duration of the segment this frame belongs to, 0 outside)
  col 2: frac_through_segment    (0–1 position within segment, 0 outside)
  col 3: density_300ms           (local overlap density, ±150 ms window, 0–1)

The clip_overlap_ratio scalar (previously col 3 of a 5-channel layout) was REMOVED
because it's also an SFS-evaluated feature; feeding it as model input was data leakage.
The model now has to *infer* overlap_ratio from audio + the temporal channels above.

Overlap ground truth comes from feature_extractor_mix.py (--overlap min_max_vad run on
the clean s1/s2 stems of Libri2Mix). The CSV's `overlap_segments` column stores segments
as semicolon-separated `start_sample-end_sample` integer pairs.

Usage:
    python src/preprocess.py \\
        --audio_dir    /ocean/.../Libri2Mix/.../train-100/mix_clean \\
        --features_csv /ocean/.../features/train-100.csv \\
        --output_dir   /ocean/.../processed/train
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel

# WavLM-Large hop length in samples at 16 kHz → one output frame every 20 ms (50 Hz).
WAVLM_HOP_SAMPLES = 320
# Local density smoothing window: ±150 ms ≈ 15 frames at 50 Hz → 31-frame symmetric window.
DENSITY_WINDOW_FRAMES = 31


def load_overlap_map(features_csv: str) -> dict:
    """Read feature CSV and return {filename: (overlap_segments_str, clip_overlap_ratio)}.

    `overlap_segments` is the raw semicolon-separated sample-index string.
    `overlap_ratio` is the clip-wide ratio (float 0–1), used as the clip-global context channel.
    """
    mapping = {}
    with open(features_csv) as f:
        for row in csv.DictReader(f):
            segs = (row.get("overlap_segments") or "").strip()
            if segs.lower() in ("", "nan", "n/a"):
                segs = ""
            ratio_raw = (row.get("overlap_ratio") or "").strip()
            try:
                ratio = float(ratio_raw)
            except ValueError:
                ratio = 0.0
            mapping[row["filename"]] = (segs, ratio)
    return mapping


def build_overlap_info(
    segs_str: str,
    clip_overlap_ratio: float,
    T: int,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, list]:
    """Convert CSV overlap segments into a (T, 4) overlap feature tensor.

    Args:
        segs_str:        e.g. "3616-31712;39456-67552" (sample indices) or "" for no overlap.
        clip_overlap_ratio: ignored (used to be col 3 of a 5-channel layout; removed because
                             feeding the SFS-evaluated clip_overlap_ratio as input was data leakage).
                             Argument kept for signature compatibility with prior callers.
        T:               number of WavLM frames for this clip.
        sample_rate:     audio sample rate (16000 Hz for Libri2Mix).

    Returns:
        overlap_info  tensor of shape (T, 4)
        segments_sec  list of (start_sec, end_sec) tuples for metadata / logging.
    """
    del clip_overlap_ratio  # explicitly unused
    overlap_info = torch.zeros(T, 4)
    segments_sec = []

    if not segs_str:
        return overlap_info, segments_sec

    for seg in segs_str.split(";"):
        seg = seg.strip()
        if not seg or "-" not in seg:
            continue
        try:
            s_samp, e_samp = (int(x) for x in seg.split("-"))
        except ValueError:
            continue
        if e_samp <= s_samp:
            continue

        # Map sample range to WavLM frame range (hop=320 samples).
        f_start = max(0, s_samp // WAVLM_HOP_SAMPLES)
        f_end = min(T, (e_samp + WAVLM_HOP_SAMPLES - 1) // WAVLM_HOP_SAMPLES)
        if f_end <= f_start:
            continue

        n_frames = f_end - f_start
        duration_s = (e_samp - s_samp) / sample_rate

        overlap_info[f_start:f_end, 0] = 1.0
        overlap_info[f_start:f_end, 1] = duration_s
        # Fraction-through-segment ramps 0→1 across the frames of this segment.
        overlap_info[f_start:f_end, 2] = torch.linspace(0.0, 1.0, steps=n_frames)

        segments_sec.append((s_samp / sample_rate, e_samp / sample_rate))

    # col 3: local density = 31-frame symmetric box-filter over col 0.
    # avg_pool1d requires (1, 1, T); pad reflect to keep length T.
    k = DENSITY_WINDOW_FRAMES
    pad = k // 2
    col0 = overlap_info[:, 0].view(1, 1, -1)
    smoothed = F.avg_pool1d(F.pad(col0, (pad, pad), mode="replicate"), kernel_size=k, stride=1)
    overlap_info[:, 3] = smoothed.view(-1)

    return overlap_info, segments_sec


def preprocess(
    audio_dir: str,
    features_csv: str,
    output_dir: str,
    sample_rate: int = 16000,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # WavLM-Large → frozen encoder; we cache its outputs, no fine-tuning here.
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()  # type: ignore
    for p in wavlm.parameters():
        p.requires_grad = False

    overlap_map = load_overlap_map(features_csv)
    print(f"Loaded overlap ground truth for {len(overlap_map)} clips from {features_csv}")

    audio_files = sorted(f for f in os.listdir(audio_dir) if f.endswith((".wav", ".flac", ".mp3")))
    print(f"Processing {len(audio_files)} files...")

    n_missing_csv = 0
    for i, filename in enumerate(audio_files):
        wav_path = os.path.join(audio_dir, filename)
        stem = os.path.splitext(filename)[0]

        # Load + resample to 16 kHz mono (WavLM input contract).
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception:
            import soundfile as sf

            data, sr = sf.read(wav_path)
            waveform = torch.from_numpy(data).float()
            waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        waveform = waveform.mean(dim=0)

        # WavLM forward pass → (T, 1024).
        with torch.no_grad():
            audio_features = wavlm(waveform.unsqueeze(0).to(device)).last_hidden_state.cpu()
            audio_features = audio_features.squeeze(0)

        T = audio_features.shape[0]

        # Overlap context from CSV. If a filename isn't in the CSV, fall back to all-zeros
        # so the clip still trains/evaluates (with no overlap signal). Tracked and reported.
        segs_str, clip_ratio = overlap_map.get(filename, ("", 0.0))
        if filename not in overlap_map:
            n_missing_csv += 1

        overlap_info, segments_sec = build_overlap_info(segs_str, clip_ratio, T, sample_rate)

        torch.save(
            {
                "audio_features": audio_features,       # (T, 1024)
                "overlap_info": overlap_info,           # (T, 5)
                "overlap_segments": segments_sec,       # list of (start_s, end_s)
                "filename": filename,
            },
            os.path.join(output_dir, f"{stem}.pt"),
        )

        if (i + 1) % 100 == 0:
            print(f"\t{i + 1}/{len(audio_files)} done")

    print(f"Done. Saved {len(audio_files)} files to {output_dir}")
    if n_missing_csv:
        print(f"[WARN] {n_missing_csv} files had no matching row in {features_csv} — filled with zero overlap.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory of mix_clean .wav files for one split")
    parser.add_argument("--features_csv", type=str, required=True,
                        help="Feature CSV for this split (must contain overlap_segments + overlap_ratio columns)")
    parser.add_argument("--output_dir", type=str, default="./data/toy/processed",
                        help="Directory to write per-clip .pt files")
    args = parser.parse_args()

    preprocess(args.audio_dir, args.features_csv, args.output_dir)
