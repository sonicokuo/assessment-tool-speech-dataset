"""Render per-clip attention-overlay figures from inference outputs.

Reads `inference_results.json` (saved next to the checkpoint by src/inference.py)
and produces one multi-panel figure per clip: the log-mel spectrogram in
grayscale, with each section's attention map overlaid as a translucent hot
heatmap. One subplot per section.

These are the paper figures: "the model attended to these audio regions when
generating the noise / reverb / pitch / ... claims."

Usage:
    python scripts/plot_attention_spectrograms.py \\
        --results   checkpoints/qwen3_17b_full_ft_tagged_v1/inference_results.json \\
        --audio_dir /path/to/Libri2Mix/wav16k/min/test/mix_clean \\
        --output    docs/report/figures/attention_overlays \\
        --clips     61-70968-0028_84-121123-0007.wav,260-123286-0020_4992-23283-0013.wav \\
        --format    pdf

If --clips is omitted, the first N clips with attention maps are plotted
(default N=10). Output: one PDF/PNG per clip with one subplot per section.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


# Match BEATs's filterbank settings so the spectrogram axes line up with the
# encoder's patch grid (and therefore with the saved attention vector).
N_MELS = 128
FRAME_LENGTH = 25     # ms
FRAME_SHIFT = 10      # ms
SAMPLE_RATE = 16000


def compute_log_mel(waveform: np.ndarray) -> np.ndarray:
    """Compute a log-mel spectrogram matching BEATs's preprocessing.

    Returns: (n_mels, n_frames) float array in log-power dB.
    """
    wav_t = torch.from_numpy(waveform).float().unsqueeze(0)
    win_length = int(SAMPLE_RATE * FRAME_LENGTH / 1000)
    hop_length = int(SAMPLE_RATE * FRAME_SHIFT / 1000)
    mel = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=N_MELS,
        f_min=0.0,
        f_max=SAMPLE_RATE // 2,
    )(wav_t).squeeze(0)               # (n_mels, n_frames)
    return 10.0 * torch.log10(mel + 1e-10).numpy()


def upsample_attention(alpha_2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Upsample a (T_p, F_p) patch grid to match the (n_frames, n_mels) spec resolution.

    Nearest-neighbor — the natural fit because each patch covers a rectangular
    span of the underlying spectrogram by construction.
    """
    t_target, f_target = target_shape
    t_p, f_p = alpha_2d.shape
    t_scale = max(1, t_target // t_p)
    f_scale = max(1, f_target // f_p)
    # Block-repeat
    up = np.repeat(np.repeat(alpha_2d, t_scale, axis=0), f_scale, axis=1)
    # Trim or pad to exact target shape
    up = up[:t_target, :f_target]
    if up.shape != (t_target, f_target):
        pad = np.zeros(target_shape)
        pad[: up.shape[0], : up.shape[1]] = up
        up = pad
    return up


def plot_one_clip(
    entry: dict,
    audio_dir: str,
    output_path: str,
    section_order: list[str],
    overlap_gt: list[tuple[float, float]] | None = None,
) -> None:
    """Render the attention-overlay figure for one clip."""
    filename = entry["filename"]
    stem = os.path.splitext(filename)[0]
    wav_path = None
    for ext in (".wav", ".flac"):
        p = os.path.join(audio_dir, stem + ext)
        if os.path.exists(p):
            wav_path = p
            break
    if wav_path is None:
        print(f"  [skip] no audio for {filename}")
        return

    waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if sr != SAMPLE_RATE:
        print(f"  [skip] {filename}: sr={sr} != {SAMPLE_RATE}")
        return

    log_mel = compute_log_mel(waveform)                  # (n_mels, n_frames)
    n_mels, n_frames = log_mel.shape
    duration_s = len(waveform) / SAMPLE_RATE

    attn_dict = entry.get("attention_maps", {})
    sections_present = [s for s in section_order if s in attn_dict]
    if not sections_present:
        print(f"  [skip] {filename}: no attention_maps")
        return

    # Determine the (T_p, F_p) grid for each section. Saved JSON is a flat list
    # of length n_patches; BEATs's freq_dim is 8 (128 mel bins / 16 patch size).
    F_P = 8
    n_panels = len(sections_present)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(11, 1.6 * n_panels + 0.4),
        sharex=True, squeeze=False,
    )

    # Common spectrogram for the underlay (rendered grayscale once per panel).
    spec_t = np.linspace(0, duration_s, n_frames)
    spec_f = np.linspace(0, SAMPLE_RATE // 2, n_mels)

    for ax_row, section_name in zip(axes[:, 0], sections_present):
        alpha_vec = np.asarray(attn_dict[section_name], dtype=np.float32)
        n_patches = alpha_vec.size
        t_p = n_patches // F_P
        if t_p * F_P != n_patches:
            print(f"  [warn] {filename} sec={section_name}: "
                  f"n_patches={n_patches} not divisible by F_p={F_P}")
            continue
        alpha_2d = alpha_vec.reshape(t_p, F_P)            # (T_p, F_p)
        # Upsample to (n_frames, n_mels) — note transpose so time is dim 0.
        alpha_up = upsample_attention(alpha_2d, target_shape=(n_frames, n_mels))
        # Normalise so the brightest cell is at alpha-max=1.0 for visibility.
        alpha_norm = alpha_up / (alpha_up.max() + 1e-12)

        # Underlay: grayscale spectrogram (transpose so y-axis is freq, x is time)
        ax_row.imshow(
            log_mel, origin="lower", aspect="auto",
            cmap="gray_r",
            extent=[0, duration_s, 0, SAMPLE_RATE // 2],
            interpolation="nearest",
        )
        # Overlay: hot colormap, alpha-modulated by attention strength
        ax_row.imshow(
            alpha_norm.T, origin="lower", aspect="auto",
            cmap="hot",
            extent=[0, duration_s, 0, SAMPLE_RATE // 2],
            alpha=0.55 * alpha_norm.T,
            interpolation="nearest",
        )
        ax_row.set_ylabel(f"<sec_{section_name}>", fontsize=9)
        ax_row.set_yticks([])

        # Ground-truth overlap windows: draw as cyan bands on the overlap row so
        # the reader can SEE whether the overlap query attends inside them.
        if section_name == "overlap" and overlap_gt:
            for (s0, s1) in overlap_gt:
                ax_row.axvspan(s0, s1, ymin=0.0, ymax=1.0,
                               facecolor="cyan", alpha=0.18, zorder=3)
                ax_row.axvline(s0, color="cyan", lw=0.6, alpha=0.6)
                ax_row.axvline(s1, color="cyan", lw=0.6, alpha=0.6)
            ax_row.set_ylabel("<sec_overlap>\n(cyan = GT overlap)", fontsize=8)

    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(filename, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to inference_results.json.")
    parser.add_argument("--audio_dir", required=True,
                        help="Directory of source .wav files (test split).")
    parser.add_argument("--output", required=True,
                        help="Output directory for figures.")
    parser.add_argument("--clips", default="",
                        help="Comma-separated filenames to render. Default = first --limit clips.")
    parser.add_argument("--limit", type=int, default=10,
                        help="If --clips is empty, render up to this many clips.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                        help="Output figure format.")
    parser.add_argument("--features_csv", default="",
                        help="Optional per-split feature CSV with overlap_segments_vad. "
                             "When given, GT overlap windows are drawn (cyan) on the "
                             "<sec_overlap> row so attention-vs-GT alignment is visible.")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    # Optional GT overlap segments for the visual alignment overlay.
    overlap_gt_map: dict = {}
    if args.features_csv:
        import csv as _csv
        with open(args.features_csv) as f:
            for row in _csv.DictReader(f):
                fn = (row.get("filename") or "").strip()
                raw = row.get("overlap_segments_vad") or row.get("overlap_segments") or ""
                segs = []
                for seg in raw.split(";"):
                    seg = seg.strip()
                    if not seg:
                        continue
                    try:
                        a, b = seg.split("-", 1)
                        s0, s1 = int(a) / SAMPLE_RATE, int(b) / SAMPLE_RATE
                        if s1 > s0:
                            segs.append((s0, s1))
                    except (ValueError, IndexError):
                        continue
                overlap_gt_map[fn] = segs
        print(f"[plot] loaded GT overlap for {len(overlap_gt_map)} clips")

    target_clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    if target_clips:
        entries = [e for e in results if e.get("filename") in target_clips]
    else:
        entries = [e for e in results if "attention_maps" in e][: args.limit]
    print(f"[plot] rendering {len(entries)} clip(s) to {args.output}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Stable section ordering for the figure: matches src/section_tags.py.
    # Pulled at runtime so renames stay synced.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from section_tags import SECTION_TAGS  # noqa: E402
    section_order = [s.name for s in SECTION_TAGS]

    for entry in entries:
        stem = os.path.splitext(entry["filename"])[0]
        out_path = os.path.join(args.output, f"{stem}_attention.{args.format}")
        plot_one_clip(entry, args.audio_dir, out_path, section_order,
                      overlap_gt=overlap_gt_map.get(entry["filename"]))
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
