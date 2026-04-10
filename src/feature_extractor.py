"""
Speech Feature Extractor - Step 1 of Speech Quality Assessment Pipeline
========================================================================
Python equivalent of experiments/Feature_Extractor_Final.ipynb.

Extracts:
  - duration, sample_rate
  - snr_db, silence_ratio
  - overlap_ratio, overlap_segments  (pyannote/segmentation-3.0)
  - srmr                             (VERSA, optional)
  - f0 variation                     (Praat: mean, std, min, max, range, voiced fraction)
  - hnr_db                           (Praat)
  - shimmer_pct                      (Praat)
  - jitter_local_pct, jitter_rap_pct (Praat)
  - pause patterns                   (Praat: count, rate, mean/total duration, ratio)
  - speaking_rate, articulation_rate (Praat: de Jong & Wempe)

Dependencies:
    pip install torchaudio pyannote.audio pandas numpy torch parselmouth
    Optional: VERSA (for SRMR)

Usage:
    python src/feature_extractor.py --audio_dir ./audio_samples --output features.csv
    python src/feature_extractor.py --audio_dir ./audio_samples --output features.csv --hf_token YOUR_HF_TOKEN
    python src/feature_extractor.py --audio_dir ./audio_samples --output features.csv --no_overlap
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torchaudio

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


# ─────────────────────────────────────────────
# 1. Duration & Sample Rate
# ─────────────────────────────────────────────


def get_duration_and_sr(wav_path: str) -> dict:
    """Return duration in seconds and sample rate.
    Compatible with both old and new torchaudio versions.
    """
    try:
        # torchaudio >= 0.9
        info = torchaudio.info(wav_path)
        num_frames = info.num_frames
        sample_rate = info.sample_rate
    except TypeError:
        # torchaudio < 0.9 returns (info, encoding)
        info, _ = torchaudio.info(wav_path)
        num_frames = info.length
        sample_rate = info.rate
    except AttributeError:
        # Fallback: use soundfile
        import soundfile as sf

        info = sf.info(wav_path)
        num_frames = info.frames
        sample_rate = info.samplerate
    duration = num_frames / sample_rate
    return {
        "duration_sec": round(duration, 3),
        "sample_rate_hz": sample_rate,
    }


# ─────────────────────────────────────────────
# 2. SNR Estimation (waveform-based, no reference)
# ─────────────────────────────────────────────


def estimate_snr(waveform: torch.Tensor, frame_len: int = 2048, hop_len: int = 512) -> float:
    """
    Estimate SNR using the percentile method:
      - Top 10% energy frames  → signal power estimate
      - Bottom 10% energy frames → noise power estimate
    Returns SNR in dB. Returns NaN if audio is silent.
    """
    waveform = waveform.mean(dim=0)  # mono
    frames = waveform.unfold(0, frame_len, hop_len)
    energies = (frames**2).mean(dim=1).numpy()

    if energies.max() < 1e-10:
        return float("nan")  # silent file

    noise_power = np.percentile(energies, 10)
    signal_power = np.percentile(energies, 90)

    if noise_power < 1e-10:
        noise_power = 1e-10  # avoid log(0)

    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(float(snr_db), 2)


# ─────────────────────────────────────────────
# 3. Silence Ratio
# ─────────────────────────────────────────────


def compute_silence_ratio(
    waveform: torch.Tensor,
    sr: int,
    frame_len_ms: int = 30,
    threshold_db: float = -40.0,
) -> float:
    """Fraction of 30 ms frames whose RMS energy is below threshold_db."""
    waveform = waveform.mean(dim=0)
    frame_len = int(sr * frame_len_ms / 1000)
    if frame_len == 0 or waveform.numel() < frame_len:
        return float("nan")

    frames = waveform.unfold(0, frame_len, frame_len)
    rms = (frames**2).mean(dim=1).sqrt()
    ref = rms.max().item()
    if ref < 1e-10:
        return 1.0  # fully silent

    rms_db = 20 * torch.log10(rms / ref + 1e-10)
    silence_ratio = (rms_db < threshold_db).float().mean().item()
    return round(silence_ratio, 4)


# ─────────────────────────────────────────────
# 4. Overlap Detection (pyannote/segmentation-3.0)
# ─────────────────────────────────────────────


def load_overlap_pipeline():
    """
    Load pyannote segmentation model for overlap detection.
    Requires a Hugging Face token with access to pyannote models.
    Set env var: HF_TOKEN=your_token
    Or pass --hf_token on the command line.
    """
    try:
        from pyannote.audio import Inference, Model

        hf_token = os.environ.get("HF_TOKEN", None)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token)
        model = model.to(device)
        inference = Inference(model, step=2.5)
        print("Pyannote segmentation pipeline loaded")
        return inference
    except Exception as e:
        print(f"[WARNING] Could not load pyannote pipeline: {e}")
        print("          overlap fields will be set to NaN.")
        return None


def _get_overlap_mask(pipeline_output) -> tuple[np.ndarray, object]:
    """Extract per-frame binary overlap mask from segmentation model output.

    The segmentation model outputs per-frame posteriors (n_frames, n_classes).
    Overlap = 2+ speakers active simultaneously.
    """
    posteriors = pipeline_output.data  # (n_frames, n_classes)
    n_classes = posteriors.shape[1]

    if n_classes <= 3:
        n_active = (posteriors > 0.5).astype(int).sum(axis=1)
    else:
        best_class = posteriors.argmax(axis=1)
        n_spk = 3
        n_active = np.where(
            best_class == 0,
            0,
            np.where(best_class <= n_spk, 1, 2),
        )

    overlap_mask = (n_active >= 2).ravel().astype(bool)
    return overlap_mask, pipeline_output.sliding_window


def compute_overlap(wav_path: str, pipeline, sample_rate: int, duration_sec: float) -> dict:
    """Compute overlap ratio and segment timestamps for CSV output."""
    if pipeline is None or duration_sec == 0:
        return {"overlap_ratio": float("nan"), "overlap_segments": float("nan")}
    try:
        output = pipeline(wav_path)
        overlap_mask, frames = _get_overlap_mask(output)
        overlap_ratio = float(overlap_mask.mean())

        segments = []
        in_overlap = False
        start_time = 0.0

        for i, is_overlap in enumerate(overlap_mask.tolist()):
            frame_start = frames[i].start
            if is_overlap and not in_overlap:
                start_time = frame_start
                in_overlap = True
            elif not is_overlap and in_overlap:
                start_s = int(round(start_time * sample_rate))
                end_s = int(round(frame_start * sample_rate))
                segments.append(f"{start_s}-{end_s}")
                in_overlap = False

        if in_overlap:
            start_s = int(round(start_time * sample_rate))
            end_s = int(round(frames[-1].end * sample_rate))
            segments.append(f"{start_s}-{end_s}")

        return {
            "overlap_ratio": round(overlap_ratio, 4),
            "overlap_segments": ";".join(segments) if segments else float("nan"),
        }
    except Exception as e:
        print(f"[WARNING] Overlap detection failed on {wav_path}: {e}")
        return {"overlap_ratio": float("nan"), "overlap_segments": float("nan")}


def extract_overlap_info(wav_path: str, pipeline, sample_rate: int = 16000) -> tuple[torch.Tensor, list]:
    """Returns frame-level overlap tensor and segment timestamps for the adapter.

    Used by preprocess.py to create .pt files for training.
    Output tensor shape: (T, 2) aligned to WavLM frames (hop=320 at 16kHz).
      Column 0: is_overlap (binary)
      Column 1: overlap confidence (binary, matches column 0)
    """
    info = torchaudio.info(wav_path)
    T = int(info.num_frames / info.sample_rate * sample_rate) // 320

    overlap_info = torch.zeros(T, 2)
    segments = []

    if pipeline is None:
        return overlap_info, segments

    try:
        output = pipeline(wav_path)
        overlap_mask, frames = _get_overlap_mask(output)

        # Map segmentation model frames → WavLM-aligned frames
        in_overlap = False
        start_time = 0.0

        for i, is_overlap in enumerate(overlap_mask.tolist()):
            if is_overlap:
                frame_start_sec = frames[i].start
                frame_end_sec = frames[i].end
                start_frame = max(0, int(frame_start_sec * sample_rate / 320))
                end_frame = min(T, int(frame_end_sec * sample_rate / 320))
                overlap_info[start_frame:end_frame, 0] = 1.0
                overlap_info[start_frame:end_frame, 1] = 1.0

            # Build segment list (in seconds)
            if is_overlap and not in_overlap:
                start_time = frames[i].start
                in_overlap = True
            elif not is_overlap and in_overlap:
                segments.append((start_time, frames[i].start))
                in_overlap = False

        if in_overlap:
            segments.append((start_time, frames[-1].end))

    except Exception as e:
        print(f"[WARNING] extract_overlap_info failed on {wav_path}: {e}")

    return overlap_info, segments


# ─────────────────────────────────────────────
# 5. SRMR (Reverberation, optional)
# ─────────────────────────────────────────────


def load_srmr_model(config: dict):
    """Validate SRMR is importable and return config."""
    try:
        from versa.utterance_metrics.srmr import srmr_metric  # noqa: F401

        print("SRMR ready")
        return config
    except Exception as e:
        print(f"[WARNING] SRMR not available: {e}")
        print("          srmr will be set to NaN.")
        return None


def compute_srmr(wav_path: str, srmr_model) -> float:
    """Compute SRMR score. Higher = less reverberation = better quality."""
    if srmr_model is None:
        return float("nan")
    try:
        import soundfile as sf
        from versa.utterance_metrics.srmr import srmr_metric

        audio, sr = sf.read(wav_path)
        score = srmr_metric(
            audio,
            sr,
            n_cochlear_filters=srmr_model.get("n_cochlear_filters", 23),
            low_freq=srmr_model.get("low_freq", 125),
            min_cf=srmr_model.get("min_cf", 4),
            max_cf=srmr_model.get("max_cf", 128),
            fast=srmr_model.get("fast", True),
            norm=srmr_model.get("norm", False),
        )
        return round(score["srmr"], 4)
    except Exception as e:
        print(f"[WARNING] SRMR failed on {wav_path}: {e}")
        return float("nan")


# ─────────────────────────────────────────────
# 6. F0 Variation (Praat)
# ─────────────────────────────────────────────


def compute_f0_variation(
    wav_path: str,
    min_pitch: float = 75.0,
    max_pitch: float = 500.0,
    time_step: float = 0.01,
) -> dict:
    """
    Extract F0 variation features using Praat via parselmouth.

    Returns:
        f0_mean_hz, f0_sd_hz, f0_min_hz, f0_max_hz,
        f0_range_hz, f0_range_st, f0_voiced_frac
    """
    empty = {
        "f0_mean_hz": float("nan"),
        "f0_sd_hz": float("nan"),
        "f0_min_hz": float("nan"),
        "f0_max_hz": float("nan"),
        "f0_range_hz": float("nan"),
        "f0_range_st": float("nan"),
        "f0_voiced_frac": float("nan"),
    }
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(
            time_step=time_step,
            pitch_floor=min_pitch,
            pitch_ceiling=max_pitch,
        )

        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        n_total = len(f0_values)
        n_voiced = len(voiced)

        if n_voiced < 2:
            return empty

        f0_mean = float(np.mean(voiced))
        f0_sd = float(np.std(voiced, ddof=1))
        f0_min = float(np.min(voiced))
        f0_max = float(np.max(voiced))
        range_hz = f0_max - f0_min
        range_st = 12.0 * np.log2(f0_max / f0_min) if f0_min > 0 else float("nan")
        voiced_frac = n_voiced / n_total if n_total > 0 else float("nan")

        return {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_sd_hz": round(f0_sd, 2),
            "f0_min_hz": round(f0_min, 2),
            "f0_max_hz": round(f0_max, 2),
            "f0_range_hz": round(range_hz, 2),
            "f0_range_st": round(range_st, 2),
            "f0_voiced_frac": round(voiced_frac, 4),
        }
    except Exception as e:
        print(f"[WARNING] F0 extraction failed on {wav_path}: {e}")
        return empty


# ─────────────────────────────────────────────
# 7. HNR (Praat)
# ─────────────────────────────────────────────


def compute_hnr(
    wav_path: str,
    min_pitch: float = 75.0,
    time_step: float = 0.01,
    silence_threshold: float = 0.1,
    periods_per_window: float = 1.0,
) -> float:
    """
    Compute mean HNR (dB) using Praat via parselmouth.
    Higher = more harmonic (clearer voice). Returns NaN on failure.
    """
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        harmonicity = snd.to_harmonicity_cc(
            time_step=time_step,
            minimum_pitch=min_pitch,
            silence_threshold=silence_threshold,
            periods_per_window=periods_per_window,
        )
        hnr_values = harmonicity.values[0]
        voiced_hnr = hnr_values[hnr_values != -200]  # -200 = unvoiced sentinel

        if len(voiced_hnr) == 0:
            return float("nan")

        return round(float(np.mean(voiced_hnr)), 2)
    except Exception as e:
        print(f"[WARNING] HNR extraction failed on {wav_path}: {e}")
        return float("nan")


# ─────────────────────────────────────────────
# 8. Shimmer (Praat)
# ─────────────────────────────────────────────


def compute_shimmer(
    wav_path: str,
    min_pitch: float = 75.0,
    max_pitch: float = 500.0,
) -> float:
    """
    Compute local shimmer (%) using Praat via parselmouth.
    Measures cycle-to-cycle variation in amplitude.
    Lower = more stable voice.
    """
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", min_pitch, max_pitch
        )
        shimmer = parselmouth.praat.call(
            [snd, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
        return round(float(shimmer) * 100, 4)
    except Exception as e:
        print(f"[WARNING] Shimmer extraction failed on {wav_path}: {e}")
        return float("nan")


# ─────────────────────────────────────────────
# 9. Jitter (Praat)
# ─────────────────────────────────────────────


def compute_jitter(
    wav_path: str,
    min_pitch: float = 75.0,
    max_pitch: float = 500.0,
) -> dict:
    """
    Compute local jitter and RAP jitter (%) using Praat via parselmouth.
    Measures cycle-to-cycle variation in the fundamental period.
    Lower = more stable pitch.
    """
    _nan = {"jitter_local_pct": float("nan"), "jitter_rap_pct": float("nan")}
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", min_pitch, max_pitch
        )

        jitter_local = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3,
        )
        jitter_rap = parselmouth.praat.call(
            point_process,
            "Get jitter (rap)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        return {
            "jitter_local_pct": round(float(jitter_local) * 100, 4),
            "jitter_rap_pct": round(float(jitter_rap) * 100, 4),
        }
    except Exception as e:
        print(f"[WARNING] Jitter extraction failed on {wav_path}: {e}")
        return _nan


# ─────────────────────────────────────────────
# 10. Pause Patterns (Praat)
# ─────────────────────────────────────────────


def compute_praat_pause_patterns(wav_path: str, min_pause_dur: float = 0.3) -> dict:
    """
    Detect inter-speech pauses using Praat's "To TextGrid (silences)".
    The silence threshold adapts to each recording (25 dB below max intensity).
    """
    _nan = {
        "praat_pause_count": float("nan"),
        "praat_pause_rate_per_min": float("nan"),
        "praat_mean_pause_dur_sec": float("nan"),
        "praat_total_pause_dur_sec": float("nan"),
        "praat_pause_to_speech_ratio": float("nan"),
    }
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        duration = snd.duration
        if duration < 0.1:
            return _nan

        intensity = snd.to_intensity(minimum_pitch=50.0, subtract_mean=True)

        tg = parselmouth.praat.call(
            intensity,
            "To TextGrid (silences)",
            -25,             # silence threshold (dB below max)
            min_pause_dur,   # minimum silent interval (s)
            0.1,             # minimum sounding interval (s)
            "silent",
            "sounding",
        )

        n_intervals = parselmouth.praat.call(tg, "Get number of intervals", 1)

        pauses_sec = []
        for i in range(1, n_intervals + 1):
            label = parselmouth.praat.call(tg, "Get label of interval", 1, i)
            if label == "silent":
                t0 = parselmouth.praat.call(tg, "Get start time of interval", 1, i)
                t1 = parselmouth.praat.call(tg, "Get end time of interval", 1, i)
                dur = t1 - t0
                if dur >= min_pause_dur:
                    pauses_sec.append(dur)

        n_pauses = len(pauses_sec)
        total_pause = sum(pauses_sec)
        pause_rate = (n_pauses / duration) * 60 if duration > 0 else 0.0
        mean_pause = float(np.mean(pauses_sec)) if n_pauses > 0 else 0.0
        p2s_ratio = total_pause / duration if duration > 0 else 0.0

        return {
            "praat_pause_count": n_pauses,
            "praat_pause_rate_per_min": round(float(pause_rate), 3),
            "praat_mean_pause_dur_sec": round(float(mean_pause), 4),
            "praat_total_pause_dur_sec": round(float(total_pause), 4),
            "praat_pause_to_speech_ratio": round(float(p2s_ratio), 4),
        }
    except Exception as e:
        print(f"[WARNING] Praat pause patterns failed on {wav_path}: {e}")
        return _nan


# ─────────────────────────────────────────────
# 11. Speaking Rate (Praat, de Jong & Wempe 2009)
# ─────────────────────────────────────────────


def compute_praat_speaking_rate(wav_path: str) -> dict:
    """
    Estimate speaking / articulation rate via Praat intensity analysis.

    Method (de Jong & Wempe 2009):
      1. Compute intensity contour.
      2. Detect syllable nuclei as local-max peaks above adaptive threshold,
         separated by dips of at least 2 dB.
      3. Divide by total duration → speaking rate.
      4. Divide by speech-only duration → articulation rate.
    """
    _nan = {
        "praat_speaking_rate_syl_sec": float("nan"),
        "praat_articulation_rate_syl_sec": float("nan"),
    }
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        duration = snd.duration
        if duration < 0.1:
            return _nan

        intensity = snd.to_intensity(minimum_pitch=50.0, subtract_mean=True)
        int_values = intensity.values[0]

        # Speech duration via Praat silence segmentation
        try:
            tg = parselmouth.praat.call(
                intensity,
                "To TextGrid (silences)",
                -25, 0.1, 0.1,
                "silent",
                "sounding",
            )
            n_int = parselmouth.praat.call(tg, "Get number of intervals", 1)
            speech_dur = 0.0
            for i in range(1, n_int + 1):
                if parselmouth.praat.call(tg, "Get label of interval", 1, i) == "sounding":
                    t0 = parselmouth.praat.call(tg, "Get start time of interval", 1, i)
                    t1 = parselmouth.praat.call(tg, "Get end time of interval", 1, i)
                    speech_dur += t1 - t0
        except Exception:
            # Fallback: frames above threshold
            times = intensity.xs()
            max_int = float(np.max(int_values))
            dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
            speech_dur = float(np.sum(int_values > (max_int - 25)) * dt)

        # Syllable-nuclei detection (de Jong & Wempe style)
        max_int = float(np.max(int_values))
        threshold = max_int - 25.0
        min_dip = 2.0

        nuclei = 0
        n = len(int_values)
        for i in range(1, n - 1):
            v = int_values[i]
            if v <= threshold:
                continue
            if v < int_values[i - 1] or v <= int_values[i + 1]:
                continue  # not a local maximum
            left_min = float(np.min(int_values[max(0, i - 10) : i]))
            right_min = float(np.min(int_values[i + 1 : min(n, i + 11)]))
            if (v - left_min >= min_dip) or (v - right_min >= min_dip):
                nuclei += 1

        speaking_rate = nuclei / duration if duration > 0 else float("nan")
        articulation_rate = nuclei / speech_dur if speech_dur > 0 else float("nan")

        return {
            "praat_speaking_rate_syl_sec": round(float(speaking_rate), 3),
            "praat_articulation_rate_syl_sec": round(float(articulation_rate), 3),
        }
    except Exception as e:
        print(f"[WARNING] Praat speaking rate failed on {wav_path}: {e}")
        return _nan


# ─────────────────────────────────────────────
# Main Extractor
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

COLUMN_ORDER = [
    "filename", "filepath",
    "duration_sec", "sample_rate_hz",
    "snr_db", "silence_ratio", "overlap_ratio", "overlap_segments",
    "srmr",
    "f0_mean_hz", "f0_sd_hz",
    "f0_min_hz", "f0_max_hz",
    "f0_range_hz", "f0_range_st",
    "f0_voiced_frac", "hnr_db",
    "shimmer_pct",
    "jitter_local_pct", "jitter_rap_pct",
    "praat_pause_count", "praat_pause_rate_per_min",
    "praat_mean_pause_dur_sec", "praat_total_pause_dur_sec",
    "praat_pause_to_speech_ratio",
    "praat_speaking_rate_syl_sec", "praat_articulation_rate_syl_sec",
]


def extract_features(wav_path: str, overlap_pipeline, srmr_model) -> dict:
    """Extract all features for a single audio file."""
    filename = os.path.basename(wav_path)
    result = {"filename": filename, "filepath": wav_path}

    # Duration & SR
    try:
        meta = get_duration_and_sr(wav_path)
        result.update(meta)
    except Exception as e:
        print(f"    [ERROR] Could not read file: {e}")
        result.update({"duration_sec": float("nan"), "sample_rate_hz": float("nan")})
        return result

    # Load waveform
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f"    [ERROR] Could not load waveform: {e}")
        return result  # remaining fields will be NaN in the DataFrame

    # SNR
    result["snr_db"] = estimate_snr(waveform)

    # Silence Ratio
    result["silence_ratio"] = compute_silence_ratio(waveform, sr)

    # Overlap (ratio + segments)
    result.update(compute_overlap(wav_path, overlap_pipeline, sr, result.get("duration_sec", 0)))

    # SRMR (Reverberation)
    result["srmr"] = compute_srmr(wav_path, srmr_model)

    # F0 Variation (Praat)
    result.update(compute_f0_variation(wav_path))

    # HNR (Praat)
    result["hnr_db"] = compute_hnr(wav_path)

    # Shimmer (Praat)
    result["shimmer_pct"] = compute_shimmer(wav_path)

    # Jitter (Praat)
    result.update(compute_jitter(wav_path))

    # Pause Patterns (Praat)
    result.update(compute_praat_pause_patterns(wav_path))

    # Speaking Rate (Praat)
    result.update(compute_praat_speaking_rate(wav_path))

    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Feature Extractor")
    parser.add_argument("--audio_dir", type=str, default="./audio_samples", help="Directory containing audio files")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV path")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token for pyannote (or set HF_TOKEN env var)")
    parser.add_argument("--no_overlap", action="store_true", help="Skip overlap detection (faster, no pyannote needed)")
    parser.add_argument("--no_srmr", action="store_true", help="Skip SRMR computation (no VERSA needed)")
    parser.add_argument("--srmr_max_cf", type=int, default=128, help="SRMR max center frequency")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set HF token if provided via CLI
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Collect audio files
    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(f"[ERROR] Directory not found: {audio_dir}")
        return

    audio_files = sorted(
        p
        for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True)
    )

    if not audio_files:
        print(f"[ERROR] No audio files found in: {audio_dir}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) in '{audio_dir}'")

    # Load pipelines
    overlap_pipeline = None
    if not args.no_overlap:
        print("Loading pyannote overlap detection pipeline...")
        overlap_pipeline = load_overlap_pipeline()

    srmr_model = None
    if not args.no_srmr:
        srmr_config = {"max_cf": args.srmr_max_cf, "fast": True, "norm": False}
        print("Loading SRMR model...")
        srmr_model = load_srmr_model(srmr_config)

    # Extract features
    print(f"\nExtracting features...\n" + "-" * 50)
    records = []
    for wav_path in audio_files:
        print(f"  Processing: {os.path.basename(wav_path)}")
        record = extract_features(wav_path, overlap_pipeline, srmr_model)
        records.append(record)

    # Save to CSV
    df = pd.DataFrame(records)
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[[c for c in COLUMN_ORDER if c in df.columns] + extra_cols]

    df.to_csv(args.output, index=False)
    print("\n" + "=" * 50)
    print(f"Done! Features saved to: {args.output}")
    print(f"Total files processed: {len(df)}")
    print("\nPreview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
