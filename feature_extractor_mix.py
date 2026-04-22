# -*- coding: utf-8 -*-
"""
Feature_Extractor_MIX — standalone Python script
Step 1 of Speech Quality Assessment Pipeline

Extracts the following features from audio files:
  duration_sec, sample_rate_hz, snr_db, silence_ratio,
  overlap_ratio, overlap_segments, srmr,
  f0_mean_hz, f0_sd_hz, f0_min_hz, f0_max_hz, f0_range_hz, f0_range_st,
  f0_voiced_frac, hnr_db, shimmer_pct,
  jitter_local_pct, jitter_rap_pct,
  praat_speaking_rate_syl_sec, praat_articulation_rate_syl_sec,
  praat_pause_count, praat_pause_rate_per_min,
  praat_mean_pause_dur_sec, praat_total_pause_dur_sec,
  praat_pause_to_speech_ratio

Output: features_mix2.csv

Dependencies (install manually before running):
  pip install torchaudio pyannote.audio pandas numpy torch soundfile praat-parselmouth
  # VERSA (for SRMR):
  #   git clone https://github.com/wavlab-speech/versa.git && cd versa && pip install .
  #   git clone https://github.com/shimhz/SRMRpy.git && cd SRMRpy && pip install .
"""

# =============================================================================
# 1. Imports & Setup
# =============================================================================
import os
import sys
import glob
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Add VERSA to path if cloned locally (adjust path as needed)
# sys.path.insert(0, './versa')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name())

torch.manual_seed(73)
np.random.seed(73)
random.seed(73)

SAMPLE_RATE = 16000
print('Imports complete')

# =============================================================================
# 2. Configuration
# =============================================================================
# Directory containing audio files to process (mix_clean WAVs from Libri2Mix)
AUDIO_DIR  = '/content/data/Libri2Mix/wav16k/min/test/mix_clean/'
OUTPUT_CSV = 'features_mix2.csv'

# Libri2Mix root — used only when OVERLAP == 'min_max_vad'
# The mix_clean filename must also exist under s1/ and s2/ subdirectories
LIBRI2MIX_ROOT = '/content/data/Libri2Mix/wav16k/min/test'

# Hugging Face token — required only when OVERLAP == 'pyannote'
HF_TOKEN = ''  # paste your token here

# Overlap detection method:
#   'min_max_vad' : Silero VAD on ground-truth s1/s2 stems (Libri2Mix structure)
#   'pyannote'    : pyannote/segmentation-3.0 (needs HF_TOKEN)
#   'none'        : skip overlap detection (overlap columns will be NaN)
OVERLAP = 'min_max_vad'

# SRMR configuration
SRMR_CONFIG = {'max_cf': 128, 'fast': True, 'norm': False}

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print('Logged in to HuggingFace')

# =============================================================================
# 3. Feature Extraction Functions
# =============================================================================

# ── 3.1 Duration & Sample Rate ───────────────────────────────────────────────
def get_duration_and_sr(wav_path: str) -> dict:
    try:
        info = torchaudio.info(wav_path)
        num_frames  = info.num_frames
        sample_rate = info.sample_rate
    except TypeError:
        info, _ = torchaudio.info(wav_path)
        num_frames  = info.length
        sample_rate = info.rate
    except AttributeError:
        info = sf.info(wav_path)
        num_frames  = info.frames
        sample_rate = info.samplerate
    return {
        'duration_sec':   round(num_frames / sample_rate, 3),
        'sample_rate_hz': sample_rate,
    }


# ── 3.2 SNR ──────────────────────────────────────────────────────────────────
def estimate_snr(waveform: torch.Tensor, frame_len: int = 2048, hop_len: int = 512) -> float:
    waveform = waveform.mean(dim=0)
    frames   = waveform.unfold(0, frame_len, hop_len)
    energies = (frames ** 2).mean(dim=1).numpy()
    if energies.max() < 1e-10:
        return float('nan')
    noise_power  = np.percentile(energies, 10)
    signal_power = np.percentile(energies, 90)
    if noise_power < 1e-10:
        noise_power = 1e-10
    return round(float(10 * np.log10(signal_power / noise_power)), 2)


# ── 3.3 Silence Ratio ────────────────────────────────────────────────────────
def compute_silence_ratio(
    waveform: torch.Tensor,
    sr: int,
    frame_len_ms: int = 30,
    threshold_db: float = -40.0,
) -> float:
    waveform  = waveform.mean(dim=0)
    frame_len = int(sr * frame_len_ms / 1000)
    if frame_len == 0 or waveform.numel() < frame_len:
        return float('nan')
    frames = waveform.unfold(0, frame_len, frame_len)
    rms    = (frames ** 2).mean(dim=1).sqrt()
    ref    = rms.max().item()
    if ref < 1e-10:
        return 1.0
    rms_db = 20 * torch.log10(rms / ref + 1e-10)
    return round((rms_db < threshold_db).float().mean().item(), 4)


# ── 3.4 Overlap Detection ────────────────────────────────────────────────────
def load_overlap_pipeline():
    """Load pyannote overlap pipeline. Only called when OVERLAP == 'pyannote'."""
    try:
        from pyannote.audio import Model, Inference
        hf_token = os.environ.get('HF_TOKEN', None)
        model     = Model.from_pretrained('pyannote/segmentation-3.0', use_auth_token=hf_token)
        model     = model.to(device)
        inference = Inference(model, step=2.5)
        print('Pyannote pipeline loaded')
        return inference
    except Exception as e:
        print(f'[WARNING] Could not load pyannote pipeline: {e}')
        return None


def compute_overlap_pyannote(wav_path: str, pipeline, sample_rate: int, duration_sec: float) -> dict:
    if pipeline is None or duration_sec == 0:
        return {'overlap_ratio': float('nan'), 'overlap_segments': float('nan')}
    try:
        output    = pipeline(wav_path)
        posteriors = output.data
        n_classes  = posteriors.shape[1]
        if n_classes <= 3:
            n_active = (posteriors > 0.5).astype(int).sum(axis=1)
        else:
            best_class = posteriors.argmax(axis=1)
            n_active   = np.where(best_class == 0, 0,
                         np.where(best_class <= 3, 1, 2))
        overlap_mask  = (n_active >= 2).ravel().astype(bool)
        overlap_ratio = float(overlap_mask.mean())
        frames        = output.sliding_window
        segments, in_overlap, start_time = [], False, 0.0
        for i, is_overlap in enumerate(overlap_mask.tolist()):
            frame_start = frames[i].start
            if is_overlap and not in_overlap:
                start_time, in_overlap = frame_start, True
            elif not is_overlap and in_overlap:
                segments.append(f'{int(round(start_time*sample_rate))}-{int(round(frame_start*sample_rate))}')
                in_overlap = False
        if in_overlap:
            segments.append(f'{int(round(start_time*sample_rate))}-{int(round(frames[-1].end*sample_rate))}')
        return {
            'overlap_ratio':    round(overlap_ratio, 4),
            'overlap_segments': ';'.join(segments) if segments else float('nan'),
        }
    except Exception as e:
        print(f'[WARNING] Overlap detection failed on {wav_path}: {e}')
        return {'overlap_ratio': float('nan'), 'overlap_segments': float('nan')}


def load_vad_model():
    """Load Silero VAD. Only called when OVERLAP == 'min_max_vad'."""
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
    )
    get_speech_timestamps, _, read_audio, *_ = utils
    print('Silero VAD loaded')
    return vad_model, get_speech_timestamps


def get_overlap_segments(wav1, wav2, sr, vad_model, get_speech_timestamps, min_overlap_sec=0.1):
    """
    Detect overlapping speech regions between two speakers using Silero VAD.
    Returns a list of dicts with start/end sample indices and duration.
    """
    min_overlap_samples = int(min_overlap_sec * sr)
    t1 = torch.from_numpy(wav1).float()
    t2 = torch.from_numpy(wav2).float()
    segs1 = get_speech_timestamps(t1, vad_model, sampling_rate=sr, return_seconds=False)
    segs2 = get_speech_timestamps(t2, vad_model, sampling_rate=sr, return_seconds=False)
    overlaps = []
    for s1 in segs1:
        for s2 in segs2:
            overlap_start = max(s1['start'], s2['start'])
            overlap_end   = min(s1['end'],   s2['end'])
            if overlap_end - overlap_start < min_overlap_samples:
                continue
            overlap_wav = wav1[overlap_start:overlap_end]
            overlap_t   = torch.from_numpy(overlap_wav).float()
            speech_in   = get_speech_timestamps(overlap_t, vad_model, sampling_rate=sr, return_seconds=False)
            silence_segs = []
            prev_end = 0
            for seg in speech_in:
                if seg['start'] > prev_end:
                    silence_segs.append((overlap_start + prev_end, overlap_start + seg['start']))
                prev_end = seg['end']
            overlap_len = overlap_end - overlap_start
            if prev_end < overlap_len:
                silence_segs.append((overlap_start + prev_end, overlap_end))
            overlaps.append({
                'start':            overlap_start,
                'end':              overlap_end,
                'start_sec':        round(overlap_start / sr, 3),
                'end_sec':          round(overlap_end   / sr, 3),
                'duration_sec':     round((overlap_end - overlap_start) / sr, 3),
                'silence_segments': silence_segs,
            })
    return overlaps


def compute_overlap_vad(wav_path: str, sr: int, vad_model, get_speech_timestamps) -> dict:
    """Wrapper around get_overlap_segments for use inside extract_features."""
    try:
        filename = os.path.basename(wav_path)
        s1_path  = os.path.join(LIBRI2MIX_ROOT, 's1', filename)
        s2_path  = os.path.join(LIBRI2MIX_ROOT, 's2', filename)
        if not os.path.exists(s1_path) or not os.path.exists(s2_path):
            return {'overlap_ratio': float('nan'), 'overlap_segments': float('nan')}
        wav1, _ = sf.read(s1_path, dtype='float32')
        wav2, _ = sf.read(s2_path, dtype='float32')
        if wav1.ndim > 1: wav1 = wav1.mean(axis=1)
        if wav2.ndim > 1: wav2 = wav2.mean(axis=1)
        overlaps      = get_overlap_segments(wav1, wav2, sr, vad_model, get_speech_timestamps)
        total_samples = max(len(wav1), len(wav2))
        if total_samples == 0 or not overlaps:
            return {'overlap_ratio': 0.0, 'overlap_segments': float('nan')}
        overlap_samples = sum(o['end'] - o['start'] for o in overlaps)
        return {
            'overlap_ratio':    round(overlap_samples / total_samples, 4),
            'overlap_segments': ';'.join(f"{o['start']}-{o['end']}" for o in overlaps),
        }
    except Exception as e:
        print(f'[WARNING] VAD overlap failed on {wav_path}: {e}')
        return {'overlap_ratio': float('nan'), 'overlap_segments': float('nan')}


# ── 3.5 SRMR ─────────────────────────────────────────────────────────────────
def load_srmr_model(config: dict):
    try:
        from versa.utterance_metrics.srmr import srmr_metric  # noqa: F401
        print('SRMR ready')
        return config
    except Exception as e:
        print(f'[WARNING] SRMR not available: {e} — srmr will be NaN')
        return None


def compute_srmr(wav_path: str, srmr_model) -> float:
    if srmr_model is None:
        return float('nan')
    try:
        from versa.utterance_metrics.srmr import srmr_metric
        audio, sr = sf.read(wav_path)
        score = srmr_metric(
            audio, sr,
            n_cochlear_filters=srmr_model.get('n_cochlear_filters', 23),
            low_freq=srmr_model.get('low_freq', 125),
            min_cf=srmr_model.get('min_cf', 4),
            max_cf=srmr_model.get('max_cf', 128),
            fast=srmr_model.get('fast', True),
            norm=srmr_model.get('norm', False),
        )
        return round(score['srmr'], 4)
    except Exception as e:
        print(f'[WARNING] SRMR failed on {wav_path}: {e}')
        return float('nan')


# ── 3.6 Praat features ───────────────────────────────────────────────────────
import parselmouth


def compute_f0_variation(
    wav_path: str,
    min_pitch: float = 75.0,
    max_pitch: float = 500.0,
    time_step: float = 0.01,
) -> dict:
    empty = {k: float('nan') for k in (
        'f0_mean_hz', 'f0_sd_hz', 'f0_min_hz', 'f0_max_hz',
        'f0_range_hz', 'f0_range_st', 'f0_voiced_frac',
    )}
    try:
        snd      = parselmouth.Sound(wav_path)
        pitch    = snd.to_pitch(time_step=time_step, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
        f0_values = pitch.selected_array['frequency']
        voiced    = f0_values[f0_values > 0]
        n_total, n_voiced = len(f0_values), len(voiced)
        if n_voiced < 2:
            return empty
        f0_min, f0_max = float(np.min(voiced)), float(np.max(voiced))
        range_st = 12.0 * np.log2(f0_max / f0_min) if f0_min > 0 else float('nan')
        return {
            'f0_mean_hz':     round(float(np.mean(voiced)), 2),
            'f0_sd_hz':       round(float(np.std(voiced, ddof=1)), 2),
            'f0_min_hz':      round(f0_min, 2),
            'f0_max_hz':      round(f0_max, 2),
            'f0_range_hz':    round(f0_max - f0_min, 2),
            'f0_range_st':    round(float(range_st), 2),
            'f0_voiced_frac': round(n_voiced / n_total, 4) if n_total > 0 else float('nan'),
        }
    except Exception as e:
        print(f'[WARNING] F0 failed on {wav_path}: {e}')
        return empty


def compute_hnr(
    wav_path: str,
    min_pitch: float = 75.0,
    time_step: float = 0.01,
    silence_threshold: float = 0.1,
    periods_per_window: float = 1.0,
) -> float:
    try:
        snd = parselmouth.Sound(wav_path)
        harmonicity = snd.to_harmonicity_cc(
            time_step=time_step, minimum_pitch=min_pitch,
            silence_threshold=silence_threshold, periods_per_window=periods_per_window,
        )
        hnr_values = harmonicity.values[0]
        voiced_hnr = hnr_values[hnr_values != -200]
        if len(voiced_hnr) == 0:
            return float('nan')
        return round(float(np.mean(voiced_hnr)), 2)
    except Exception as e:
        print(f'[WARNING] HNR failed on {wav_path}: {e}')
        return float('nan')


def compute_shimmer(wav_path: str, min_pitch: float = 75.0, max_pitch: float = 500.0) -> float:
    try:
        snd = parselmouth.Sound(wav_path)
        pp  = parselmouth.praat.call(snd, 'To PointProcess (periodic, cc)', min_pitch, max_pitch)
        shimmer = parselmouth.praat.call([snd, pp], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return round(float(shimmer) * 100, 4)
    except Exception as e:
        print(f'[WARNING] Shimmer failed on {wav_path}: {e}')
        return float('nan')


def compute_jitter(wav_path: str, min_pitch: float = 75.0, max_pitch: float = 500.0) -> dict:
    _nan = {'jitter_local_pct': float('nan'), 'jitter_rap_pct': float('nan')}
    try:
        snd = parselmouth.Sound(wav_path)
        pp  = parselmouth.praat.call(snd, 'To PointProcess (periodic, cc)', min_pitch, max_pitch)
        jitter_local = parselmouth.praat.call(pp, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = parselmouth.praat.call(pp, 'Get jitter (rap)',   0, 0, 0.0001, 0.02, 1.3)
        return {
            'jitter_local_pct': round(float(jitter_local) * 100, 4),
            'jitter_rap_pct':   round(float(jitter_rap)   * 100, 4),
        }
    except Exception as e:
        print(f'[WARNING] Jitter failed on {wav_path}: {e}')
        return _nan


def compute_praat_pause_patterns(wav_path: str, min_pause_dur: float = 0.3) -> dict:
    _nan = {
        'praat_pause_count': float('nan'), 'praat_pause_rate_per_min': float('nan'),
        'praat_mean_pause_dur_sec': float('nan'), 'praat_total_pause_dur_sec': float('nan'),
        'praat_pause_to_speech_ratio': float('nan'),
    }
    try:
        snd      = parselmouth.Sound(wav_path)
        duration = snd.duration
        if duration < 0.1:
            return _nan
        intensity = snd.to_intensity(minimum_pitch=50.0, subtract_mean=True)
        tg = parselmouth.praat.call(
            intensity, 'To TextGrid (silences)', -25, min_pause_dur, 0.1, 'silent', 'sounding'
        )
        n_intervals = parselmouth.praat.call(tg, 'Get number of intervals', 1)
        pauses_sec = []
        for i in range(1, n_intervals + 1):
            if parselmouth.praat.call(tg, 'Get label of interval', 1, i) == 'silent':
                t0 = parselmouth.praat.call(tg, 'Get start time of interval', 1, i)
                t1 = parselmouth.praat.call(tg, 'Get end time of interval', 1, i)
                if t1 - t0 >= min_pause_dur:
                    pauses_sec.append(t1 - t0)
        n_pauses, total_pause = len(pauses_sec), sum(pauses_sec)
        return {
            'praat_pause_count':           n_pauses,
            'praat_pause_rate_per_min':    round((n_pauses / duration) * 60, 3) if duration > 0 else 0.0,
            'praat_mean_pause_dur_sec':    round(float(np.mean(pauses_sec)), 4) if n_pauses > 0 else 0.0,
            'praat_total_pause_dur_sec':   round(float(total_pause), 4),
            'praat_pause_to_speech_ratio': round(total_pause / duration, 4) if duration > 0 else 0.0,
        }
    except Exception as e:
        print(f'[WARNING] Pause patterns failed on {wav_path}: {e}')
        return _nan


def compute_praat_speaking_rate(wav_path: str) -> dict:
    _nan = {'praat_speaking_rate_syl_sec': float('nan'), 'praat_articulation_rate_syl_sec': float('nan')}
    try:
        snd      = parselmouth.Sound(wav_path)
        duration = snd.duration
        if duration < 0.1:
            return _nan
        intensity  = snd.to_intensity(minimum_pitch=50.0, subtract_mean=True)
        int_values = intensity.values[0]
        try:
            tg = parselmouth.praat.call(intensity, 'To TextGrid (silences)', -25, 0.1, 0.1, 'silent', 'sounding')
            n_int = parselmouth.praat.call(tg, 'Get number of intervals', 1)
            speech_dur = sum(
                parselmouth.praat.call(tg, 'Get end time of interval', 1, i) -
                parselmouth.praat.call(tg, 'Get start time of interval', 1, i)
                for i in range(1, n_int + 1)
                if parselmouth.praat.call(tg, 'Get label of interval', 1, i) == 'sounding'
            )
        except Exception:
            times      = intensity.xs()
            dt         = (times[-1] - times[0]) / max(len(times) - 1, 1)
            speech_dur = float(np.sum(int_values > (float(np.max(int_values)) - 25)) * dt)
        max_int   = float(np.max(int_values))
        threshold = max_int - 25.0
        min_dip   = 2.0
        n         = len(int_values)
        nuclei    = 0
        for i in range(1, n - 1):
            v = int_values[i]
            if v <= threshold or v < int_values[i - 1] or v <= int_values[i + 1]:
                continue
            left_min  = float(np.min(int_values[max(0, i - 10): i]))
            right_min = float(np.min(int_values[i + 1: min(n, i + 11)]))
            if (v - left_min >= min_dip) or (v - right_min >= min_dip):
                nuclei += 1
        return {
            'praat_speaking_rate_syl_sec':     round(nuclei / duration,   3) if duration   > 0 else float('nan'),
            'praat_articulation_rate_syl_sec': round(nuclei / speech_dur, 3) if speech_dur > 0 else float('nan'),
        }
    except Exception as e:
        print(f'[WARNING] Speaking rate failed on {wav_path}: {e}')
        return _nan


# ── 3.7 Main per-file extractor ───────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}

COLUMN_ORDER = [
    'filename', 'filepath',
    'duration_sec', 'sample_rate_hz',
    'snr_db', 'silence_ratio', 'overlap_ratio', 'overlap_segments',
    'srmr',
    'f0_mean_hz', 'f0_sd_hz', 'f0_min_hz', 'f0_max_hz',
    'f0_range_hz', 'f0_range_st', 'f0_voiced_frac',
    'hnr_db', 'shimmer_pct',
    'jitter_local_pct', 'jitter_rap_pct',
    'praat_pause_count', 'praat_pause_rate_per_min',
    'praat_mean_pause_dur_sec', 'praat_total_pause_dur_sec',
    'praat_pause_to_speech_ratio',
    'praat_speaking_rate_syl_sec', 'praat_articulation_rate_syl_sec',
]


def extract_features(wav_path: str, overlap_handle, srmr_model) -> dict:
    """
    Extract all features for a single audio file.

    overlap_handle:
        OVERLAP == 'min_max_vad'  → (vad_model, get_speech_timestamps)
        OVERLAP == 'pyannote'     → pyannote inference pipeline
        OVERLAP == 'none'         → None
    """
    filename = os.path.basename(wav_path)
    result   = {'filename': filename, 'filepath': wav_path}

    try:
        result.update(get_duration_and_sr(wav_path))
    except Exception as e:
        print(f'  [ERROR] Could not read file: {e}')
        result.update({'duration_sec': float('nan'), 'sample_rate_hz': float('nan')})
        return result

    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f'  [ERROR] Could not load waveform: {e}')
        return result

    result['snr_db']        = estimate_snr(waveform)
    result['silence_ratio'] = compute_silence_ratio(waveform, sr)

    if OVERLAP == 'min_max_vad' and overlap_handle is not None:
        vad_model_, get_ts_ = overlap_handle
        result.update(compute_overlap_vad(wav_path, sr, vad_model_, get_ts_))
    elif OVERLAP == 'pyannote' and overlap_handle is not None:
        result.update(compute_overlap_pyannote(wav_path, overlap_handle, sr, result.get('duration_sec', 0)))
    else:
        result['overlap_ratio']    = float('nan')
        result['overlap_segments'] = float('nan')

    result['srmr']        = compute_srmr(wav_path, srmr_model)
    result.update(compute_f0_variation(wav_path))
    result['hnr_db']      = compute_hnr(wav_path)
    result['shimmer_pct'] = compute_shimmer(wav_path)
    result.update(compute_jitter(wav_path))
    result.update(compute_praat_pause_patterns(wav_path))
    result.update(compute_praat_speaking_rate(wav_path))

    return result


# =============================================================================
# 4. Main
# =============================================================================
if __name__ == '__main__':
    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(f'Directory not found: {AUDIO_DIR}')

    audio_files = sorted(
        p for ext in SUPPORTED_EXTENSIONS
        for p in glob.glob(os.path.join(AUDIO_DIR, f'**/*{ext}'), recursive=True)
    )
    if not audio_files:
        raise FileNotFoundError(f'No supported audio files found in: {AUDIO_DIR}')
    print(f'Found {len(audio_files)} audio file(s) in {AUDIO_DIR!r}\n')

    # Load overlap model
    overlap_handle = None
    if OVERLAP == 'min_max_vad':
        overlap_handle = load_vad_model()   # returns (vad_model, get_speech_timestamps)
        print('Overlap method: Silero VAD (min_max_vad)')
    elif OVERLAP == 'pyannote':
        overlap_handle = load_overlap_pipeline()
        print('Overlap method: pyannote')
    else:
        print('Overlap method: none')

    srmr_model = load_srmr_model(SRMR_CONFIG)

    print('\nExtracting features...\n' + '-' * 50)
    records = []
    for wav_path in tqdm(audio_files, desc='Extracting'):
        records.append(extract_features(wav_path, overlap_handle, srmr_model))

    df         = pd.DataFrame(records)
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    df         = df[COLUMN_ORDER + extra_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print('\n' + '=' * 50)
    print(f'Done! Features saved to: {OUTPUT_CSV}')
    print(f'   Overlap method : {OVERLAP}')
    print(f'   Files processed: {len(df)}')
    print(df.head())
