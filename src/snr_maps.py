"""snr_maps.py — DENSE, oracle SNR supervision targets from the clean Libri2Mix stems.

WHY (the strategic upgrade)
---------------------------
A single global SNR number is the energy-weighted aggregate of a DENSE
per-time-frequency local-SNR field. The speech-SEPARATION literature has
directly-supervised that field for a decade as the Ideal Ratio Mask:

    IRM(t,f) = ( SNR(t,f) / (SNR(t,f) + 1) )^0.5
             = ( |S(t,f)|^2 / (|S(t,f)|^2 + |N(t,f)|^2) )^0.5

(Wang, Narayanan, Wang, TASLP 2014; survey arXiv:1708.07524). Because Libri2Mix
ships the CLEAN s1 (target) and s2 (interferer) stems and the mixture is exactly
mix_clean = s1 + s2 (verified: max|mix-(s1+s2)|=3e-5, corr(mix-s1,s2)=1.0 — see
clean_features.py), we can compute these targets EXACTLY (oracle), not estimate
them. This converts the 2D quality-map from a category error into a directly
supervised, novel contribution: the audio-quality GLaMM/Kosmos-2, where the
generated text is tied to oracle dense maps + the abstention head.

This module builds two oracle targets per clip:

  (1) snr_timeline  (T,)        — per-FRAME instantaneous target-vs-interferer SNR,
                                   a TIME timeline at the WavLM/BEATs frame rate.
                                   SNR_t = 10*log10( (sum s1[t]^2 + eps)
                                                   / (sum s2[t]^2 + eps) ),
                                   clamped to [-30, 40] dB. This is the cleanest,
                                   most directly-oracle signal: it needs no STFT
                                   and falls straight out of the two stems' frame
                                   energies. s1-active frames are flagged (within
                                   `active_db_below_peak` dB of the s1 peak frame
                                   energy) so silence does not masquerade as a real
                                   SNR reading.

  (2) irm_map       (T_p,F_bins)— per-T-F-bin Ideal Ratio Mask, the literal
                                   separation target, downsampled to the head grid.
                                   IRM(t,f) = ( |S1|^2 / (|S1|^2 + |S2|^2 + eps) )^0.5
                                   in [0,1]; ~1 where s1 dominates, ~0.5 where the
                                   two are equal, ~0 where s2 dominates.

PURE NUMPY. No torch, no torchaudio, no Praat — unit-testable anywhere. STFT is a
hand-rolled framed rFFT (Hann window) so the frame grid is exactly controllable and
matches the documented WavLM 50 Hz / BEATs T_p convention.

FRAME-RATE CONVENTIONS (must match the model, see src/preprocess.py + spec_encoder.py)
-------------------------------------------------------------------------------------
  * WavLM-Large emits one frame every 320 samples at 16 kHz == 20 ms == 50 Hz.
    `snr_timeline` is computed on a 320-sample hop with a 320-sample frame so its
    length T equals the clip's WavLM frame count n_samples // 320 (the same count
    preprocess.py uses). Each timeline bin therefore aligns 1:1 with a WavLM frame
    and an overlap_info row.
  * BEATs patch grid for a clip is T_p = (fbank_frames // 16), F_p = 128//16 = 8,
    where fbank uses 25 ms window / 10 ms hop (100 Hz fbank). For a 5 s clip
    T_p ~= 31, F_p = 8 (spec_encoder.py docstring). The IRM map is pooled to
    (T_p, F_bins) with F_bins defaulting to that 8-bin BEATs frequency grid; T_p is
    passed in by the caller (read off the cached BEATs grid) so the supervised map
    lands on the SAME bins the head predicts. If T_p is not known at extraction
    time, a default derived from the BEATs fbank math is used and the alignment is
    re-pooled at train time — see `wavlm_t` / `beats_t_p` helpers and the
    ALIGNMENT note in compute_snr_maps.py.
"""
from __future__ import annotations

import numpy as np

# ── frame-rate constants (mirror src/preprocess.py + src/spec_encoder.py) ─────
WAVLM_HOP_SAMPLES = 320          # 20 ms @ 16 kHz → 50 Hz WavLM frame rate
WAVLM_FRAME_SAMPLES = 320        # non-overlapping energy frames for the timeline
SNR_CLAMP_DB = (-30.0, 40.0)     # instantaneous-SNR clamp from the task spec

# BEATs fbank → patch grid (spec_encoder.py): 25 ms win / 10 ms hop fbank,
# patch_embedding kernel=stride=16 over both axes, mel bins = 128.
BEATS_FBANK_WIN_SAMPLES = 400    # 25 ms @ 16 kHz
BEATS_FBANK_HOP_SAMPLES = 160    # 10 ms @ 16 kHz
BEATS_PATCH = 16
BEATS_MEL_BINS = 128
F_BINS_DEFAULT = BEATS_MEL_BINS // BEATS_PATCH   # 8

# IRM STFT (matches the BEATs fbank time grid so the T axis pools cleanly).
STFT_WIN_SAMPLES = 400           # 25 ms Hann window
STFT_HOP_SAMPLES = 160           # 10 ms hop  → 100 Hz STFT frame rate
STFT_N_FFT = 512                 # >= win; 257 rFFT bins

EPS = 1e-10


# ── (helpers) frame-grid sizes ───────────────────────────────────────────────
def wavlm_n_frames(n_samples: int, hop: int = WAVLM_HOP_SAMPLES) -> int:
    """WavLM frame count for a clip == n_samples // hop (preprocess.py convention)."""
    return int(n_samples) // int(hop)


def beats_t_p(n_samples: int) -> int:
    """BEATs time-patch count T_p for a clip of n_samples at 16 kHz.

    fbank frames n = (n_samples - win)//hop + 1 (kaldi fbank, 25ms/10ms);
    patch_embedding stride 16 over time → T_p = n // 16. Matches
    spec_encoder._forward_beats (T_p ~= 31 for a 5 s clip).
    """
    n_samples = int(n_samples)
    if n_samples < BEATS_FBANK_WIN_SAMPLES:
        return 0
    n_fbank = (n_samples - BEATS_FBANK_WIN_SAMPLES) // BEATS_FBANK_HOP_SAMPLES + 1
    return n_fbank // BEATS_PATCH


# ── (1) per-frame instantaneous SNR timeline (oracle, no STFT) ───────────────
def snr_timeline_from_stems(
    s1: np.ndarray,
    s2: np.ndarray,
    hop: int = WAVLM_HOP_SAMPLES,
    frame: int = WAVLM_FRAME_SAMPLES,
    clamp_db: tuple[float, float] = SNR_CLAMP_DB,
    active_db_below_peak: float = 40.0,
    n_frames: int | None = None,
    eps: float = EPS,
) -> dict:
    """Per-frame instantaneous target-vs-interferer SNR timeline from the stems.

    For non-overlapping frames (hop == frame == 320 samples) the n-th frame covers
    samples [n*320, n*320+320), exactly the support of WavLM frame n, so the
    returned timeline aligns 1:1 with the model's WavLM frames / overlap_info rows.

        SNR_t = 10 * log10( (sum_t s1^2 + eps) / (sum_t s2^2 + eps) )

    clamped to `clamp_db`. A frame is flagged s1-active when its s1 energy is within
    `active_db_below_peak` dB of the clip's peak s1 frame energy; outside the active
    set the SNR reading is dominated by the noise floor and should not be treated as
    a real local SNR (the head/metric masks on `s1_active`).

    Returns
    -------
    dict with numpy arrays:
        snr_timeline : (T,) float32, clamped instantaneous SNR in dB.
        s1_active    : (T,) bool, True where s1 is locally active.
        s1_energy    : (T,) float32, per-frame s1 power (for diagnostics).
        s2_energy    : (T,) float32, per-frame s2 power (for diagnostics).
    T == n_samples // hop (WavLM frame count).
    """
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    s2 = np.asarray(s2, dtype=np.float64).ravel()
    n = min(s1.shape[0], s2.shape[0])
    s1, s2 = s1[:n], s2[:n]

    T = n // hop
    if T <= 0:
        return {
            "snr_timeline": np.zeros(0, np.float32),
            "s1_active": np.zeros(0, bool),
            "s1_energy": np.zeros(0, np.float32),
            "s2_energy": np.zeros(0, np.float32),
        }

    # Build T frames of `frame` samples each, anchored at n*hop. With hop==frame this
    # is a clean reshape; we index explicitly so hop != frame stays correct.
    idx = np.arange(T) * hop
    s1_pow = np.empty(T, np.float64)
    s2_pow = np.empty(T, np.float64)
    for t in range(T):
        a, b = idx[t], min(idx[t] + frame, n)
        s1_pow[t] = np.sum(s1[a:b] ** 2)
        s2_pow[t] = np.sum(s2[a:b] ** 2)

    snr = 10.0 * np.log10((s1_pow + eps) / (s2_pow + eps))
    snr = np.clip(snr, clamp_db[0], clamp_db[1])

    peak = float(s1_pow.max())
    if peak <= eps:
        s1_active = np.zeros(T, bool)
    else:
        thresh = peak * (10.0 ** (-active_db_below_peak / 10.0))
        s1_active = s1_pow >= thresh

    snr = snr.astype(np.float32)
    s1_pow = s1_pow.astype(np.float32)
    s2_pow = s2_pow.astype(np.float32)

    # Optional exact-length alignment to the clip's cached WavLM frame count. The
    # per-frame conv can drift by ±1 frame vs n_samples//hop; forcing the length keeps
    # collate alignment exact. Pads are inactive (mask 0) so they never enter the loss.
    if n_frames is not None and int(n_frames) != T:
        nf = int(n_frames)
        snr = _fit_len_1d(snr, nf, pad_val=0.0)
        s1_active = _fit_len_1d(s1_active.astype(np.float32), nf, pad_val=0.0).astype(bool)
        s1_pow = _fit_len_1d(s1_pow, nf, pad_val=0.0)
        s2_pow = _fit_len_1d(s2_pow, nf, pad_val=0.0)

    return {
        "snr_timeline": snr,
        "s1_active": s1_active,
        "s1_energy": s1_pow,
        "s2_energy": s2_pow,
    }


def _fit_len_1d(a: np.ndarray, length: int, pad_val: float = 0.0) -> np.ndarray:
    """Trim or zero-pad a 1-D array to exactly `length`."""
    if a.shape[0] == length:
        return a
    if a.shape[0] > length:
        return a[:length]
    out = np.full(length, pad_val, dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


# ── (2) per-T-F-bin Ideal Ratio Mask (oracle separation target) ──────────────
def _stft_power(
    x: np.ndarray,
    n_fft: int = STFT_N_FFT,
    win: int = STFT_WIN_SAMPLES,
    hop: int = STFT_HOP_SAMPLES,
) -> np.ndarray:
    """Hann-windowed framed power spectrogram |X(t,f)|^2 → (n_frames, n_fft//2+1).

    Pure numpy rFFT. Frame grid matches the BEATs fbank time grid (25 ms / 10 ms),
    so pooling the time axis to BEATs T_p is a clean mean-pool.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.shape[0]
    if n < win:
        x = np.pad(x, (0, win - n))
        n = win
    n_frames = (n - win) // hop + 1
    if n_frames < 1:
        n_frames = 1
    window = np.hanning(win).astype(np.float64)
    frames = np.empty((n_frames, win), np.float64)
    for t in range(n_frames):
        a = t * hop
        frames[t] = x[a:a + win] * window
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (spec.real ** 2 + spec.imag ** 2)   # (n_frames, n_fft//2+1)


def irm_map_from_stems(
    s1: np.ndarray,
    s2: np.ndarray,
    t_p: int | None = None,
    f_bins: int = F_BINS_DEFAULT,
    n_fft: int = STFT_N_FFT,
    win: int = STFT_WIN_SAMPLES,
    hop: int = STFT_HOP_SAMPLES,
    active_db_below_peak: float = 40.0,
    eps: float = EPS,
) -> dict:
    """Per-T-F-bin oracle IRM, pooled to the (T_p, F_bins) head grid.

        IRM(t,f) = ( |S1(t,f)|^2 / (|S1(t,f)|^2 + |S2(t,f)|^2 + eps) )^0.5   ∈ [0,1]

    ~1 where the target s1 dominates the bin, ~0.5 where the two stems carry equal
    energy, ~0 where the interferer s2 dominates. This is exactly the Ideal Ratio
    Mask the separation field regresses; here it is computed from the oracle stems.

    Pooling
    -------
      * frequency: the 257 rFFT bins are mean-pooled into `f_bins` contiguous bands
        (default 8, the BEATs F_p grid). Pooling is done on POWER (|S1|^2, |S2|^2)
        and the ratio is taken AFTER pooling, so each coarse bin's IRM is the
        energy-correct band ratio rather than an average of per-bin ratios.
      * time: the STFT frames (100 Hz) are mean-pooled into `t_p` bins. If `t_p` is
        None it defaults to BEATs T_p via beats_t_p(n_samples) so the map lands on
        the BEATs grid by default. Pooling is on power, ratio after.

    Returns
    -------
    dict:
        irm_map    : (T_p, f_bins) float32 in [0,1].
        irm_energy : (T_p, f_bins) float32, per-bin total stem power (|S1|^2+|S2|^2).
        irm_active : (T_p, f_bins) bool, energy within `active_db_below_peak` dB of
                     the clip's peak bin — the bins where IRM is a real ratio (silent
                     bins default to ~0.707 and must be masked out of the loss).
        t_p        : int, time bins.
        f_bins     : int, frequency bins.
    """
    s1 = np.asarray(s1, dtype=np.float64).ravel()
    s2 = np.asarray(s2, dtype=np.float64).ravel()
    n = min(s1.shape[0], s2.shape[0])
    s1, s2 = s1[:n], s2[:n]

    if t_p is None:
        t_p = beats_t_p(n)
    t_p = max(1, int(t_p))
    f_bins = max(1, int(f_bins))

    p1 = _stft_power(s1, n_fft=n_fft, win=win, hop=hop)   # (F_t, n_fft//2+1)
    p2 = _stft_power(s2, n_fft=n_fft, win=win, hop=hop)
    n_frames = p1.shape[0]
    n_freq = p1.shape[1]

    # ── frequency pooling on POWER (ratio after) ──
    p1f = _mean_pool_axis(p1, f_bins, axis=1)   # (n_frames, f_bins)
    p2f = _mean_pool_axis(p2, f_bins, axis=1)

    # ── time pooling on POWER (ratio after) ──
    p1tf = _mean_pool_axis(p1f, t_p, axis=0)    # (t_p, f_bins)
    p2tf = _mean_pool_axis(p2f, t_p, axis=0)

    denom = p1tf + p2tf
    irm = np.sqrt(p1tf / (denom + eps))
    irm = np.clip(irm, 0.0, 1.0).astype(np.float32)
    # Per-bin total power, so the loss can MASK silent (s1≈s2≈0) bins. In a silent
    # bin IRM defaults to sqrt(eps/(2eps))≈0.707 — undefined, NOT a real ratio — so
    # the head should only be supervised where there is energy. `irm_energy` and the
    # boolean `irm_active` (relative to the clip's peak bin) carry that mask.
    irm_energy = denom.astype(np.float32)
    peak = float(irm_energy.max())
    if peak <= eps:
        irm_active = np.zeros_like(irm, dtype=bool)
    else:
        irm_active = irm_energy >= peak * (10.0 ** (-active_db_below_peak / 10.0))
    return {
        "irm_map": irm,
        "irm_energy": irm_energy,
        "irm_active": irm_active,
        "t_p": int(t_p),
        "f_bins": int(f_bins),
    }


def _mean_pool_axis(arr: np.ndarray, out_len: int, axis: int) -> np.ndarray:
    """Mean-pool `arr` along `axis` into `out_len` contiguous, near-equal bins.

    Handles the common case where the input length is not an exact multiple of
    out_len by assigning each output bin the floor/ceil split (np.array_split), so
    no samples are dropped and every output bin is non-empty as long as
    in_len >= out_len. If in_len < out_len, output bins beyond in_len repeat the
    nearest input via index clamping (rare; only for very short clips).
    """
    in_len = arr.shape[axis]
    out_len = max(1, int(out_len))
    if in_len == out_len:
        return arr.astype(np.float64, copy=False)
    if in_len < out_len:
        # upsample by nearest-index repeat (short clip / over-fine grid)
        idx = np.minimum((np.arange(out_len) * in_len) // out_len, in_len - 1)
        return np.take(arr, idx, axis=axis).astype(np.float64, copy=False)
    # downsample: split into out_len contiguous chunks, mean each
    chunks = np.array_split(arr, out_len, axis=axis)
    pooled = np.stack([c.mean(axis=axis) for c in chunks], axis=axis)
    return pooled


# ── convenience: both targets for one clip ───────────────────────────────────
def snr_maps_from_stems(
    s1: np.ndarray,
    s2: np.ndarray,
    t_p: int | None = None,
    f_bins: int = F_BINS_DEFAULT,
) -> dict:
    """Compute both oracle targets for one (s1, s2) pair.

    Returns a JSON/NPZ-friendly dict:
        snr_timeline : (T,)  list/array, instantaneous SNR (dB), WavLM frame rate.
        s1_active    : (T,)  bool, s1-active frame flags.
        irm_map      : (T_p, f_bins) IRM in [0,1].
        t_p, f_bins  : grid sizes.
        T            : timeline length (== WavLM frame count).
    """
    tl = snr_timeline_from_stems(s1, s2)
    irm = irm_map_from_stems(s1, s2, t_p=t_p, f_bins=f_bins)
    return {
        "snr_timeline": tl["snr_timeline"],
        "s1_active": tl["s1_active"],
        "irm_map": irm["irm_map"],
        "irm_active": irm["irm_active"],
        "t_p": irm["t_p"],
        "f_bins": irm["f_bins"],
        "T": int(tl["snr_timeline"].shape[0]),
    }
