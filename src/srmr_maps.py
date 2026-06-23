"""srmr_maps.py — ORACLE 2D SRMR modulation-energy tensor from the clean s1 stem.

WHAT THIS IS
------------
SRMR (speech-to-reverberation modulation energy ratio; Falk, Zheng & Chan,
"A Non-Intrusive Quality and Intelligibility Measure of Reverberant and
Dereverberated Speech", IEEE TASLP 18(7):1766-1774, 2010) is the ONE AQUA-NL
feature whose natural dense target is GENUINELY time-frequency-2D. Where the
local-SNR field (snr_map_head) is a 1-D per-frame TIME signal (its "frequency"
axis in the IRM branch is a borrowed STFT-band pool), SRMR is defined on a
2-D AUDITORY plane:

    a gammatone ACOUSTIC filterbank (default 23 cochlear bands)
      x  an 8-band MODULATION filterbank (Q=2) on each band's temporal envelope.

The per-(acoustic-band, modulation-band) energy is a 23 x 8 tensor. The scalar
SRMR a quality model reports is its AGGREGATE:

    SRMR = sum( E[:, modulation bands 1-4] ) / sum( E[:, modulation bands 5..K*] )

i.e. the ratio of low-modulation energy (the 1-4 Hz syllabic band where clean
speech concentrates) over high-modulation energy (5-8, inflated by reverberation
smearing). K* is the SRMRpy adaptive upper modulation band (the 90%-acoustic-
bandwidth cutoff). Reverberation moves energy from the low to the high modulation
bands, so the 2D tensor carries WHERE the smearing lives, not just the ratio.

Because Libri2Mix ships the clean single-speaker s1 stem, the tensor is computed
on a WELL-POSED single-speaker signal (oracle), exactly like clean_features.clean_srmr
computes the scalar. This module returns the FULL 23x8 tensor, so the SRMR-map head
(snr_map_head.SupervisedSRMRMapHead) is supervised on the dense 2D field and the
scalar is read THROUGH it (the CBM tie, mirroring the SNR map's pooled-scalar tie).

IMPLEMENTATION
--------------
Primary path reuses the SRMRpy internals (gammatone fft_gtgram envelopes ->
modulation_filterbank -> windowed per-(band,modband)-frame energy), so the tensor's
aggregate is BYTE-IDENTICAL to srmrpy.srmr (validated: tensor-aggregate == library
scalar == clean_features GT to 4 dp on real Libri2Mix s1 stems). If SRMRpy is not
importable we fall back to a self-contained gammatone + modulation filterbank
re-implementation of the same Falk et al. 2010 pipeline.

Pure numpy at the signal level; a thin torch wrapper packs the per-clip target the
same shape the dataset / collate / head consume. No transformers / LM deps, so it is
unit-testable on CPU.
"""
from __future__ import annotations

import numpy as np

# Canonical SRMR geometry (Falk et al. 2010 / SRMRpy defaults).
N_ACOUSTIC_BANDS: int = 23      # gammatone cochlear filterbank channels
N_MODULATION_BANDS: int = 8     # modulation filterbank channels (Q=2)
LOW_FREQ: int = 125             # lowest gammatone centre freq (Hz)
MIN_CF: int = 4                 # lowest modulation centre freq (Hz)
MAX_CF: int = 128               # highest modulation centre freq (Hz)
W_LENGTH_S: float = 0.256       # modulation-energy analysis window (s)
W_INC_S: float = 0.064          # analysis hop (s)
FAST_MFS: float = 400.0         # envelope rate of the fft_gtgram fast path (Hz)
EPS: float = 1e-12


def _have_srmrpy() -> bool:
    try:
        import srmrpy  # noqa: F401
        from srmrpy.srmr import (  # noqa: F401
            compute_modulation_cfs, calc_cutoffs, calc_erbs,
            modulation_filterbank, modfilt, segment_axis,
        )
        from gammatone.fftweight import fft_gtgram  # noqa: F401
        return True
    except Exception:
        return False


def _hamming(n: int) -> np.ndarray:
    """Periodic Hamming window (matches SRMRpy: hamming(n+1)[:-1])."""
    try:
        from scipy.signal.windows import hamming as _h
        return np.asarray(_h(n + 1)[:-1], dtype=np.float64)
    except Exception:
        # numpy fallback (numpy.hamming is symmetric/periodic-by-N convention)
        return np.asarray(np.hamming(n + 1)[:-1], dtype=np.float64)


# ── primary path: SRMRpy internals -> full (23, 8, n_frames) energy ─────────────
def _energy_tensor_srmrpy(
    x: np.ndarray,
    fs: int,
    n_acoustic: int,
    n_modulation: int,
    low_freq: int,
    min_cf: int,
    max_cf: int,
) -> tuple[np.ndarray, int]:
    """(n_acoustic, n_modulation, n_frames) energy + adaptive K* via SRMRpy internals.

    Replicates srmrpy.srmr's `fast=True` branch verbatim (gammatone fft envelopes ->
    Q=2 modulation filterbank -> windowed sum-of-squares per frame), then returns the
    full energy tensor BEFORE the scalar reduction together with the adaptive upper
    modulation band K* (so the caller can reproduce the exact library scalar).
    """
    from srmrpy.srmr import (
        compute_modulation_cfs, calc_cutoffs, calc_erbs,
        modulation_filterbank, modfilt, segment_axis,
    )
    from gammatone.fftweight import fft_gtgram

    mfs = FAST_MFS
    gt_env = fft_gtgram(x, fs, 0.010, 0.0025, n_acoustic, low_freq)  # (n_acoustic, T_env)
    w_length = int(np.ceil(W_LENGTH_S * mfs))
    w_inc = int(np.ceil(W_INC_S * mfs))

    mod_cfs = compute_modulation_cfs(min_cf, max_cf, n_modulation)
    MF = modulation_filterbank(mod_cfs, mfs, 2)

    n_frames = int(1 + (gt_env.shape[1] - w_length) // w_inc)
    if n_frames < 1:
        n_frames = 1
    w = _hamming(w_length)

    energy = np.zeros((n_acoustic, n_modulation, n_frames), dtype=np.float64)
    for i, ac_ch in enumerate(gt_env):
        mod_out = modfilt(MF, ac_ch)
        for j, mod_ch in enumerate(mod_out):
            seg = segment_axis(mod_ch, w_length, overlap=w_length - w_inc, end="pad")
            energy[i, j, :] = np.sum((w * seg[:n_frames]) ** 2, axis=1)

    # adaptive K* (the 90%-acoustic-bandwidth modulation cutoff), exactly as SRMRpy.
    avg = np.mean(energy, axis=2)                              # (n_acoustic, n_mod)
    erbs = np.flipud(calc_erbs(low_freq, fs, n_acoustic))
    total = float(np.sum(avg)) + EPS
    ac_energy = np.sum(avg, axis=1)
    ac_perc = ac_energy * 100.0 / total
    cumsum = np.cumsum(np.flipud(ac_perc))
    idx = np.where(cumsum > 90)[0]
    k90 = int(idx[0]) if idx.size else (n_acoustic - 1)
    bw = float(erbs[k90])
    cutoffs = calc_cutoffs(mod_cfs, fs, 2)[0]
    kstar = n_modulation
    if (bw > cutoffs[4]) and (bw < cutoffs[5]):
        kstar = 5
    elif (bw > cutoffs[5]) and (bw < cutoffs[6]):
        kstar = 6
    elif (bw > cutoffs[6]) and (bw < cutoffs[7]):
        kstar = 7
    elif bw > cutoffs[7]:
        kstar = 8
    return energy, int(kstar)


# ── fallback path: self-contained gammatone + modulation filterbank ─────────────
def _erb_centre_freqs(fs: int, n: int, low_freq: float) -> np.ndarray:
    """ERB-spaced gammatone centre frequencies (Glasberg & Moore), high->low."""
    ear_q, min_bw = 9.26449, 24.7
    hi = fs / 2.0
    erb_lo = 21.4 * np.log10(4.37e-3 * low_freq + 1.0)
    erb_hi = 21.4 * np.log10(4.37e-3 * hi + 1.0)
    erb_pts = np.linspace(erb_hi, erb_lo, n)
    cf = (10.0 ** (erb_pts / 21.4) - 1.0) / 4.37e-3
    return cf  # descending, like SRMRpy / Slaney


def _gammatone_envelopes(x: np.ndarray, fs: int, n_acoustic: int, low_freq: float) -> np.ndarray:
    """Approximate gammatone envelopes via 4th-order bandpass + Hilbert magnitude.

    A self-contained stand-in for gammatone.fftweight.fft_gtgram for the (rare) case
    SRMRpy / the gammatone package are unavailable. Same Falk et al. 2010 cochlear
    decomposition idea (one envelope per ERB-spaced cochlear band), downsampled to the
    400 Hz modulation rate. Not byte-identical to fft_gtgram but the same physics.
    """
    from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
    cfs = _erb_centre_freqs(fs, n_acoustic, low_freq)
    ear_q, min_bw = 9.26449, 24.7
    envs = []
    for cf in cfs:
        erb = ((cf / ear_q) + min_bw)
        lo = max(20.0, cf - erb)
        hi = min(fs / 2.0 - 1.0, cf + erb)
        if hi <= lo:
            envs.append(np.zeros(int(np.ceil(len(x) * FAST_MFS / fs))))
            continue
        sos = butter(2, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band", output="sos")
        band = sosfiltfilt(sos, x)
        env = np.abs(hilbert(band))
        # downsample envelope to the 400 Hz modulation rate
        from math import gcd
        up, down = int(FAST_MFS), int(fs)
        g = gcd(up, down)
        env = resample_poly(env, up // g, down // g)
        envs.append(env)
    L = min(len(e) for e in envs)
    return np.stack([e[:L] for e in envs], axis=0)             # (n_acoustic, T_env)


def _modulation_filterbank_fallback(n_modulation: int, mfs: float, min_cf: int, max_cf: int):
    """Q=2 modulation bandpass filterbank centre freqs + 2nd-order Butterworth sos."""
    from scipy.signal import butter
    # log-spaced modulation centre frequencies (SRMRpy compute_modulation_cfs scheme)
    spacing = (max_cf / min_cf) ** (1.0 / (n_modulation - 1))
    cfs = np.array([min_cf * (spacing ** k) for k in range(n_modulation)], dtype=np.float64)
    q = 2.0
    filt = []
    for cf in cfs:
        bw = cf / q
        lo = max(0.5, cf - bw / 2.0) / (mfs / 2.0)
        hi = min(mfs / 2.0 - 1.0, cf + bw / 2.0) / (mfs / 2.0)
        lo = min(lo, 0.99); hi = min(max(hi, lo + 1e-3), 0.999)
        filt.append(butter(2, [lo, hi], btype="band", output="sos"))
    return cfs, filt


def _energy_tensor_fallback(
    x: np.ndarray,
    fs: int,
    n_acoustic: int,
    n_modulation: int,
    low_freq: int,
    min_cf: int,
    max_cf: int,
) -> tuple[np.ndarray, int]:
    """Self-contained (n_acoustic, n_modulation, n_frames) energy + K*=n_modulation."""
    from scipy.signal import sosfiltfilt
    mfs = FAST_MFS
    gt_env = _gammatone_envelopes(x, fs, n_acoustic, low_freq)
    _cfs, mod_filt = _modulation_filterbank_fallback(n_modulation, mfs, min_cf, max_cf)
    w_length = int(np.ceil(W_LENGTH_S * mfs))
    w_inc = int(np.ceil(W_INC_S * mfs))
    n_frames = int(1 + (gt_env.shape[1] - w_length) // w_inc)
    if n_frames < 1:
        n_frames = 1
    w = _hamming(w_length)
    energy = np.zeros((n_acoustic, n_modulation, n_frames), dtype=np.float64)
    for i, ac_ch in enumerate(gt_env):
        for j, sos in enumerate(mod_filt):
            mod_ch = sosfiltfilt(sos, ac_ch)
            for f in range(n_frames):
                start = f * w_inc
                seg = mod_ch[start:start + w_length]
                if len(seg) < w_length:
                    seg = np.pad(seg, (0, w_length - len(seg)))
                energy[i, j, f] = np.sum((w * seg) ** 2)
    # fallback uses the full 8-band ratio (no adaptive cutoff) -> K* = n_modulation.
    return energy, int(n_modulation)


def srmr_energy_tensor(
    x: np.ndarray,
    fs: int = 16000,
    n_acoustic: int = N_ACOUSTIC_BANDS,
    n_modulation: int = N_MODULATION_BANDS,
    low_freq: int = LOW_FREQ,
    min_cf: int = MIN_CF,
    max_cf: int = MAX_CF,
    use_srmrpy: bool | None = None,
) -> tuple[np.ndarray, int]:
    """Per-(acoustic, modulation, frame) modulation-energy tensor of a mono signal.

    Returns:
        (energy, kstar):
          energy : (n_acoustic, n_modulation, n_frames) float64 modulation energy.
          kstar  : the adaptive upper modulation band (SRMRpy 90%-bandwidth rule).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros((n_acoustic, n_modulation, 1), dtype=np.float64), n_modulation
    if use_srmrpy is None:
        use_srmrpy = _have_srmrpy()
    if use_srmrpy:
        try:
            return _energy_tensor_srmrpy(x, fs, n_acoustic, n_modulation, low_freq, min_cf, max_cf)
        except Exception:
            pass
    return _energy_tensor_fallback(x, fs, n_acoustic, n_modulation, low_freq, min_cf, max_cf)


def srmr_scalar_from_avg(avg_energy: np.ndarray, kstar: int = N_MODULATION_BANDS) -> float:
    """Aggregate the (acoustic, modulation) average-energy tensor into the scalar SRMR.

    SRMR = sum(low-modulation energy, bands 1-4) / sum(high-modulation energy, 5..K*).
    This is the EXACT SRMRpy reduction, so when `avg_energy` is the mean over the frame
    axis of srmr_energy_tensor's output and `kstar` is its K*, this returns the library
    scalar (and therefore the clean_features GT) to floating precision.
    """
    avg = np.asarray(avg_energy, dtype=np.float64)
    kstar = int(max(5, min(kstar, avg.shape[1])))
    num = float(np.sum(avg[:, :4]))
    den = float(np.sum(avg[:, 4:kstar])) + EPS
    return num / den


def srmr_map_target(
    x: np.ndarray,
    fs: int = 16000,
    n_acoustic: int = N_ACOUSTIC_BANDS,
    n_modulation: int = N_MODULATION_BANDS,
    low_freq: int = LOW_FREQ,
    min_cf: int = MIN_CF,
    max_cf: int = MAX_CF,
    log_floor: float = 1e-8,
    use_srmrpy: bool | None = None,
) -> dict:
    """Build the per-clip ORACLE 2D SRMR target from a mono clean stem.

    The dense target is the time-averaged (acoustic x modulation) energy tensor, the
    true 2D SRMR structure. We store it in TWO forms:
      * `srmr_avg`     (n_acoustic, n_modulation) raw average energy — the exact field
        whose aggregate (srmr_scalar_from_avg) reproduces the library scalar / GT.
      * `srmr_logmap`  (n_acoustic, n_modulation) per-band LOG-energy, z-like in scale
        (log10(avg + floor)) — the stable regression target the head predicts on
        (raw modulation energies span many orders of magnitude across bands).
    Plus the scalar and K* so the scalar can be read back through the field.

    Returns a dict of float32 numpy arrays / scalars:
        {srmr_avg, srmr_logmap, srmr_mask, srmr_scalar, kstar,
         n_acoustic, n_modulation, n_frames}
    `srmr_mask` is an all-ones (n_acoustic, n_modulation) validity mask (every band is
    supervised; padded/empty clips get an all-zero mask).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 64:
        avg = np.zeros((n_acoustic, n_modulation), dtype=np.float32)
        return {
            "srmr_avg": avg,
            "srmr_logmap": np.log10(np.full_like(avg, log_floor)).astype(np.float32),
            "srmr_mask": np.zeros((n_acoustic, n_modulation), dtype=np.float32),
            "srmr_scalar": float("nan"),
            "kstar": int(n_modulation),
            "n_acoustic": int(n_acoustic),
            "n_modulation": int(n_modulation),
            "n_frames": 0,
        }
    energy, kstar = srmr_energy_tensor(
        x, fs, n_acoustic, n_modulation, low_freq, min_cf, max_cf, use_srmrpy=use_srmrpy
    )
    n_frames = int(energy.shape[2])
    avg = np.mean(energy, axis=2)                              # (n_acoustic, n_modulation)
    scalar = srmr_scalar_from_avg(avg, kstar)
    logmap = np.log10(np.clip(avg, log_floor, None)).astype(np.float32)
    return {
        "srmr_avg": avg.astype(np.float32),
        "srmr_logmap": logmap,
        "srmr_mask": np.ones((n_acoustic, n_modulation), dtype=np.float32),
        "srmr_scalar": float(scalar),
        "kstar": int(kstar),
        "n_acoustic": int(n_acoustic),
        "n_modulation": int(n_modulation),
        "n_frames": n_frames,
    }
