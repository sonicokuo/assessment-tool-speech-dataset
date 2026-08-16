"""Bit-reproduction check for the corrected-audio lineage. READ-ONLY on live data.

Replays regenerate_corrected's per-clip pipeline for the first N clips of the test
split and compares sample-exactly against the LIVE audio_corrected/test. If it matches,
xr and g*noise are recoverable and snr gets an exact contribution map on our real test
clips. If not, the lineage cannot be reconstructed and snr attribution needs a fresh
evaluation set.

`idx` must come from the position in the SORTED mix glob, because the per-clip RNG is
default_rng(seed*1_000_003 + idx).
"""
import glob, os, sys
import numpy as np, soundfile as sf, scipy.signal as sps

SH = "/ocean/projects/cis260125p/shared"
sys.path.insert(0, f"{SH}/repo_verify/scripts")
sys.path.insert(0, f"{SH}/repo_verify/src")

MIX_DIR = f"{SH}/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean"
LIVE = f"{SH}/data/audio_corrected/test"
WHAM = f"{SH}/data/wham_noise/tr"
RIR_GLOB = f"{SH}/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*.wav"
SEED, P_REAL, SNR_LO, SNR_HI, N = 0, 0.5, 0.0, 40.0, 20

mixes = sorted(glob.glob(os.path.join(MIX_DIR, "*.wav")))
print(f"mix glob: {len(mixes)} files  (live dir has {len(os.listdir(LIVE))})")
if not mixes:
    print("[fatal] mix_dir empty — wrong path, cannot test"); sys.exit(1)
real_rirs = [f for f in sorted(glob.glob(RIR_GLOB))
             if "noise" not in os.path.basename(f).lower()]   # script line 141-142
wham_files = glob.glob(os.path.join(WHAM, "*.wav"))     # UNSORTED, as the original
print(f"real_rirs: {len(real_rirs)}   wham: {len(wham_files)}")

def add_noise(x, noise, snr_db, rng):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    st = rng.integers(0, max(1, len(noise) - len(x) + 1))
    noise = noise[st:st + len(x)]
    ps = np.mean(x ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    g = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return x + g * noise, float(g)

ok = 0
for idx, mp in enumerate(mixes[:N]):
    stem = os.path.basename(mp)
    live_p = os.path.join(LIVE, stem)
    if not os.path.exists(live_p):
        continue
    rng = np.random.default_rng(SEED * 1_000_003 + idx)
    x, sr = sf.read(mp)
    x = (x.mean(1) if x.ndim > 1 else x).astype(np.float64)
    use_real = len(real_rirs) and rng.random() < P_REAL
    if not use_real:
        print(f"  [{stem[:28]}] simulated RIR -> needs pyroomacoustics replay, skipping")
        continue
    hp = real_rirs[rng.integers(len(real_rirs))]
    h, hsr = sf.read(hp)
    if h.ndim > 1:
        h = h[:, 0]                      # channel 0, NOT the mean
    if hsr != sr:
        h = sps.resample_poly(h, sr, hsr)
    h = h.astype(np.float64)
    h = h / (np.max(np.abs(h)) or 1.0)   # RIR is PEAK-NORMALISED — omitting this alone breaks it
    xr = sps.fftconvolve(x, h)[:len(x)]
    snr = float(rng.uniform(SNR_LO, SNR_HI))
    nz, nsr = sf.read(wham_files[rng.integers(len(wham_files))])
    nz = nz.mean(1) if nz.ndim > 1 else nz
    if nsr != sr:
        nz = sps.resample_poly(nz.astype(np.float64), sr, nsr)
    xd, g = add_noise(xr, nz.astype(np.float64), snr, rng)
    peak = np.max(np.abs(xd)) or 1.0
    xd = 0.98 * xd / peak
    live, _ = sf.read(live_p)
    n = min(len(xd), len(live))
    err = float(np.max(np.abs(xd[:n] - live[:n]))) if n else float("nan")
    match = err < 1e-6
    ok += match
    print(f"  [{stem[:28]}] max|diff| = {err:.3e}  {'MATCH' if match else 'DIFFERS'}")
print(f"\nreproduced {ok} of the real-RIR clips attempted")
print("MATCH => components recoverable on our real test clips; DIFFERS => lineage not reconstructible")
