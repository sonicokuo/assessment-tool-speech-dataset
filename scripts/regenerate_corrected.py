#!/usr/bin/env python
"""Corrected-data regeneration for the AQUA-NL rebuild (Stage A/B, scaled).

Fixes the two GT confounds in the 2-speaker train mixtures:
  - srmr<->f0 (0.80): reverberate the mix (real SLR28 RIR w/ prob, else simulated) so SRMR
    measures real reverberation instead of degenerating to a pitch proxy on anechoic audio.
  - snr<->temporal (0.57-0.66): add wham noise at an INDEPENDENT uniform SNR so the SNR label
    is orthogonal to speech content; take prosody/f0 from the CLEAN stem, not the mixture.

Writes degraded WAVs (model trains on these via WavLM) + features_corrected_train.csv.
Prosody/f0 are REUSED from clean_features_train.json / clean_f0_train.json (already extracted);
jitter/shimmer/hnr are extracted on the clean s1 stem ONLY if --voice (gated by positive control).

Run on a compute node (versa/SRMR = torch). Parallel over --workers processes.
"""
import sys, os, glob, json, argparse, tempfile, random
sys.path.insert(0, "src")
import numpy as np, soundfile as sf, scipy.signal as sps, pyroomacoustics as pra, pandas as pd
from scipy.stats import spearmanr
from multiprocessing import Pool

SH = "/ocean/projects/cis260125p/shared"

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix_dir",   default=f"{SH}/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/mix_clean")
    ap.add_argument("--s1_dir",    default=f"{SH}/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/s1")
    ap.add_argument("--wham_dir",  default=f"{SH}/data/wham_noise/tr")
    ap.add_argument("--real_rir_glob", default=f"{SH}/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*.wav")
    ap.add_argument("--out_audio", default=f"{SH}/data/audio_corrected/train-100")
    ap.add_argument("--out_csv",   default=f"{SH}/data/features_corrected_train.csv")
    ap.add_argument("--clean_feat", default=f"{SH}/data/clean_features_train.json")
    ap.add_argument("--clean_f0",   default=f"{SH}/data/clean_f0_train.json")
    ap.add_argument("--p_real", type=float, default=0.5, help="prob of using a real recorded RIR vs a simulated one")
    ap.add_argument("--snr_lo", type=float, default=0.0)
    ap.add_argument("--snr_hi", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N clips (smoke test); 0 = all")
    ap.add_argument("--voice", action="store_true", help="also extract jitter/shimmer/hnr on the clean s1 stem")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

# ---- lazy per-worker globals (SRMR model + RIR list loaded once per process) ----
_G = {}
def _init(args_dict, real_rirs):
    # CUDA init inside a forked multiprocessing worker DEADLOCKS. Force SRMR onto CPU by
    # hiding the GPU BEFORE importing feature_extractor_mix (which probes torch.cuda at import).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import feature_extractor_mix as fx
    _G["fx"] = fx
    _G["srmr"] = fx.load_srmr_model(getattr(fx, "SRMR_CONFIG", {}))
    _G["real_rirs"] = real_rirs
    _G["args"] = argparse.Namespace(**args_dict)
    _G["clean_feat"] = json.load(open(args_dict["clean_feat"]))
    _G["clean_f0"] = json.load(open(args_dict["clean_f0"]))

def _sim_rir(sr, rng):
    dims = [rng.uniform(4, 9), rng.uniform(3, 7), rng.uniform(2.6, 4)]
    rt60 = rng.uniform(0.3, 0.8)
    e, mo = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e), max_order=int(min(mo, 40)))
    room.add_source([rng.uniform(0.5, dims[0]-0.5), rng.uniform(0.5, dims[1]-0.5), rng.uniform(1, 2)])
    room.add_microphone([rng.uniform(0.5, dims[0]-0.5), rng.uniform(0.5, dims[1]-0.5), rng.uniform(1, 2)])
    room.compute_rir()
    return room.rir[0][0]

def _load_real_rir(sr, rng):
    p = _G["real_rirs"][rng.integers(len(_G["real_rirs"]))]
    h, hsr = sf.read(p)
    if h.ndim > 1: h = h[:, 0]
    if hsr != sr: h = sps.resample_poly(h, sr, hsr)
    h = h.astype(np.float64)
    m = np.max(np.abs(h)) or 1.0
    return h / m

def _add_noise(x, noise, snr_db, rng):
    # tile/crop noise to len(x), scale to target SNR relative to signal power
    if len(noise) < len(x):
        reps = int(np.ceil(len(x) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    off = rng.integers(0, max(1, len(noise) - len(x) + 1))
    noise = noise[off:off + len(x)]
    ps = np.mean(x ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    g = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return x + g * noise

def _process(item):
    idx, mix_path = item
    a = _G["args"]; fx = _G["fx"]
    rng = np.random.default_rng(a.seed * 1_000_003 + idx)  # deterministic per-clip, varies by index
    stem = os.path.basename(mix_path)                        # e.g. '103-..._1235-...wav' == clean-feat key
    try:
        x, sr = sf.read(mix_path)
        x = x.mean(1) if x.ndim > 1 else x
        x = x.astype(np.float64)
        # 1) reverberate
        h = _load_real_rir(sr, rng) if (len(_G["real_rirs"]) and rng.random() < a.p_real) else _sim_rir(sr, rng)
        xr = sps.fftconvolve(x, h)[:len(x)]
        # 2) noise at independent SNR
        snr = float(rng.uniform(a.snr_lo, a.snr_hi))
        wham_files = glob.glob(os.path.join(a.wham_dir, "*.wav"))
        nz, nsr = sf.read(wham_files[rng.integers(len(wham_files))])
        nz = nz.mean(1) if nz.ndim > 1 else nz
        if nsr != sr: nz = sps.resample_poly(nz.astype(np.float64), sr, nsr)
        xd = _add_noise(xr, nz.astype(np.float64), snr, rng)
        # normalize to avoid clipping, write degraded wav
        peak = np.max(np.abs(xd)) or 1.0
        xd = 0.98 * xd / peak
        os.makedirs(a.out_audio, exist_ok=True)
        out_wav = os.path.join(a.out_audio, stem)
        sf.write(out_wav, xd, sr)
        # 3) SRMR on the DEGRADED audio (real reverb+noise) = GT for srmr
        srmr = fx.compute_srmr(out_wav, _G["srmr"])
        # 4) prosody/f0 from CLEAN features (already extracted); jitter/shimmer/hnr on clean stem if --voice
        cf = _G["clean_feat"].get(stem, {})
        cf0 = _G["clean_f0"].get(stem, {})
        row = dict(filename=stem, snr_db=snr, srmr=srmr,
                   f0_mean_hz=cf0.get("f0_mean_hz"),
                   praat_speaking_rate_syl_sec=cf.get("praat_speaking_rate_syl_sec"),
                   praat_articulation_rate_syl_sec=cf.get("praat_articulation_rate_syl_sec"),
                   praat_pause_count=cf.get("praat_pause_count"),
                   praat_pause_rate_per_min=cf.get("praat_pause_rate_per_min"))
        if a.voice:
            s1 = os.path.join(a.s1_dir, stem)
            if os.path.exists(s1):
                try:
                    vq = fx.compute_voice_quality(s1) if hasattr(fx, "compute_voice_quality") else {}
                    row.update(hnr_db=vq.get("hnr_db"), jitter=vq.get("jitter_local"), shimmer=vq.get("shimmer_local"))
                except Exception:
                    pass
        return row
    except Exception as e:
        if idx < 5: print("skip", idx, repr(e)[:100], flush=True)
        return None

def main():
    a = build_args()
    # real_rirs_isotropic_noises/ mixes real RIRs (air_*, RWCP_*_rir_*) with isotropic NOISE (RVB_*_noise_*);
    # keep only the RIRs (exclude 'noise' in the basename).
    real_rirs = [f for f in sorted(glob.glob(a.real_rir_glob))
                 if "noise" not in os.path.basename(f).lower()]
    print(f"real RIRs found: {len(real_rirs)} (p_real={a.p_real}); simulated fallback via pyroomacoustics", flush=True)
    mixes = sorted(glob.glob(os.path.join(a.mix_dir, "*.wav")))
    if a.limit: mixes = mixes[:a.limit]
    items = list(enumerate(mixes))
    print(f"clips: {len(items)}  workers: {a.workers}  out_audio: {a.out_audio}", flush=True)
    args_dict = vars(a)
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(args_dict, real_rirs)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process, items, chunksize=8)):
            if r is not None: rows.append(r)
            if (i + 1) % 500 == 0: print(f"{i+1}/{len(items)} kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"WROTE {a.out_csv}  N={len(df)}", flush=True)
    # confound matrix on the regenerated GT (validate decorrelation)
    d = df.dropna(subset=["srmr", "f0_mean_hz", "snr_db", "praat_pause_count",
                          "praat_pause_rate_per_min", "praat_speaking_rate_syl_sec"])
    def sp(x, y): return spearmanr(d[x], d[y]).correlation
    print("=== CORRECTED confound matrix (orig in parens) ===", flush=True)
    print(f"  srmr <-> f0            : {sp('srmr','f0_mean_hz'):+.3f}   (orig +0.80)", flush=True)
    print(f"  snr  <-> pause_count   : {sp('snr_db','praat_pause_count'):+.3f}   (orig +0.57)", flush=True)
    print(f"  snr  <-> pause_rate    : {sp('snr_db','praat_pause_rate_per_min'):+.3f}   (orig +0.66)", flush=True)
    print(f"  snr  <-> speaking_rate : {sp('snr_db','praat_speaking_rate_syl_sec'):+.3f}   (orig -0.58)", flush=True)

if __name__ == "__main__":
    main()
