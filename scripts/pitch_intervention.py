#!/usr/bin/env python3
"""pitch_intervention.py — value-space causal probe for f0_mean / f0_sd.

WHY THIS EXISTS
The white-box control (D2) showed our feature-space attribution ANTI-correlates with truth:
on a model that grounds BY CONSTRUCTION, d(sd)/d(qhat) scored +0.3661 while grad*input over
the 1024 WavLM dims scored -0.2868 — same model, same forward pass. So every model-side
attribution number in this project measured our EXTRACTOR, not the model.

The deeper reason: phi_t = keep_t (q_t - mu)/((N-1) sd) is a sensitivity in PITCH space, and
grad*input is a sensitivity in WavLM-FEATURE space. Different charts. Worse, grad*input is the
first-order Taylor term for ZEROING a frame's features — the linearised form of exactly the OOD
deletion protocol we already renounced on Hase et al. (NeurIPS 2021, arXiv:2106.00786). It was
not merely the wrong chart; it was a rejected intervention in disguise.

THIS PROBE WORKS IN SIGNAL SPACE — the one chart the model and the functional share. We edit
the pitch contour and resynthesise, so the intervention is expressed in Hz, the same units as
the reference.

WHAT IS MEASURED, AND WHY IT NEEDS NO ATTRIBUTION MAP
Two GLOBAL edits whose effect on the ground truth is EXACT ALGEBRA, giving a 2x2 dissociation:

  ARM 1  uniform additive shift   q -> q + d   (voiced frames)
             Delta f0_mean = d      EXACTLY
             Delta f0_sd   = 0      EXACTLY   (sd is shift-invariant)

  ARM 2  spread scaling           q -> mu + (1+e)(q - mu)
             Delta f0_sd   = e*sd   EXACTLY
             Delta f0_mean = 0      EXACTLY

A model that measures the STATISTICS gives slope 1 on the diagonal and 0 off it. A model
reading a generic "pitch-ness" correlate CANNOT produce the off-diagonal zeros — shifting and
scaling both change the contour, and only the statistic distinguishes them. This is strictly
stronger than a map correlation and immune to every defect that killed the feature-space panel.

UNITS TRAP: pitch tools shift in CENTS (multiplicative), which scales mean AND sd together and
would collapse the dissociation. We edit the contour array directly, ADDITIVELY IN HZ.

CONTROLS (each one killed a previous experiment in this project)
  * delta=0 RESYNTHESIS NULL. Vocoder analysis-synthesis is not transparent. If the artifact
    response is comparable to the edit response the probe is dead — so the baseline for every
    contrast is resynth-0, NOT the original audio, and every arm is paired.
  * REALISATION FIDELITY. GT is RE-EXTRACTED with Praat on every edited stem and compared to
    the algebraic prediction. If WORLD did not implement the intended edit, everything
    downstream is noise. (WORLD's Harvest contour and Praat's tracker need not agree exactly.)
  * VOICING-MASK EQUALITY. An edit near a voicing boundary can flip the tracker's mask, and GT
    then jumps discontinuously through N and mu — the negative voicing path the white-box
    identified. Violating clips are DISCARDED and the rate reported.
  * MULTIVARIATE Delta-GT. Resynthesis moves jitter/shimmer/hnr too. All 11 features are
    recorded so co-movement cannot be misread as a pitch response (same confound class as the
    measured -0.19/dB amplitude term in the SNR dose-response).
  * SPEAKER-CLUSTERED bootstrap: test clips share speakers, so per-clip resampling overstates n.

SCOPE. Run on the CLEAN s1 stems: `_s1clean` twins are real test inputs (the model is scored on
them), so this is in-distribution, needs no re-mix, and isolates the question. If the model
cannot track its own statistic on clean audio it certainly cannot in mixtures. The mixture arm
(fixed reverb/noise draw held constant across the sweep) is a follow-up, not a prerequisite.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from model.adapter import build_adapter           # noqa: E402

SR = 16000
FRAME_PERIOD = 10.0      # ms -> 100 Hz contour, EXACTLY the oracle map grid (TIME_STEP 0.01)
MIN_PITCH, MAX_PITCH, TIME_STEP = 75.0, 500.0, 0.01   # match build_oracle_maps.py:49


# ----------------------------------------------------------------------------- GT
def praat_f0(wav: np.ndarray):
    """f0 contour + voiced mask, with build_oracle_maps.py's exact settings.

    Clean stems have an EMPTY overlap set, so `keep = voiced & ~in_overlap` collapses to
    `voiced` and the reference is entirely audio-derived.
    """
    import parselmouth
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR)
    p = snd.to_pitch(time_step=TIME_STEP, pitch_floor=MIN_PITCH, pitch_ceiling=MAX_PITCH)
    q = p.selected_array["frequency"].astype(float)      # 0 where unvoiced
    return q, q > 0.0


def gt_stats(wav: np.ndarray):
    q, keep = praat_f0(wav)
    if keep.sum() < 3:
        return None
    v = q[keep]
    # ddof=1: phi_t = keep(q-mu)/((N-1) sd) presumes the SAMPLE sd, so the map and this
    # statistic must use the same denominator or the algebra below is off by sqrt(N/(N-1)).
    return {"f0_mean": float(v.mean()), "f0_sd": float(v.std(ddof=1)),
            "n_voiced": int(keep.sum()), "keep": keep}


# ------------------------------------------------------------------- interventions
def world_analyze(x: np.ndarray):
    import pyworld as pw
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, SR, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return f0, sp, ap, t


def world_synth(f0, sp, ap):
    import pyworld as pw
    y = pw.synthesize(np.ascontiguousarray(f0), sp, ap, SR, frame_period=FRAME_PERIOD)
    return y.astype(np.float32)


def safe_magnitudes(f0: np.ndarray):
    """Largest edits that keep EVERY voiced frame inside Praat's [75, 500] Hz bracket.

    WHY THIS IS NOT OPTIONAL: the first run discarded 12 of 12 clips on voicing-mask flips.
    A fixed -20 Hz shift on a low-pitched speaker (f0 ~85 Hz) drives frames below the 75 Hz
    FLOOR, where Praat re-labels them UNVOICED. The mask collapses, and GT then jumps
    discontinuously through N and mu -- the negative voicing path the white-box identified.
    The edit was destroying the reference rather than moving it.

    Adaptive magnitudes are FREE here: Delta-mean = d and Delta-sd = e*sd hold EXACTLY for any
    d and e, and the regression is on the RE-EXTRACTED Delta-GT rather than on d, so per-clip
    magnitudes cost nothing and are absorbed automatically.
    """
    v = f0 > 0
    if v.sum() < 3:
        return 0.0, 0.0
    q = f0[v]
    # ROBUST percentiles, NOT min/max. Praat emits occasional single-frame outliers near the
    # bracket edges, and one frame at 76 Hz collapsed d_max to 0.5 Hz for the whole clip --
    # skipping 72 of 72 while the measured flip rate was 0.0000 everywhere. The mask-flip
    # control is the real guard; this bound only has to be SENSIBLE, and a handful of
    # outlier frames crossing the floor costs a flip rate well under the threshold.
    lo, hi = float(np.percentile(q, 5)), float(np.percentile(q, 95))
    d_max = float(np.clip(min(lo - MIN_PITCH, MAX_PITCH - hi) * 0.5, 0.0, 20.0))
    mu = float(q.mean())
    dev = max(abs(hi - mu), abs(mu - lo))
    if dev < 1e-6:
        return d_max, 0.0
    e_max = min((MAX_PITCH - mu) / dev, (mu - MIN_PITCH) / dev) - 1.0
    e_max = float(np.clip(e_max * 0.5, 0.0, 0.30))
    return d_max, e_max


def edit_shift(f0: np.ndarray, d: float) -> np.ndarray:
    """ARM 1: additive Hz shift on voiced frames. sd invariant, mean moves by exactly d.

    No clipping: `safe_magnitudes` already guarantees the bracket, and clipping would
    SILENTLY break the algebra (a clipped frame changes both mean and sd by unknown amounts).
    """
    g = f0.copy()
    v = g > 0
    g[v] = g[v] + d
    return g


def edit_spread(f0: np.ndarray, e: float) -> np.ndarray:
    """ARM 2: scale deviations about the mean. mean invariant, sd moves by exactly e*sd."""
    g = f0.copy()
    v = g > 0
    if v.sum() < 3:
        return g
    mu = g[v].mean()
    g[v] = mu + (1.0 + e) * (g[v] - mu)
    return g


# ------------------------------------------------------------------------- model
def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}) or {}
    # reliability_head SWITCHES THE HEAD TYPE (proj -> 2F vs Linear -> F). Omitting it left the
    # head RANDOMLY INITIALISED while reporting only 2 missing keys, and every map captured that
    # way was backprop through noise. Strict from the start.
    adapter = build_adapter(
        variant=cfg.get("adapter_variant", "attn-concat"),
        audio_dim=cfg.get("audio_dim", 1024), lm_dim=cfg.get("lm_dim", 4096),
        compression=cfg.get("compression", 8), aux_pool=cfg.get("aux_pool") or "mean",
        reliability_head=bool(cfg.get("reliability_head", False)))
    miss, unexp = adapter.load_state_dict(
        ck.get("adapter_state_dict") or ck.get("adapter") or {}, strict=False)
    if miss or unexp:
        raise SystemExit(f"[fatal] state_dict mismatch missing={list(miss)} unexpected={list(unexp)}")
    print(f"[load] variant={cfg.get('adapter_variant')} reliability_head="
          f"{bool(cfg.get('reliability_head', False))} zero_overlap={cfg.get('zero_overlap_input')} "
          f"state_dict=EXACT MATCH", flush=True)
    return adapter.eval().to(device), cfg


def calibrate_layer(wavlm, test_dir: str, audio_dir: str, device: str) -> int:
    """Find WHICH hidden_states index the cached features came from, by matching bit-for-bit.

    The layer-7 cache was built by a script that is not in the tree, so the index convention
    (hidden_states[0] is the EMBEDDING output, not layer 1) is not documented anywhere. Guessing
    would feed the model a distribution it never saw — the same class of error as the pyannote
    lineage, which was only caught by a bit-level match. Measure it; do not assume.
    """
    import soundfile as sf
    pts = sorted(f for f in os.listdir(test_dir) if f.endswith(".pt"))
    for fn in pts[:20]:
        stem = fn[:-3]
        wp = os.path.join(audio_dir, stem + ".wav")
        if not os.path.exists(wp):
            continue
        ref = torch.load(os.path.join(test_dir, fn), map_location="cpu",
                         weights_only=False)["audio_features"].float()
        x, _ = sf.read(wp)
        x = (x.mean(1) if np.ndim(x) > 1 else x).astype(np.float32)
        with torch.no_grad():
            hs = wavlm(torch.from_numpy(x).unsqueeze(0).to(device),
                       output_hidden_states=True).hidden_states
        diffs = {}
        for k, h in enumerate(hs):
            hh = h[0].float().cpu()
            n = min(hh.shape[0], ref.shape[0])
            if n < 10:
                continue
            diffs[k] = float((hh[:n] - ref[:n]).abs().mean())
        if not diffs:
            continue
        order = sorted(diffs, key=diffs.get)
        best, second = order[0], order[1]
        bd, sd2 = diffs[best], diffs[second]
        scale = float(ref.abs().mean())
        print(f"[calib] {stem} dtype={ref.dtype} |feat|={scale:.3f}\n"
              f"        best  index={best:2d} mean|diff|={bd:.3e}  (rel {bd/max(scale,1e-9):.2%})\n"
              f"        2nd   index={second:2d} mean|diff|={sd2:.3e}\n"
              f"        separation ratio 2nd/best = {sd2/max(bd,1e-12):.1f}x", flush=True)
        # SEPARATION, not an absolute bound. The cache is stored at reduced precision, so an
        # exact match is impossible — the first attempt hard-failed at 7.4e-3 (0.7% relative,
        # i.e. bf16 rounding) even though index 7 was unambiguously correct. A WRONG layer is
        # off by orders of magnitude, so "best is far better than the runner-up" is the test
        # that actually distinguishes right-layer-lossy-storage from wrong-layer.
        if bd <= 0.05 * scale and sd2 >= 5.0 * bd:
            return best
        raise SystemExit(f"[fatal] layer identification ambiguous (best {best} diff {bd:.3e}, "
                         f"2nd {second} diff {sd2:.3e}). Refusing to guess the layer.")
    raise SystemExit("[fatal] no clip had both a .pt and a .wav — cannot calibrate the layer")


def make_predict(wavlm, adapter, layer_idx: int, device: str, n_feat: int):
    @torch.no_grad()
    def predict(wav: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
        h = wavlm(x, output_hidden_states=True).hidden_states[layer_idx]
        # zero_overlap_input: this arm was TRAINED with the channel zeroed, so feeding
        # anything else is off-distribution for it.
        oi = torch.zeros(h.shape[1], 4).unsqueeze(0).to(device)
        r = adapter(h.float(), oi.float())
        v = r[1] if isinstance(r, (tuple, list)) else r
        if isinstance(v, (tuple, list)):
            v = v[0]
        if v.dim() == 3:
            v = v.mean(dim=1)
        return v[0, :n_feat].float().cpu().numpy()
    return predict


# ------------------------------------------------------------------------- stats
def speaker_of(stem: str) -> str:
    m = re.match(r"(\d+)-", stem)
    return m.group(1) if m else stem


def _ols2(dm, dsd, y):
    """y ~ b0 + b1*d(mean) + b2*d(sd). Returns (b1, b2)."""
    A = np.column_stack([np.ones_like(dm), dm, dsd])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(coef[2])


def cluster_boot_multi(dmean, dsd, y, clusters, n_boot=2000, seed=0):
    """PARTIAL slopes of the model response on BOTH true changes, speaker-clustered.

    WHY MULTIPLE AND NOT SIMPLE REGRESSION. The smoke test's realisation-fidelity diagnostic
    showed the two arms are NOT clean: a pure additive shift moved the re-extracted sd by
    -6.59 Hz where the algebra says 0, and spread scaling at +1.0 moved sd the WRONG WAY
    (-1.71 vs +5.42 predicted). WORLD's Harvest contour, edited and resynthesised, is simply
    not recovered by Praat's independent tracker.

    That does NOT sink the experiment. The arms do not need to be clean -- they need to SPAN
    the (d_mean, d_sd) plane, and they do, because they contaminate in different patterns.
    Regressing on BOTH measured changes at once recovers the partial derivatives that the
    contaminated single-arm slopes cannot. A faithful model gives:

        d(yhat_f0mean) / d(gt_mean) = 1     and   d(yhat_f0mean) / d(gt_sd)   = 0
        d(yhat_f0sd)   / d(gt_sd)   = 1     and   d(yhat_f0sd)   / d(gt_mean) = 0

    Identifiability is reported as corr(d_mean, d_sd): near +/-1 means the arms are collinear
    and the partials are NOT separable, which must be checked before reading any coefficient.
    """
    dm, ds, y = (np.asarray(v, float) for v in (dmean, dsd, y))
    ok = np.isfinite(dm) & np.isfinite(ds) & np.isfinite(y)
    dm, ds, y, cl = dm[ok], ds[ok], y[ok], np.asarray(clusters)[ok]
    if dm.size < 12 or dm.std() < 1e-9 or ds.std() < 1e-9:
        return None
    r = float(np.corrcoef(dm, ds)[0, 1])
    b1, b2 = _ols2(dm, ds, y)
    uniq = np.unique(cl)
    idx = {u: np.flatnonzero(cl == u) for u in uniq}
    rng, B1, B2 = np.random.default_rng(seed), [], []
    for _ in range(n_boot):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        if dm[pick].std() < 1e-9 or ds[pick].std() < 1e-9:
            continue
        try:
            c1, c2 = _ols2(dm[pick], ds[pick], y[pick])
        except np.linalg.LinAlgError:
            continue
        B1.append(c1); B2.append(c2)
    if len(B1) < 50:
        return None
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return (b1, *q(B1)), (b2, *q(B2)), dm.size, uniq.size, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True, help="processed_layer7/test (for layer calib)")
    ap.add_argument("--clean_dir", required=True, help="audio_corrected/test-s1clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    # FRACTIONS of each clip's own safe magnitude (see safe_magnitudes). Fixed absolute
    # magnitudes discarded 12/12 clips by driving low-pitched speakers below Praat's floor.
    ap.add_argument("--levels", default="-1,-0.5,0.5,1")
    ap.add_argument("--max_flip", type=float, default=0.05,
                    help="max tolerated voiced-mask disagreement vs the resynth-0 baseline")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import soundfile as sf
    from transformers import WavLMModel

    short = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    I_MEAN, I_SD = short.index("f0_mean"), short.index("f0_sd")
    levels = [float(s) for s in a.levels.split(",")]

    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(a.device).eval()
    adapter, _cfg = load_model(a.checkpoint, a.device)
    layer_idx = calibrate_layer(wavlm, a.test_dir, a.clean_dir, a.device)
    print(f"[calib] USING hidden_states[{layer_idx}]", flush=True)
    predict = make_predict(wavlm, adapter, layer_idx, a.device, len(short))

    # SPEAKER ROUND-ROBIN, not sorted order. Sorted filenames put every early clip in the same
    # speaker directory: the smoke test drew 12 clips from ONE speaker, leaving the clustered
    # bootstrap with a single cluster and degenerate zero-width CIs.
    allf = sorted(f for f in os.listdir(a.clean_dir) if f.endswith(".wav"))
    by_spk: dict[str, list[str]] = {}
    for f in allf:
        by_spk.setdefault(speaker_of(f[:-4]), []).append(f)
    order, spk_keys = [], sorted(by_spk)
    for j in range(max(len(v) for v in by_spk.values())):
        for s in spk_keys:
            if j < len(by_spk[s]):
                order.append(by_spk[s][j])
    files = order[: a.n * 6]
    print(f"[data] {len(allf)} clips across {len(spk_keys)} speakers; "
          f"round-robin pool={len(files)}", flush=True)
    recs, n_maskfail, n_err, n_narrow, flips, mags = [], 0, 0, 0, [], []

    for fn in files:
        if len(recs) >= a.n:
            break
        stem = fn[:-4]
        try:
            x, sr = sf.read(os.path.join(a.clean_dir, fn))
            x = (x.mean(1) if np.ndim(x) > 1 else x).astype(np.float64)
            if sr != SR or len(x) < SR // 2:
                continue
            f0, sp, apx, _ = world_analyze(x)
            if (f0 > 0).sum() < 20:
                continue

            # delta=0 resynthesis null — the paired BASELINE for every arm below
            y0 = world_synth(f0, sp, apx)
            g0 = gt_stats(y0)
            gx = gt_stats(x.astype(np.float32))
            if g0 is None or gx is None:
                continue
            p0, px = predict(y0), predict(x.astype(np.float32))

            d_max, e_max = safe_magnitudes(f0)
            mags.append((d_max, e_max))
            if d_max < 2.0 or e_max < 0.03:
                n_narrow += 1                 # speaker sits too close to Praat's bracket
                continue

            rec = {"stem": stem, "spk": speaker_of(stem), "d_max": d_max, "e_max": e_max,
                   "gt_orig": {k: gx[k] for k in ("f0_mean", "f0_sd", "n_voiced")},
                   "gt_r0": {k: g0[k] for k in ("f0_mean", "f0_sd", "n_voiced")},
                   "pred_orig": px.tolist(), "pred_r0": p0.tolist(),
                   "shift": {}, "spread": {}}

            bad_mask = False
            for tag, mx, editor in (("shift", d_max, edit_shift),
                                    ("spread", e_max, edit_spread)):
                for lv in levels:
                    amt = lv * mx                       # per-clip magnitude, exact algebra kept
                    yv = world_synth(editor(f0, amt), sp, apx)
                    gv = gt_stats(yv)
                    if gv is None:
                        continue
                    # voicing-mask equality vs the resynth-0 baseline
                    n = min(gv["keep"].size, g0["keep"].size)
                    flip = float((gv["keep"][:n] != g0["keep"][:n]).mean())
                    flips.append(flip)
                    if flip > a.max_flip:
                        bad_mask = True
                    rec[tag][str(lv)] = {
                        "amt": amt, "gt_mean": gv["f0_mean"], "gt_sd": gv["f0_sd"],
                        "pred": predict(yv).tolist(), "mask_flip": flip}
            if bad_mask:
                n_maskfail += 1
                continue
            recs.append(rec)
            if len(recs) % 10 == 0:
                print(f"  {len(recs)}/{a.n}", flush=True)
        except Exception as e:                                  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"[warn] {stem}: {type(e).__name__}: {e}", flush=True)

    json.dump(recs, open(a.out, "w"))
    fl = np.array(flips) if flips else np.zeros(1)
    print(f"\nwrote {a.out}  clips={len(recs)}  mask-flip discards={n_maskfail}  "
          f"narrow-bracket skips={n_narrow}  errors={n_err}")
    print(f"  voiced-mask flip rate: median {np.median(fl):.4f}  p90 {np.percentile(fl, 90):.4f} "
          f" max {fl.max():.4f}   (threshold {a.max_flip})")
    if mags:
        dm = np.array([m[0] for m in mags]); em = np.array([m[1] for m in mags])
        # printed even when everything is skipped: the first two runs discarded 100% of clips
        # and I could not see WHY without this.
        print(f"  d_max Hz: median {np.median(dm):.1f} p10 {np.percentile(dm, 10):.1f} "
              f"max {dm.max():.1f}  |  e_max: median {np.median(em):.3f} "
              f"p10 {np.percentile(em, 10):.3f}   (need d>=2.0, e>=0.03)")
    if recs:
        print(f"  per-clip safe magnitudes: d_max median {np.median([r['d_max'] for r in recs]):.1f} Hz, "
              f"e_max median {np.median([r['e_max'] for r in recs]):.3f}")
    if len(recs) < 10:
        print("[fatal] too few clips to conclude anything")
        return 1

    # ---------------------------------------------------------------- ARTIFACT FLOOR
    print("\n=== CONTROL 1: RESYNTHESIS NULL (delta=0) ===")
    print("    If this is comparable to the edit responses below, the probe is DEAD.")
    for nm, i in (("f0_mean", I_MEAN), ("f0_sd", I_SD)):
        dp = np.array([r["pred_r0"][i] - r["pred_orig"][i] for r in recs])
        dg = np.array([r["gt_r0"][nm] - r["gt_orig"][nm] for r in recs])
        print(f"    {nm:<8} model |shift| {np.abs(dp).mean():7.3f}  (sd {dp.std():.3f})   "
              f"GT |shift| {np.abs(dg).mean():7.3f}")

    # -------------------------------------------------- REALISATION FIDELITY + 2x2
    print("\n=== CONTROL 2: REALISATION FIDELITY (did WORLD do what the algebra says?) ===")
    print("    predicted: shift -> d(mean)=amt, d(sd)=0 | spread -> d(mean)=0, d(sd)=amt*sd_r0")
    for tag in ("shift", "spread"):
        for lv in levels:
            am, asd, pm, psd = [], [], [], []
            for r in recs:
                d = r[tag].get(str(lv))
                if not d:
                    continue
                am.append(d["gt_mean"] - r["gt_r0"]["f0_mean"])
                asd.append(d["gt_sd"] - r["gt_r0"]["f0_sd"])
                pm.append(d["amt"] if tag == "shift" else 0.0)
                psd.append(0.0 if tag == "shift" else d["amt"] * r["gt_r0"]["f0_sd"])
            if not am:
                continue
            print(f"    {tag:<7} lvl {lv:<5} d(mean) actual {np.mean(am):+7.3f} vs predicted "
                  f"{np.mean(pm):+7.3f}   |   d(sd) actual {np.mean(asd):+7.3f} vs predicted "
                  f"{np.mean(psd):+7.3f}")

    print("\n=== THE 2x2 DISSOCIATION — PARTIAL derivatives of the model's emitted value ===")
    print("    Both arms are POOLED and both true changes enter as regressors, because the")
    print("    single-arm slopes are contaminated (see realisation fidelity above).")
    print("    Faithful model: d(yhat_X)/d(gt_X) = +1  and  d(yhat_X)/d(gt_other) = 0.\n")

    DM, DS, C = [], [], []
    Y = {I_MEAN: [], I_SD: []}
    for r in recs:
        for tag in ("shift", "spread"):
            for _lv, d in r[tag].items():
                DM.append(d["gt_mean"] - r["gt_r0"]["f0_mean"])
                DS.append(d["gt_sd"] - r["gt_r0"]["f0_sd"])
                C.append(r["spk"])
                for i in (I_MEAN, I_SD):
                    Y[i].append(d["pred"][i] - r["pred_r0"][i])

    for nm, i in (("f0_mean", I_MEAN), ("f0_sd", I_SD)):
        res = cluster_boot_multi(DM, DS, Y[i], C)
        if res is None:
            print(f"    yhat_{nm:<8} insufficient variation / too few clusters")
            continue
        (b1, l1, h1), (b2, l2, h2), n, nspk, r_xy = res
        t1, t2 = (1.0, 0.0) if nm == "f0_mean" else (0.0, 1.0)
        m1 = "MATCHES" if l1 <= t1 <= h1 else "off-target"
        m2 = "MATCHES" if l2 <= t2 <= h2 else "off-target"
        print(f"    yhat_{nm:<8} d/d(gt_mean) {b1:+.3f} [{l1:+.3f},{h1:+.3f}] want {t1:+.0f} {m1}")
        print(f"    {'':<13} d/d(gt_sd)   {b2:+.3f} [{l2:+.3f},{h2:+.3f}] want {t2:+.0f} {m2}"
              f"   n={n} spk={nspk}")
        print(f"    {'':<13} identifiability: corr(d_mean, d_sd) = {r_xy:+.3f}"
              f"{'   <-- COLLINEAR, partials NOT separable' if abs(r_xy) > 0.9 else ''}")
    if len(set(C)) < 5:
        print("\n    [warn] fewer than 5 speakers — the clustered CI is not meaningful yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
