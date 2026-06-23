#!/usr/bin/env python3
"""DETERMINISTIC re-score of v21 under the FORMAL SFS framework (Tier-3).

Every choice is PINNED so the canonical v21 table regenerates BYTE-IDENTICALLY
on re-run:

  * sigma_f   = robust MAD-based per-estimator scale from
                metrics_calibrated.estimate_noise_model(...).sigma(robust=True).
  * GT        = CLEAN-stem oracle (clean_features_{test,dev}.json +
                clean_f0_{test,dev}.json). The model's claims are parsed from
                v21's generated text and compared against the clean GT.
  * alpha     = 0.05.
  * family    = selected per feature by its MEASURED tail
                (metrics_calibrated.select_family on the GT-disagreement excess
                kurtosis): heavy-tailed -> distribution-free VP, light -> the
                two-sided Gaussian band.
  * band      = coverage_tolerance(sigma_f, JND_f, alpha, family). For a feature
                flagged heteroscedastic by the noise model, sigma is taken
                magnitude-dependent via NoiseModel.sigma_at(|gt|), so the band
                widens with |gt| THROUGH sigma (Tier-3; no rel_frac knob).
  * seed      = 12345 for every bootstrap (paired skill CI + p-value).
  * GT join   = EXACT by filename; clips iterated in SORTED filename order so the
                resampling stream is deterministic.

The framework is IMPORTED from the committed src modules (no inline math). The
table (per-feature coverage-tol, skill, bootstrap 95% CI, Holm/BH p,
identifiability, P_max vs achieved) is printed and dumped to JSON.

Run:
  env python scripts/rescore_v21_formal.py
"""
import json
import math
import statistics as st
import sys

SHARED = "/ocean/projects/cis260125p/shared"
sys.path.insert(0, f"{SHARED}/assessment-tool-redirect/src")

from sfs import HybridClaimParser, SFSScorer, PERCEPTUAL_JND  # noqa: E402
from metrics_calibrated import (  # noqa: E402
    estimate_noise_model,
    select_family,
    coverage_tolerance,
    p_max_ceiling,
    identifiability,
    bootstrap_skill_ci,
    holm_correction,
    bh_correction,
)

# ── PINNED constants ─────────────────────────────────────────────────────────
DATA = f"{SHARED}/data"
PRED = f"{SHARED}/checkpoints/v21_observability/inference_results.json"
OUT = f"{SHARED}/rescore_v21/rescore_v21_formal_table.json"
SPLITS = ("test", "dev")
ALPHA = 0.05
N_BOOT = 4000
SEED = 12345
KURT_THRESHOLD = 1.0     # select_family heavy-tail cutoff

# feature -> (mix csv column, clean source {feat|f0}, clean key). Only features
# with a clean second estimator are scorable under the clean-GT framing; the
# rest (hnr/jitter/shimmer/overlap_ratio) have no clean oracle here and are
# reported as "no clean sigma".
FEAT = {
    "snr": ("snr_db", "feat", "snr_db"),
    "srmr": ("srmr", "feat", "srmr"),
    "f0_mean": ("f0_mean_hz", "f0", "f0_mean_hz"),
    "f0_sd": ("f0_sd_hz", "f0", "f0_sd_hz"),
    "speaking_rate": ("praat_speaking_rate_syl_sec", "feat",
                      "praat_speaking_rate_syl_sec"),
    "articulation_rate": ("praat_articulation_rate_syl_sec", "feat",
                          "praat_articulation_rate_syl_sec"),
    "pause_count": ("praat_pause_count", "feat", "praat_pause_count"),
    "pause_rate": ("praat_pause_rate_per_min", "feat", "praat_pause_rate_per_min"),
}
ALL_FEATS = list(FEAT.keys())
# snr/srmr read the same acoustic content; the mix-vs-clean disagreement is a
# definitional bias, not two independent draws, so do NOT apply the sqrt(2)
# split (independent=False -> conservative larger sigma).
NON_INDEP = {"snr", "srmr"}
# constant-baseline kind: mode for the integer count, median elsewhere.
MODE_BASELINE = {"pause_count"}


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ── load mix (for the second estimator) + clean GT, keyed by filename ────────
import csv  # noqa: E402

mix_rows: dict = {}
clean_feat: dict = {}
clean_f0: dict = {}
for sp in SPLITS:
    with open(f"{DATA}/features/{sp}.csv") as fh:
        for r in csv.DictReader(fh):
            mix_rows[r["filename"]] = r
    for k, v in json.load(open(f"{DATA}/clean_features_{sp}.json")).items():
        clean_feat[k] = v
    for k, v in json.load(open(f"{DATA}/clean_f0_{sp}.json")).items():
        clean_f0[k] = v


def mix_val(feat, fname):
    col = FEAT[feat][0]
    r = mix_rows.get(fname)
    return fnum(r[col]) if r and col in r else None


def clean_gt(feat, fname):
    col, src, ckey = FEAT[feat]
    if src == "feat":
        d = clean_feat.get(fname)
        return fnum(d[ckey]) if d and ckey in d else None
    d = clean_f0.get(fname)
    return fnum(d.get(ckey)) if d else None


# ── fit per-feature noise model on the FULL paired corpus (sorted) ───────────
# Pairs: (mix estimate, clean oracle) per clip. sigma_f = robust MAD per-
# estimator scale. iterate filenames in SORTED order for determinism.
def fit_models():
    models: dict = {}
    fnames = sorted(mix_rows)
    for feat in ALL_FEATS:
        a, b = [], []      # a = mix estimate, b = clean oracle
        for fn in fnames:
            mv = mix_val(feat, fn)
            cv = clean_gt(feat, fn)
            if mv is None or cv is None:
                continue
            a.append(mv)
            b.append(cv)
        indep = feat not in NON_INDEP
        models[feat] = estimate_noise_model(feat, a, b, independent=indep,
                                            hetero_corr_threshold=0.2)
    return models


MODELS = fit_models()

# ── parse v21 model claims from generated text (sorted by filename) ──────────
results = json.load(open(PRED))
results = sorted(results, key=lambda e: e["filename"])
parser = HybridClaimParser()


def scalar_claims(text):
    out = {}
    for c in parser.parse(text):
        if c.feature in SFSScorer.TOLERANCES and c.feature not in out:
            out[c.feature] = c.value
    return out


model_claims: dict = {}
for entry in results:
    fn = entry["filename"]
    gen = entry.get("generated") or entry.get("generated_clean") or ""
    model_claims[fn] = scalar_claims(gen)

N_EVAL = len(results)


# ── score one feature under the clean-GT Tier-3 band ─────────────────────────
def score_feature(feat):
    m = MODELS[feat]
    jnd_f = PERCEPTUAL_JND.get(feat, 0.0)
    family = select_family(m.ex_kurtosis, kurt_threshold=KURT_THRESHOLD)
    sigma_const = m.sigma(robust=True)

    rec = {
        "feature": feat, "family": family,
        "sigma_robust": sigma_const, "rel_sigma": m.rel_sigma,
        "heteroscedastic": m.heteroscedastic, "ex_kurtosis": m.ex_kurtosis,
        "jnd": jnd_f, "n_pairs_sigma": m.n,
    }

    # per-clip Tier-3 band: sigma_f(|gt|) = sigma_at(|gt|) folds heteroscedasticity
    # into sigma; band = JND + k(alpha, family)*sigma_f(|gt|).
    def tol_fn(_feat, gt):
        sig = m.sigma_at(gt, robust=True)
        return coverage_tolerance(sig, jnd_f, ALPHA, family)

    # representative (homoscedastic) tol at the median |gt| for the table column.
    preds, gts = [], []
    for entry in results:               # already filename-sorted
        fn = entry["filename"]
        claimed = model_claims.get(fn, {}).get(feat)
        g = clean_gt(feat, fn)
        if claimed is None or g is None:
            continue
        preds.append(claimed)
        gts.append(g)
    rec["n"] = len(preds)

    med_gt = st.median(gts) if gts else 0.0
    rec["tol_at_median_gt"] = tol_fn(feat, med_gt)
    # flat-band representative (sigma_const) for P_max / identifiability columns.
    rec["tol_const"] = coverage_tolerance(sigma_const, jnd_f, ALPHA, family)

    # identifiability (Fano) on the clean-GT corpus range; quartile cell (R/4),
    # half-width R/8 (sigma-independent effect size).
    cvals = sorted(g for g in (clean_gt(feat, fn) for fn in sorted(mix_rows))
                   if g is not None)
    if cvals:
        def pct(a, p):
            i = p / 100.0 * (len(a) - 1)
            lo, hi = math.floor(i), math.ceil(i)
            return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (i - lo)
        dyn = pct(cvals, 97.5) - pct(cvals, 2.5)
    else:
        dyn = m.dynamic_range
    delta_q = dyn / 8.0 if dyn > 0 else jnd_f
    ident = identifiability(feat, sigma_const, jnd_f, dyn, delta_f=delta_q)
    rec["identifiable"] = ident.scorable
    rec["channel_capacity"] = ident.channel_capacity
    rec["log2_cells"] = ident.log2_cells
    rec["dyn_range"] = dyn

    # observability ceiling at the representative const band.
    rec["p_max"] = p_max_ceiling(sigma_const, rec["tol_const"])

    if not preds:
        rec.update({"achieved": None, "skill": None, "ci_lo": None,
                    "ci_hi": None, "p_value": None, "baseline_value": None,
                    "baseline_precision": None, "at_ceiling": None,
                    "headroom": None})
        return rec

    # achieved precision under the per-clip Tier-3 band.
    achieved = sum(1 for p, g in zip(preds, gts)
                   if abs(p - g) <= tol_fn(feat, g)) / len(preds)
    rec["achieved"] = achieved

    # constant baseline (median, or mode for the integer count).
    if feat in MODE_BASELINE:
        try:
            base_val = st.mode([round(g) for g in gts])
        except st.StatisticsError:
            base_val = st.median(gts)
    else:
        base_val = st.median(gts)
    rec["baseline_value"] = base_val
    rec["baseline_precision"] = (sum(1 for g in gts
                                     if abs(base_val - g) <= tol_fn(feat, g))
                                 / len(gts))

    # seeded paired bootstrap skill CI + one-sided p-value (model > baseline).
    sk = bootstrap_skill_ci(preds, gts, feat, tol_fn, base_val,
                            n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
    rec["skill"] = sk["point"]
    rec["ci_lo"] = sk["lo"]
    rec["ci_hi"] = sk["hi"]
    rec["p_value"] = sk["p_value"]

    # at-ceiling verdict: achieved within 2 binomial SE of P_max.
    se = math.sqrt(max(achieved * (1 - achieved), 1e-9) / len(preds))
    rec["at_ceiling"] = achieved >= rec["p_max"] - 2 * se
    rec["headroom"] = rec["p_max"] - achieved
    return rec


def main():
    rows = {f: score_feature(f) for f in ALL_FEATS}
    # multiple-comparison correction across features with a defined p.
    pvals = {f: r["p_value"] for f, r in rows.items() if r.get("p_value") is not None}
    holm = holm_correction(pvals, alpha=ALPHA) if pvals else {}
    bh = bh_correction(pvals, alpha=ALPHA) if pvals else {}
    for f in rows:
        rows[f]["holm"] = holm.get(f)
        rows[f]["bh"] = bh.get(f)

    def fmt(x, n=3):
        if x is None:
            return "—"
        if isinstance(x, bool):
            return "Y" if x else "N"
        if isinstance(x, float):
            if x != x:
                return "nan"
            return f"{x:.{n}f}"
        return str(x)

    print(f"# DETERMINISTIC v21 formal re-score (clean-GT, Tier-3)")
    print(f"# n_eval_clips={N_EVAL}  alpha={ALPHA}  n_boot={N_BOOT}  seed={SEED}")
    print(f"# sigma_f = robust MAD per-estimator; family per measured tail; "
          f"GT join EXACT by filename (sorted)")
    print()
    print("## noise model (per feature, full paired corpus)\n")
    print("| feature | n_pairs | sigma_robust | rel_sigma | hetero | exKurt | family |")
    print("|---|---|---|---|---|---|---|")
    for f in ALL_FEATS:
        r = rows[f]
        print(f"| {f} | {r['n_pairs_sigma']} | {fmt(r['sigma_robust'])} | "
              f"{fmt(r['rel_sigma'])} | {fmt(r['heteroscedastic'])} | "
              f"{fmt(r['ex_kurtosis'], 2)} | {r['family']} |")

    print("\n## CANONICAL v21 TABLE (clean-GT, Tier-3 coverage band)\n")
    hdr = ["feature", "n", "tol@med", "skill", "95% CI", "holm_p", "bh_p",
           "identif?", "P_max", "achieved", "at_ceiling?"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for f in ALL_FEATS:
        r = rows[f]
        ci = (f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]"
              if r.get("ci_lo") is not None else "—")
        hp = r.get("holm")
        bp = r.get("bh")
        hp_s = (f"{fmt(hp['p_adj'], 4)}{'*' if hp['reject'] else ''}"
                if hp else "—")
        bp_s = (f"{fmt(bp['p_adj'], 4)}{'*' if bp['reject'] else ''}"
                if bp else "—")
        ident_s = ("SCORABLE" if r.get("identifiable") else "UNIDENTIF")
        cap = r.get("channel_capacity")
        ident_s += f" ({fmt(cap, 2)}b)" if cap is not None else ""
        print("| " + " | ".join([
            f, str(r["n"]), fmt(r.get("tol_at_median_gt")),
            fmt(r.get("skill")), ci, hp_s, bp_s, ident_s,
            fmt(r.get("p_max")), fmt(r.get("achieved")),
            fmt(r.get("at_ceiling"))]) + " |")

    payload = {
        "meta": {"n_eval_clips": N_EVAL, "alpha": ALPHA, "n_boot": N_BOOT,
                 "seed": SEED, "gt": "clean", "sigma": "robust_mad",
                 "family": "per_measured_tail", "kurt_threshold": KURT_THRESHOLD},
        "rows": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
    print(f"\n# wrote {OUT}")


if __name__ == "__main__":
    main()
