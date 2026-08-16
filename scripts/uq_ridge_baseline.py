#!/usr/bin/env python3
"""uq_ridge_baseline.py — the baseline that decides whether contribution III survives.

WHY THIS IS THE MOST DANGEROUS UN-RUN EXPERIMENT
Our abstention numbers (AURC +12.2% ill-posed, +19.9% robust5) are measured against EMIT-ALWAYS
only. That is not the comparison a reviewer makes. The honest baseline is not a bare ridge — a
ridge has 100% coverage by construction and cannot abstain — but a **ridge EQUIPPED with its own
uncertainty**: fit the point estimate, then fit a second model to predict its absolute error, and
threshold that. Jaeger et al. (ICLR 2023, arXiv:2211.15259) find post-hoc confidence usually
matches or beats trained selectors, so this is the standard, cheap, and hostile comparison.

If the equipped ridge traces a better risk-coverage curve than our sigma-gated system on the
ill-posed panel, then "observability-driven abstention" is not a contribution on this corpus, and
the paper must pivot to the ceiling measurement + the LM's hedging + the failure-mode analysis.
A reviewer builds this in an afternoon; we should have the number first.

WHAT IS COMPARED
Both systems produce (prediction, confidence) per clip per feature, so both trace a risk-coverage
curve by thresholding confidence. We report AURC (area under the risk-coverage curve, lower is
better) computed identically for both — the SAME loss, stated, on the SAME clips.

  ours   : aux-head prediction, sigma head as confidence
  theirs : ridge on mean+std pooled frozen features, error-predictor as confidence

Everything is cross-validated; the error predictor never sees the target, only the pooled features
and the point estimate (given both y and yhat it would compute the error exactly).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.feature_set import ILL_POSED_UNDER_OVERLAP_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402
from eval.results_io import write_result  # noqa: E402

# Panels derived from the single sources of truth, not retyped. These literals were
# byte-identical to canonical when this was written (verified 2026-08-16) — the point is that a
# future edit to feature_set.py can no longer silently disagree with three separate copies.
# Four defects in this repo came from exactly that: hnr_db/shimmer_pct in the target builder, a
# short-name GT lookup, slot_frames() ordering, and FEATS in score_matched_test.py which
# silently reported 7 of 11 features.
ROBUST5 = list(HEADLINE_FEATURES)
ILLPOSED = sorted(ILL_POSED_UNDER_OVERLAP_FEATURES)



def _auc_from_order(e: np.ndarray) -> float:
    risks = np.cumsum(e) / np.arange(1, e.size + 1)
    return float(risks.mean())


def aurc(conf: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """(AURC, E-AURC) for a confidence signal.

    ⚠️ RAW AURC CANNOT COMPARE TWO DIFFERENT SYSTEMS. Each system's risk is ITS OWN absolute
    error, so a more ACCURATE model gets a lower AURC even when its confidence ordering is no
    better — and the abstention claim is only about the ordering. The first run of this script
    reported ours winning 11 of 11 on raw AURC, which is exactly what a pure accuracy advantage
    would produce.

    E-AURC (Geifman, Uziel & El-Yaniv, ICLR 2019, arXiv:1805.08206) subtracts the AURC that a
    PERFECT confidence ordering would achieve on the SAME error vector, leaving only the quality
    of the ordering. That is the comparable quantity.
    """
    ok = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[ok], err[ok]
    if conf.size < 20:
        return float("nan"), float("nan")
    a = _auc_from_order(err[np.argsort(conf)])     # ascending uncertainty = keep low ones first
    a_opt = _auc_from_order(np.sort(err))          # oracle ordering: smallest errors first
    return a, a - a_opt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_sigma", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    cols = [f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0]
            for f in SUPERVISED_FEATURES]

    gt = {}
    for r in csv.DictReader(open(a.features_csv)):
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        row = {}
        for nm, cl in zip(names, cols):
            try:
                row[nm] = float(r.get(cl, r.get(nm, "")))
            except (TypeError, ValueError):
                pass
        gt[stem] = row

    recs = [r for r in json.load(open(a.aux_sigma)) if r.get("aux_mean")]
    keep = []
    for r in recs:
        stem = r["filename"]
        base = stem[:-8] if stem.endswith("_s1clean") else stem
        g = gt.get(stem) or gt.get(base) or {}
        if g:
            keep.append({"stem": stem, "aux": np.asarray(r["aux_mean"], float),
                         "sigma": np.asarray(r.get("sigma") or [], float), "gt": g})
    print(f"clips: {len(keep)}", flush=True)

    X = []
    for r in keep:
        af = torch.load(os.path.join(a.pt_dir, r["stem"] + ".pt"),
                        map_location="cpu", weights_only=False)["audio_features"].float()
        X.append(np.concatenate([af.mean(0).numpy(), af.std(0).numpy()]).astype(np.float32))
    X = np.stack(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    is_clean = np.array([r["stem"].endswith("_s1clean") for r in keep])
    print(f"pooled encoder features  (clean twins: {int(is_clean.sum())})", flush=True)

    print(f"\n{'feature':<15}{'E-AURC ours':>13}{'E-AURC ridge':>14}{'winner':>10}"
          f"{'(raw ours)':>12}{'(raw ridge)':>13}")
    print("-" * 78)
    summary = {}
    kf = KFold(n_splits=a.folds, shuffle=True, random_state=0)
    for j, nm in enumerate(names):
        ok = np.array([nm in r["gt"] and np.isfinite(r["gt"][nm]) for r in keep])
        if ok.sum() < 200:
            continue
        y = np.array([r["gt"][nm] for r, k in zip(keep, ok) if k])
        yh_ours = np.array([r["aux"][j] for r, k in zip(keep, ok) if k])
        sg = np.array([r["sigma"][j] if r["sigma"].size > j else np.nan
                       for r, k in zip(keep, ok) if k])
        Xf = X[ok]

        # BASELINE: ridge point estimate, then a SECOND model predicting its |error|.
        yh_ridge = np.zeros_like(y)
        for tr, te in kf.split(Xf):
            yh_ridge[te] = Ridge(alpha=10.0).fit(Xf[tr], y[tr]).predict(Xf[te])
        err_ridge = np.abs(yh_ridge - y)
        conf_ridge = np.zeros_like(err_ridge)
        Xe = np.column_stack([Xf, yh_ridge])       # never y — that would leak the error exactly
        for tr, te in kf.split(Xe):
            conf_ridge[te] = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_depth=6,
                random_state=0).fit(Xe[tr], err_ridge[tr]).predict(Xe[te])

        err_ours = np.abs(yh_ours - y)
        # ⚠️ CONSTANT-GT LEAK GUARD (ported from oracle_error_ceiling.py, where the same defect
        # scored 0.965). Where GT is constant within a condition, err = |yhat - const| is a
        # DETERMINISTIC FUNCTION of yhat — and yhat is one of the error-predictor's inputs, so
        # the ridge's confidence model predicts its own input and wins spuriously. On clean clips
        # overlap_ratio GT is exactly 0 for all 3000, which produced a 4x "ridge win" out of line
        # with every other feature. Void the row rather than report an artifact.
        cl_ = is_clean[ok]
        const_cell = ((cl_.sum() > 50 and float(np.std(y[cl_])) < 1e-6)
                      or ((~cl_).sum() > 50 and float(np.std(y[~cl_])) < 1e-6))
        if const_cell:
            print(f"{nm:<15}{'VOID — GT constant within a condition (err = f(input))':>60}")
            summary[nm] = {"voided": "constant GT within condition"}
            continue
        # Scale-free comparison: both risks are normalised by the feature's own error scale, so
        # AURC values are comparable across features and neither system is flattered by units.
        # Normalise each system by ITS OWN mean error, so E-AURC is a scale-free statement about
        # the ORDERING alone and neither side is flattered by being the more accurate estimator.
        so = float(np.mean(err_ours)) or 1.0
        sr = float(np.mean(err_ridge)) or 1.0
        a_ours, e_ours = (aurc(sg, err_ours / so) if np.isfinite(sg).any()
                          else (float("nan"), float("nan")))
        a_ridge, e_ridge = aurc(conf_ridge, err_ridge / sr)
        win = ("ours" if np.isfinite(e_ours) and e_ours < e_ridge else
               "ridge+UQ" if np.isfinite(e_ridge) else "-")
        print(f"{nm:<15}{e_ours:13.4f}{e_ridge:14.4f}{win:>10}"
              f"{a_ours:12.4f}{a_ridge:13.4f}")
        summary[nm] = {"eaurc_ours": e_ours, "eaurc_ridge_uq": e_ridge,
                       "aurc_ours": a_ours, "aurc_ridge_uq": a_ridge, "winner": win}

    for panel, feats in (("ROBUST5", ROBUST5), ("ILL-POSED", ILLPOSED)):
        o = np.nanmean([summary[f]["eaurc_ours"] for f in feats if f in summary])
        r = np.nanmean([summary[f]["eaurc_ridge_uq"] for f in feats if f in summary])
        print(f"\n{panel} E-AURC: ours {o:.4f} vs ridge+UQ {r:.4f} -> "
              f"{'OURS WINS' if o < r else 'RIDGE+UQ WINS'}")
        summary[f"_panel_{panel}"] = {"eaurc_ours": float(o), "eaurc_ridge_uq": float(r)}

    write_result(summary, out_path=a.out, producer="uq_ridge_baseline",
                 checkpoint=a.aux_sigma, n=len(keep))
    print(f"\nwrote {a.out}")
    print("READ: lower AURC is better. If ridge+UQ wins the ILL-POSED panel, abstention is not")
    print("a contribution on this corpus and the paper pivots to the ceiling + hedging + the")
    print("failure-mode analysis. Report this row either way — a reviewer builds it in an hour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
