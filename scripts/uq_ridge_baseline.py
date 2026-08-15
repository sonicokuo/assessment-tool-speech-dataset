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
from data.feature_set import SUPERVISED_FEATURES  # noqa: E402

ROBUST5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
ILLPOSED = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]


def aurc(conf: np.ndarray, err: np.ndarray) -> float:
    """Area under the risk-coverage curve. Sort by confidence (most confident first), sweep
    coverage 1/n..1, risk = mean error over the retained set. Lower is better."""
    ok = np.isfinite(conf) & np.isfinite(err)
    conf, err = conf[ok], err[ok]
    if conf.size < 20:
        return float("nan")
    order = np.argsort(conf)                       # ascending "uncertainty" = keep the low ones
    e = err[order]
    risks = np.cumsum(e) / np.arange(1, e.size + 1)
    return float(risks.mean())


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
    print("pooled encoder features", flush=True)

    print(f"\n{'feature':<15}{'AURC ours':>11}{'AURC ridge+UQ':>15}{'winner':>10}")
    print("-" * 52)
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
        # Scale-free comparison: both risks are normalised by the feature's own error scale, so
        # AURC values are comparable across features and neither system is flattered by units.
        s = float(np.mean(err_ours) + np.mean(err_ridge)) / 2.0 or 1.0
        a_ours = aurc(sg, err_ours / s) if np.isfinite(sg).any() else float("nan")
        a_ridge = aurc(conf_ridge, err_ridge / s)
        win = ("ours" if np.isfinite(a_ours) and a_ours < a_ridge else
               "ridge+UQ" if np.isfinite(a_ridge) else "-")
        print(f"{nm:<15}{a_ours:11.4f}{a_ridge:15.4f}{win:>10}")
        summary[nm] = {"aurc_ours": a_ours, "aurc_ridge_uq": a_ridge, "winner": win}

    for panel, feats in (("ROBUST5", ROBUST5), ("ILL-POSED", ILLPOSED)):
        o = np.nanmean([summary[f]["aurc_ours"] for f in feats if f in summary])
        r = np.nanmean([summary[f]["aurc_ridge_uq"] for f in feats if f in summary])
        print(f"\n{panel}: ours {o:.4f} vs ridge+UQ {r:.4f} -> "
              f"{'OURS WINS' if o < r else 'RIDGE+UQ WINS'}")
        summary[f"_panel_{panel}"] = {"ours": float(o), "ridge_uq": float(r)}

    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("READ: lower AURC is better. If ridge+UQ wins the ILL-POSED panel, abstention is not")
    print("a contribution on this corpus and the paper pivots to the ceiling + hedging + the")
    print("failure-mode analysis. Report this row either way — a reviewer builds it in an hour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
