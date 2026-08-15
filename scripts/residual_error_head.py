#!/usr/bin/env python3
"""residual_error_head.py — a better abstention signal than sigma, and the test of whether it is.

WHY THIS IS JUSTIFIED NOW
The oracle error-ceiling test showed within-condition error ranking is ACHIEVABLE (6 of 11
features above 0.30, none below 0.15) while the trained sigma head reaches only 0.05-0.26. So the
shortfall is estimator quality, not missing information, and a dedicated error predictor is worth
building. Jaeger et al. (ICLR 2023, arXiv:2211.15259) find post-hoc confidence usually matches or
beats trained selectors, so this is the cheap, literature-standard first move — not an exotic one.

WHAT IT PRODUCES
Per-clip predicted |error| for every feature, cross-validated, which can be dropped straight into
`slot_decode(abstain_mask=...)` as the gate signal in place of sigma. The comparison against sigma
on the SAME clips is the deliverable: if this ranks errors better within a condition, the
abstention contribution improves without touching the model.

THE INPUT RULE THAT MAKES IT HONEST
The predictor sees the pooled encoder representation and the model's own prediction — both
available at inference — and NEVER the target. Given `y` and `yhat` it could compute the error
exactly and score 1.0 by construction, which measures nothing. Cells whose GT is constant within a
condition are voided for the same reason (there `err = |yhat - const|` is a deterministic function
of an input; `overlap_ratio` on clean clips scored 0.965 that way).
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, float)
        sx = x[o]
        i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and sx[j] == sx[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1)
            i = j
        return r
    if a.size < 5 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_sigma", required=True, help="dump_aux_sigma.py output (aux + sigma)")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor
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

    recs = json.load(open(a.aux_sigma))
    keep = []
    for r in recs:
        stem = r.get("filename") or ""
        base = stem[:-8] if stem.endswith("_s1clean") else stem
        g = gt.get(stem) or gt.get(base) or {}
        if r.get("aux_mean") and g:
            keep.append({"stem": stem, "aux": np.asarray(r["aux_mean"], float),
                         "sigma": np.asarray(r.get("sigma") or [], float), "gt": g,
                         "clean": stem.endswith("_s1clean")})
    print(f"clips: {len(keep)}", flush=True)

    X_enc = []
    for r in keep:
        p = os.path.join(a.pt_dir, r["stem"] + ".pt")
        af = torch.load(p, map_location="cpu", weights_only=False)["audio_features"].float()
        X_enc.append(np.concatenate([af.mean(0).numpy(), af.std(0).numpy()]).astype(np.float32))
    X_enc = np.stack(X_enc)
    X_enc = (X_enc - X_enc.mean(0)) / (X_enc.std(0) + 1e-6)
    is_clean = np.array([r["clean"] for r in keep])
    print("pooled encoder features", flush=True)

    print(f"\n{'feature':<15}{'sigma':>9}{'residual':>10}{'delta':>9}   note")
    print("-" * 56)
    out, summary = {}, {}
    for j, nm in enumerate(names):
        ok = np.array([nm in r["gt"] and np.isfinite(r["gt"][nm]) for r in keep])
        if ok.sum() < 200:
            continue
        y = np.array([r["gt"][nm] for r, k in zip(keep, ok) if k])
        yh = np.array([r["aux"][j] for r, k in zip(keep, ok) if k])
        sg = np.array([r["sigma"][j] if r["sigma"].size > j else np.nan
                       for r, k in zip(keep, ok) if k])
        err = np.abs(yh - y)
        cl = is_clean[ok]

        X = np.column_stack([X_enc[ok], yh])
        pred = np.zeros_like(err)
        for tr, te in KFold(n_splits=a.folds, shuffle=True, random_state=0).split(X):
            pred[te] = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_depth=6,
                random_state=0).fit(X[tr], err[tr]).predict(X[te])

        def within(v):
            # constant-GT cells are voided: err becomes a function of an input, not a ranking
            s_m = (spearman(v[~cl], err[~cl])
                   if (~cl).sum() > 50 and np.std(y[~cl]) > 1e-6 else np.nan)
            s_c = (spearman(v[cl], err[cl])
                   if cl.sum() > 50 and np.std(y[cl]) > 1e-6 else np.nan)
            return np.nanmax([s_m, s_c]) if np.any(np.isfinite([s_m, s_c])) else np.nan

        w_sig = within(sg) if np.isfinite(sg).any() else float("nan")
        w_res = within(pred)
        d = w_res - w_sig if np.isfinite(w_sig) else float("nan")
        note = ("residual head WINS" if np.isfinite(d) and d > 0.05 else
                "no gain — keep sigma" if np.isfinite(d) else "")
        print(f"{nm:<15}{w_sig:9.3f}{w_res:10.3f}{d:9.3f}   {note}")
        summary[nm] = {"sigma_within": float(w_sig), "residual_within": float(w_res),
                       "delta": float(d)}
        for r, k, p_ in zip(keep, ok, pred.tolist() if ok.all() else pred.tolist()):
            pass
        out[nm] = {r["stem"]: float(p_) for r, p_ in
                   zip([r for r, k in zip(keep, ok) if k], pred.tolist())}

    json.dump({"summary": summary, "predicted_abs_error": out}, open(a.out, "w"))
    print(f"\nwrote {a.out}")
    print("READ: `predicted_abs_error` drops into slot_decode(abstain_mask=...) as the gate")
    print("signal in place of sigma. A positive delta means abstention improves with NO retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
