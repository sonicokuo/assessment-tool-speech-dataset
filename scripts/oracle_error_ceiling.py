#!/usr/bin/env python3
"""oracle_error_ceiling.py — is per-claim abstention ACHIEVABLE, or provably capped?

THE DECISION THIS MAKES
Our sigma head ranks errors WITHIN a condition at Spearman 0.05-0.26, against a ~0.24-0.30
requirement. That shortfall has been measured three independent ways (Mondrian certified 0 of 15
cells; the sigma error-ranking audit; the graded-ladder analysis). Two explanations are still
open and they imply OPPOSITE actions:

  (A) SIGMA IS WEAK.        A better uncertainty estimator (MDN, a residual head, evidence-region
                            pooling) would rank errors well. -> spend GPU on it.
  (B) THE ERROR IS IRREDUCIBLE. Given what is observable at inference time, per-claim error is
                            essentially unpredictable, and CONDITION-LEVEL abstention is a real
                            ceiling, not a shortfall. -> stop, and report the ceiling AS the result.

Nobody has run the test that separates them, and it costs no GPU.

THE TEST
Train the STRONGEST error predictor the information permits, then measure how well it ranks
errors within a condition. This upper-bounds any uncertainty head, including a perfect one.

  input  : the pooled encoder representation (what sigma sees) + the model's own prediction
  target : |yhat - y|, the realised absolute error
  scored : held-out, and SPLIT BY CONDITION (mixtures / clean), because ranking ACROSS
           conditions is the easy part we can already do at rho ~1.0

CRITICAL — WHY THE TARGET CANNOT INCLUDE GT: if the predictor sees both `y` and `yhat` it can
compute the error exactly and scores 1.0 by construction, which measures nothing. The oracle here
is "best predictor given inference-time-observable information", NOT "predictor that peeks".
That is exactly the quantity a deployed selective-prediction system is bounded by
(Chow, IEEE Trans. Inf. Theory 1970; Geifman & El-Yaniv, NeurIPS 2017, arXiv:1705.08500; the
reject rule for regression thresholds conditional variance, Zaoui/Denis/Hebiri, NeurIPS 2020,
arXiv:2006.16597).

READING THE RESULT (pre-registered before running)
  ceiling >= 0.30 within-condition  -> (A). Sigma is the weak link; MDN/residual head justified.
  ceiling 0.15-0.30                 -> partial; report both the ceiling and sigma's gap to it.
  ceiling <  0.15                   -> (B). Per-claim abstention is DEAD ON THIS DATA, and the
                                       ceiling is a publishable measured limit rather than a
                                       failure of our estimator.
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
    """Average-rank Spearman. Ordinal ranks break ties by array position, which on the heavily
    tied error vectors here produced non-monotone results in an earlier analysis."""
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


def pooled_features(pt_dir: str, stems: list[str]) -> dict[str, np.ndarray]:
    """mean+std pooling — the same representation the linear skyline probe uses."""
    import torch
    out = {}
    for i, s in enumerate(stems):
        p = os.path.join(pt_dir, s + ".pt")
        if not os.path.exists(p):
            continue
        af = torch.load(p, map_location="cpu", weights_only=False)["audio_features"].float()
        out[s] = np.concatenate([af.mean(0).numpy(), af.std(0).numpy()]).astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f"  pooled {i+1}/{len(stems)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="inference_results.json (aux_mean + sigma)")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--pt_dir", required=True)
    ap.add_argument("--out", default="oracle_error_ceiling.json")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    cols = [f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0] for f in SUPERVISED_FEATURES]

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

    res = json.load(open(a.results))
    recs = []
    for r in res:
        fn = r.get("filename") or ""
        stem = fn[:-4] if fn.endswith(".wav") else fn
        aux = r.get("aux_mean")
        if not aux:
            continue
        base = stem[:-8] if stem.endswith("_s1clean") else stem
        g = gt.get(stem) or gt.get(base) or {}
        recs.append({"stem": stem, "aux": np.asarray(aux, float), "gt": g,
                     "clean": stem.endswith("_s1clean")})
    print(f"clips with predictions: {len(recs)}", flush=True)

    print("pooling encoder features (CPU)...", flush=True)
    feats = pooled_features(a.pt_dir, [r["stem"] for r in recs])
    recs = [r for r in recs if r["stem"] in feats]
    print(f"clips with features: {len(recs)}", flush=True)
    if len(recs) < 200:
        print("[fatal] too few clips")
        return 1

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    X_enc = np.stack([feats[r["stem"]] for r in recs])
    mu, sd = X_enc.mean(0), X_enc.std(0) + 1e-6
    X_enc = (X_enc - mu) / sd
    is_clean = np.array([r["clean"] for r in recs])

    print(f"\n{'feature':<15}{'n':>6}{'ALL':>9}{'MIXTURES':>11}{'CLEAN':>9}  {'model':<6}verdict")
    print("-" * 68)
    summary = {}
    for j, nm in enumerate(names):
        ok = np.array([nm in r["gt"] and np.isfinite(r["gt"][nm]) for r in recs])
        if ok.sum() < 200:
            continue
        y = np.array([r["gt"][nm] for r in recs if nm in r["gt"] and np.isfinite(r["gt"][nm])])
        yh = np.array([r["aux"][j] for r, k in zip(recs, ok) if k])
        err = np.abs(yh - y)
        # the predictor sees the encoder representation AND the model's own prediction,
        # but NEVER the target -- see the module docstring.
        X = np.column_stack([X_enc[ok], yh])
        # TWO predictor classes. A linear ceiling is only a ceiling WITHIN the linear class —
        # quoting it as "the" bound would understate what any uncertainty head could reach, and
        # this project's failure mode is exactly that kind of unearned upper bound. The
        # gradient-boosted model is the nonlinear check; the reported ceiling is the MAX.
        best = None
        for tag, mk in (("ridge", lambda: Ridge(alpha=10.0)),
                        ("gbm", lambda: HistGradientBoostingRegressor(
                            max_iter=200, learning_rate=0.1, max_depth=6, random_state=0))):
            pred = np.zeros_like(err)
            for tr, te in KFold(n_splits=a.folds, shuffle=True, random_state=0).split(X):
                pred[te] = mk().fit(X[tr], err[tr]).predict(X[te])
            cl_ = is_clean[ok]
            cand = (spearman(pred, err),
                    spearman(pred[~cl_], err[~cl_]) if (~cl_).sum() > 50 else float("nan"),
                    spearman(pred[cl_], err[cl_]) if cl_.sum() > 50 else float("nan"), tag)
            if best is None or np.nanmax(cand[1:3]) > np.nanmax(best[1:3]):
                best = cand
        s_all, s_mix, s_cln, who = best
        cl = is_clean[ok]
        # ⚠️ CONSTANT-GT LEAK GUARD. Where GT is (near-)constant within a condition,
        # err = |yhat - const| is a DETERMINISTIC FUNCTION OF yhat -- and yhat is one of the
        # predictor's inputs, so it predicts its own input and the "ceiling" approaches 1.0.
        # `overlap_ratio` on clean clips is exactly 0 for every clip and scored 0.965 this way.
        # That is a leak, not an achievable ranking. Void such cells rather than reporting them.
        MIN_GT_SD = 1e-6
        if cl.sum() > 50 and float(np.std(y[cl])) < MIN_GT_SD:
            s_cln = float("nan")
        if (~cl).sum() > 50 and float(np.std(y[~cl])) < MIN_GT_SD:
            s_mix = float("nan")
        within = np.nanmax([s_mix, s_cln]) if np.any(np.isfinite([s_mix, s_cln])) else float("nan")
        verdict = ("VOID (constant GT in-condition — leak)" if not np.isfinite(within) else
                   "(A) sigma is the weak link" if within >= 0.30 else
                   "partial" if within >= 0.15 else
                   "(B) IRREDUCIBLE — condition-level is a real ceiling")
        print(f"{nm:<15}{ok.sum():>6}{s_all:>9.3f}{s_mix:>11.3f}{s_cln:>9.3f}  {who:<6}{verdict}")
        summary[nm] = {"n": int(ok.sum()), "all": s_all, "mixtures": s_mix,
                       "clean": s_cln, "best_predictor": who, "within": float(within)}

    print("\nREAD: the MIXTURES/CLEAN columns are the ones that matter. Ranking ACROSS")
    print("conditions is already solved (rho ~1.0); the open question is ranking WITHIN one.")
    print("This is an UPPER BOUND on any uncertainty head, sigma included.")
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
