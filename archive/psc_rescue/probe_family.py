"""P3 — STRENGTHEN THE BASELINE ON PURPOSE.

The point of a baseline FAMILY is that a reviewer cannot improve it in an afternoon. Our
skyline so far is ONE probe: untuned ridge (alpha=10) on mean+std pooled features. Zaiem et al.
(Interspeech 2023, arXiv:2306.00452) show single-probe conclusions are fragile -- leaderboards
reorder when downstream capacity changes -- so a lone untuned ridge is exactly the kind of
baseline someone strengthens in review.

So we strengthen it OURSELVES, three ways, on BOTH the layer we currently use (24) and the
layer the sweep picked (7):
  ridge-untuned  alpha=10, the recorded recipe (control -- must reproduce 0.6887 at layer 24)
  ridge-tuned    alpha chosen per feature by 5-fold CV over 6 decades
  mlp            one hidden layer, early stopping -- gives the probe NON-LINEAR capacity

This can only make our reported margin SMALLER. Run it anyway: a margin measured against the
strongest baseline we can build is the only one worth quoting, and the alternative is a
reviewer building it for us.

Usage: probe_family.py <train.npz> <test.npz> <train_csv> <test_csv> <out.json>
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402

TR_NPZ, TE_NPZ, TR_CSV, TE_CSV, OUT = sys.argv[1:6]
LAYERS = [24, 7]
ILL = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
COL = {n: c for n, c, _ in SUPERVISED_FEATURES}
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def load_gt(path):
    gt = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            stem = os.path.splitext(os.path.basename(r.get("filename", "")))[0]
            d = {}
            for n in FEATURE_NAMES:
                try:
                    v = float(r.get(COL[n], ""))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    d[n] = v
            gt[stem] = d
    return gt


def align(npz_path, gt):
    z = np.load(npz_path, allow_pickle=True)
    X, names = z["X"], [str(x) for x in z["names"]]
    keep = [i for i, n in enumerate(names) if gt.get(n)]
    return X[keep].astype(np.float32), [names[i] for i in keep]


gtr, gte = load_gt(TR_CSV), load_gt(TE_CSV)
Xtr, ntr = align(TR_NPZ, gtr)
Xte, nte = align(TE_NPZ, gte)
print(f"[data] train {Xtr.shape} test {Xte.shape}", flush=True)


def score(model_fn, xtr, xte, feat):
    itr = [i for i, n in enumerate(ntr) if feat in gtr[n]]
    ite = [i for i, n in enumerate(nte) if feat in gte[n]]
    if len(itr) < 500 or len(ite) < 100:
        return float("nan")
    ytr = np.array([gtr[ntr[i]][feat] for i in itr])
    yte = np.array([gte[nte[i]][feat] for i in ite])
    mu, sd = xtr[itr].mean(0), xtr[itr].std(0) + 1e-6
    a, b = (xtr[itr] - mu) / sd, (xte[ite] - mu) / sd
    try:
        m = model_fn().fit(a, ytr)
        return float(spearmanr(m.predict(b), yte).correlation)
    except Exception as e:
        print(f"    [{feat}] {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


PROBES = {
    "ridge-untuned": lambda: Ridge(alpha=10.0),
    "ridge-tuned": lambda: RidgeCV(alphas=ALPHAS, cv=5),
    "mlp-256": lambda: MLPRegressor(hidden_layer_sizes=(256,), max_iter=120,
                                    early_stopping=True, n_iter_no_change=8,
                                    random_state=0),
}

res = {}
print(f"\n{'layer':<7}{'probe':<16}" + "".join(f"{f[:9]:>10}" for f in FEATURE_NAMES)
      + f"{'ROBUST5':>10}{'ILL':>9}")
print("-" * (23 + 10 * len(FEATURE_NAMES) + 19))
for L in LAYERS:
    for pname, fn in PROBES.items():
        per = {f: score(fn, Xtr[:, L], Xte[:, L], f) for f in FEATURE_NAMES}
        rob = float(np.mean([per[f] for f in HEADLINE_FEATURES if np.isfinite(per[f])]))
        ill = float(np.mean([per[f] for f in ILL if np.isfinite(per[f])]))
        res[f"L{L}_{pname}"] = {"per_feature": per, "robust5": rob, "illposed": ill}
        print(f"{L:<7}{pname:<16}" + "".join(f"{per[f]:>10.3f}" for f in FEATURE_NAMES)
              + f"{rob:>10.4f}{ill:>9.4f}", flush=True)

best = max(res.items(), key=lambda kv: kv[1]["robust5"] if np.isfinite(kv[1]["robust5"]) else -9)
print(f"\nSTRONGEST PROBE: {best[0]}  robust5 {best[1]['robust5']:.4f}  "
      f"ill-posed {best[1]['illposed']:.4f}")
print("Compare against OUR audio-only aux head 0.7066 (layer 24) — and note our head reads")
print("layer 24 while the best probe may read layer 7, which is UNFAIR TO US until we retrain.")
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
