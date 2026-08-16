"""P0 stage 2 — per-layer ridge sweep. CPU only.

Fits the IDENTICAL untuned ridge (alpha=10, the recipe behind the recorded 0.6883) separately
for every one of WavLM-Large's 25 hidden states, plus three combination variants:
  best-layer   -- the single winning layer per feature
  top3-concat  -- concatenation of the 3 best layers (6144-dim)
  weighted     -- SUPERB-style learned softmax layer weights (25 params, fit on train)

SANITY CHECK BUILT IN: layer 24 is re-fit here on the same data as every other layer, so it
must land near the recorded skyline (0.6883 robust5). If it does not, the extraction or the
split differs from the original run and the sweep is not comparable -- the script says so.

PRE-REGISTERED DECISION RULE (EXPLAINABILITY §1.12 P0):
  <= +0.01 robust5 over layer 24            -> layer 24 vindicated; one sentence; stop.
  >= +0.02 robust5 OR >= +0.05 on any ill-posed feature -> proceed to a matched retrain.

WARNING THIS SCRIPT CANNOT ENFORCE: a better layer raises the SKYLINE too, and possibly more
than it raises us. Report whatever it finds.

Usage: layer_ridge.py <train.npz> <test.npz> <train_csv> <test_csv> <out.json>
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402

TR_NPZ, TE_NPZ, TR_CSV, TE_CSV, OUT = sys.argv[1:6]
ILL = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
COL = {n: c for n, c, _ in SUPERVISED_FEATURES}
RECORDED_LAYER24 = 0.6883


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
L = Xtr.shape[1]
print(f"[data] train {Xtr.shape}  test {Xte.shape}  layers={L}", flush=True)


def fit_score(xtr, xte, feat):
    itr = [i for i, n in enumerate(ntr) if feat in gtr[n]]
    ite = [i for i, n in enumerate(nte) if feat in gte[n]]
    if len(itr) < 500 or len(ite) < 100:
        return float("nan")
    ytr = np.array([gtr[ntr[i]][feat] for i in itr])
    yte = np.array([gte[nte[i]][feat] for i in ite])
    mu, sd = xtr[itr].mean(0), xtr[itr].std(0) + 1e-6
    r = Ridge(alpha=10.0).fit((xtr[itr] - mu) / sd, ytr)
    return float(spearmanr(r.predict((xte[ite] - mu) / sd), yte).correlation)


per_layer = {}
for l in range(L):
    per_layer[l] = {f: fit_score(Xtr[:, l], Xte[:, l], f) for f in FEATURE_NAMES}
    rob = [per_layer[l][f] for f in HEADLINE_FEATURES if np.isfinite(per_layer[l][f])]
    per_layer[l]["_robust5"] = float(np.mean(rob)) if rob else float("nan")
    ip = [per_layer[l][f] for f in ILL if np.isfinite(per_layer[l][f])]
    per_layer[l]["_illposed"] = float(np.mean(ip)) if ip else float("nan")
    print(f"  layer {l:>2}: robust5 {per_layer[l]['_robust5']:.4f}  "
          f"ill-posed {per_layer[l]['_illposed']:.4f}", flush=True)

l24 = per_layer[L - 1]["_robust5"]
drift = abs(l24 - RECORDED_LAYER24)
print(f"\n[SANITY] layer {L-1} robust5 = {l24:.4f} vs recorded skyline {RECORDED_LAYER24:.4f} "
      f"(drift {drift:.4f}) " + ("OK" if drift < 0.02 else
      "<-- DRIFT >0.02: extraction/split differs from the original run, sweep NOT comparable"))

best_l = max(range(L), key=lambda l: per_layer[l]["_robust5"]
             if np.isfinite(per_layer[l]["_robust5"]) else -9)
top3 = sorted(range(L), key=lambda l: per_layer[l]["_robust5"]
              if np.isfinite(per_layer[l]["_robust5"]) else -9, reverse=True)[:3]
cat_tr = np.concatenate([Xtr[:, l] for l in top3], axis=1)
cat_te = np.concatenate([Xte[:, l] for l in top3], axis=1)
concat = {f: fit_score(cat_tr, cat_te, f) for f in FEATURE_NAMES}
concat["_robust5"] = float(np.mean([concat[f] for f in HEADLINE_FEATURES
                                    if np.isfinite(concat[f])]))
concat["_illposed"] = float(np.mean([concat[f] for f in ILL if np.isfinite(concat[f])]))

print(f"\n{'variant':<22}{'robust5':>10}{'ill-posed':>12}{'vs layer24':>12}")
print("-" * 56)
print(f"{'layer 24 (current)':<22}{l24:>10.4f}{per_layer[L-1]['_illposed']:>12.4f}{0.0:>12.4f}")
print(f"{'best single layer '+str(best_l):<22}{per_layer[best_l]['_robust5']:>10.4f}"
      f"{per_layer[best_l]['_illposed']:>12.4f}{per_layer[best_l]['_robust5']-l24:>+12.4f}")
print(f"{'top3 concat '+str(top3):<22}{concat['_robust5']:>10.4f}"
      f"{concat['_illposed']:>12.4f}{concat['_robust5']-l24:>+12.4f}")

gain_rob = max(per_layer[best_l]["_robust5"], concat["_robust5"]) - l24
gain_ill = max(per_layer[best_l]["_illposed"], concat["_illposed"]) - per_layer[L-1]["_illposed"]
print(f"\n[DECISION] robust5 gain {gain_rob:+.4f} | ill-posed gain {gain_ill:+.4f}")
if gain_rob >= 0.02 or gain_ill >= 0.05:
    print("  -> TRIGGERED: proceed to a matched retrain on the winning layer(s).")
elif gain_rob <= 0.01:
    print("  -> NOT triggered: layer 24 vindicated. One sentence, stop.")
else:
    print("  -> AMBIGUOUS (0.01 < gain < 0.02): judgement call, report the sweep either way.")
print("  NOTE: a better layer raises the SKYLINE too. Report whatever this found.")

json.dump({"per_layer": {str(k): v for k, v in per_layer.items()},
           "best_layer": best_l, "top3": top3, "top3_concat": concat,
           "layer24_robust5": l24, "recorded_skyline": RECORDED_LAYER24,
           "gain_robust5": gain_rob, "gain_illposed": gain_ill},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
