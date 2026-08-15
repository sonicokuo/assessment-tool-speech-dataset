#!/usr/bin/env python3
"""ridge_probe_2x2.py — does SUPERVISION MASKING explain the ridge deficit, or POOLING?

THE DEFICIT
Our aux head loses to a tuned layer-7 ridge on exactly the five ill-posed features
(f0_mean 0.308 vs 0.5485, f0_sd 0.116 vs 0.2517), and wins on all five robust ones — 11 of 11
matching whether the feature is masked. Two causes are on the table and they imply different fixes:

  MASKING — `train.py:1068` drops the 5 ill-posed features from the aux-MSE on every clip with
      overlap >= 0.5, and `stirn_stop_grad` detaches the mean from the NLL, so those heads train
      on CLEAN CLIPS ONLY while the ridge trains on 100%. Fix = retrain unmasked.
  POOLING — the ridge pools mean+std; our aux head is `W . mean_t(z_t)`, mean only. **f0_sd IS a
      variability statistic**, so mean-pooling destroys precisely the information it needs. Fix =
      change the head, which is a different (and larger) change.

Attributing the whole gap to masking without testing pooling would send a retrain after the wrong
thing — so run both factors at once, on the SAME frozen features, with no training involved.

THE DESIGN
A linear probe on frozen layer-7 features, 2x2:
    {train on CLEAN CLIPS ONLY, train on ALL CLIPS} x {mean-pool, mean+std-pool}
Every cell is scored on the SAME full test set, so cells are directly comparable.

READING (pre-registered):
  clean-only+mean ~ 0.308  AND  all+mean ~ 0.5485   -> MASKING is the whole cause; retrain unmasked.
  mean+std closes f0_sd but all-vs-clean does not   -> POOLING is the cause for f0_sd.
  both matter                                       -> fix both; report the decomposition.
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

ILLPOSED = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
ROBUST5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]


def srcc(a, b):
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(x.size, float); sx = x[o]; i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and sx[j] == sx[i]:
                j += 1
            r[o[i:j]] = 0.5 * (i + j - 1); i = j
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 10 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="processed_layer7/train")
    ap.add_argument("--test_dir", required=True, help="processed_layer7/test")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit_train", type=int, default=6000)
    a = ap.parse_args()

    import torch
    from sklearn.linear_model import Ridge

    names = [f[0] if isinstance(f, (tuple, list)) else str(f) for f in SUPERVISED_FEATURES]
    cols = [f[1] if isinstance(f, (tuple, list)) and len(f) > 1 else f[0]
            for f in SUPERVISED_FEATURES]

    def load_gt(p):
        g = {}
        for r in csv.DictReader(open(p)):
            fn = r.get("filename") or ""
            stem = fn[:-4] if fn.endswith(".wav") else fn
            row = {}
            for nm, cl in zip(names, cols):
                try:
                    row[nm] = float(r.get(cl, r.get(nm, "")))
                except (TypeError, ValueError):
                    pass
            g[stem] = row
        return g

    def pool(d, stems):
        M, S, keep = [], [], []
        for s in stems:
            p = os.path.join(d, s + ".pt")
            if not os.path.exists(p):
                continue
            af = torch.load(p, map_location="cpu", weights_only=False)["audio_features"].float()
            M.append(af.mean(0).numpy()); S.append(af.std(0).numpy()); keep.append(s)
        return np.stack(M), np.stack(S), keep

    gt_tr, gt_te = load_gt(a.train_csv), load_gt(a.test_csv)
    tr_stems = sorted(f[:-3] for f in os.listdir(a.train_dir) if f.endswith(".pt"))[: a.limit_train]
    te_stems = sorted(f[:-3] for f in os.listdir(a.test_dir) if f.endswith(".pt"))
    print(f"train {len(tr_stems)}  test {len(te_stems)}", flush=True)

    Mtr, Str, tr_stems = pool(a.train_dir, tr_stems)
    Mte, Ste, te_stems = pool(a.test_dir, te_stems)
    print("pooled", flush=True)

    # "clean-only" reproduces the masked head's training set: the aux-MSE drops the ill-posed
    # features on every clip with overlap >= 0.5, i.e. effectively every mixture.
    tr_clean = np.array([s.endswith("_s1clean") for s in tr_stems])

    print(f"\n{'feature':<15}{'clean|mean':>12}{'all|mean':>10}{'clean|m+s':>11}"
          f"{'all|m+s':>10}   attribution")
    print("-" * 74)
    out = {}
    for nm in names:
        ytr = np.array([gt_tr.get(s, {}).get(nm, np.nan) for s in tr_stems])
        yte = np.array([gt_te.get(s, {}).get(nm, np.nan)
                        if not s.endswith("_s1clean")
                        else gt_te.get(s, gt_te.get(s[:-8], {})).get(nm, np.nan)
                        for s in te_stems])
        if np.isfinite(ytr).sum() < 200 or np.isfinite(yte).sum() < 200:
            continue
        row = {}
        for sub, subname in ((tr_clean, "clean"), (np.ones_like(tr_clean, bool), "all")):
            for feats, fname in (((Mtr, Mte), "mean"),
                                 ((np.hstack([Mtr, Str]), np.hstack([Mte, Ste])), "m+s")):
                Xtr, Xte = feats
                m = sub & np.isfinite(ytr)
                if m.sum() < 200:
                    row[f"{subname}|{fname}"] = float("nan")
                    continue
                mu, sd = Xtr[m].mean(0), Xtr[m].std(0) + 1e-6
                r = Ridge(alpha=10.0).fit((Xtr[m] - mu) / sd, ytr[m])
                row[f"{subname}|{fname}"] = srcc(r.predict((Xte - mu) / sd), yte)
        d_mask = row["all|mean"] - row["clean|mean"]
        d_pool = row["clean|m+s"] - row["clean|mean"]
        att = ("MASKING dominates" if d_mask > d_pool + 0.05 else
               "POOLING dominates" if d_pool > d_mask + 0.05 else
               "both / neither")
        print(f"{nm:<15}{row['clean|mean']:12.3f}{row['all|mean']:10.3f}"
              f"{row['clean|m+s']:11.3f}{row['all|m+s']:10.3f}   {att}")
        out[nm] = row | {"delta_masking": d_mask, "delta_pooling": d_pool, "attribution": att}

    for panel, feats in (("ILL-POSED", ILLPOSED), ("ROBUST5", ROBUST5)):
        have = [f for f in feats if f in out]
        if have:
            print(f"\n{panel} mean: clean|mean {np.nanmean([out[f]['clean|mean'] for f in have]):.4f}"
                  f"  all|m+s {np.nanmean([out[f]['all|m+s'] for f in have]):.4f}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("READ: 'all|mean' vs 'clean|mean' isolates SUPERVISION MASKING; 'clean|m+s' vs")
    print("'clean|mean' isolates POOLING. Our aux head is the clean|mean cell by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
