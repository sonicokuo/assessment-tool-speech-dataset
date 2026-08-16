"""P3 — the NON-SSL baseline: log-mel filterbank + ridge.

WHY THIS EXISTS. Our baseline family currently runs from a closed-form linear probe on FROZEN
WavLM straight to our full stack, with nothing testing whether a LEARNED ENCODER is needed at
all. XANE (Interspeech 2024) found that for SNR / T60 / C50 regression, WavLM features do NO
BETTER than mel filterbanks -- the same target class as our strongest features. That is an
obvious question a reviewer asks, and "we didn't run it" is a worse answer than any number.

COMMITMENT ATTACHED (EXPLAINABILITY §1.12 P3): once run, this gets REPORTED. Running a
baseline and quietly dropping an uncomfortable number is what actually sinks papers.

EXPECT mel to be competitive on `snr`. snr_db is an INJECTED CONSTRUCTION PARAMETER (rho 0.939
with the s1clean noise floor), not a measured signal property -- a spectral feature detecting
how much noise was added is unsurprising. Pre-empt it; report PER-FEATURE, never just the mean.
snr is 1 of 5 in srcc_robust, so if mel matches us there, a fifth of the headline is free.

TWO VARIANTS, so the baseline cannot be dismissed as under-powered:
  mel80     -- 80 log-mel, mean+std pooled (160-dim). The XANE-style comparison.
  mel80+d   -- plus delta and delta-delta (480-dim). The classic ASR frontend; gives the
               baseline temporal-derivative information that static pooling throws away.

Usage: mel_ridge.py <train_audio_dirs_csv> <test_audio_dirs_csv> <train_csv> <test_csv> <out.json> [n_train]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torchaudio
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402

TR_DIRS, TE_DIRS, TR_CSV, TE_CSV, OUT = sys.argv[1:6]
NTR = int(sys.argv[6]) if len(sys.argv) > 6 else 0
SR = 16000
ILL = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
COL = {n: c for n, c, _ in SUPERVISED_FEATURES}

melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=400, hop_length=160, n_mels=80, power=2.0)
todb = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
d1 = torchaudio.transforms.ComputeDeltas()


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


def feats_for(dirs_csv, gt, limit=0):
    files = []
    for d in dirs_csv.split(","):
        files.extend(sorted(glob.glob(os.path.join(d, "*.wav"))))
    files.sort()
    if limit:
        rng = np.random.default_rng(0)
        idx = sorted(rng.permutation(len(files))[:limit])
        files = [files[i] for i in idx]
    X, names = [], []
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not gt.get(stem):
            continue
        try:
            w, sr = torchaudio.load(f)
        except Exception:
            continue
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
        w = w.mean(dim=0)
        if w.numel() < SR // 2:
            continue
        m = todb(melspec(w))                       # (80, T)
        dd1 = d1(m)
        dd2 = d1(dd1)
        stat = lambda t: np.concatenate([t.mean(1).numpy(), t.std(1).numpy()])
        X.append(np.concatenate([stat(m), stat(dd1), stat(dd2)]).astype(np.float32))
        names.append(stem)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    return np.stack(X), names


gtr, gte = load_gt(TR_CSV), load_gt(TE_CSV)
print("[mel] extracting train ...", flush=True)
Xtr, ntr = feats_for(TR_DIRS, gtr, NTR)
print("[mel] extracting test ...", flush=True)
Xte, nte = feats_for(TE_DIRS, gte)
print(f"[data] train {Xtr.shape}  test {Xte.shape}", flush=True)

VARIANTS = {"mel80": slice(0, 160), "mel80+deltas": slice(0, 480)}


def fit(xtr, xte, feat):
    itr = [i for i, n in enumerate(ntr) if feat in gtr[n]]
    ite = [i for i, n in enumerate(nte) if feat in gte[n]]
    if len(itr) < 500 or len(ite) < 100:
        return float("nan")
    ytr = np.array([gtr[ntr[i]][feat] for i in itr])
    yte = np.array([gte[nte[i]][feat] for i in ite])
    mu, sd = xtr[itr].mean(0), xtr[itr].std(0) + 1e-6
    r = Ridge(alpha=10.0).fit((xtr[itr] - mu) / sd, ytr)
    return float(spearmanr(r.predict((xte[ite] - mu) / sd), yte).correlation)


res = {}
print(f"\n{'variant':<14}" + "".join(f"{f[:10]:>11}" for f in FEATURE_NAMES)
      + f"{'ROBUST5':>10}{'ILL':>9}")
print("-" * (14 + 11 * len(FEATURE_NAMES) + 19))
for vname, sl in VARIANTS.items():
    per = {f: fit(Xtr[:, sl], Xte[:, sl], f) for f in FEATURE_NAMES}
    rob = float(np.mean([per[f] for f in HEADLINE_FEATURES if np.isfinite(per[f])]))
    ill = float(np.mean([per[f] for f in ILL if np.isfinite(per[f])]))
    res[vname] = {"per_feature": per, "robust5": rob, "illposed": ill}
    print(f"{vname:<14}" + "".join(f"{per[f]:>11.3f}" for f in FEATURE_NAMES)
          + f"{rob:>10.4f}{ill:>9.4f}")

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ per-feature, NOT the mean. If mel matches us on `snr` that is EXPECTED (injected "
      "construction parameter) and must be stated, not buried -- snr is 1 of 5 in robust5.")
