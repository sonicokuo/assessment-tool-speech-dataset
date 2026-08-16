"""REGRESSION SKYLINE: ridge on stats-pooled FROZEN WavLM -> the 11 scalars.

The control that decides what this paper is allowed to claim. If a linear model on the same
frozen features matches the full adapter + LoRA-8B pipeline, then the accuracy is the
ENCODER's, not the architecture's, and the headline must move to abstention/interface.
Precedent for the attack: Tan et al., NeurIPS 2024 (arXiv 2406.16964) -- removing the LLM
did not degrade LLM-for-time-series methods. ALLD (ICLR 2025) reports this row.

Pooling is mean+std over time (2048-dim), the standard SUPERB-style frozen-feature probe.
"""
import glob, os, sys, csv, math, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from data.feature_set import FEATURE_NAMES, SUPERVISED_FEATURES
from scipy.stats import spearmanr

B = "/ocean/projects/cis260125p/shared"
CSV_COL = {n: c for n, c, _ in SUPERVISED_FEATURES}

def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def load_split(pt_dir, csv_path, limit=None):
    gt = {r["filename"]: r for r in csv.DictReader(open(csv_path))}
    files = sorted(glob.glob(f"{pt_dir}/*.pt"))
    if limit:
        files = files[:limit]
    X, Y, names = [], [], []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        af = d.get("audio_features")
        fn = d.get("filename") or os.path.basename(f)
        row = gt.get(fn) or gt.get(fn + ".wav")
        if af is None or row is None:
            continue
        a = af.float().numpy()
        X.append(np.concatenate([a.mean(0), a.std(0)]))
        Y.append([num(row.get(CSV_COL[k])) for k in FEATURE_NAMES])
        names.append(fn)
        if (i + 1) % 5000 == 0:
            print(f"  loaded {i+1}/{len(files)}", flush=True)
    return np.array(X, dtype=np.float32), Y, names

print("loading TRAIN ...", flush=True)
Xtr, Ytr, _ = load_split(f"{B}/data/processed_corrected/train", f"{B}/data/features_corrected_merged/train-100.csv")
print(f"train X {Xtr.shape}", flush=True)
print("loading TEST ...", flush=True)
Xte, Yte, _ = load_split(f"{B}/data/processed_corrected/test", f"{B}/data/features_corrected_merged/test.csv")
print(f"test  X {Xte.shape}", flush=True)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

from sklearn.linear_model import Ridge
MODEL = {"snr": .942, "srmr": .851, "f0_mean": .359, "speaking_rate": .541,
         "pause_count": .626, "pause_rate": .534, "overlap_ratio": .988}
ROB = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
print(f"\n{'feature':<14}{'RIDGE SRCC':>12}{'MODEL SRCC':>12}{'delta':>9}{'n_te':>7}")
print("-" * 56)
res = {}
for j, feat in enumerate(FEATURE_NAMES):
    tr_ok = [i for i, y in enumerate(Ytr) if y[j] is not None]
    te_ok = [i for i, y in enumerate(Yte) if y[j] is not None]
    if len(tr_ok) < 500 or len(te_ok) < 100:
        continue
    ytr = np.array([Ytr[i][j] for i in tr_ok], dtype=np.float64)
    yte = np.array([Yte[i][j] for i in te_ok], dtype=np.float64)
    r = Ridge(alpha=10.0).fit(Xtr[tr_ok], ytr)
    pred = r.predict(Xte[te_ok])
    rho = spearmanr(pred, yte).correlation
    res[feat] = float(rho)
    m = MODEL.get(feat)
    ms = f"{m:.3f}" if m else "--"
    dl = f"{rho - m:+.3f}" if m else "--"
    print(f"{feat:<14}{rho:>12.3f}{ms:>12}{dl:>9}{len(te_ok):>7}")
rob = [res[f] for f in ROB if f in res]
print(f"\nRIDGE mean over 5 robust = {np.mean(rob):.4f}   |   MODEL = 0.6986   |   "
      f"delta = {np.mean(rob)-0.6986:+.4f}")
json.dump(res, open(f"{B}/skyline_ridge_results.json", "w"), indent=2)
print("wrote skyline_ridge_results.json")
