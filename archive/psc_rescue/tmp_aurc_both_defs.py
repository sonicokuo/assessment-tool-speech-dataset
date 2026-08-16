import csv, json, math, sys
import numpy as np
sys.path.insert(0, "/ocean/projects/cis260125p/shared/repo_verify/src")
from data.feature_set import FEATURE_NAMES, FEATURE_SCALES, SUPERVISED_FEATURES, ILL_POSED_UNDER_OVERLAP_FEATURES
COL = {n: c for n, c, _ in SUPERVISED_FEATURES}
SH = "/ocean/projects/cis260125p/shared"
cap = json.load(open(f"{SH}/sigma_capture_fw2_TEST.json"))
gt = {r["filename"]: r for r in csv.DictReader(open(f"{SH}/data/features_corrected_merged/test.csv"))}
sc = dict(zip(FEATURE_NAMES, FEATURE_SCALES))
ROBUST5 = ["snr", "srmr", "speaking_rate", "pause_count", "pause_rate"]
ILL = ["f0_mean", "f0_sd", "jitter", "shimmer", "hnr"]
def num(v):
    try:
        f = float(v); return f if math.isfinite(f) else None
    except Exception: return None
def aurc_ordered(e):
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))
print("feature           n | binary(k=1) gain% | continuous |err| gain%")
res = {}
for i, f in enumerate(FEATURE_NAMES):
    E, SG = [], []
    for k, v in cap.items():
        r = gt.get(k)
        if not r: continue
        y = num(r.get(COL[f]))
        if y is None: continue
        E.append(abs(float(v["aux_mean"][i]) - y))
        SG.append(math.exp(0.5 * float(v["aux_log_var"][i])))
    E, SG = np.array(E), np.array(SG)
    if len(E) < 100: continue
    order = np.argsort(SG, kind="stable")
    rng = np.random.default_rng(0)
    # binary (original 2026-07-29 def): miscoverage at k=1.0 in FEATURE_SCALES units
    L = (E / sc[f] > 1.0).astype(float)
    b_sig = aurc_ordered(L[order])
    b_rand = float(np.mean([aurc_ordered(L[rng.permutation(len(L))]) for _ in range(20)]))
    b_gain = 100*(b_rand-b_sig)/b_rand if b_rand > 0 else float("nan")
    # continuous (aurc_eval def): raw |err|
    rng = np.random.default_rng(0)
    c_sig = aurc_ordered(E[order])
    c_rand = float(np.mean([aurc_ordered(E[rng.permutation(len(E))]) for _ in range(20)]))
    c_gain = 100*(c_rand-c_sig)/c_rand
    res[f] = (b_gain, c_gain)
    tag = " *ILL*" if f in ILL_POSED_UNDER_OVERLAP_FEATURES else ""
    print(f"{f:<14}{len(E):>6} | {b_gain:>13.1f} | {c_gain:>18.1f}{tag}")
for panel, feats in (("ROBUST5", ROBUST5), ("ILL-POSED", ILL)):
    bg = [res[f][0] for f in feats if f in res and np.isfinite(res[f][0])]
    cg = [res[f][1] for f in feats if f in res and np.isfinite(res[f][1])]
    print(f"{panel}: binary mean {np.mean(bg):+.1f}%  |  continuous mean {np.mean(cg):+.1f}%")
