"""Item 0.14 — overlap_ratio against the CONSTANT-PREDICTOR FLOOR on the restricted
mixture range. SRCC alone cannot decide inclusion; nMAE vs floor can."""
import json, csv, numpy as np, sys
sys.path.insert(0, "scripts")
from score_attribution import spearman, boot_ci
SH = "/ocean/projects/cis260125p/shared"
gt = {r["filename"].replace(".wav", ""): r
      for r in csv.DictReader(open(f"{SH}/data/features_corrected_merged/test.csv"))}
res = json.load(open(f"{SH}/checkpoints/full/audioonly_attnconcat_seed73/inference_results.json"))
rows = res if isinstance(res, list) else res.get("results", [])

def get_pred(e):
    a = e.get("_aux") or e.get("aux_mean")
    if isinstance(a, dict):
        a = a.get("aux_mean")
    return float(a[7]) if isinstance(a, list) and len(a) > 7 else None   # overlap_ratio idx 7

for label, sel in (("MIXTURES", lambda s: "_s1clean" not in s),
                   ("CLEAN", lambda s: "_s1clean" in s),
                   ("POOLED", lambda s: True)):
    P, G = [], []
    for e in rows:
        stem = str(e.get("filename", "")).replace(".wav", "")
        if stem not in gt or not sel(stem):
            continue
        p = get_pred(e)
        try:
            g = float(gt[stem]["overlap_ratio"])
        except (TypeError, ValueError):
            continue
        if p is None or not np.isfinite(p) or not np.isfinite(g):
            continue
        P.append(p); G.append(g)
    if len(P) < 20:
        print(f"{label:<9} n={len(P)} too few"); continue
    P, G = np.array(P), np.array(G)
    srcc = spearman(P, G)
    scale = G.std() if G.std() > 0 else 1.0
    nmae = float(np.abs(P - G).mean() / scale)
    floor = float(np.abs(G - G.mean()).mean() / scale)     # constant-predictor floor
    gain = (floor - nmae) / floor * 100 if floor > 0 else float("nan")
    print(f"{label:<9} n={len(P):<5} SRCC {srcc:+.4f}   nMAE {nmae:.3f}   floor {floor:.3f}   "
          f"gain vs constant {gain:+.1f}%   GT range [{G.min():.2f},{G.max():.2f}] sd={G.std():.3f}")
