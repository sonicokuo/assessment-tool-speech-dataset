"""overlap_ratio was MISCLASSIFIED as FIXED in the causal test.

Its GT is stem-VAD, so attenuating s2 changes it — and the dose-response ALREADY
recorded that change as `ovl_frac` (0.696 -> 0.000 across alpha). So the sensitivity
test is free: does the model's overlap_ratio prediction track its own moving GT?
"""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from causal_dose_analyze import spearman, boot
SH = "/ocean/projects/cis260125p/shared"
rows = json.load(open(f"{SH}/causal_dose.json"))
alphas = sorted({float(k) for r in rows for k in r["doses"]})
a0 = str(alphas[0])

X, Y, per_clip = [], [], []
for r in rows:
    b = r["doses"].get(a0)
    if not b:
        continue
    g0, p0 = b.get("ovl_frac"), b["pred"].get("overlap_ratio")
    if g0 is None or p0 is None:
        continue
    xs, ys = [], []
    for al in alphas[1:]:
        d = r["doses"].get(str(al))
        if not d:
            continue
        g1, p1 = d.get("ovl_frac"), d["pred"].get("overlap_ratio")
        if g1 is None or p1 is None or not np.isfinite(g1) or not np.isfinite(p1):
            continue
        X.append(g1 - g0); Y.append(p1 - p0)
        xs.append(g1 - g0); ys.append(p1 - p0)
    if len(xs) >= 3:
        per_clip.append(spearman(np.array(xs), np.array(ys)))

X, Y = np.array(X, float), np.array(Y, float)
print(f"=== overlap_ratio SENSITIVITY (was wrongly classified FIXED) ===")
print(f"    n pairs = {X.size}   |dGT| median = {np.median(np.abs(X)):.4f}")
slope = float(np.polyfit(X, Y, 1)[0])
r_p = float(np.corrcoef(X, Y)[0, 1])
rng = np.random.default_rng(0)
bs = [np.corrcoef(X[i], Y[i])[0, 1] for i in (rng.integers(0, X.size, X.size) for _ in range(2000)) if X[i].std() > 1e-12]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"    pooled: slope={slope:+.4f}  r={r_p:+.4f} [{lo:+.4f},{hi:+.4f}]")
m, l, h = boot(np.array(per_clip, float), 2000)
print(f"    per-clip spearman(dGT, dyhat): {m:+.4f} [{l:+.4f},{h:+.4f}]   n_clips={len(per_clip)}")
print(f"    slope 1.0 would be perfect tracking; the model predicts a RATIO so units match.")
