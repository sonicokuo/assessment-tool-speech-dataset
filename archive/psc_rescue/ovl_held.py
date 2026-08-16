"""Was overlap_ratio's r=+0.879 the input echo?

With the overlap channel HELD at its alpha=0 state, the model can no longer read the
changed overlap from its input. If tracking survives, it is audio-derived; if it
collapses, the original number was the rho-0.9999 echo.
"""
import json, sys, numpy as np
sys.path.insert(0, "scripts")
from causal_dose_analyze import spearman
SH = "/ocean/projects/cis260125p/shared"

def track(path, label):
    rows = json.load(open(path))
    alphas = sorted({float(k) for r in rows for k in r["doses"]})
    a0 = str(alphas[0])
    X, Y = [], []
    for r in rows:
        b = r["doses"].get(a0)
        if not b:
            continue
        g0, p0 = b.get("ovl_frac"), b["pred"].get("overlap_ratio")
        if g0 is None or p0 is None:
            continue
        for al in alphas[1:]:
            d = r["doses"].get(str(al))
            if not d:
                continue
            g1, p1 = d.get("ovl_frac"), d["pred"].get("overlap_ratio")
            if g1 is None or p1 is None or not np.isfinite(g1) or not np.isfinite(p1):
                continue
            X.append(g1 - g0); Y.append(p1 - p0)
    X, Y = np.array(X, float), np.array(Y, float)
    slope = float(np.polyfit(X, Y, 1)[0])
    r_p = float(np.corrcoef(X, Y)[0, 1])
    rng = np.random.default_rng(0)
    bs = [np.corrcoef(X[i], Y[i])[0, 1] for i in
          (rng.integers(0, X.size, X.size) for _ in range(2000)) if X[i].std() > 1e-12]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {label:<26} n={X.size}  slope={slope:+.4f}  r={r_p:+.4f} [{lo:+.4f},{hi:+.4f}]")
    return r_p

print("=== overlap_ratio: does tracking survive when the input cannot change? ===")
a = track(f"{SH}/causal_dose.json", "input UPDATED (original)")
b = track(f"{SH}/causal_dose_held.json", "input HELD at alpha=0")
print(f"\n  drop = {a - b:+.4f}")
print("  a large drop means the original was substantially the INPUT ECHO;")
print("  a small drop means overlap_ratio is genuinely inferred from AUDIO.")
