"""Does the overlap map score track clip overlap ratio v, flipping sign near v=0.5?

EXACT DERIVATION being tested. For v = sum_t o_t / T, the single-frame deletion
contribution is
        Delta_t = (o_t - v) / (T - 1)
so overlapped frames get +(1-v)/(T-1) and NON-overlapped get -v/(T-1). Unsigned
magnitudes are (1-v) vs v, so for v > 0.5 the NON-overlapped frames carry the LARGER
unsigned contribution and an IDEAL estimator's unsigned saliency is ANTI-ALIGNED with
the overlap mask. Our test mixtures are 99.1% above v=0.5.

PREDICTION: per-clip spearman(saliency, mask) should DECREASE with v and cross zero
near v = 0.5. If it does, the observed -0.15/-0.46 is the ideal-estimator signature,
not a model failure, and the framework is validated rather than embarrassed.
"""
import csv, sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
gt = {r["filename"].replace(".wav", ""): r
      for r in csv.DictReader(open(f"{SH}/data/features_corrected_merged/test.csv"))}

for tag, f in (("oracle-input", "v2_grad.npz"), ("ZEROED-input", "v2_grad_zero.npz")):
    m = np.load(f"{SH}/{f}")
    V, R = [], []
    for n in names:
        if n.endswith("_s1clean"):
            continue
        k = f"overlap_ratio/{n}"
        if k not in o or k not in m or n not in gt:
            continue
        try:
            v = float(gt[n]["overlap_ratio"])
        except (TypeError, ValueError):
            continue
        ref = np.asarray(o[k], dtype=float)
        if ref.size < 8 or not (ref > 0).any() or (ref > 0).all():
            continue
        r = spearman(degrade(np.asarray(m[k], dtype=float))[:ref.size], ref)
        if np.isfinite(r):
            V.append(v); R.append(r)
    V, R = np.array(V), np.array(R)
    print(f"\n=== {tag}  n={V.size}   clip overlap v range [{V.min():.2f}, {V.max():.2f}] ===")
    print(f"    rho(v, per-clip map score) = {spearman(V, R):+.4f}   "
          f"(theory: NEGATIVE -- higher overlap => more anti-aligned)")
    edges = [0.0, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
    print(f"    {'v bin':<14}{'n':>6}{'mean map score':>18}")
    for a, b in zip(edges[:-1], edges[1:]):
        s = (V >= a) & (V < b)
        if s.sum() >= 5:
            print(f"    [{a:.2f},{b:.2f}){s.sum():>6}{R[s].mean():>18.4f}")
    lo, hi = R[V < 0.55], R[V >= 0.75]
    if lo.size >= 5 and hi.size >= 5:
        print(f"    low-v (<0.55) mean {lo.mean():+.4f}   high-v (>=0.75) mean {hi.mean():+.4f}"
              f"   difference {lo.mean()-hi.mean():+.4f}")
