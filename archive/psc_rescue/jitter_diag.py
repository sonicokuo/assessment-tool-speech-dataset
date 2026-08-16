"""Why is jitter 92% energy-explained when shimmer (0.054) is clean?

Both are per-period measures from the SAME Praat point process, so a large asymmetry
suggests MY aggregation rather than the feature. Hypothesis: my jitter map places
deviation mass at pulse times weighted by |dT|, and large TIMING deviations cluster at
voicing onsets/offsets -- exactly where energy transitions. That would make the map a
boundary detector, not a jitter localiser.

Structural comparison only: no audio, no model.
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman
SH = "/ocean/projects/cis260125p/shared"
z = np.load(f"{SH}/oracle_voice_test.npz", allow_pickle=True)
names = [str(x) for x in z["names"]][:400]

def stats(feat):
    sp, frac, edge, xcorr = [], [], [], []
    for n in names:
        k, ks = f"{feat}/{n}", f"shimmer/{n}"
        if k not in z:
            continue
        v = np.asarray(z[k], dtype=float)
        if v.size < 20 or np.allclose(v, 0):
            continue
        nz = np.abs(v) > 0
        frac.append(nz.mean())                       # support sparsity
        # is the mass at the EDGES of contiguous support runs?
        s = nz.astype(int)
        d = np.diff(np.concatenate([[0], s, [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs = ends - starts
        edge.append(float(np.median(runs)) if runs.size else np.nan)
        if ks in z and feat != "shimmer":
            w = np.asarray(z[ks], dtype=float)[:v.size]
            xcorr.append(spearman(np.abs(v), np.abs(w[:v.size])))
    return (np.nanmean(frac), np.nanmedian(edge), np.nanmean(xcorr) if xcorr else np.nan, len(frac))

print(f"{'feature':<10}{'support frac':>14}{'median run (frames)':>22}{'|corr| with shimmer':>22}   n")
for f in ("jitter", "shimmer", "hnr"):
    fr, rn, xc, n = stats(f)
    print(f"{f:<10}{fr:>14.3f}{rn:>22.1f}{xc:>22.3f}   {n}")
print("\nIf jitter's support is much SPARSER and its runs SHORTER than shimmer's, the map is")
print("sitting on isolated pulses (plausibly at voicing boundaries) rather than spread over")
print("the voiced region -- an aggregation artifact, not a property of jitter.")
