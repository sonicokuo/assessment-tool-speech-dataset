"""Price the VALUE-DEPENDENT features against the same trivial nulls.

PREDICTION (falsifiable, stated before running): support-defined references are trivially
predictable (pause_count -energy 0.576 / voicing 0.676; f0_mean voicing 0.463), while
value-dependent ones are not (f0_sd all nulls |0.066|). hnr, jitter and shimmer are
value-dependent means, so their nulls should be SMALL. If they are, the taxonomy row is
about the FEATURES; if they are not, it was about my metric.

Also scores the pause_count BOUNDARY variant: if a count's influence really sits at
threshold crossings rather than over quiet interiors, the energy null should stop
explaining it.
"""
import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
CLEAN = f"{SH}/data/audio_corrected/test-s1clean"
FRAME = 160

def envelope(stem):
    base = stem if stem.endswith("_s1clean") else stem + "_s1clean"
    p = os.path.join(CLEAN, base + ".wav")
    if not os.path.exists(p):
        return None
    x, _ = sf.read(p)
    x = x.mean(1) if x.ndim > 1 else x
    nf = len(x) // FRAME
    if nf < 8:
        return None
    e = (x[:nf * FRAME].reshape(nf, FRAME) ** 2).mean(axis=1) + 1e-12
    return 10 * np.log10(e)

z = np.load(f"{SH}/oracle_voice_test.npz", allow_pickle=True)
names = [str(x) for x in z["names"]]
rng = np.random.default_rng(0)
print(f"{'feature':<22}{'CEILING':>9}{'N1 rand':>10}{'-energy':>10}{'+energy':>10}{'voicing':>10}   n")
for feat in ("hnr", "jitter", "shimmer", "pause_count_boundary"):
    C, R, NE, PE, V = [], [], [], [], []
    for n in names:
        k = f"{feat}/{n}"
        if k not in z:
            continue
        ref = np.asarray(z[k], dtype=float)
        if ref.size < 8 or np.allclose(ref, ref[0]):
            continue
        edb = envelope(n)
        if edb is None:
            continue
        m = min(edb.size, ref.size)
        ref, edb = ref[:m], edb[:m]
        vad = (edb > (edb.max() - 25.0)).astype(float)
        C.append(spearman(degrade(ref), ref))
        R.append(spearman(degrade(rng.random(m)), ref))
        NE.append(spearman(degrade(-edb), ref))
        PE.append(spearman(degrade(edb), ref))
        V.append(spearman(degrade(vad), ref))
        if len(C) >= 600:
            break
    if len(C) < 30:
        print(f"{feat:<22} n={len(C)} too few"); continue
    g = lambda v: boot_ci(np.array(v, float), 400)[0]      # noqa: E731
    print(f"{feat:<22}{g(C):>9.3f}{g(R):>10.3f}{g(NE):>10.3f}{g(PE):>10.3f}{g(V):>10.3f}   {len(C)}")
print("\nSmall nulls => the reference is NOT purchasable from energy/voicing => the feature")
print("can support a localisation claim. Large nulls => it cannot, like pause_count.")
