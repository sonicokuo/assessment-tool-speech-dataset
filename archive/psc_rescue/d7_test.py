"""D7: does removing the Praat grid offset change the panel? It should only ever HELP the
model, since the offset penalises the candidate and nothing else."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, align_to_oracle
SH = "/ocean/projects/cis260125p/shared"
SPEC = [("f0_sd", "oracle_maps_test.npz", "v4_ig_signed.npz", 0.743, 0.066),
        ("hnr", "oracle_voice_test.npz", "v4_ig_voice.npz", 0.550, 0.086)]
print(f"{'feature':<9}{'offset':<9}{'model':>24}{'effect':>9}   n")
for feat, onp, mnp, ceil, null in SPEC:
    o = np.load(f"{SH}/{onp}", allow_pickle=True); m = np.load(f"{SH}/{mnp}")
    names = [str(x) for x in o["names"]]
    for off in (0, 1, 2, 3):
        sc = []
        for n in names:
            k = f"{feat}/{n}"
            if k not in o or k not in m or not n.endswith("_s1clean"):
                continue
            ref = np.asarray(o[k], dtype=float)
            if ref.size < 8 or np.allclose(ref, ref[0]):
                continue
            cand = align_to_oracle(np.asarray(m[k], dtype=float), off)
            d = degrade(cand)[:ref.size]
            if d.size >= 8:
                sc.append(spearman(d, ref))
        if len(sc) < 30:
            continue
        a, lo, hi = boot_ci(np.array(sc, float), 400)
        print(f"{feat:<9}{off:<9}{a:+.4f} [{lo:+.4f},{hi:+.4f}]{(a-null)/(ceil-null)*100:>8.1f}%   {len(sc)}")
    print()
