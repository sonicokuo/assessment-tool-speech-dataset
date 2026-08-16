import numpy as np, sys
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
for feat in ("f0_sd", "f0_mean"):
    for tag, f in (("ORACLE-INPUT", "saliency_test.npz"), ("ZEROED-OVL", "saliency_test_zeroovl.npz")):
        m = np.load(f"{SH}/{f}")
        row = {}
        for panel, sel in (("MIX", lambda n: not n.endswith("_s1clean")),
                           ("CLEAN", lambda n: n.endswith("_s1clean"))):
            sc = []
            for n in names:
                k = f"{feat}/{n}"
                if k not in o or k not in m or not sel(n):
                    continue
                ref = np.asarray(o[k], dtype=float)
                if ref.size < 8 or np.allclose(ref, ref[0]):
                    continue
                sc.append(spearman(degrade(np.asarray(m[k], dtype=float))[:ref.size], ref))
            row[panel] = boot_ci(np.array(sc, dtype=float), 600) + (len(sc),)
        mi, cl = row["MIX"], row["CLEAN"]
        print(f"{feat:<9} {tag:<13} MIX {mi[0]:+.4f} [{mi[1]:+.4f},{mi[2]:+.4f}] n={mi[3]:<5} "
              f"CLEAN {cl[0]:+.4f} [{cl[1]:+.4f},{cl[2]:+.4f}] n={cl[3]}")
