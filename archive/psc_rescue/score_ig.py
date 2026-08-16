"""Score hnr/shimmer under signed IG. IG beat grad*input on f0_sd (0.1159 -> 0.1285,
7.4% -> 9.2%), so it is the better candidate; this is the last chance for shimmer before
its negative is settled."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_voice_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
CEIL = {"hnr": 0.550, "shimmer": 0.379}
NULL = {"hnr": 0.086, "shimmer": 0.054}
print(f"{'feature':<10}{'cand':<12}{'panel':<7}{'model':>24}{'effect':>9}   n")
for feat in ("hnr", "shimmer"):
    for tag, f in (("gradxinput", "v4_gxi_voice.npz"), ("IG", "v4_ig_voice.npz")):
        try:
            m = np.load(f"{SH}/{f}")
        except FileNotFoundError:
            continue
        for panel, sel in (("CLEAN", lambda n: n.endswith("_s1clean")),
                           ("MIX", lambda n: not n.endswith("_s1clean"))):
            sc = []
            for n in names:
                k = f"{feat}/{n}"
                if k not in o or k not in m or not sel(n):
                    continue
                ref = np.asarray(o[k], dtype=float)
                if ref.size < 8 or np.allclose(ref, ref[0]):
                    continue
                d = degrade(np.asarray(m[k], dtype=float))[:ref.size]
                if d.size >= 8:
                    sc.append(spearman(d, ref))
            if len(sc) < 30:
                continue
            a, lo, hi = boot_ci(np.array(sc, float), 500)
            eff = (a - NULL[feat]) / (CEIL[feat] - NULL[feat]) * 100
            flag = "  <-- ABOVE NULL" if lo > NULL[feat] else ""
            print(f"{feat:<10}{tag:<12}{panel:<7}{a:+.4f} [{lo:+.4f},{hi:+.4f}]{eff:>8.1f}%   {len(sc)}{flag}")
    print()
