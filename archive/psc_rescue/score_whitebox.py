"""Does the extractor recover the ceiling for a model that grounds BY CONSTRUCTION?

ceiling 0.743 | trivial null 0.066 | real model 10.7% (signed IG, D7-corrected)
  white-box near ceiling -> extractor faithful, 10.7% is a real weak result
  white-box near 10%     -> the EXTRACTOR is the bottleneck and no attribution number
                            in this project is readable yet
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, align_to_oracle
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
m = np.load(f"{SH}/whitebox_f0sd.npz")
SPACES = [("FEATURE-space (our extractor)", "f0_sd"), ("VALUE-space (type-matched)", "value")]
names = [str(x) for x in o["names"]]
CEIL, NULL = 0.743, 0.066
for label, pfx in SPACES:
  for panel, sel in (("CLEAN", lambda n: n.endswith("_s1clean")),
                     ("MIX", lambda n: not n.endswith("_s1clean"))):
      sc = []
      for n in names:
          k = f"f0_sd/{n}"; mk = f"{pfx}/{n}"
          if k not in o or mk not in m or not sel(n):
              continue
          ref = np.asarray(o[k], dtype=float)
          if ref.size < 8 or np.allclose(ref, ref[0]):
              continue
          d = degrade(align_to_oracle(np.asarray(m[mk], dtype=float), 2))[:ref.size]
          if d.size >= 8:
              sc.append(spearman(d, ref))
      if len(sc) < 20:
          print(f"{label} {panel}: n={len(sc)} too few"); continue
      a, lo, hi = boot_ci(np.array(sc, float), 500)
      print(f"WHITE-BOX {label:<30} {panel:<6} {a:+.4f} [{lo:+.4f},{hi:+.4f}]  "
            f"= {(a-NULL)/(CEIL-NULL)*100:.1f}% of achievable   n={len(sc)}")
print("\nreal model for comparison: f0_sd CLEAN 10.7%")
