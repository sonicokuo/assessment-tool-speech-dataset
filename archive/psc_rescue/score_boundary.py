"""R1 verdict: does the BOUNDARY reference rescue pause_count?

Old reference (uniform mass over quiet interiors): ceiling 0.917, worst null 0.676
(inverted VAD), model 0.524 -> BELOW the trivial baseline. Retracted.

Boundary reference (mass at threshold CROSSINGS, weighted toward pauses near the 0.3 s
rule): ceiling 0.613, worst null 0.283. Much less trivially predictable. Does the model
clear it?
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_voice_test.npz", allow_pickle=True)
m = np.load(f"{SH}/v4_gxi_signed_evt.npz")
names = [str(x) for x in o["names"]]
CEIL, NULL = 0.613, 0.283
print(f"{'panel':<8}{'model':>24}{'ceiling':>9}{'null':>8}{'effect':>9}   n")
for panel, sel in (("CLEAN", lambda n: n.endswith("_s1clean")),
                   ("MIX", lambda n: not n.endswith("_s1clean"))):
    sc = []
    for n in names:
        ok, mk = f"pause_count_boundary/{n}", f"pause_count/{n}"
        if ok not in o or mk not in m or not sel(n):
            continue
        ref = np.asarray(o[ok], dtype=float)
        if ref.size < 8 or np.allclose(ref, ref[0]):
            continue
        d = degrade(np.asarray(m[mk], dtype=float))[:ref.size]
        if d.size >= 8:
            sc.append(spearman(d, ref))
    if len(sc) < 30:
        print(f"{panel:<8} n={len(sc)} too few"); continue
    a, lo, hi = boot_ci(np.array(sc, float), 600)
    eff = (a - NULL) / (CEIL - NULL) * 100
    flag = "  <-- ABOVE NULL (RESCUED)" if lo > NULL else "  below null"
    print(f"{panel:<8}{a:+.4f} [{lo:+.4f},{hi:+.4f}]{CEIL:>9.3f}{NULL:>8.3f}{eff:>8.1f}%   {len(sc)}{flag}")
print("\nFor reference, the OLD uniform-interior reference: model 0.524 vs null 0.676 = below.")
