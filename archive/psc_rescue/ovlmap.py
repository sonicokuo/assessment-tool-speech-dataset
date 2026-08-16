"""Score the model's overlap_ratio MAP against its exact oracle map.

Never run before. The oracle map is the overlap indicator (exact, ceiling 0.883).
The ORACLE-INPUT capture is contaminated -- overlap_info channel 0 IS this map, so a
model echoing its input scores perfectly for free. The ZEROED capture is the honest one.
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
rng = np.random.default_rng(0)
print(f"{'method':<12}{'input':<9}{'panel':<7}{'spearman':>24}{'mass-conc':>24}   n")
for meth, f in (("grad", "v2_grad.npz"), ("grad", "v2_grad_zero.npz"),
                ("gradxinput", "v2_gxi.npz"), ("gradxinput", "v2_gxi_zero.npz")):
    try:
        m = np.load(f"{SH}/{f}")
    except FileNotFoundError:
        continue
    tag = "ZEROED" if "_zero" in f else "oracle"
    sp, mc, n1 = [], [], []
    for n in names:
        if n.endswith("_s1clean"):
            continue                      # clean twins have an all-zero overlap map
        k = f"overlap_ratio/{n}"
        if k not in o or k not in m:
            continue
        ref = np.asarray(o[k], dtype=float)
        if ref.size < 8 or not (ref > 0).any() or (ref > 0).all():
            continue
        d = degrade(np.asarray(m[k], dtype=float))[:ref.size]
        sp.append(spearman(d, ref)); mc.append(mass_concentration(d, ref))
        n1.append(mass_concentration(rng.random(ref.size), ref))
    if len(sp) < 20:
        continue
    s0, sl, sh = boot_ci(np.array(sp, float), 600)
    m0, ml, mh = boot_ci(np.array(mc, float), 600)
    nn, nl, nh = boot_ci(np.array(n1, float), 600)
    print(f"{meth:<12}{tag:<9}{'MIX':<7}{s0:+.4f} [{sl:+.4f},{sh:+.4f}]{m0:+.4f} [{ml:+.4f},{mh:+.4f}]  {len(sp)}")
    print(f"{'':<28}N1 random mass-conc {nn:+.4f} [{nl:+.4f},{nh:+.4f}]   (ceiling spearman ~0.883)")
