"""Items 0.8b / 0.29 / 0.30 — attribution variants, mass concentration, sigma link."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
IDX = {"f0_mean": 2, "f0_sd": 3}          # aux-head index, SUPERVISED_FEATURES order

variants = [("grad", "v2_grad.npz"), ("grad", "v2_grad_zero.npz"),
            ("gradxinput", "v2_gxi.npz"), ("gradxinput", "v2_gxi_zero.npz"),
            ("ig", "v2_ig.npz"), ("ig", "v2_ig_zero.npz")]

print(f"{'feature':<8} {'method':<11} {'input':<7} {'panel':<6} "
      f"{'spearman':>22} {'mass-conc':>22}   n")
for feat in ("f0_sd", "f0_mean"):
    for meth, fname in variants:
        try:
            m = np.load(f"{SH}/{fname}")
        except FileNotFoundError:
            continue
        tag = "zeroed" if ("zeroovl" in fname or "_zero" in fname) else "oracle"
        for panel, sel in (("MIX", lambda n: not n.endswith("_s1clean")),
                           ("CLEAN", lambda n: n.endswith("_s1clean"))):
            sp, mc = [], []
            for n in names:
                k = f"{feat}/{n}"
                if k not in o or k not in m or not sel(n):
                    continue
                ref = np.asarray(o[k], dtype=float)
                if ref.size < 8 or np.allclose(ref, ref[0]):
                    continue
                d = degrade(np.asarray(m[k], dtype=float))[:ref.size]
                sp.append(spearman(d, ref))
                mc.append(mass_concentration(d, ref))
            if len(sp) < 5:
                continue
            s0, sl, sh = boot_ci(np.array(sp, dtype=float), 400)
            m0, ml, mh = boot_ci(np.array(mc, dtype=float), 400)
            print(f"{feat:<8} {meth:<11} {tag:<7} {panel:<6} "
                  f"{s0:+.4f} [{sl:+.4f},{sh:+.4f}] {m0:+.4f} [{ml:+.4f},{mh:+.4f}]  {len(sp)}")

# ---- item 0.29: does grounding track uncertainty? WITHIN CONDITION ----
# A pooled correlation is confounded: clean clips differ from mixtures in BOTH
# grounding and sigma, so pooling measures the condition contrast, not a per-clip
# relationship. This is the same trap that made sigma look like a per-claim ranker
# when it was really a condition detector.
print("\n=== 0.29  per-clip grounding vs per-clip sigma (WITHIN CONDITION) ===")
print("    negative rho = MORE uncertain -> LESS grounded (the predicted direction)")
for meth, fname in variants:
    try:
        m = np.load(f"{SH}/{fname}")
    except FileNotFoundError:
        continue
    if not any(k.startswith("sigma/") for k in m.files):
        continue
    tag = "zeroed" if (("zeroovl" in fname or "_zero" in fname) or "_zero" in fname) else "oracle"
    for feat in ("f0_sd", "f0_mean"):
        for panel, sel in (("MIX", lambda n: not n.endswith("_s1clean")),
                           ("CLEAN", lambda n: n.endswith("_s1clean")),
                           ("POOLED", lambda n: True)):
            g, sg = [], []
            for n in names:
                k, sk = f"{feat}/{n}", f"sigma/{n}"
                if k not in o or k not in m or sk not in m or not sel(n):
                    continue
                ref = np.asarray(o[k], dtype=float)
                if ref.size < 8 or np.allclose(ref, ref[0]):
                    continue
                v = spearman(degrade(np.asarray(m[k], dtype=float))[:ref.size], ref)
                lv = float(np.asarray(m[sk], dtype=float)[IDX[feat]])
                if np.isfinite(v) and np.isfinite(lv):
                    g.append(v); sg.append(lv)
            if len(g) > 50:
                r = spearman(np.array(sg), np.array(g))
                print(f"  {feat:<8} {meth:<11} {tag:<7} {panel:<7} rho = {r:+.4f}   n={len(g)}")
