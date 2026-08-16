"""Score the model's maps for the EVENT features against their exact oracle maps.

speaking_rate's reference is a SPIKE TRAIN (~5.9% support), so Spearman is dominated by
tied zeros — the regime where f0_mean's noise ladder plateaued. Mass-concentration is the
meaningful statistic: fraction of |map| mass inside the true support MINUS the support's
own share of the timeline, so 0.0 = chance at any sparsity.
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_events_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
rng = np.random.default_rng(0)

print(f"{'feature':<14}{'method':<11}{'panel':<7}{'mass-conc (lead)':>24}{'spearman':>24}   n")
for feat in ("speaking_rate", "pause_count"):
    for meth, f in (("grad", "v2_grad_evt.npz"), ("gradxinput", "v2_gxi_evt.npz")):
        try:
            m = np.load(f"{SH}/{f}")
        except FileNotFoundError:
            continue
        for panel, sel in (("MIX", lambda n: not n.endswith("_s1clean")),
                           ("CLEAN", lambda n: n.endswith("_s1clean"))):
            mc, sp, n1mc = [], [], []
            for n in names:
                k = f"{feat}/{n}"
                if k not in o or k not in m or not sel(n):
                    continue
                ref = np.asarray(o[k], dtype=float)
                if ref.size < 8 or not (ref > 0).any() or (ref > 0).all():
                    continue
                d = degrade(np.asarray(m[k], dtype=float))[:ref.size]
                if d.size < 8:
                    continue
                mc.append(mass_concentration(d, ref))
                sp.append(spearman(d, ref))
                n1mc.append(mass_concentration(rng.random(ref.size), ref))   # N1 null
            if len(mc) < 20:
                continue
            a, lo, hi = boot_ci(np.array(mc, float), 800)
            n = len(mc)
            s0, sl, sh = boot_ci(np.array(sp, float), 800)
            nn, nl, nh = boot_ci(np.array(n1mc, float), 800)
            flag = "  ABOVE-CHANCE" if lo > nh else ("  at chance" if lo <= nh <= hi else "")
            print(f"{feat:<14}{meth:<11}{panel:<7}{a:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"{s0:+.4f} [{sl:+.4f},{sh:+.4f}]  {n}{flag}")
            print(f"{'':<32}N1 random mass-conc {nn:+.4f} [{nl:+.4f},{nh:+.4f}]")
