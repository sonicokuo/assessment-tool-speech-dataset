"""D1 verdict: does the SIGNED reduction rescue f0_sd?

f0_sd's reference phi_t = keep_t(q_t-mu)/((N-1)sd) is SIGN-VARYING, so an unsigned
candidate is structurally capped against it. capture_saliency applied ||.||_2 over the
1024 dims to ALL methods, destroying the sign that makes IG/grad*input contributions.
Signed = attr.sum(dim=-1), which preserves completeness.

Reported against the trivial nulls measured on the same panel, so the margin is honest:
f0_sd CLEAN nulls are |0.066| or below; ceiling 0.743.
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
CEIL = {"f0_sd": {"CLEAN": 0.743, "MIX": 0.545}, "f0_mean": {"CLEAN": 0.719, "MIX": 0.663}}
NULL = {"f0_sd": 0.066, "f0_mean": 0.488}      # strongest trivial null on the CLEAN panel

print(f"{'feature':<9}{'reduction':<11}{'panel':<7}{'score':>22}{'vs ceiling':>12}{'above null':>12}")
for feat in ("f0_sd", "f0_mean"):
    for tag, f in (("UNSIGNED", "v2_gxi.npz"), ("SIGNED", "v4_gxi_signed.npz"),
               ("SIGNED-IG", "v4_ig_signed.npz")):
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
            c = CEIL[feat][panel]
            nl = NULL[feat] if panel == "CLEAN" else 0.03
            eff = (a - nl) / (c - nl) * 100
            print(f"{feat:<9}{tag:<11}{panel:<7}{a:+.4f} [{lo:+.4f},{hi:+.4f}]{a/c*100:>11.1f}%{eff:>11.1f}%")
    print()
print("'above null' = (score - strongest trivial null)/(ceiling - null); the honest effect size.")

# GUARD: v3 produced "signed" files byte-identical to unsigned because the fix was never
# transferred. If UNSIGNED and SIGNED agree to 4 decimals on every cell, that has recurred.
try:
    u = np.load(f"{SH}/v2_gxi.npz"); g = np.load(f"{SH}/v4_gxi_signed.npz")
    ks = [k for k in u.files if k.startswith("f0_sd/")][:50]
    same = sum(np.allclose(u[k], g[k]) for k in ks if k in g.files)
    print(f"\nSANITY: {same}/{len(ks)} sampled maps IDENTICAL between unsigned and signed"
          f"  {'<-- FIX DID NOT TAKE EFFECT' if same > len(ks)//2 else '(differ, as expected)'}")
except Exception as e:
    print(f"sanity check unavailable: {e}")
