"""Score the model against the THREE CLEAN references (f0_sd, hnr, shimmer) plus the
boundary pause variant. Effect size = (model - worst trivial null)/(ceiling - null).

Clean = no trivial null (loudness/voicing/random) reproduces the reference:
  f0_sd 0.066 | hnr 0.086 | shimmer 0.054     vs   jitter 0.454 (confounded, excluded)
  pause_count_boundary 0.283 (down from 0.676 under the old uniform-interior reference)
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
SPEC = [
    ("f0_sd",  "oracle_maps_test.npz",   "v4_gxi_signed.npz", 0.743, 0.066, True),
    ("hnr",    "oracle_voice_test.npz",  "v4_gxi_voice.npz",  0.550, 0.086, True),
    ("shimmer","oracle_voice_test.npz",  "v4_gxi_voice.npz",  0.379, 0.054, True),
    ("jitter", "oracle_voice_test.npz",  "v4_gxi_voice.npz",  0.491, 0.454, True),
    ("pause_count_boundary", "oracle_voice_test.npz", "v4_gxi_signed_evt.npz", 0.613, 0.283, False),
]
print(f"{'feature':<22}{'panel':<7}{'model':>22}{'ceil':>7}{'null':>7}{'effect':>9}   n")
for feat, onp, mnp, ceil, null, clean in SPEC:
    try:
        o = np.load(f"{SH}/{onp}", allow_pickle=True); m = np.load(f"{SH}/{mnp}")
    except FileNotFoundError as e:
        print(f"{feat:<22} missing: {e}"); continue
    names = [str(x) for x in o["names"]]
    mkey = "pause_count" if feat == "pause_count_boundary" else feat
    for panel, sel in (("CLEAN", lambda n: n.endswith("_s1clean")),
                       ("MIX", lambda n: not n.endswith("_s1clean"))):
        sc = []
        for n in names:
            ok, mk = f"{feat}/{n}", f"{mkey}/{n}"
            if ok not in o or mk not in m or not sel(n):
                continue
            ref = np.asarray(o[ok], dtype=float)
            if ref.size < 8 or np.allclose(ref, ref[0]):
                continue
            d = degrade(np.asarray(m[mk], dtype=float))[:ref.size]
            if d.size >= 8:
                sc.append(spearman(d, ref))
        if len(sc) < 30:
            continue
        a, lo, hi = boot_ci(np.array(sc, float), 500)
        eff = (a - null) / (ceil - null) * 100
        flag = "  <-- ABOVE NULL" if lo > null else ""
        print(f"{feat:<22}{panel:<7}{a:+.4f} [{lo:+.4f},{hi:+.4f}]{ceil:>7.3f}{null:>7.3f}{eff:>8.1f}%   {len(sc)}{flag}")
print("\neffect = (model - worst trivial null)/(ceiling - null). ABOVE NULL requires the CI")
print("lower bound to clear the null, not just the point estimate.")
