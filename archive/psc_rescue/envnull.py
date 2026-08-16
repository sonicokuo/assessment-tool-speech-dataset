"""How much of the snr map is just the energy envelope? Answerable from the oracle maps
alone -- no model needed. This is the GATE: if the envelope reproduces phi, then any model
that tracks loudness scores well for free and the map row proves nothing."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/snr_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
raw, pooled, flat_b, ceil_list = [], [], [], []
for n in names:
    k, ek = f"snr/{n}", f"energy/{n}"
    if k not in o or ek not in o:
        continue
    phi = np.asarray(o[k], dtype=float)
    env = np.asarray(o[ek], dtype=float)[:phi.size]
    if phi.size < 8 or np.allclose(phi, phi[0]):
        continue
    raw.append(spearman(env, phi))                       # envelope at native 100 Hz
    f = 16
    n2 = (phi.size // f) * f
    if n2 >= f:
        pl = phi[:n2].reshape(-1, f).sum(axis=1)
        ceil_list.append(spearman(np.repeat(pl / f, f), phi[:n2]))
    pooled.append(spearman(degrade(env)[:phi.size], phi))  # envelope with the model's handicap
    # how flat is b_t/B really? (the premise of the whole concern)
    # phi = (10/ln10)(a/A - b/B); if b/B were exactly flat, phi would be an affine map of a/A
    flat_b.append(spearman(env / env.sum(), phi))
for lbl, v in (("CEILING (perfect map @6.25Hz)", ceil_list),
               ("envelope @100Hz vs phi", raw),
               ("envelope POOLED to 6.25Hz vs phi", pooled),
               ("normalised envelope vs phi", flat_b)):
    a, lo, hi = boot_ci(np.array(v, float), 800)
    print(f"  {lbl:<36}{a:+.4f} [{lo:+.4f},{hi:+.4f}]  n={len(v)}")
print("\n  ~1.0 means phi IS the energy envelope and the map row cannot distinguish")
print("  a real SNR reader from a loudness tracker.")
