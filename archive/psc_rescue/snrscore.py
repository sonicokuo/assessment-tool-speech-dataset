"""Score the model's snr map against the exact oracle map, with the MANDATORY null.

THE ENVELOPE NULL IS THE GATE. phi_t = (10/ln10)(a_t/A - b_t/B), and with quasi-stationary
WHAM noise b_t/B is near-flat, so phi is dominated by the SPEECH ENERGY ENVELOPE. A model
with no notion of SNR scores well simply by tracking loudness. If the raw envelope scores
at or above the model, the map row is uninformative and the closed-form dose test has to
carry snr instead. This is the overlap-echo problem in new clothes, so it gets the same
N5-style treatment.

snr's map is a RATIO -- signed and zero-sum -- so per the attribution algebra:
  * mass-concentration is MEANINGLESS here (sum phi = 0); Spearman only.
  * the model's UNSIGNED saliency cannot match a signed reference in general, so we also
    report the score against |phi| as the interpretable variant.
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci
SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/snr_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
print(f"snr oracle maps: {len(names)} clips (real-RIR subset)\n")
rng = np.random.default_rng(0)

def run(tag, path):
    try:
        m = np.load(f"{SH}/{path}")
    except FileNotFoundError:
        return
    sp_signed, sp_abs, env_null, rnd_null, ceil = [], [], [], [], []
    for n in names:
        k, ek = f"snr/{n}", f"energy/{n}"
        mk = f"snr/{n}"
        if k not in o or ek not in o or mk not in m:
            continue
        phi = np.asarray(o[k], dtype=float)
        if phi.size < 8 or np.allclose(phi, phi[0]):
            continue
        d = degrade(np.asarray(m[mk], dtype=float))[:phi.size]
        if d.size < 8:
            continue
        sp_signed.append(spearman(d, phi))
        sp_abs.append(spearman(d, np.abs(phi)))
        env = np.asarray(o[ek], dtype=float)[:phi.size]
        env_null.append(spearman(degrade(env)[:phi.size], phi))     # THE GATE
        rnd_null.append(spearman(degrade(rng.random(phi.size))[:phi.size], phi))
        f = 16
        pooled = phi[:(phi.size // f) * f].reshape(-1, f).sum(axis=1)
        ceil.append(spearman(np.repeat(pooled / f, f), phi[:pooled.size * f]))
    if len(sp_signed) < 20:
        print(f"{tag}: n={len(sp_signed)} too few"); return
    for lbl, v in (("MODEL vs signed phi", sp_signed), ("MODEL vs |phi|", sp_abs),
                   ("ENERGY-ENVELOPE NULL", env_null), ("N1 random", rnd_null),
                   ("CEILING (perfect map)", ceil)):
        a, lo, hi = boot_ci(np.array(v, float), 600)
        print(f"  {tag:<14}{lbl:<24}{a:+.4f} [{lo:+.4f},{hi:+.4f}]  n={len(v)}")
    print()

for tag, p in (("grad", "v2_grad_snr.npz"), ("gradxinput", "v2_gxi_snr.npz"),
               ("grad-ZEROED", "v2_grad_snr_zero.npz")):
    run(tag, p)
print("GATE: if ENERGY-ENVELOPE NULL >= MODEL, the snr map row is uninformative and the")
print("closed-form dose test (snr_db - 20log10(c)) carries the snr story instead.")
