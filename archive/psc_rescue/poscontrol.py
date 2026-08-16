"""Item 0.18 — POSITIVE CONTROL: what does the metric return for a model that
genuinely grounds?

A model using DEFINING-STATISTIC POOLING predicts per-frame f0 qhat and a keep
mask mhat, then computes f0_sd the way the formula does. Its attribution map is
therefore the analytic sensitivity of ITS OWN computation:

    phihat_t = mhat_t (qhat_t - muhat) / ((Nhat-1) sdhat)

We sweep prediction error and score phihat against the TRUE oracle map. This
calibrates the metric in PREDICTION space (what a real model gets wrong) rather
than map space, and it is the ONLY thing that makes the observed 17%-of-ceiling
interpretable: without it we cannot separate "the model grounds weakly" from
"the metric saturates for everything."
"""
import sys, numpy as np
sys.path.insert(0, "scripts")
from score_attribution import spearman, degrade, boot_ci, mass_concentration

SH = "/ocean/projects/cis260125p/shared"
o = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in o["names"]]
rng = np.random.default_rng(0)


def sd_map(q, keep):
    idx = np.flatnonzero(keep)
    if idx.size < 3:
        return None
    v = q[idx]; mu = v.mean()
    sd = float(np.sqrt(((v - mu) ** 2).sum() / (idx.size - 1)))
    if sd <= 0:
        return None
    phi = np.zeros(q.shape, dtype=float)
    phi[idx] = (v - mu) / ((idx.size - 1) * sd)
    return phi


print(f"{'f0 err':>7} {'voicing err':>12} {'panel':<6} {'spearman':>22} {'mass-conc':>22}   n")
for f0_err, v_err in ((0.00, 0.0), (0.05, 0.0), (0.15, 0.0), (0.30, 0.0),
                      (0.05, 0.10), (0.15, 0.20), (0.30, 0.30)):
    for panel, sel in (("CLEAN", lambda n: n.endswith("_s1clean")),
                       ("MIX", lambda n: not n.endswith("_s1clean"))):
        sp, mc = [], []
        for n in names:
            qk, kk, mk = f"q/{n}", f"keep/{n}", f"f0_sd/{n}"
            if qk not in o or kk not in o or mk not in o or not sel(n):
                continue
            q = np.asarray(o[qk], dtype=float)
            keep = np.asarray(o[kk], dtype=float) > 0
            ref = np.asarray(o[mk], dtype=float)
            if keep.sum() < 5 or ref.size < 8 or np.allclose(ref, ref[0]):
                continue
            # imperfect per-frame f0 prediction
            qh = q.copy()
            voiced = q > 0
            if f0_err > 0:
                qh[voiced] = q[voiced] * (1.0 + f0_err * rng.standard_normal(int(voiced.sum())))
            # imperfect voicing / keep detection
            kh = keep.copy()
            if v_err > 0:
                flip = rng.random(kh.size) < v_err
                kh = np.where(flip, ~kh, kh)
            phih = sd_map(qh, kh)
            if phih is None:
                continue
            d = degrade(phih)[:ref.size]
            sp.append(spearman(d, ref))
            mc.append(mass_concentration(d, ref))
        if len(sp) < 5:
            continue
        s0, sl, sh = boot_ci(np.array(sp, dtype=float), 300)
        m0, ml, mh = boot_ci(np.array(mc, dtype=float), 300)
        print(f"{f0_err:>7.2f} {v_err:>12.2f} {panel:<6} "
              f"{s0:+.4f} [{sl:+.4f},{sh:+.4f}] {m0:+.4f} [{ml:+.4f},{mh:+.4f}]  {len(sp)}")
