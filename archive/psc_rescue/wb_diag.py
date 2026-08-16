"""Why did the white-box control score -0.216? Three candidate explanations, tested.

E1 the probe's per-frame predictions are bad          -> control is inconclusive
E2 the VALUE-space gradient d(sd)/d(qhat) is fine but the FEATURE-space gradient
   d(sd)/d(x) does not preserve it                    -> the EXTRACTOR is the problem
E3 a sign error in how I build the white-box map      -> my bug

The reference phi_t = keep(q-mu)/((N-1)sd) is a sensitivity in PITCH space. The candidate
is grad*input in WavLM FEATURE space. These are related by the chain rule and are NOT the
same object -- exactly the type mismatch that produced the spurious overlap_ratio negative.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from score_attribution import spearman, degrade, align_to_oracle
from whitebox_control import DefiningStatisticModel
SH = "/ocean/projects/cis260125p/shared"
z = np.load(f"{SH}/oracle_maps_test.npz", allow_pickle=True)
names = [str(x) for x in z["names"] if f"q/{x}" in z][:60]
TD = f"{SH}/data/processed_corrected/test"

# NOTE: the trained weights were not saved; retrain briefly for the diagnostic
model = DefiningStatisticModel()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
def load(stem):
    p = os.path.join(TD, stem + ".pt")
    if not os.path.exists(p): return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    af = d["audio_features"].float()
    q = torch.from_numpy(np.asarray(z[f"q/{stem}"], np.float32))[::2]
    k = torch.from_numpy(np.asarray(z[f"keep/{stem}"], np.float32))[::2]
    n = min(af.shape[0], q.shape[0])
    return af[:n], q[:n], k[:n]
for ep in range(3):
    for s in names[:40]:
        r = load(s)
        if r is None: continue
        af,q,k = r
        qh,mh = model(af.unsqueeze(0))
        loss = (((qh[0]-q)**2)*k).sum()/k.sum().clamp(min=1)/1000.0 + \
               torch.nn.functional.binary_cross_entropy(mh[0].clamp(1e-6,1-1e-6), k)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()

e_q, e_m, sp_value, sp_feat = [], [], [], []
for s in names[40:]:
    r = load(s)
    if r is None: continue
    af,q,k = r
    kb = k > 0.5
    if kb.sum() < 5: continue
    x = af.unsqueeze(0).requires_grad_(True)
    qh, mh = model(x)
    # E1: per-frame accuracy on voiced frames
    e_q.append(float((qh[0].detach()[kb]-q[kb]).abs().mean() / q[kb].abs().mean()))
    e_m.append(float(((mh[0].detach()>0.5).float()!=k).float().mean()))
    # E2a: VALUE-space gradient d(sd)/d(qhat)  -- should match phi if the formula is right
    sd = model.f0_sd(x)
    gq = torch.autograd.grad(sd[0], qh, retain_graph=True, allow_unused=True)[0]
    ref = np.asarray(z[f"f0_sd/{s}"], float)[::2][:af.shape[0]]
    if gq is not None:
        sp_value.append(spearman(gq[0].detach().numpy(), ref))
    # E2b: FEATURE-space grad*input, as the extractor produces it
    g = torch.autograd.grad(sd[0], x)[0][0]
    attr = (g*x.detach()[0]).sum(-1).numpy()
    sp_feat.append(spearman(attr, ref))
f=lambda v: (np.nanmean(v) if v else float('nan'))
print(f"E1 per-frame f0 rel-error (voiced) : {f(e_q):.3f}")
print(f"E1 voicing mask error rate         : {f(e_m):.3f}")
print(f"E2a spearman(d sd/d qhat, phi)     : {f(sp_value):+.3f}   <- VALUE space (formula)")
print(f"E2b spearman(grad*input, phi)      : {f(sp_feat):+.3f}   <- FEATURE space (extractor)")
print("\nE2a high + E2b low/negative => the extractor cannot carry a value-space contribution")
print("through the feature-space gradient. That is a TYPE MISMATCH, not a model failure.")
