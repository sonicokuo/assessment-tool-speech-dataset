import sys, torch
sys.path.insert(0, "/tmp/srctest")
from model.adapter import build_adapter
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, T, D = 3, 400, 1024
a = torch.randn(B, T, D, device=dev); o = torch.randn(B, T, 4, device=dev)
L = torch.tensor([400, 250, 120], device=dev)

def mk(pool, rel):
    torch.manual_seed(7)
    return build_adapter("film-attn", lm_dim=512, reliability_head=rel, aux_pool=pool).to(dev).eval()

# 1. default must be byte-identical to the old mean path
m_def, m_mean = mk("mean", True), mk("mean", True)
with torch.no_grad():
    _, (d1, _) = m_def(a, o); _, (d2, _) = m_mean(a, o)
print("1. default==mean, deterministic:", torch.allclose(d1, d2))

# 2. linear_softmax builds and runs, both head types, with and without lengths
for rel in (True, False):
    m = mk("linear_softmax", rel)
    with torch.no_grad():
        out = m(a, o); out_l = m(a, o, lengths=L)
    sp = out[1][0] if rel else out[1]
    spl = out_l[1][0] if rel else out_l[1]
    print(f"2. linear_softmax rel={rel}: {tuple(sp.shape)} / with lengths {tuple(spl.shape)}  finite={torch.isfinite(spl).all().item()}")

# 3. THE POINT: does it differ from mean, and does padding actually get zero weight?
mm, ml = mk("mean", True), mk("linear_softmax", True)
with torch.no_grad():
    _, (vm, _) = mm(a, o, lengths=L); _, (vl, _) = ml(a, o, lengths=L)
print("3. linear_softmax != mean:", not torch.allclose(vm, vl), f"(max abs diff {(vm-vl).abs().max():.4f})")

# 4. padding invariance: garbage in padded frames must NOT change the answer
a2 = a.clone(); a2[1, 250:] = 99.0; a2[2, 120:] = -99.0
with torch.no_grad():
    _, (v1, _) = ml(a, o, lengths=L); _, (v2, _) = ml(a2, o, lengths=L)
print("4. padded frames ignored:", torch.allclose(v1[1:], v2[1:], atol=1e-3),
      f"(max drift {(v1[1:]-v2[1:]).abs().max():.5f})")

# 5. attribution identity still holds: per-frame contributions sum to the emitted value
m = mk("linear_softmax", True)
with torch.no_grad():
    prefix = m.inner(a, o)
    z = m.regress_head(prefix); mean_t = z[0]
    w = mean_t.abs(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
    recon = (w * mean_t).sum(1)
    _, (v, _) = m(a, o)
print("5. attribution sums to emitted value:", torch.allclose(recon, v, atol=1e-4))
