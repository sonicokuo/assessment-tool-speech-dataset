import sys, torch
sys.path.insert(0, "/tmp/srctest")
from model.adapter import build_adapter
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, T, D = 3, 400, 1024
a = torch.randn(B, T, D, device=dev); o = torch.randn(B, T, 4, device=dev)
L = torch.tensor([400, 250, 120], device=dev)
a2 = a.clone(); a2[1, 250:] = 99.0; a2[2, 120:] = -99.0

def mk(pool):
    torch.manual_seed(7)
    return build_adapter("film-attn", lm_dim=512, reliability_head=True, aux_pool=pool).to(dev).eval()

print("Is padding leakage PRE-EXISTING (mean) or introduced by linear_softmax?\n")
for pool in ("mean", "linear_softmax"):
    m = mk(pool)
    with torch.no_grad():
        _, (v1, _) = m(a, o, lengths=L)
        _, (v2, _) = m(a2, o, lengths=L)
        # also: does the PREFIX itself change at valid positions?
        p1 = m.inner(a, o, lengths=L); p2 = m.inner(a2, o, lengths=L)
    drift_out = (v1[1:] - v2[1:]).abs().max().item()
    n_valid = [ (int(L[i].item())//8) for i in range(B) ]
    drift_pref = max((p1[i, :n_valid[i]] - p2[i, :n_valid[i]]).abs().max().item() for i in (1, 2))
    print(f"  {pool:<15} output drift {drift_out:8.4f}   | prefix drift at VALID positions {drift_pref:8.4f}")
print("\n(prefix drift > 0 at valid positions => the conv/attention mixes padded input frames")
print(" in at the boundary, which is upstream of ANY pooling choice)")
