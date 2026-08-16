import sys, torch
sys.path.insert(0, "/tmp/srctest")
from model.adapter import build_adapter
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, T, D = 2, 400, 1024
a = torch.randn(B, T, D, device=dev); o = torch.randn(B, T, 4, device=dev)
def mk(v, **kw): return build_adapter(v, lm_dim=512, reliability_head=True, **kw).to(dev).eval()
def np_(m): return sum(p.numel() for p in m.parameters())
res = {}
for v in ("film-attn", "attn-concat", "film-mamba", "mamba-concat"):
    m = mk(v)
    with torch.no_grad(): pre, (mu, lv) = m(a, o)
    res[v] = (tuple(pre.shape), tuple(mu.shape), np_(m))
    print(f"{v:<14} prefix {res[v][0]}  aux {res[v][1]}  params {res[v][2]:,}")
# the ONLY structural difference between film-attn and attn-concat must be conditioning
fa = dict(mk("film-attn").named_parameters()); ac = dict(mk("attn-concat").named_parameters())
only_fa = sorted(k for k in fa if k not in ac); only_ac = sorted(k for k in ac if k not in fa)
print("\nparams only in film-attn :", only_fa)
print("params only in attn-concat:", only_ac)
shared_same = all(fa[k].shape == ac[k].shape for k in fa if k in ac)
print("all shared params identical shape:", shared_same, f"({len(set(fa)&set(ac))} shared)")
# lengths path still works on the new arm
m = mk("attn-concat")
with torch.no_grad(): p, _ = m(a, o, lengths=torch.tensor([400,200], device=dev))
print("attn-concat + lengths (masked pool):", tuple(p.shape))
