import sys, torch
sys.path.insert(0, "/tmp/srctest")
from model.adapter import build_adapter
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, T, D = 2, 400, 1024
a = torch.randn(B, T, D, device=dev); o = torch.randn(B, T, 4, device=dev)
def mk(v, **kw): return build_adapter(v, lm_dim=512, reliability_head=True, **kw).to(dev).eval()
for comp in (8, 4):
    m = mk('film-attn', compression=comp)
    with torch.no_grad(): pre, (mu, lv) = m(a, o)
    print(f'film-attn compression={comp}: T={T} -> prefix {tuple(pre.shape)}  tok/s={50/comp:.2f}  aux {tuple(mu.shape)}')
m8 = mk('film-attn')
with torch.no_grad(): p8, _ = m8(a, o)
print('DEFAULT (no compression arg) prefix:', tuple(p8.shape), '-> equals compression=8:', p8.shape[1] == 49)
m4 = mk('film-attn', compression=4)
with torch.no_grad(): p4, (mu4, _) = m4(a, o, lengths=torch.tensor([400, 200], device=dev))
print('compression=4 + lengths (masked pool):', tuple(p4.shape), 'aux', tuple(mu4.shape))
for comp in (8, 4):
    mm = mk('film-mamba', compression=comp)
    with torch.no_grad(): pm, _ = mm(a, o)
    print(f'film-mamba compression={comp}: prefix {tuple(pm.shape)}')
print('get_output_length(400): comp8 ->', m8.inner.compressor.get_output_length(400),
      '| comp4 ->', m4.inner.compressor.get_output_length(400))
