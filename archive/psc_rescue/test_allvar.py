import sys, torch
sys.path.insert(0, "/tmp/srctest")
from model.adapter import build_adapter
dev = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(2, 400, 1024, device=dev); o = torch.randn(2, 400, 4, device=dev)
VAR = ["concat-only","sigmoid-gate","film","film-attn","film-attn-2L","film-mamba",
       "film-mamba-2L","qformer","attn-concat","mamba-concat"]
bad = []
for v in VAR:
    try:
        # exactly how train.py builds it: compression always passed
        m = build_adapter(v, lm_dim=512, reliability_head=True, compression=8).to(dev).eval()
        with torch.no_grad(): pre, sp = m(a, o)
        mu = sp[0] if isinstance(sp, tuple) else sp
        print(f"  OK   {v:<15} prefix {tuple(pre.shape)}  aux {tuple(mu.shape)}")
    except Exception as e:
        bad.append(v); print(f"  FAIL {v:<15} {type(e).__name__}: {str(e)[:90]}")
print("\nfailures:", bad or "none")
