"""A1 probe: is the SelfAttentionContextBlock delocalizing the prefix?

Runs CPU-only on a handful of clips. Four probes:
  P0  checkpoint metadata + head bias magnitudes (how much of the DC term is bias).
  P1  attention matrix stats: row entropy, self/diagonal mass, column concentration.
  P2  effective receptive field: zero one 8-frame (160 ms) input window, measure the
      L2 delta per token at conv-out (must be exactly local), post-context, final
      prefix, and per-frame head output z. Participation ratio: 1/N = one token,
      1.0 = uniformly spread.
  P3  gradient saliency at the PRE-context conv output: ||d v_f / d conv_t||;
      concentration + AUC vs overlap. If this is concentrated while the prefix-level
      map is flat, the head-side flatness is post-mixing and a pre-context head could
      localize.
  P4  token-norm artifact: corr(||prefix_t||, |z_dev_t,f|) per feature; also
      corr with pooled WavLM-frame norm (weak energy proxy) and the overlap mask.

Usage: python a1_probe.py <ckpt> <test_dir> [n_clips]
"""
import glob, os, sys, types
import numpy as np

# mamba_ssm needs CUDA at import on some builds; the attn variants never touch it.
try:
    raise ImportError
except Exception:
    m = types.ModuleType("mamba_ssm")
    class _NoMamba:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("Mamba unavailable in this probe env")
    m.Mamba = _NoMamba
    sys.modules["mamba_ssm"] = m

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from model.adapter import build_adapter          # noqa: E402
from data.feature_set import FEATURE_NAMES        # noqa: E402

torch.manual_seed(0)
CKPT, TEST = sys.argv[1], sys.argv[2]
NCLIP = int(sys.argv[3]) if len(sys.argv) > 3 else 12

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
print("P0 checkpoint:", os.path.basename(os.path.dirname(CKPT)))
for k in ("adapter_variant", "aux_pool", "seed", "reliability_head", "compression",
          "lambda_mse", "lambda_nll", "epochs"):
    print(f"   {k} = {cfg.get(k)}")
adapter = build_adapter(cfg["adapter_variant"], lm_dim=4096,
                        reliability_head=bool(cfg.get("reliability_head", False)),
                        compression=int(cfg.get("compression", 8)),
                        aux_pool=str(cfg.get("aux_pool", "mean")))
missing, unexpected = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
assert not [k for k in missing if "head" in k or "regress" in k], missing
adapter = adapter.eval()
NF = len(FEATURE_NAMES)

# head bias per feature (mean rows only for reliability head)
head = adapter.regress_head
W = head.proj.weight if hasattr(head, "proj") else head.weight
Bv = head.proj.bias if hasattr(head, "proj") else head.bias
bias_mean = Bv.detach()[:NF]
print("P0 head bias (mean rows): " +
      " ".join(f"{n}={b:+.2f}" for n, b in zip(FEATURE_NAMES, bias_mean.tolist())))

inner = adapter.inner
is_attn = type(inner.context).__name__ == "SelfAttentionContextBlock"
print(f"   context={type(inner.context).__name__}  conditioning={getattr(inner,'conditioning',None)}")

try:
    torch.backends.mha.set_fastpath_enabled(False)
except Exception:
    pass

# capture attention weights
attn_store = {}
if is_attn:
    layer = inner.context.layers.layers[0]
    orig_fwd = layer.self_attn.forward
    def wrapped(q, k, v, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        out, w = orig_fwd(q, k, v, **kw)
        attn_store["w"] = w.detach()
        return out, w
    layer.self_attn.forward = wrapped

def head_z(prefix):
    out = adapter.regress_head(prefix)
    if isinstance(out, tuple):
        out = out[0]
    return out

def manual_forward(af, oi):
    """Replicate ReliabilityAwareAdapter.forward with stage outputs exposed."""
    xc = inner.compressor(af)
    xt = inner.context(xc) if not is_attn else inner.context(xc, lengths=None)
    xu = inner.proj_up(xt)
    N = xu.shape[1]
    o = inner.overlap_embed(oi).transpose(1, 2)
    o = F.adaptive_avg_pool1d(o, N).transpose(1, 2)
    if getattr(inner, "conditioning", "film") == "film":
        xcond = inner.film(xu, o)
    else:
        xcond = inner.cond_proj(torch.cat([xu, o], dim=-1))
    prefix = inner.mlp(xcond)
    return xc, xt, prefix

files = sorted(glob.glob(f"{TEST}/*.pt"))
# mix of mixtures and clean, deterministic
picks = files[:NCLIP]

ent_rows, self_mass, diag1, diag3, col_ent, rowsim = [], [], [], [], [], []
pr_ctx, pr_pre, pr_z, loc_conv, loc_ctx, loc_pre = [], [], [], [], [], []
grad_conc, grad_auc, prefz_conc = {f: [] for f in FEATURE_NAMES}, {f: [] for f in FEATURE_NAMES}, {f: [] for f in FEATURE_NAMES}
norm_corr, wav_corr, ov_corr = {f: [] for f in FEATURE_NAMES}, {f: [] for f in FEATURE_NAMES}, {f: [] for f in FEATURE_NAMES}

def norm_entropy(x):
    x = np.abs(x) + 1e-12
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

def auc(scores, labels):
    l = (labels > 0.5).astype(int)
    if l.sum() in (0, len(l)):
        return np.nan
    order = np.argsort(scores)
    r = np.empty(len(scores)); r[order] = np.arange(1, len(scores) + 1)
    n1, n0 = l.sum(), len(l) - l.sum()
    return (r[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

for f in picks:
    d = torch.load(f, map_location="cpu", weights_only=False)
    af = d["audio_features"].unsqueeze(0).float()
    oi = d["overlap_info"].unsqueeze(0).float()
    with torch.no_grad():
        xc, xt, prefix = manual_forward(af, oi)
        ref = inner(af, oi)
        assert torch.allclose(prefix, ref, atol=1e-4), "manual forward mismatch"
        N = prefix.shape[1]
        z = head_z(prefix)[0]                      # (N, F)
        ovp = F.adaptive_avg_pool1d(oi[0, :, 0][None, None, :], N)[0, 0].numpy()

    # P1 attention stats
    if is_attn and "w" in attn_store:
        Wm = attn_store["w"][0].numpy()            # (H, N, N)
        H = Wm.shape[0]
        for h in range(H):
            A = Wm[h]
            ent_rows.append(np.mean([norm_entropy(A[i]) for i in range(N)]))
            self_mass.append(np.mean(np.diag(A)))
            idx = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
            diag1.append(float(A[idx <= 1].sum() / N))
            diag3.append(float(A[idx <= 3].sum() / N))
            col = A.mean(0)
            col_ent.append(norm_entropy(col))
            # row similarity: mean pairwise corr of attention rows
            Ac = A - A.mean(1, keepdims=True)
            sd = Ac.std(1, keepdims=True); sd[sd == 0] = 1
            Z = Ac / sd
            Cm = Z @ Z.T / N
            rowsim.append(float(Cm[~np.eye(N, dtype=bool)].mean()))

    # P2 ERF: zero the 8 input frames feeding token k
    with torch.no_grad():
        for k in (N // 4, N // 2, (3 * N) // 4):
            af2 = af.clone()
            af2[:, 8 * k: 8 * k + 8, :] = 0.0
            xc2, xt2, pre2 = manual_forward(af2, oi)
            z2 = head_z(pre2)[0]
            dc = (xc2 - xc)[0].norm(dim=-1).numpy()
            dt = (xt2 - xt)[0].norm(dim=-1).numpy()
            dp = (pre2 - prefix)[0].norm(dim=-1).numpy()
            dz = (z2 - z).abs().mean(-1).numpy()
            for dvec, loc, prl in ((dc, loc_conv, None), (dt, loc_ctx, pr_ctx),
                                   (dp, loc_pre, pr_pre), (dz, None, pr_z)):
                s = dvec.sum() + 1e-12
                if loc is not None:
                    loc.append(float(dvec[max(0, k - 1): k + 2].sum() / s))
                if prl is not None:
                    prl.append(float((dvec.sum() ** 2) / (len(dvec) * (dvec ** 2).sum() + 1e-12)))

    # P3 gradient saliency at conv level
    afg = af.clone()
    xcg = inner.compressor(afg)
    xcg.retain_grad()
    xtg = inner.context(xcg, lengths=None) if is_attn else inner.context(xcg)
    xug = inner.proj_up(xtg)
    o = inner.overlap_embed(oi).transpose(1, 2)
    o = F.adaptive_avg_pool1d(o, xug.shape[1]).transpose(1, 2)
    if getattr(inner, "conditioning", "film") == "film":
        xcondg = inner.film(xug, o)
    else:
        xcondg = inner.cond_proj(torch.cat([xug, o], dim=-1))
    pg = inner.mlp(xcondg)
    zg = head_z(pg)[0]                              # (N, F)
    if str(cfg.get("aux_pool", "mean")) == "linear_softmax":
        w = zg.abs(); w = w / w.sum(0, keepdim=True).clamp(min=1e-6)
        v = (w * zg).sum(0)
    else:
        v = zg.mean(0)
    for fi, fname in enumerate(FEATURE_NAMES):
        g = torch.autograd.grad(v[fi], xcg, retain_graph=True)[0][0]   # (N, d)
        sal = g.norm(dim=-1).detach().numpy()
        grad_conc[fname].append(norm_entropy(sal))
        a = auc(sal, ovp)
        if a == a:
            grad_auc[fname].append(a)
        zdev = (z[:, fi] - z[:, fi].mean()).abs().numpy()
        prefz_conc[fname].append(norm_entropy(zdev))
        # P4 norm correlations
        pn = prefix[0].norm(dim=-1).numpy()
        wn = F.adaptive_avg_pool1d(af[0].norm(dim=-1)[None, None, :], N)[0, 0].numpy()
        for tgt, store in ((pn, norm_corr), (wn, wav_corr), (ovp, ov_corr)):
            if zdev.std() > 0 and np.std(tgt) > 0:
                store[fname].append(float(np.corrcoef(zdev, tgt)[0, 1]))

if is_attn:
    print(f"\nP1 attention ({len(picks)} clips x 8 heads): "
          f"row-entropy mean {np.mean(ent_rows):.3f} (1.0=uniform)  "
          f"self-mass {np.mean(self_mass):.3f} (uniform={1/np.mean([29]):.3f})  "
          f"mass|i-j|<=1 {np.mean(diag1):.3f}  <=3 {np.mean(diag3):.3f}")
    print(f"   column-mean entropy {np.mean(col_ent):.3f}   row-to-row attention corr {np.mean(rowsim):.3f}"
          f"  (1.0 = every token attends identically => attention acts as a global pool)")

print(f"\nP2 effective receptive field (zero one 160ms input window):")
print(f"   local mass (perturbed token +-1): conv {np.mean(loc_conv):.3f}  "
      f"post-context {np.mean(loc_ctx):.3f}  final-prefix {np.mean(loc_pre):.3f}")
print(f"   participation ratio (1/N~{1/29:.3f}=local, 1=uniform): "
      f"post-context {np.mean(pr_ctx):.3f}  final-prefix {np.mean(pr_pre):.3f}  head-z {np.mean(pr_z):.3f}")

print(f"\nP3 conv-level gradient saliency vs prefix-level |z-dev| concentration (entropy, 1=flat):")
print(f"{'feature':<14}{'grad-conc':>10}{'grad-AUCov':>11}{'prefix-conc':>12}")
for fname in FEATURE_NAMES:
    ga = np.mean(grad_auc[fname]) if grad_auc[fname] else float("nan")
    print(f"{fname:<14}{np.mean(grad_conc[fname]):>10.3f}{ga:>11.3f}{np.mean(prefz_conc[fname]):>12.3f}")

print(f"\nP4 corr of |z_dev_t,f| with token norm / wavlm norm / overlap:")
print(f"{'feature':<14}{'||prefix||':>11}{'||wavlm||':>11}{'overlap':>9}")
for fname in FEATURE_NAMES:
    print(f"{fname:<14}{np.mean(norm_corr[fname]):>11.3f}{np.mean(wav_corr[fname]):>11.3f}"
          f"{np.mean(ov_corr[fname]):>9.3f}")
