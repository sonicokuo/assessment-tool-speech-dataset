"""EXACT per-feature temporal attribution for the verified-slot readout.

The emitted value is  v_f = W_f . mean_t(prefix_t) + b_f , so

        v_f = (1/N) * SUM_t (W_f . prefix_t)  +  b_f

and the contribution of prefix token t to feature f is EXACTLY (1/N)*(W_f . prefix_t).
This is not an approximation (unlike attention weights or gradient saliency) -- it is the
literal arithmetic decomposition of the number the model emitted, and it sums to v_f - b_f
by construction. We assert that identity per clip so a silent bug cannot pass.

WHAT THIS IS FOR: mean-pooling gives the model NO INCENTIVE to localise -- only the mean has
to be right, so contributions may be spread arbitrarily. That is the same failure that made
the SRMR map reproduce the training-set average. So we also emit the controls that decide
whether the map means anything:
  (a) ALIGNMENT   -- does the overlap_ratio attribution concentrate on frames that are
                     actually overlapped? (per-frame GT available from overlap_info[:,0])
  (b) SPECIFICITY -- is clip A's map different from clip B's, or is it the average map?
  (c) CONCENTRATION -- normalised entropy; 1.0 = perfectly flat = no localisation at all.

Usage: attribution_capture.py <ckpt> <test_dir> <out.npz> [n_clips]
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from model.adapter import build_adapter          # noqa: E402
from data.feature_set import FEATURE_NAMES        # noqa: E402

CKPT, TEST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NCLIP = int(sys.argv[4]) if len(sys.argv) > 4 else 400

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["config"]
# aux_pool MUST come from the checkpoint. Omitting it builds a mean-pooling adapter and
# loads linear_softmax weights into it -- the head then reads the prefix at the wrong
# granularity and EVERY value is wrong, not just the attributions. Caught 2026-08-06 by
# the reconstruct-against-the-model assertion below (max err 18.47).
adapter = build_adapter(cfg["adapter_variant"], lm_dim=4096,
                        reliability_head=bool(cfg.get("reliability_head", False)),
                        compression=int(cfg.get("compression", 8)),
                        aux_pool=str(cfg.get("aux_pool", "mean")))
missing, _ = adapter.load_state_dict(ck["adapter_state_dict"], strict=False)
head_missing = [k for k in missing if "head" in k or "regress" in k]
assert not head_missing, f"HEAD WEIGHTS DID NOT LOAD: {head_missing}"
dev = "cuda" if torch.cuda.is_available() else "cpu"
adapter = adapter.to(dev).eval()

# THE DECOMPOSITION DEPENDS ON THE POOLING. Getting this wrong silently produces an
# attribution that does not correspond to the number the model actually emitted:
#   mean            v_f = mean_t(z_t,f) + ...   -> contribution_t = z_t,f / N
#   linear_softmax  v_f = SUM_t w_t,f z_t,f     -> contribution_t = w_t,f z_t,f,
#                                                  w_t,f = |z_t,f| / SUM_t|z_t,f|
# Both are exact and both sum to v_f, but they are DIFFERENT functions of the prefix.
AUX_POOL = str(cfg.get("aux_pool", "mean"))
NF = len(FEATURE_NAMES)
print(f"variant={cfg['adapter_variant']}  aux_pool={AUX_POOL}  device={dev}", flush=True)

def per_frame_z(prefix):
    """(1, N, lm_dim) -> (N, F) per-frame head outputs (means only), bias included."""
    out = adapter.regress_head(prefix)
    if isinstance(out, tuple):
        out = out[0]
    return out[0].float()

files = sorted(glob.glob(f"{TEST}/*.pt"))[:NCLIP]
contribs, ovmasks, names, maxlen = [], [], [], 0
with torch.no_grad():
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        af = d["audio_features"].unsqueeze(0).to(dev).float()
        oi = d["overlap_info"].unsqueeze(0).to(dev).float()
        prefix = adapter.inner(af, oi)                       # (1, N, lm_dim)
        N = prefix.shape[1]
        z = per_frame_z(prefix)                              # (N, F)
        if AUX_POOL == "linear_softmax":
            w = z.abs()
            w = w / w.sum(0, keepdim=True).clamp(min=1e-6)   # (N, F), sums to 1 over time
            C = (w * z).T                                    # (F, N) EXACT contributions
        else:
            C = (z / N).T                                    # (F, N)
        # Identity check against what the MODEL ACTUALLY EMITS, not against a formula we
        # assumed. This is the check that would have caught using the mean decomposition
        # on a linear_softmax checkpoint.
        emitted = adapter(af, oi)[1]
        if isinstance(emitted, tuple):
            emitted = emitted[0]
        emitted = emitted[0].float()
        assert torch.allclose(C.sum(1), emitted, atol=2e-2), (
            f"attribution does not reconstruct the emitted value "
            f"(max err {(C.sum(1) - emitted).abs().max():.4f}) — wrong pooling formula?")
        # per-frame overlap GT, pooled to prefix resolution for alignment scoring
        ov = oi[0, :, 0]
        ovp = torch.nn.functional.adaptive_avg_pool1d(ov[None, None, :], N)[0, 0]
        contribs.append(C.cpu().numpy()); ovmasks.append(ovp.cpu().numpy())
        names.append(d.get("filename", os.path.basename(f)))
        maxlen = max(maxlen, N)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

Cpad = np.full((len(contribs), len(FEATURE_NAMES), maxlen), np.nan, dtype=np.float32)
Opad = np.full((len(contribs), maxlen), np.nan, dtype=np.float32)
for i, (C, O) in enumerate(zip(contribs, ovmasks)):
    Cpad[i, :, : C.shape[1]] = C
    Opad[i, : O.shape[0]] = O
np.savez_compressed(OUT, contrib=Cpad, overlap=Opad,
                    names=np.array(names), features=np.array(FEATURE_NAMES))
print(f"wrote {OUT}  contrib={Cpad.shape}")
