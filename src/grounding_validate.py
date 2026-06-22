"""grounding_validate.py — validate the v18 bottleneck keep-mask against the oracle.

WHAT THIS IS
------------
The currently-missing validation piece (design doc §7). It wires
`src/grounding_metrics.py` to the bottleneck head's extracted per-feature keep-mask
λ̄ and the oracle `overlap_segments` that live in each preprocessed `.pt`. For the
`overlap_ratio` feature (and pauses if a span set materializes) it reports, on the
overlap-bearing clips:

  - IoU + pointing-game + time-concentration ratio vs the oracle overlap spans,
  - Wilcoxon signed-rank of per-clip concentration / IoU vs a random-λ baseline,
  - RISE-style DELETION faithfulness (replace the highest-λ patches with the head's
    OWN noise baseline R̄ and watch the scalar move — for the bottleneck this is
    EXACTLY the operation the head already does, so a high-λ patch provably matters
    and a low-λ patch provably does not; the softmax head has no such guarantee),
  - the two Adebayo (2018) SANITY checks: model-parameter randomization (re-init the
    head's queries/K_proj/readout → the faithful map should DECORRELATE from the
    trained map; the softmax map barely moves) and a label-randomization HOOK (the
    orchestration for a head-only retrain on shuffled scalars; the heavy run is later).

It is offline (no retrain for the main metrics, no GPU train), consumes the cached
`.pt` clips directly, and reconstructs the head from a training checkpoint's embedded
config + `decoupled_head_state_dict`. Pure-torch/numpy metric core lives in
`grounding_metrics.py`; this module is orchestration + IO.

USAGE
-----
    python src/grounding_validate.py \
        --checkpoint   /.../v18_bottleneck/best.pt \
        --processed_dir /.../data/processed/test \
        --feature overlap_ratio \
        --out          grounding_eval.json \
        [--sanity model_rand] [--max_clips N] [--deletion]

The killer experiment (design §9) is `--feature overlap_ratio --deletion --sanity
model_rand` on the overlap-bearing test clips for the v18 head, compared against the
same run on the v17 softmax checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoupled_grounding import DecoupledGroundingHead  # noqa: E402
from feature_set import N_FEATURES  # noqa: E402
import grounding_metrics as gm  # noqa: E402

WAVLM_FRAME_RATE_HZ = 50.0  # matches src/inference.py measured_duration_sec


# ── head reconstruction from a training checkpoint ──────────────────────────────
def load_head(checkpoint_path: str, device: str = "cpu") -> DecoupledGroundingHead:
    """Rebuild the DecoupledGroundingHead from a checkpoint's embedded config + weights.

    Reads grounding_mode / decoupled_* / bits_beta_per_feature / concrete_temp_* from
    the pickled training config so the SAME forward (softmax vs bottleneck) is rebuilt,
    then loads `decoupled_head_state_dict`. Raises if the checkpoint has no head.
    """
    ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg = ck.get("config", {})
    if "decoupled_head_state_dict" not in ck:
        raise ValueError(
            f"{checkpoint_path} has no decoupled_head_state_dict — not a grounding run."
        )
    head = DecoupledGroundingHead(
        d_model=int(cfg.get("decoupled_d_model", 256)),
        d_patch=int(cfg.get("spec_d_patch", 768)),
        n_features=N_FEATURES,
        n_heads=int(cfg.get("decoupled_n_heads", 1)),
        readout_hidden=cfg.get("decoupled_readout_hidden"),
        huber_delta=float(cfg.get("decoupled_huber_delta", 1.0)),
        grounding_mode=str(cfg.get("grounding_mode", "softmax")),
        bits_beta_per_feature=cfg.get("bits_beta_per_feature"),
        concrete_temp=float(cfg.get("concrete_temp_end",
                                    cfg.get("concrete_temp_start", 1.0))),
    )
    # Load tolerantly: a softmax (v17-era) checkpoint predates the bottleneck-only
    # `concrete_temp` buffer (and any other non-parameter buffer added later), so a
    # strict load wrongly rejects it. We allow MISSING BUFFERS (the freshly-built head
    # keeps its constructed default, e.g. concrete_temp from the config) but still
    # require every trainable PARAMETER to be present, so we never silently score a
    # head with randomly-initialised weights.
    sd = ck["decoupled_head_state_dict"]
    missing, unexpected = head.load_state_dict(sd, strict=False)
    param_names = {n for n, _ in head.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise ValueError(
            f"{checkpoint_path}: head checkpoint is missing trainable parameters "
            f"{missing_params} — refusing to score a partially-initialised head."
        )
    if unexpected:
        # extra keys in the checkpoint that the rebuilt head does not have: harmless
        # (e.g. a renamed/removed buffer), but surface it so it is not silent.
        print(f"[load_head] note: ignored {len(unexpected)} unexpected key(s) "
              f"in {os.path.basename(checkpoint_path)}: {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}")
    head.to(device).eval()
    return head


# ── per-clip map extraction ─────────────────────────────────────────────────────
def clip_duration_sec(sample: dict) -> float:
    """Clip duration: from the WavLM frame count when present, else the max overlap end."""
    af = sample.get("audio_features")
    if af is not None:
        return float(af.shape[0]) / WAVLM_FRAME_RATE_HZ
    segs = sample.get("overlap_segments") or []
    return float(max((e for _s, e in segs), default=0.0))


@torch.no_grad()
def extract_mask(
    head: DecoupledGroundingHead,
    sample: dict,
    feature_idx: int,
    device: str = "cpu",
) -> tuple[np.ndarray, int, int] | None:
    """Run the (eval) head on a clip's beats_patches → flat keep-map for one feature.

    Returns (alpha_flat (P,), time_dim, freq_dim) or None when the clip has no
    beats_patches. In bottleneck mode the returned map is the deterministic λ̄.
    """
    patches = sample.get("beats_patches")
    if patches is None:
        return None
    meta = sample.get("beats_grid_meta", {}) or {}
    t_dim = int(meta.get("time_dim", 0))
    f_dim = int(meta.get("freq_dim", gm.F_P_DEFAULT))
    p = patches.unsqueeze(0).to(device).to(head.K_proj.weight.dtype)  # (1,P,d)
    pad = sample.get("beats_patches_mask")
    valid = None if pad is None else (~pad.to(torch.bool)).unsqueeze(0).to(device)
    head.eval()
    mp, _z, _pred = head(p, patch_mask=valid)        # (1, n_features, P)
    alpha = mp[0, feature_idx].detach().cpu().numpy()
    if t_dim <= 0:
        t_dim = alpha.size // max(f_dim, 1)
    return alpha, t_dim, f_dim


# ── RISE deletion faithfulness (uses the head's own noise substitution) ─────────
@torch.no_grad()
def deletion_curve(
    head: DecoupledGroundingHead,
    sample: dict,
    feature_idx: int,
    n_steps: int = 10,
    device: str = "cpu",
) -> dict | None:
    """RISE deletion AUC for one feature (design §7.2.4) — the CAUSAL faithfulness number.

    Order valid patches by λ̄ (descending). Progressively replace the top-ranked
    patches with the head's noise baseline R̄ (= masked mean of V over valid patches,
    detached), re-read the scalar through the readout, and record |pred − pred_full|.
    Low AUC = faithful (the high-λ patches really are the ones that move the scalar).
    For the BOTTLENECK head this deletion IS the head's own substitution, so it is
    exactly faithful; the softmax head is run by zeroing attention on the deleted
    patches and renormalizing, which is the standard occlusion analogue.

    Returns {auc_deletion, frac_steps, deltas, pred_full} or None when no patches.
    """
    patches = sample.get("beats_patches")
    if patches is None:
        return None
    p = patches.unsqueeze(0).to(device).to(head.K_proj.weight.dtype)  # (1,P,d)
    pad = sample.get("beats_patches_mask")
    valid = None if pad is None else (~pad.to(torch.bool)).unsqueeze(0).to(device)

    ext = extract_mask(head, sample, feature_idx, device)
    if ext is None:
        return None
    alpha, _t, _f = ext
    valid_idx = (np.arange(alpha.size) if valid is None
                 else np.where(valid[0].cpu().numpy())[0])
    if valid_idx.size == 0:
        return None
    order = valid_idx[np.argsort(-alpha[valid_idx])]  # high-λ first

    def _readout_with_kept(keep_bool: np.ndarray) -> float:
        """Scalar from the readout when only `keep_bool` patches survive (rest → R̄)."""
        R = head.V_proj(p)                                   # (1,P,d_model)
        vf = (torch.ones_like(R[..., :1]) if valid is None
              else valid.view(1, R.shape[1], 1).to(R.dtype))  # (1,P,1)
        denom = vf.sum(dim=1).clamp(min=1.0)                 # (1,1)
        R_bar = (R.detach() * vf).sum(dim=1) / denom         # (1,d_model)
        keep = torch.zeros(1, R.shape[1], 1, dtype=R.dtype, device=R.device)
        keep[0, keep_bool, 0] = 1.0
        keep = keep * vf
        Z = keep * R.detach() + (1.0 - keep) * R_bar.unsqueeze(1)  # (1,P,d_model)
        z_f = (Z * vf).sum(dim=1) / denom                    # (1,d_model)
        # build a (1, n_features, d_model) so the per-feature readout indexes feature.
        z_all = z_f.unsqueeze(1).expand(1, head.n_features, R.shape[2]).contiguous()
        pred = head._readout(z_all)                          # (1, n_features)
        return float(pred[0, feature_idx].item())

    keep_all = np.ones(alpha.size, dtype=bool)
    keep_all[~np.isin(np.arange(alpha.size), valid_idx)] = False
    pred_full = _readout_with_kept(np.where(keep_all)[0])

    deltas = []
    fracs = []
    n = order.size
    for k in range(0, n_steps + 1):
        n_del = int(round(n * k / n_steps))
        keep = keep_all.copy()
        keep[order[:n_del]] = False
        pred_k = _readout_with_kept(np.where(keep)[0])
        deltas.append(abs(pred_k - pred_full))
        fracs.append(k / n_steps)
    auc = float(_trapz(deltas, fracs))
    return {"auc_deletion": auc, "frac_steps": fracs, "deltas": deltas,
            "pred_full": pred_full}


def _trapz(y, x) -> float:
    """numpy-version-safe trapezoidal integration (trapz removed in numpy 2.x)."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


# ── model-parameter randomization sanity check (Adebayo 2018) ───────────────────
@torch.no_grad()
def model_randomization_spearman(
    head: DecoupledGroundingHead,
    samples: list[dict],
    feature_idx: int,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Re-init the head's query/K/readout, recompute λ̄, Spearman vs the trained map.

    A FAITHFUL map should DECORRELATE (Spearman → 0) once the weights that produced
    it are randomized; an attention map driven by q·Kᵀ geometry barely moves and
    stays correlated (→ fails the check). Reported per clip, aggregated.
    """
    import copy
    rand_head = copy.deepcopy(head)
    g = torch.Generator().manual_seed(seed)
    for name, prm in rand_head.named_parameters():
        if any(k in name for k in ("queries", "K_proj", "readout")):
            prm.copy_(torch.randn(prm.shape, generator=g) * prm.detach().std().clamp(min=1e-3))
    rand_head.to(device).eval()

    rhos = []
    for s in samples:
        a = extract_mask(head, s, feature_idx, device)
        b = extract_mask(rand_head, s, feature_idx, device)
        if a is None or b is None:
            continue
        rho = _spearman(a[0], b[0])
        if rho is not None:
            rhos.append(rho)
    arr = np.asarray(rhos, dtype=float)
    return {"n": int(arr.size),
            "mean_spearman": float(arr.mean()) if arr.size else float("nan"),
            "median_spearman": float(np.median(arr)) if arr.size else float("nan")}


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation, pure numpy (no scipy dep)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size < 2 or x.size != y.size:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return None
    return float((rx * ry).sum() / denom)


def _wilcoxon(diffs: np.ndarray) -> dict:
    """Wilcoxon signed-rank statistic + normal-approx p (two-sided), pure numpy.

    Tests whether per-clip (trained − random) values are centered above 0. Returns
    {W, z, p, n}. Mirrors scripts/attention_gt_alignment.py's report without scipy.
    """
    d = np.asarray(diffs, float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return {"W": 0.0, "z": 0.0, "p": 1.0, "n": 0}
    ranks = np.argsort(np.argsort(np.abs(d))).astype(float) + 1.0
    W_plus = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    se_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W_plus - mean_w) / se_w if se_w > 0 else 0.0
    # two-sided normal-approx p
    p = float(2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / np.sqrt(2.0)))))
    return {"W": W_plus, "z": float(z), "p": p, "n": n}


def _erf(x: float) -> float:
    import math
    return math.erf(x)


# ── orchestration ───────────────────────────────────────────────────────────────
def validate(
    head: DecoupledGroundingHead,
    samples: list[dict],
    feature_name: str = "overlap_ratio",
    do_deletion: bool = False,
    sanity: str | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Run the region + faithfulness suite for one feature over the clip set.

    Region metrics (IoU/pointing/concentration/Wilcoxon-vs-random) are computed on
    OVERLAP-BEARING clips only (non-empty overlap_segments). Deletion + sanity are
    label-free and run on all clips with patches.
    """
    from feature_set import SUPERVISED_FEATURES
    names = [n for n, _c, _f in SUPERVISED_FEATURES]
    if feature_name not in names:
        raise ValueError(f"unknown feature {feature_name!r}; options {names}")
    fidx = names.index(feature_name)

    rng = np.random.default_rng(seed)
    ious, soft_ious, points, ratios, rand_ratios = [], [], [], [], []
    rand_soft_ious = []
    n_overlap = 0
    n_collapsed = 0          # overlap clips whose map carries NO positive mass (uniform fallback)
    del_aucs = []

    for s in samples:
        ext = extract_mask(head, s, fidx, device)
        if ext is None:
            continue
        alpha, _t, f_dim = ext
        dur = clip_duration_sec(s)
        segs = s.get("overlap_segments") or []

        if segs and dur > 0:
            n_overlap += 1
            # a map with no positive mass localizes nothing — track it so a collapsed
            # (e.g. bits-penalty-zeroed) bottleneck mask reads as DIFFUSE/low, not null.
            if float(np.clip(np.asarray(alpha, float), 0.0, None).sum()) <= 0.0:
                n_collapsed += 1
            # primary IoU: threshold-FREE soft histogram IoU (robust on flat/collapsed
            # maps where the median-thresholded hard IoU degenerates to 0).
            r_siou = gm.soft_iou_time(alpha, segs, dur, f_dim)
            r_iou = gm.iou_time(alpha, segs, dur, f_dim, thresh="median")   # hard, secondary
            r_pt = gm.pointing_game(alpha, segs, dur, f_dim)
            r_cr = gm.time_concentration_ratio(alpha, segs, dur, f_dim)
            rand_alpha = rng.permutation(alpha)
            if r_siou is not None:
                soft_ious.append(r_siou["soft_iou"])
                rr_si = gm.soft_iou_time(rand_alpha, segs, dur, f_dim)
                rand_soft_ious.append(rr_si["soft_iou"] if rr_si is not None else 0.0)
            if r_iou is not None:
                ious.append(r_iou["iou"])
            if r_pt is not None:
                points.append(1.0 if r_pt["hit"] else 0.0)
            if r_cr is not None:
                ratios.append(r_cr["ratio"])
                # random-λ baseline concentration (same shape, shuffled mass).
                rr = gm.time_concentration_ratio(rand_alpha, segs, dur, f_dim)
                rand_ratios.append(rr["ratio"] if rr is not None else 1.0)

        if do_deletion:
            dc = deletion_curve(head, s, fidx, device=device)
            if dc is not None:
                del_aucs.append(dc["auc_deletion"])

    out: dict = {
        "feature": feature_name,
        "grounding_mode": head.grounding_mode,
        "n_clips": len(samples),
        "n_overlap_clips": n_overlap,
        # n_collapsed: overlap clips whose extracted map had NO positive mass (the
        # bottleneck keep-mask clamped fully OFF for this feature). Such maps localize
        # nothing and are scored as the uninformative floor (soft_iou ~ gt_frac,
        # concentration ~ 1.0), never null. A high n_collapsed/n_overlap flags a
        # DIFFUSE / ungrounded map (often a partially-trained bottleneck), not a bug.
        "n_collapsed_maps": n_collapsed,
        "frac_collapsed": float(n_collapsed / n_overlap) if n_overlap else None,
        # primary IoU is the threshold-free soft histogram IoU (always defined).
        "soft_iou_mean": float(np.mean(soft_ious)) if soft_ious else None,
        "soft_iou_random_mean": float(np.mean(rand_soft_ious)) if rand_soft_ious else None,
        # hard median-thresholded IoU kept for continuity (brittle on flat maps).
        "iou_mean": float(np.mean(ious)) if ious else None,
        "pointing_game_hit_rate": float(np.mean(points)) if points else None,
        "concentration_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "concentration_ratio_random_mean": float(np.mean(rand_ratios)) if rand_ratios else None,
    }
    if ratios and rand_ratios and len(ratios) == len(rand_ratios):
        diffs = np.asarray(ratios) - np.asarray(rand_ratios)
        out["wilcoxon_vs_random"] = _wilcoxon(diffs)
    if do_deletion:
        out["deletion_auc_mean"] = float(np.mean(del_aucs)) if del_aucs else None
        out["n_deletion_clips"] = len(del_aucs)
    if sanity == "model_rand":
        out["sanity_model_randomization"] = model_randomization_spearman(
            head, samples, fidx, device, seed)
    return out


def load_samples(processed_dir: str, max_clips: int | None = None) -> list[dict]:
    """Load preprocessed .pt clips (those carrying beats_patches)."""
    paths = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    if max_clips is not None:
        paths = paths[:max_clips]
    out = []
    for p in paths:
        try:
            s = torch.load(p, weights_only=False, map_location="cpu")
        except Exception:
            continue
        if "beats_patches" in s:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Validate the bottleneck grounding mask vs oracle spans.")
    ap.add_argument("--checkpoint", required=True, help="training ckpt with decoupled_head_state_dict")
    ap.add_argument("--processed_dir", required=True, help="dir of preprocessed .pt clips (test split)")
    ap.add_argument("--feature", default="overlap_ratio")
    ap.add_argument("--out", default="grounding_eval.json")
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--deletion", action="store_true", help="run RISE deletion AUC (needs the head)")
    ap.add_argument("--sanity", choices=["model_rand"], default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    head = load_head(args.checkpoint, args.device)
    samples = load_samples(args.processed_dir, args.max_clips)
    print(f"[grounding_validate] mode={head.grounding_mode} clips={len(samples)} "
          f"feature={args.feature}")
    res = validate(head, samples, args.feature, args.deletion, args.sanity, args.device, args.seed)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"[grounding_validate] wrote {args.out}")


if __name__ == "__main__":
    main()
