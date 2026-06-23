"""srmr_map_validate.py — validate the SUPERVISED dense 2D SRMR-modulation-map head
against the stem-derived oracle (the headline faithfulness experiment for the TRUE-2D
map, the unique contribution of the v25 model).

WHAT THIS IS
------------
The 2D analogue of snr_map_validate.py. Where the SNR map head was validated against the
oracle per-frame SNR TIMELINE, the SupervisedSRMRMapHead is validated against the oracle
clean-s1 23x8 SRMR LOG-energy map (srmr_logmap on the s1 stem, the time-AVERAGED
acoustic x modulation grid Falk et al. 2010 / SRMRpy compute). It reports, over the
SUPERVISED (srmr_mask=1) cells of the 23x8 grid:

  1. AGREEMENT — Pearson r AND Spearman SRCC between the predicted log-map cells and the
                 oracle log-map cells, reported two ways: POOLED (every supervised cell of
                 every clip flattened into one vector, then correlated) and PER-CLIP
                 (correlate each clip's 23x8 cells, then average across clips). Plus the
                 srmr_map_mae (mean |error| on the log-map, the SAME metric the head's
                 map_loss emits) so the validator number ties to the training metric.

  2. MODEL-RANDOMIZATION SANITY (Adebayo 2018) — re-init a FRESH head of the same shape
                 (random weights) and re-run on the same audio; the trained-vs-random
                 correlation must collapse to ~0. A high correlation would mean the map is
                 an input artifact of the mean-pooled features, not a learned function.

The metric core (correlation / MAE / randomization) is pure torch+scipy so it is unit-
testable on CPU with synthetic tensors; the checkpoint + .pt orchestration sits on top.

This head is CLIP-LEVEL (mean-pool WavLM frames -> MLP -> 23x8 grid), unlike the SNR
timeline head, so there is no per-frame deletion test — the faithfulness claim here is
"the predicted 2D modulation grid matches the oracle grid", measured by the cell-wise
correlation + the randomization sanity.

USAGE
-----
    python src/srmr_map_validate.py \
        --checkpoint    /.../v25_srmr2d/last.pt \
        --processed_dir /.../data/processed_aug/test \
        --srmr_map_dir  /.../data/srmr_map_targets/test \
        --out           srmr_map_eval.json [--max_clips N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snr_map_head import SupervisedSRMRMapHead  # noqa: E402

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:  # pragma: no cover - scipy is present on PSC
    _scipy_spearmanr = None


# ── pure-torch metric core (unit-testable) ──────────────────────────────────────
def _masked_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Pearson correlation between two flattened series over the masked entries.

    Returns 0.0 when fewer than 2 supervised cells or a degenerate (zero-variance)
    series — so a flat constant prediction reads as no correlation, not NaN.
    """
    m = mask.reshape(-1).to(torch.bool)
    if int(m.sum()) < 2:
        return 0.0
    p = pred.reshape(-1)[m].float()
    t = target.reshape(-1)[m].float()
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.norm() * t.norm())
    if float(denom) <= 1e-8:
        return 0.0
    return float((p @ t) / denom)


def _masked_spearman(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Spearman rank correlation over the masked cells.

    Uses scipy when available; otherwise falls back to a Pearson-on-ranks computed in
    torch. Returns 0.0 on <2 cells / zero-variance / NaN so a degenerate map reads as no
    correlation rather than NaN.
    """
    m = mask.reshape(-1).to(torch.bool)
    if int(m.sum()) < 2:
        return 0.0
    p = pred.reshape(-1)[m].float()
    t = target.reshape(-1)[m].float()
    if _scipy_spearmanr is not None:
        rho, _ = _scipy_spearmanr(p.cpu().numpy(), t.cpu().numpy())
        if rho != rho:  # NaN (zero variance)
            return 0.0
        return float(rho)
    # torch fallback: Pearson on ranks (average-rank tie handling omitted; argsort ranks)
    pr = torch.argsort(torch.argsort(p)).float()
    tr = torch.argsort(torch.argsort(t)).float()
    return _masked_pearson(pr, tr, torch.ones_like(pr))


def map_agreement(
    pred: torch.Tensor,        # (A, M) or (B, A, M) predicted log-energy map
    target: torch.Tensor,      # (A, M) or (B, A, M) oracle log-energy map
    mask: torch.Tensor,        # (A, M) or (B, A, M) supervised-cell mask (1 = supervise)
) -> dict[str, float]:
    """Per-clip cell-wise agreement: Pearson r + Spearman SRCC + MAE over supervised cells.

    Accepts a single clip (2-D) or a batch (3-D); for a batch the correlations are the
    MEAN per-clip correlation and the MAE is pooled over all supervised cells.
    """
    if pred.dim() == 2:
        pred, target, mask = pred[None], target[None], mask[None]
    B = pred.shape[0]
    mf = mask.to(pred.dtype)
    abs_err = (pred - target).abs() * mf
    denom = mf.sum().clamp(min=1.0)
    mae = float(abs_err.sum() / denom)
    pcorrs = [_masked_pearson(pred[b], target[b], mask[b]) for b in range(B)]
    scorrs = [_masked_spearman(pred[b], target[b], mask[b]) for b in range(B)]
    return {
        "map_pearson": float(sum(pcorrs) / max(1, len(pcorrs))),
        "map_srcc": float(sum(scorrs) / max(1, len(scorrs))),
        "srmr_map_mae": mae,
        "n_cells": float(denom.item()),
    }


def model_randomization(
    head: SupervisedSRMRMapHead,
    audio_features: torch.Tensor,    # (B, T, audio_dim)
    mask: torch.Tensor,              # (B, A, M) supervised-cell mask
    seed: int = 0,
) -> dict[str, float]:
    """Adebayo (2018) model-parameter randomization sanity check.

    Re-initializes a FRESH head of the same shape (random weights), runs both heads on
    the same audio, and reports the mean per-clip Pearson correlation between the trained
    map and the random-weight map. A faithful map should DECORRELATE (corr near 0); a high
    correlation would mean the prediction is an artifact of the mean-pooled input,
    independent of training.
    """
    torch.manual_seed(int(seed))
    with torch.no_grad():
        trained = head.forward(audio_features)                                 # (B, A, M)
        rand_head = SupervisedSRMRMapHead(
            audio_dim=head.audio_dim, n_acoustic=head.n_acoustic,
            n_modulation=head.n_modulation, hidden=head.hidden,
            huber_delta=head.huber_delta,
        ).to(audio_features.device).to(head.in_proj.weight.dtype)
        rand = rand_head.forward(audio_features)                               # (B, A, M)
    B = trained.shape[0]
    corrs = [_masked_pearson(trained[b], rand[b], mask[b]) for b in range(B)]
    return {"model_rand_corr": float(sum(corrs) / max(1, len(corrs)))}


# ── checkpoint reconstruction + .pt orchestration ───────────────────────────────
def _load_head(checkpoint: dict, device: str) -> SupervisedSRMRMapHead:
    cfg = checkpoint.get("config", {})
    head = SupervisedSRMRMapHead(
        audio_dim=int(cfg.get("srmr_map_audio_dim", 1024)),
        n_acoustic=int(cfg.get("srmr_map_n_acoustic", 23)),
        n_modulation=int(cfg.get("srmr_map_n_modulation", 8)),
        hidden=int(cfg.get("srmr_map_hidden", 256)),
        huber_delta=float(cfg.get("srmr_map_huber_delta", 1.0)),
    ).to(device)
    sd = checkpoint.get("srmr_map_head_state_dict")
    if sd is None:
        raise KeyError("checkpoint has no srmr_map_head_state_dict")
    # reliability / extra heads are training-only; tolerate extras like the SNR validator.
    head.load_state_dict(sd, strict=False)
    head.eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--srmr_map_dir", required=True,
                    help="dir of oracle dense SRMR targets (srmr_logmap + srmr_mask .pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    head = _load_head(ckpt, args.device)
    head_dtype = head.in_proj.weight.dtype

    manifest_path = os.path.join(args.srmr_map_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else None

    pts = sorted(f for f in os.listdir(args.processed_dir) if f.endswith(".pt"))
    if args.max_clips:
        pts = pts[: args.max_clips]

    per_clip_pearson, per_clip_srcc = [], []
    pooled_pred, pooled_tgt = [], []     # flattened supervised cells across ALL clips
    mae_num, mae_den = 0.0, 0.0
    mr_corrs = []
    n_scored = 0
    n_bands_seen = 0
    for ptname in pts:
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")

        # resolve oracle target: prefer manifest, fall back to <stem>.pt by filename.
        if manifest is not None and filename in manifest:
            tgt_path = os.path.join(args.srmr_map_dir, manifest[filename])
        else:
            stem = os.path.splitext(filename)[0]
            tgt_path = os.path.join(args.srmr_map_dir, stem + ".pt")
        if not os.path.exists(tgt_path):
            continue
        tgt = torch.load(tgt_path, weights_only=False)
        target = tgt["srmr_logmap"].to(args.device).float()                    # (A, M)
        mask = tgt.get("srmr_mask", torch.ones_like(target)).to(args.device).float()

        af = cached["audio_features"].to(args.device).to(head_dtype).unsqueeze(0)  # (1,T,D)
        with torch.no_grad():
            pred = head.forward(af)[0].float()                                 # (A, M)

        # align grids defensively (targets are fixed 23x8).
        A = min(pred.shape[0], target.shape[0])
        M = min(pred.shape[1], target.shape[1])
        pred, target, mask = pred[:A, :M], target[:A, :M], mask[:A, :M]

        a = map_agreement(pred, target, mask)
        per_clip_pearson.append(a["map_pearson"])
        per_clip_srcc.append(a["map_srcc"])
        mae_num += a["srmr_map_mae"] * a["n_cells"]
        mae_den += a["n_cells"]
        n_bands_seen = max(n_bands_seen, int(a["n_cells"]))

        # accumulate supervised cells for the POOLED correlation.
        mb = mask.reshape(-1).to(torch.bool)
        pooled_pred.append(pred.reshape(-1)[mb])
        pooled_tgt.append(target.reshape(-1)[mb])

        mr = model_randomization(head, af, mask[None])
        mr_corrs.append(mr["model_rand_corr"])
        n_scored += 1

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    # pooled correlation over every supervised cell of every clip.
    pooled_pearson = pooled_srcc = 0.0
    if pooled_pred:
        pp = torch.cat(pooled_pred)
        tt = torch.cat(pooled_tgt)
        ones = torch.ones_like(pp)
        pooled_pearson = _masked_pearson(pp, tt, ones)
        pooled_srcc = _masked_spearman(pp, tt, ones)

    result = {
        "n_clips": n_scored,
        "n_bands": n_bands_seen,
        # headline (per-clip-then-averaged) numbers the task asks for:
        "srmr_map_pearson_mean": _mean(per_clip_pearson),
        "srmr_map_srcc_mean": _mean(per_clip_srcc),
        "srmr_map_mae_mean": float(mae_num / mae_den) if mae_den else 0.0,
        "model_rand_corr_mean": _mean(mr_corrs),
        # pooled-across-clips correlation (one flat vector of all supervised cells):
        "srmr_map_pearson_pooled": float(pooled_pearson),
        "srmr_map_srcc_pooled": float(pooled_srcc),
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
