"""snr_map_validate.py — validate the SUPERVISED dense local-SNR-map head against
the stem-derived oracle (the headline faithfulness experiment for the dense map).

WHAT THIS IS
------------
The dense-map analogue of grounding_validate.py. Where the bottleneck head was
validated against the oracle OVERLAP spans, the SupervisedSNRMapHead is validated
against the oracle per-frame SNR TIMELINE (clean_features.snr_timeline_db on the
s1/s2 stems). It reports, on the supervised (s1-active) frames:

  1. AGREEMENT      — Pearson correlation + MAE (dB) between the predicted timeline
                      and the oracle timeline (per clip + pooled). This is the direct
                      "is the dense field right" number.
  2. DELETION-FAITHFULNESS — zero out the frames the model predicts are HIGHEST-SNR
                      and re-pool the scalar; a faithful timeline must DROP the pooled
                      SNR more than deleting RANDOM frames does. (RISE/insertion-
                      deletion faithfulness, Petsiuk 2018, adapted to a regression
                      readout.) The pool is the same energy-weighted mean the CBM tie
                      uses, so "delete high-SNR frames → pooled SNR falls" is the exact
                      faithfulness claim the head makes.
  3. MODEL-RANDOMIZATION SANITY (Adebayo 2018) — re-init the head's weights and re-run;
                      a faithful map must DECORRELATE from the trained map (a high
                      trained-vs-random correlation would mean the map is an input
                      artifact, not a learned function). Reported as the trained-vs-
                      random correlation (should be near 0).

The metric core (correlation / MAE / deletion / randomization) is pure torch so it is
unit-testable on CPU with synthetic tensors; the checkpoint + .pt orchestration sits
on top.

USAGE
-----
    python src/snr_map_validate.py \
        --checkpoint    /.../v23_snrmap/best.pt \
        --processed_dir /.../data/processed/test \
        --snr_map_dir   /.../data/snr_map_targets/test \
        --out           snr_map_eval.json [--max_clips N] [--delete_frac 0.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snr_map_head import SupervisedSNRMapHead  # noqa: E402


# ── pure-torch metric core (unit-testable) ──────────────────────────────────────
def _masked_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Pearson correlation between two 1-D series over the masked entries.

    Returns 0.0 when fewer than 2 supervised frames or a degenerate (zero-variance)
    series — so a flat constant prediction reads as no correlation, not NaN.
    """
    m = mask.to(torch.bool)
    if int(m.sum()) < 2:
        return 0.0
    p = pred[m].float()
    t = target[m].float()
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.norm() * t.norm())
    if float(denom) <= 1e-8:
        return 0.0
    return float((p @ t) / denom)


def timeline_agreement(
    pred: torch.Tensor,        # (T,) or (B, T) predicted per-frame SNR (dB)
    target: torch.Tensor,      # (T,) or (B, T) oracle per-frame SNR (dB)
    mask: torch.Tensor,        # (T,) or (B, T) s1-active mask (1 = supervised)
) -> dict[str, float]:
    """Per-frame agreement: Pearson r + MAE (dB) over supervised frames.

    Accepts a single clip (1-D) or a batch (2-D); for a batch the correlation is the
    MEAN per-clip correlation and the MAE is pooled over all supervised frames.
    """
    if pred.dim() == 1:
        pred, target, mask = pred[None], target[None], mask[None]
    B = pred.shape[0]
    mf = mask.to(pred.dtype)
    abs_err = (pred - target).abs() * mf
    denom = mf.sum().clamp(min=1.0)
    mae = float(abs_err.sum() / denom)
    corrs = [_masked_pearson(pred[b], target[b], mask[b]) for b in range(B)]
    corr = float(sum(corrs) / max(1, len(corrs)))
    return {"timeline_pearson": corr, "timeline_mae_db": mae,
            "n_frames": float(denom.item())}


def _energy_pool(snr_frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pool (B,) — the same readout the CBM scalar tie uses."""
    m = mask.to(snr_frame.dtype)
    denom = m.sum(dim=-1).clamp(min=1e-8)
    return (snr_frame * m).sum(dim=-1) / denom


def deletion_faithfulness(
    snr_frame: torch.Tensor,    # (B, T) predicted per-frame SNR
    mask: torch.Tensor,         # (B, T) active mask
    delete_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Deletion test: removing the predicted-HIGHEST-SNR frames must drop the pooled
    SNR MORE than removing random frames.

    For each clip we pool the predicted timeline three ways: (a) full, (b) after
    masking out the top-`delete_frac` predicted-high-SNR active frames, (c) after
    masking out a RANDOM `delete_frac` of active frames. A faithful map has
    drop_high >> drop_random (the high-SNR frames it points at provably carry the
    pooled value). Returns the mean pooled drops + the fraction of clips where
    drop_high > drop_random (the per-clip win rate).

    The pool is monotone in every frame, so drop_high >= drop_random is the
    structural expectation; the magnitude/winrate quantify how concentrated (faithful)
    the predicted field is vs a flat one (where the two drops coincide).
    """
    g = torch.Generator(device=snr_frame.device).manual_seed(int(seed))
    B, T = snr_frame.shape
    full = _energy_pool(snr_frame, mask)                       # (B,)
    drops_high, drops_rand, wins = [], [], 0
    for b in range(B):
        m = mask[b].to(torch.bool)
        n_active = int(m.sum())
        if n_active < 2:
            continue
        k = max(1, int(round(delete_frac * n_active)))
        active_idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        # (b) delete top-k predicted-high frames
        vals = snr_frame[b][active_idx]
        top = active_idx[torch.topk(vals, k).indices]
        m_high = m.clone()
        m_high[top] = False
        pooled_high = _energy_pool(snr_frame[b:b + 1], m_high[None])[0]
        # (c) delete k random active frames
        perm = active_idx[torch.randperm(n_active, generator=g, device=snr_frame.device)[:k]]
        m_rand = m.clone()
        m_rand[perm] = False
        pooled_rand = _energy_pool(snr_frame[b:b + 1], m_rand[None])[0]

        dh = float(full[b] - pooled_high)
        dr = float(full[b] - pooled_rand)
        drops_high.append(dh)
        drops_rand.append(dr)
        wins += int(dh > dr)
    n = max(1, len(drops_high))
    return {
        "deletion_drop_high_db": float(sum(drops_high) / n),
        "deletion_drop_random_db": float(sum(drops_rand) / n),
        "deletion_win_rate": float(wins / n),
        "deletion_n_clips": float(len(drops_high)),
    }


def model_randomization(
    head: SupervisedSNRMapHead,
    audio_features: torch.Tensor,    # (B, T, audio_dim)
    mask: torch.Tensor,             # (B, T)
    seed: int = 0,
) -> dict[str, float]:
    """Adebayo (2018) model-parameter randomization sanity check.

    Re-initializes a FRESH head of the same shape (random weights), runs both heads on
    the same audio, and reports the mean per-clip Pearson correlation between the
    trained timeline and the random-weight timeline. A faithful map should DECORRELATE
    (corr near 0); a high correlation would mean the prediction is an input artifact
    independent of training.
    """
    with torch.no_grad():
        trained = head.forward_timeline(audio_features)        # (B, T)
        rand_head = SupervisedSNRMapHead(
            audio_dim=head.audio_dim, d_patch=head.d_patch, f_bins=head.f_bins,
            hidden=head.hidden, kernel_size=head.kernel_size,
            predict_irm=head.predict_irm, huber_delta=head.huber_delta,
        ).to(audio_features.device).to(head.in_proj.weight.dtype)
        rand = rand_head.forward_timeline(audio_features)      # (B, T)
    B = trained.shape[0]
    corrs = [_masked_pearson(trained[b], rand[b], mask[b]) for b in range(B)]
    return {"model_rand_corr": float(sum(corrs) / max(1, len(corrs)))}


# ── checkpoint reconstruction + .pt orchestration ───────────────────────────────
def _load_head(checkpoint: dict, device: str) -> SupervisedSNRMapHead:
    cfg = checkpoint.get("config", {})
    head = SupervisedSNRMapHead(
        audio_dim=int(cfg.get("snr_map_audio_dim", 1024)),
        d_patch=int(cfg.get("spec_d_patch", 768)),
        f_bins=int(cfg.get("snr_map_f_bins", 8)),
        hidden=int(cfg.get("snr_map_hidden", 256)),
        kernel_size=int(cfg.get("snr_map_kernel_size", 5)),
        predict_irm=bool(cfg.get("snr_map_predict_irm", False)),
        huber_delta=float(cfg.get("snr_map_huber_delta", 1.0)),
    ).to(device)
    sd = checkpoint.get("snr_map_head_state_dict")
    if sd is None:
        raise KeyError("checkpoint has no snr_map_head_state_dict")
    head.load_state_dict(sd)
    head.eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--snr_map_dir", required=True,
                    help="dir of oracle dense targets (compute_snr_map_targets.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_clips", type=int, default=0)
    ap.add_argument("--delete_frac", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    head = _load_head(ckpt, args.device)
    head_dtype = head.in_proj.weight.dtype

    manifest = json.load(open(os.path.join(args.snr_map_dir, "manifest.json")))
    pts = sorted(f for f in os.listdir(args.processed_dir) if f.endswith(".pt"))
    if args.max_clips:
        pts = pts[: args.max_clips]

    # accumulate per-clip predictions for batched-ish metrics (kept per-clip for memory)
    agg = {"timeline_pearson": [], "timeline_mae_db": []}
    del_high, del_rand, del_wins, n_del = [], [], 0, 0
    mr_corrs = []
    n_scored = 0
    for ptname in pts:
        cached = torch.load(os.path.join(args.processed_dir, ptname), weights_only=False)
        filename = cached.get("filename", os.path.splitext(ptname)[0] + ".wav")
        rel = manifest.get(filename)
        if rel is None:
            continue
        tgt = torch.load(os.path.join(args.snr_map_dir, rel), weights_only=False)
        af = cached["audio_features"].to(args.device).to(head_dtype).unsqueeze(0)   # (1,T,D)
        target = tgt["snr_map_target"].to(args.device)                              # (T,)
        mask = tgt.get("snr_map_mask", torch.ones_like(target)).to(args.device)     # (T,)
        T = min(af.shape[1], target.shape[0])
        af, target, mask = af[:, :T], target[:T], mask[:T]
        with torch.no_grad():
            pred = head.forward_timeline(af)[0, :T].float()                         # (T,)

        a = timeline_agreement(pred, target.float(), mask.float())
        agg["timeline_pearson"].append(a["timeline_pearson"])
        agg["timeline_mae_db"].append(a["timeline_mae_db"])

        d = deletion_faithfulness(pred[None], mask[None].float(),
                                  delete_frac=args.delete_frac)
        if d["deletion_n_clips"] > 0:
            del_high.append(d["deletion_drop_high_db"])
            del_rand.append(d["deletion_drop_random_db"])
            del_wins += int(d["deletion_win_rate"] > 0.5)
            n_del += 1

        mr = model_randomization(head, af, mask[None].float())
        mr_corrs.append(mr["model_rand_corr"])
        n_scored += 1

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    result = {
        "n_clips": n_scored,
        "timeline_pearson_mean": _mean(agg["timeline_pearson"]),
        "timeline_mae_db_mean": _mean(agg["timeline_mae_db"]),
        "deletion_drop_high_db": _mean(del_high),
        "deletion_drop_random_db": _mean(del_rand),
        "deletion_win_rate": float(del_wins / n_del) if n_del else 0.0,
        "model_rand_corr_mean": _mean(mr_corrs),
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
