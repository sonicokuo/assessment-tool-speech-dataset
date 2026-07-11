"""Token-coupled grounding head (M1).

The novelty this head adds over ``snr_map_head.py`` (a dense per-frame map off the
raw WavLM features) is **coupling the grounding to the number the LM actually
emits**.  GLaMM-style: the LM hidden state at an emitted number-token position is
used as a query that attends over the audio *prefix* tokens (the adapter output the
LM conditions on), producing

  - ``alpha``  : a 1-D over-prefix (==over-time, 8x-compressed) attention map, one row
                 per emitted number token, and
  - ``pooled`` : the alpha-weighted read of a learned per-prefix value, i.e. a scalar
                 that the emitted number is supervised to match.

Two losses tie this together (both pre-scaled by their lambda by the caller):

  - **pooled-scalar Huber** ``pooled vs gt_scalar`` -- the number the model says is a
    grounded pooled read of the audio, not a free-floating LM guess.  This is the lever
    the deep-research flagged #1: "grounding exists but never reaches the token".
  - **Liu-2017 attention-correctness** ``-sum_n beta * log alpha`` -- pushes the
    number token's attention onto the oracle grounded region ``beta`` (a per-prefix
    distribution, e.g. the normalised per-frame SNR target pooled to prefix
    resolution), so the map points where the feature actually lives.

The head is deliberately self-contained and LM-agnostic: it consumes already-extracted
``token_hidden`` (B, Q, lm_dim) and ``prefix`` (B, N, prefix_dim) tensors, so it can be
unit-tested on toy tensors and wired into ``train.py``'s prose forward
(``output_hidden_states=True``, which is cheap, unlike ``output_attentions``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults align with src/adapter.py (LM_DIM read at runtime; MODEL_DIM prefix).
DEFAULT_LM_DIM = 4096
DEFAULT_PREFIX_DIM = 4096
DEFAULT_HIDDEN = 256


class TokenGroundingHead(nn.Module):
    """Number-token-conditioned attention over the audio prefix.

    Args:
        lm_dim:     hidden size of the LM (query source). Read from llm.config at
                    construction time in train.py.
        prefix_dim: hidden size of the adapter prefix tokens (keys/values).
        hidden:     projection width for the q/k attention.
        huber_delta: transition point of the pooled-scalar Huber loss.
    """

    def __init__(
        self,
        lm_dim: int = DEFAULT_LM_DIM,
        prefix_dim: int = DEFAULT_PREFIX_DIM,
        hidden: int = DEFAULT_HIDDEN,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(lm_dim, hidden)
        self.k_proj = nn.Linear(prefix_dim, hidden)
        # per-prefix scalar "local feature value"; pooled by alpha into the emitted number.
        self.v_proj = nn.Linear(prefix_dim, 1)
        self.scale = hidden ** -0.5
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        token_hidden: torch.Tensor,        # (B, Q, lm_dim)
        prefix: torch.Tensor,              # (B, N, prefix_dim)
        prefix_mask: torch.Tensor | None = None,  # (B, N) bool/float, True/1 = valid
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pooled (B,Q), alpha (B,Q,N), values (B,N))."""
        q = self.q_proj(token_hidden)                       # (B, Q, h)
        k = self.k_proj(prefix)                             # (B, N, h)
        logits = torch.einsum("bqh,bnh->bqn", q, k) * self.scale  # (B, Q, N)
        if prefix_mask is not None:
            m = prefix_mask.bool().unsqueeze(1)             # (B, 1, N)
            logits = logits.masked_fill(~m, float("-inf"))
        alpha = logits.softmax(dim=-1)                      # (B, Q, N)
        # Rows that were fully masked (no valid prefix, or Q-padding) -> softmax of all
        # -inf is nan; zero them so downstream losses (gated by gt_mask) stay finite.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        values = self.v_proj(prefix).squeeze(-1)            # (B, N)
        pooled = torch.einsum("bqn,bn->bq", alpha, values)  # (B, Q)
        return pooled, alpha, values

    def pooled_loss(
        self,
        pooled: torch.Tensor,      # (B, Q)
        gt_scalar: torch.Tensor,   # (B, Q)
        gt_mask: torch.Tensor,     # (B, Q) 1 where this number-slot is present
    ) -> torch.Tensor:
        gt_mask = gt_mask.to(pooled.dtype)
        err = pooled - gt_scalar.to(pooled.dtype)
        per = F.huber_loss(
            err, torch.zeros_like(err), reduction="none", delta=self.huber_delta
        ) * gt_mask
        return per.sum() / gt_mask.sum().clamp(min=1.0)

    def liu_loss(
        self,
        alpha: torch.Tensor,       # (B, Q, N) attention (rows sum to 1 over valid N)
        beta: torch.Tensor,        # (B, Q, N) oracle region distribution (sums to 1 over N)
        valid: torch.Tensor | None = None,  # (B, Q) 1 where the slot has an oracle region
    ) -> torch.Tensor:
        """Liu-2017 attention correctness: -sum_n beta * log alpha (cross-entropy of
        the attention against the oracle region distribution)."""
        logp = torch.log(alpha.clamp(min=1e-8))
        ce = -(beta * logp).sum(dim=-1)        # (B, Q)
        if valid is not None:
            valid = valid.to(ce.dtype)
            return (ce * valid).sum() / valid.sum().clamp(min=1.0)
        return ce.mean()


def token_grounding_loss_term(
    head: TokenGroundingHead | None,
    batch: dict,
    lambda_token_grounding: float,
    device: torch.device | str = "cpu",
    lambda_liu: float = 0.0,
) -> tuple[torch.Tensor | None, dict]:
    """Compute the (pre-scaled) token-coupled grounding loss for a batch.

    Reads from ``batch`` (all optional; term no-ops if the coupled tensors are absent,
    so a clip without an oracle region simply contributes nothing):
        tg_token_hidden : (B, Q, lm_dim)  LM hidden states at the number-token slots
        tg_prefix       : (B, N, prefix_dim) adapter prefix tokens
        tg_prefix_mask  : (B, N)          valid-prefix mask
        tg_gt_scalar    : (B, Q)          GT feature value per number slot
        tg_gt_mask      : (B, Q)          slot-present mask (pooled-scalar loss)
        tg_beta         : (B, Q, N)       oracle region distribution (Liu loss)
        tg_region_mask  : (B, Q)          slot-has-region mask (Liu loss)

    Returns (weighted_loss_or_None, metrics).  The returned loss is already multiplied
    by its lambdas (matching the snr_map convention of pre-scaling inside the term).
    """
    if head is None or lambda_token_grounding <= 0.0:
        return None, {}
    th = batch.get("tg_token_hidden")
    pf = batch.get("tg_prefix")
    if th is None or pf is None:
        return None, {"loss_token_grounding": 0.0}

    th = th.to(device)
    pf = pf.to(device)
    pfm = batch.get("tg_prefix_mask")
    pfm = pfm.to(device) if pfm is not None else None

    pooled, alpha, _ = head(th, pf, pfm)

    metrics: dict[str, float] = {}
    total = torch.zeros((), device=device, dtype=pooled.dtype)

    gt = batch.get("tg_gt_scalar")
    gtm = batch.get("tg_gt_mask")
    if gt is not None and gtm is not None:
        pl = head.pooled_loss(pooled, gt.to(device), gtm.to(device))
        total = total + lambda_token_grounding * pl
        metrics["loss_tg_pooled"] = float(pl.detach())

    if lambda_liu > 0.0:
        beta = batch.get("tg_beta")
        rm = batch.get("tg_region_mask")
        if beta is not None:
            ll = head.liu_loss(
                alpha, beta.to(device),
                rm.to(device) if rm is not None else None,
            )
            total = total + lambda_liu * ll
            metrics["loss_tg_liu"] = float(ll.detach())

    metrics["loss_token_grounding"] = float(total.detach())
    return total, metrics
