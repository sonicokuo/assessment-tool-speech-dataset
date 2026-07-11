"""Number Token Loss (NTL-WAS / abs variant), arXiv 2411.02083.

Why this exists
---------------
Cross-entropy treats digit tokens as NOMINAL: predicting "9" when the target is
"8" costs exactly as much as predicting "1". A digit token is only ~12% of the
prose tokens, so its (nominal) gradient is diluted by the ~88% structural tokens,
and the model learns the sentence skeleton long before it learns the numbers
(the documented "scalar→text dead" pathology). Naively up-weighting digit CE
hurts fluency without fixing the nominal-distance problem.

NTL fixes the missing ORDINAL signal. On every position whose TARGET token is a
single digit 0-9, it adds a regression penalty between the digit the model
EXPECTS and the digit the target wants:

    E_pred  = sum_d  d * softmax(logits[:, digit_ids])[d]      # expected digit
    L_NTL   = | E_pred - target_digit |                        # WAS/abs penalty

This is the value-weighted L1 (Wasserstein-1 on the ordered digit support)
variant of NTL: it is 0 when the predicted distribution over the 10 digit tokens
concentrates on the correct digit, and grows with the absolute distance to it.

Design notes (verified against the paper + this repo's needs):
  - Plug-and-play. It is an auxiliary term added to CE (total = CE + lambda*NTL,
    paper default lambda 0.3). No architecture change, no vocab change, normal
    token head. It only needs a map from the 10 single-digit tokens to {0..9}.
  - Masked to digit-target positions ONLY. Structural tokens, the decimal point,
    "=", padding, and the prefix/prompt positions contribute nothing.
  - Restricting the softmax to the 10 digit logits (rather than the full vocab)
    keeps E_pred well-defined as an expectation over digits, which is what makes
    the penalty an ordinal distance and not a vocab-wide quantity. This matches
    the NTL formulation (the head is free to put mass elsewhere; on a digit
    target position we read off only the relative digit distribution).
  - Numerically stable: a single log-softmax over 10 logits, no manual exp of raw
    logits. Returns a 0.0 scalar (with grad if inputs require grad) when there are
    no digit-target positions, so it is a safe no-op on digit-free batches.

This module is pure torch and CPU-testable (see tests/test_ntl.py): with a tiny
synthetic logits/targets example, the loss is strictly LOWER when the predicted
digit distribution concentrates on the correct digit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# The 10 single-digit characters, in value order 0..9. digit_token_ids() below
# maps each to its single tokenizer id and asserts the 1-token / distinct-id
# invariant (true for Qwen, which tokenizes numbers digit-by-digit).
_DIGIT_CHARS: tuple[str, ...] = tuple(str(d) for d in range(10))


def digit_token_ids(tokenizer) -> torch.Tensor:
    """Return a (10,) long tensor: token id of each single digit '0'..'9', in value order.

    Identifies the ids ONCE per tokenizer (cache the result and pass it into
    number_token_loss; do not call this per step). Encodes each digit character
    with no special tokens and requires it to map to exactly one id, distinct
    across the 10 digits — the digit-level-tokenization assumption NTL relies on
    (holds for Qwen, Llama-3, PaLM-style tokenizers).

    Raises:
        ValueError: if any single digit does not encode to exactly one token, or
                    if two digits collide on the same id (NTL cannot be applied —
                    the tokenizer is not digit-level and a different number
                    representation is required).
    """
    ids: list[int] = []
    for ch in _DIGIT_CHARS:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"digit {ch!r} encodes to {len(enc)} tokens ({enc}); NTL needs "
                f"single-token digits (digit-level tokenizer). Got tokenizer "
                f"{getattr(tokenizer, 'name_or_path', '?')}."
            )
        ids.append(int(enc[0]))
    if len(set(ids)) != 10:
        raise ValueError(
            f"single-digit token ids are not distinct: {ids}; NTL cannot map a "
            f"digit token back to a unique value."
        )
    return torch.tensor(ids, dtype=torch.long)


def number_token_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    digit_ids: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """NTL-WAS/abs penalty on digit-target positions.

    Args:
        logits:      (B, S, V) float — LM logits ALREADY ALIGNED so that
                     logits[b, j] is the distribution predicting target_ids[b, j].
                     (The caller does the next-token shift; see compute_loss.)
        target_ids:  (B, S) long — the target token id at each aligned position.
        digit_ids:   (10,) long — token ids of '0'..'9' in value order, from
                     digit_token_ids(tokenizer). Must live on / move to logits.device.
        target_mask: (B, S) bool/0-1, optional — True at positions that are real
                     target tokens (e.g. attention mask, NOT padding). Positions
                     where this is False are excluded even if they happen to equal
                     a digit id. If None, all positions are eligible.

    Returns:
        Scalar tensor: mean over digit-target positions of |E_pred - target_digit|.
        Returns a 0.0 scalar (carrying grad through `logits` so it is graph-safe)
        when there are no digit-target positions in the batch.

    The expected digit E_pred is computed over a softmax RESTRICTED to the 10
    digit logits, so it is a genuine expectation over {0..9} and the penalty is an
    ordinal distance. Uses log-softmax for numerical stability.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, S, V); got shape {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"target_ids {tuple(target_ids.shape)} must match logits[:2] "
            f"{tuple(logits.shape[:2])}"
        )

    device = logits.device
    digit_ids = digit_ids.to(device)

    # Which target positions are a single digit, and which digit value it is.
    #   is_digit[b, j]   True if target_ids[b, j] is one of the 10 digit ids
    #   tgt_value[b, j]  the digit value 0..9 at those positions (else 0, masked out)
    # (B, S, 10): broadcast-compare each target id against the 10 digit ids.
    eq = target_ids.unsqueeze(-1) == digit_ids.view(1, 1, -1)   # (B, S, 10) bool
    is_digit = eq.any(dim=-1)                                    # (B, S) bool
    # Position within digit_ids → the digit VALUE (digit_ids is in 0..9 order).
    # argmax over the 10 one-hot slots gives the value; 0 where not a digit (masked).
    tgt_value = eq.float().argmax(dim=-1)                        # (B, S) long-ish, in 0..9

    if target_mask is not None:
        is_digit = is_digit & target_mask.to(device).bool()

    if not bool(is_digit.any()):
        # No digit-target positions: return a graph-connected 0 so callers can
        # always add lambda * ntl without a branch, and grad flows as zero.
        return (logits.sum() * 0.0)

    # Expected digit under the predicted distribution RESTRICTED to digit logits.
    # Gather the 10 digit logits at every position, log-softmax over them (stable),
    # then E_pred = sum_d d * p_d.
    digit_logits = logits.index_select(dim=-1, index=digit_ids)          # (B, S, 10)
    log_p = F.log_softmax(digit_logits, dim=-1)                          # (B, S, 10)
    p = log_p.exp()                                                      # (B, S, 10)
    values = torch.arange(10, device=device, dtype=p.dtype)             # (10,)
    e_pred = (p * values.view(1, 1, -1)).sum(dim=-1)                     # (B, S)

    abs_err = (e_pred - tgt_value.to(e_pred.dtype)).abs()               # (B, S)
    mask_f = is_digit.to(e_pred.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (abs_err * mask_f).sum() / denom
