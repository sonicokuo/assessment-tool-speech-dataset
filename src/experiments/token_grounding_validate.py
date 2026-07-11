"""token_grounding_validate.py — causal deletion validation for the M1 token-coupled
grounding head (``src/token_grounding_head.py``).

Why this is a *separate* module from ``grounding_validate.py``: that script validates the
decoupled bottleneck head, whose map is a per-feature keep-mask over **spectral patches**
and which never touches the LM.  The M1 head is different in kind — its query is the LM
hidden state at the emitted number token and its map ``alpha`` is over the **WavLM adapter
prefix**.  So its deletion test is a different pipeline (LM-in-the-loop) and lives here.

Two deletion levels (design §7 "deletion: mask mapped frames -> the emitted number must
change"):

  - **head-level** (this file, CPU-unit-tested): mask the top-k% ``alpha``-attended prefix
    frames and check the head's ``pooled`` read moves MORE than masking k% random frames.
    WARNING — this is CIRCULAR and is a **plumbing/smoke check only, NOT a causal test**:
    ``pooled = sum(alpha * v)``, so masking high-``alpha`` frames perturbs that weighted sum
    by construction, and even a *random-weights* head "wins" (its own high-alpha frames
    dominate its own pool).  The model_rand sanity therefore does NOT collapse here — see
    ``tests/test_token_grounding_validate.py::test_head_level_deletion_is_circular_confound``.
  - **LM-level** (``lm_level_deletion`` below; needs the adapter+LM, run on GPU): mask the
    top-k% ``alpha`` frames in the prefix the LM conditions on, regenerate, and check the
    *emitted* SNR number changes more than under random masking.  This is the ONLY valid
    causal test — the emitted number does not mechanically depend on the head's ``alpha``.

The Adebayo model-randomisation sanity (random-weights head) is meaningful only for the
LM-level test: there a genuinely grounded map shows top-k >> random while the random head
collapses to ~chance.  For head-level it stays high for both (the circularity above).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from model.token_grounding_head import TokenGroundingHead


# ── head reconstruction ─────────────────────────────────────────────────────────
def load_token_grounding_head(
    checkpoint_path: str, device: str = "cpu"
) -> TokenGroundingHead:
    """Rebuild ``TokenGroundingHead`` from a checkpoint's ``token_grounding_head_state_dict``.

    Dims are inferred from the saved weight shapes (q_proj: (hidden, lm_dim),
    k_proj: (hidden, prefix_dim), v_proj: (1, prefix_dim)) so we don't depend on the
    training config being present.  Raises if the checkpoint has no token-grounding head.
    """
    ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ck.get("token_grounding_head_state_dict")
    if sd is None:
        raise ValueError(
            f"{checkpoint_path} has no token_grounding_head_state_dict — not an M1 run. "
            "(grounding_validate.py handles the decoupled bottleneck head instead.)"
        )
    hidden, lm_dim = sd["q_proj.weight"].shape
    prefix_dim = sd["k_proj.weight"].shape[1]
    head = TokenGroundingHead(lm_dim=lm_dim, prefix_dim=prefix_dim, hidden=int(hidden))
    head.load_state_dict(sd)
    head.to(device).eval()
    return head


def randomized_head_like(head: TokenGroundingHead, seed: int = 0) -> TokenGroundingHead:
    """A model_rand sanity twin: same shape, freshly-initialised (random) weights."""
    g = torch.Generator().manual_seed(seed)
    lm_dim = head.q_proj.in_features
    prefix_dim = head.k_proj.in_features
    hidden = head.q_proj.out_features
    rnd = TokenGroundingHead(lm_dim=lm_dim, prefix_dim=prefix_dim, hidden=hidden,
                             huber_delta=head.huber_delta)
    for p in rnd.parameters():
        p.data = torch.empty_like(p).normal_(0.0, 0.02, generator=g)
    return rnd.to(next(head.parameters()).device).eval()


# ── head-level deletion (self-contained, CPU-testable) ──────────────────────────
@torch.no_grad()
def head_level_deletion(
    head: TokenGroundingHead,
    token_hidden: torch.Tensor,          # (1, 1, lm_dim) — the emitted number's LM hidden state
    prefix: torch.Tensor,                # (1, N, prefix_dim)
    prefix_mask: torch.Tensor | None = None,  # (1, N) bool; None -> all valid
    frac: float = 0.2,
    generator: torch.Generator | None = None,
) -> dict:
    """Mask the top-``frac`` alpha-attended prefix frames vs the same count at random and
    measure how far the head's ``pooled`` read moves.  Returns per-clip deltas + a win flag
    (top-k moves pooled strictly more than random == the map is causal for the scalar).
    """
    device = prefix.device
    N = prefix.shape[1]
    if prefix_mask is None:
        prefix_mask = torch.ones(1, N, dtype=torch.bool, device=device)
    prefix_mask = prefix_mask.bool()

    pooled0, alpha, _ = head(token_hidden, prefix, prefix_mask)   # pooled0 (1,1), alpha (1,1,N)
    a = alpha[0, 0].clone()                                       # (N,)
    valid = prefix_mask[0]
    nvalid = int(valid.sum().item())
    if nvalid <= 1:
        return {"delta_top": 0.0, "delta_rand": 0.0, "win": False, "nvalid": nvalid}
    k = max(1, int(round(frac * nvalid)))

    # top-k alpha frames among valid
    a[~valid] = float("-inf")
    top_idx = torch.topk(a, k).indices
    pm_top = prefix_mask.clone()
    pm_top[0, top_idx] = False
    pooled_top, _, _ = head(token_hidden, prefix, pm_top)
    delta_top = float((pooled_top - pooled0).abs().item())

    # k random valid frames
    valid_idx = valid.nonzero(as_tuple=True)[0]
    perm = valid_idx[torch.randperm(nvalid, generator=generator, device=valid_idx.device)][:k]
    pm_rand = prefix_mask.clone()
    pm_rand[0, perm] = False
    pooled_rand, _, _ = head(token_hidden, prefix, pm_rand)
    delta_rand = float((pooled_rand - pooled0).abs().item())

    return {"delta_top": delta_top, "delta_rand": delta_rand,
            "win": delta_top > delta_rand, "nvalid": nvalid}


def summarize_deletion(results: Iterable[dict]) -> dict:
    """Aggregate per-clip deletion dicts into win-rate + mean drops."""
    rs = [r for r in results if r.get("nvalid", 0) > 1]
    if not rs:
        return {"n": 0, "win_rate": float("nan"), "mean_delta_top": float("nan"),
                "mean_delta_rand": float("nan"), "drop_ratio": float("nan")}
    n = len(rs)
    wr = sum(1 for r in rs if r["win"]) / n
    mt = sum(r["delta_top"] for r in rs) / n
    mr = sum(r["delta_rand"] for r in rs) / n
    return {"n": n, "win_rate": wr, "mean_delta_top": mt, "mean_delta_rand": mr,
            "drop_ratio": (mt / mr) if mr > 0 else float("inf")}


def run_head_level(
    head: TokenGroundingHead,
    samples: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    frac: float = 0.2,
    seed: int = 0,
    with_sanity: bool = True,
) -> dict:
    """Run head-level deletion over an iterable of (token_hidden, prefix, prefix_mask) tuples.

    Returns {"trained": summary, "model_rand": summary} — a real map has trained win-rate
    well above the model_rand win-rate (~0.5).
    """
    g = torch.Generator().manual_seed(seed)
    trained = summarize_deletion(
        head_level_deletion(head, th, pf, pm, frac=frac, generator=g)
        for (th, pf, pm) in samples
    )
    out = {"trained": trained}
    if with_sanity:
        rnd = randomized_head_like(head, seed=seed)
        g2 = torch.Generator().manual_seed(seed)
        # NB: samples may be a one-shot generator; caller passes a list for both passes.
        out["model_rand"] = summarize_deletion(
            head_level_deletion(rnd, th, pf, pm, frac=frac, generator=g2)
            for (th, pf, pm) in samples
        )
    return out


# ── LM-level deletion (needs adapter+LM; run on GPU) ────────────────────────────
def lm_level_deletion(
    generate_number: Callable[[torch.Tensor, torch.Tensor | None], float],
    head: TokenGroundingHead,
    token_hidden: torch.Tensor,
    prefix: torch.Tensor,
    prefix_mask: torch.Tensor | None = None,
    frac: float = 0.2,
    generator: torch.Generator | None = None,
) -> dict:
    """The real §7 claim: mask the top-alpha prefix frames the LM conditions on, regenerate,
    and measure the change in the *emitted* number vs random masking.

    ``generate_number(prefix, prefix_mask) -> float`` is supplied by the caller (wraps
    ``inference.py``'s greedy generation + SNR-number parse for a single clip), so this
    function stays LM-agnostic and unit-testable with a stub generator.
    """
    device = prefix.device
    N = prefix.shape[1]
    if prefix_mask is None:
        prefix_mask = torch.ones(1, N, dtype=torch.bool, device=device)
    prefix_mask = prefix_mask.bool()

    base = generate_number(prefix, prefix_mask)
    with torch.no_grad():
        _, alpha, _ = head(token_hidden, prefix, prefix_mask)
    a = alpha[0, 0].clone()
    valid = prefix_mask[0]
    nvalid = int(valid.sum().item())
    if nvalid <= 1 or base is None:
        return {"delta_top": float("nan"), "delta_rand": float("nan"),
                "win": False, "nvalid": nvalid}
    k = max(1, int(round(frac * nvalid)))

    a[~valid] = float("-inf")
    top_idx = torch.topk(a, k).indices
    pm_top = prefix_mask.clone(); pm_top[0, top_idx] = False
    d_top = generate_number(prefix, pm_top)

    valid_idx = valid.nonzero(as_tuple=True)[0]
    perm = valid_idx[torch.randperm(nvalid, generator=generator, device=valid_idx.device)][:k]
    pm_rand = prefix_mask.clone(); pm_rand[0, perm] = False
    d_rand = generate_number(prefix, pm_rand)

    delta_top = abs(d_top - base) if d_top is not None else float("nan")
    delta_rand = abs(d_rand - base) if d_rand is not None else float("nan")
    win = (d_top is not None and d_rand is not None and delta_top > delta_rand)
    return {"delta_top": delta_top, "delta_rand": delta_rand, "win": win, "nvalid": nvalid}
