"""SectionQueryHead — per-section cross-attention from LM to spec patches.

This is the module that operationalises the professor's design: when the LM
emits a <sec_X> token, a per-section query attends over the spectrogram-encoder
patches, producing one 2D attention map per section per clip. That map is the
evidence overlay for the EMNLP paper figure.

Architecture:

    spec patches (B, P, d_patch)       LM hidden state at <sec_X> (B, d_lm) — v2 only
              │                                          │
              ▼                                          │ (v1 uses a static
       W_k → K (B, P, d_k)                              │  learnable query
       W_v → V (B, P, d_v)                              │  per section instead)
              │                                          │
              ▼                                          ▼
              ────►── cross-attention ────► q_t (B, d_k)
                       q · K^T / √d_k
                       softmax → α (B, P)   ← THE ATTENTION MAP
                       α · V    → z (B, d_v)
                                  │
                       W_o → e_t (B, d_lm)
                                  │
                       inject into LM input at next position

Two query modes, selectable per checkpoint:

  STATIC (single-pass training, default for backward compat):
    self.queries: nn.Parameter (n_sections, d_k). At <sec_X> the query is
    looked up by section_idx — same vector regardless of clip context. Used
    via SectionQueryHead.forward(section_idx, K, V).

  DYNAMIC (two-pass training, matches the professor's "LM generates a query"):
    self.W_q: nn.Linear(d_lm, d_k). At <sec_X> the LM's hidden state h_t is
    projected to a query: q_t = W_q · h_t. The query reflects everything the
    LM has read so far (audio prefix + prompt + earlier sections). Used via
    SectionQueryHead.forward_dynamic(h_t, K, V, batch_idx).

Parameter budget (default config):
    n_sections × d_k        learnable queries (static):  6 × 256       = 1.5 K
    d_lm × d_k              W_q (dynamic):               2048 × 256    = 524 K
    d_patch × d_k           W_k:                         768 × 256     = 197 K
    d_patch × d_v           W_v:                         768 × 256     = 197 K
    d_v × d_lm              W_o:                         256 × 2048    = 524 K
    Total: ~ 1.4 M params — still negligible vs the 1.7B LM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SectionQueryHead(nn.Module):
    """Cross-attention from per-section queries to spec patches.

    Args:
        n_sections:    number of distinct sections (e.g. 6 for the EMNLP catalog).
        d_patch:       embedding dim of the spec encoder's output (768 for BEATs).
        d_lm:          LM hidden dim (1536 for Qwen2.5-1.5B, 2048 for Qwen3-1.7B).
        d_k:           internal query/key dim. Default 256 — comfortably below d_lm.
        d_v:           internal value dim. Default 256.
        residual_inject: if True (default), `e_t` is returned as a residual to be
                         ADDED to the LM input embedding at the section position.
                         If False, returned as a replacement embedding (legacy).
    """

    def __init__(
        self,
        n_sections: int,
        d_patch: int = 768,
        d_lm: int = 2048,
        d_k: int = 256,
        d_v: int = 256,
        residual_inject: bool = True,
    ):
        super().__init__()
        self.n_sections = n_sections
        self.d_patch = d_patch
        self.d_lm = d_lm
        self.d_k = d_k
        self.d_v = d_v
        self.residual_inject = residual_inject

        # Static learnable queries, one per section. Used in static mode.
        # Initialised small so the softmax at step 0 is roughly uniform over patches.
        self.queries = nn.Parameter(torch.randn(n_sections, d_k) * 0.02)

        # Dynamic-query projection. Used in dynamic mode (forward_dynamic).
        # Maps the LM's hidden state at a <sec_X> position into the cross-attention
        # query space. Small init: at step 0 the dynamic query is near-uniform too.
        self.W_q = nn.Linear(d_lm, d_k, bias=False)
        nn.init.normal_(self.W_q.weight, mean=0.0, std=0.01)

        self.W_k = nn.Linear(d_patch, d_k, bias=False)
        self.W_v = nn.Linear(d_patch, d_v, bias=False)
        self.W_o = nn.Linear(d_v, d_lm, bias=False)

        # Small init on W_o so the residual injection doesn't perturb the LM
        # input distribution at step 0 — the cross-attention path learns to
        # contribute as training progresses.
        nn.init.normal_(self.W_o.weight, mean=0.0, std=0.01)

    def precompute_kv(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project spec patches to K and V once per clip.

        Args:
            patches: (B, P, d_patch) from SpecEncoder.

        Returns:
            K: (B, P, d_k)
            V: (B, P, d_v)
        """
        return self.W_k(patches), self.W_v(patches)

    def forward(
        self,
        section_idx: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cross-attention for one section, return (e_t, α).

        Args:
            section_idx:       (B,) long tensor — which section's query to use
                               for each batch row.
            K:                 (B, P, d_k) from precompute_kv.
            V:                 (B, P, d_v) from precompute_kv.
            key_padding_mask:  (B, P) bool — True at padded positions to mask out.

        Returns:
            e_t: (B, d_lm) — the LM-dim audio summary for this section. Caller
                 adds (residual_inject=True) or replaces (False) the LM input
                 embedding at the section-open position.
            α:   (B, P) — softmax attention weights over patches. This is the
                 explainability output, kept for the paper figure.
        """
        # (B, d_k)
        q = self.queries[section_idx]

        # scores: (B, P) = (B, 1, d_k) @ (B, d_k, P) → squeeze
        scores = torch.einsum("bd,bpd->bp", q, K) / math.sqrt(self.d_k)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        alpha = scores.softmax(dim=-1)              # (B, P) — the attention map

        # attended value: (B, d_v) = einsum (B, P) (B, P, d_v)
        z = torch.einsum("bp,bpd->bd", alpha, V)
        e_t = self.W_o(z)                            # (B, d_lm)
        return e_t, alpha

    def forward_dynamic(
        self,
        h_t: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic-query forward: query is derived from the LM's hidden state.

        Use this in the v2 / EMNLP-faithful design where each <sec_X> position
        produces a context-conditional query from `h_t = LM hidden state at that
        position`. Compared to the static `forward`, the query reflects
        everything the LM has read so far (audio + prompt + earlier sections),
        so it can ask different things of the spec depending on what's already
        been said.

        Two input shapes are supported:

          (a) `h_t` shape (N, d_lm); K, V shape (B, P, d_k|d_v); `batch_idx`
              shape (N,) maps each query to the clip whose K/V it should attend
              to. Used during training when we vectorise the per-clip queries
              for one batch.

          (b) `h_t` shape (B, d_lm); K, V shape (B, P, d_k|d_v); batch_idx None.
              Used at inference time for one query per clip in the current step.

        Returns:
            e_t: (N or B, d_lm)
            alpha: (N or B, P)
        """
        # Project hidden state to query.
        q = self.W_q(h_t)                                # (N, d_k) or (B, d_k)

        if batch_idx is None:
            # Shape (a) collapses to (b) with N == B and a 1-to-1 mapping.
            # einsum directly.
            scores = torch.einsum("bd,bpd->bp", q, K) / math.sqrt(self.d_k)
        else:
            # Gather per-query K/V using batch_idx
            K_per = K[batch_idx]                          # (N, P, d_k)
            V_per_idx = V[batch_idx]                      # (N, P, d_v)
            scores = torch.einsum("nd,npd->np", q, K_per) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            mask = key_padding_mask if batch_idx is None else key_padding_mask[batch_idx]
            scores = scores.masked_fill(mask, float("-inf"))

        alpha = scores.softmax(dim=-1)                   # (N or B, P)

        if batch_idx is None:
            z = torch.einsum("bp,bpd->bd", alpha, V)
        else:
            z = torch.einsum("np,npd->nd", alpha, V_per_idx)

        e_t = self.W_o(z)                                # (N or B, d_lm)
        return e_t, alpha

    def forward_all_sections(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cross-attention for ALL sections at once.

        Used at inference when we want to save all section attention maps in a
        single pass (each section has its own static query, so we can vectorise).

        Args:
            K:  (B, P, d_k)
            V:  (B, P, d_v)
            key_padding_mask: (B, P) bool

        Returns:
            e_all: (B, n_sections, d_lm)
            alpha_all: (B, n_sections, P)
        """
        # (n_sections, d_k) → (1, n_sections, d_k) → (B, n_sections, d_k) via broadcast
        q = self.queries.unsqueeze(0).expand(K.shape[0], -1, -1)
        scores = torch.einsum("bsd,bpd->bsp", q, K) / math.sqrt(self.d_k)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        alpha = scores.softmax(dim=-1)               # (B, n_sections, P)
        z = torch.einsum("bsp,bpd->bsd", alpha, V)
        e_all = self.W_o(z)                          # (B, n_sections, d_lm)
        return e_all, alpha
