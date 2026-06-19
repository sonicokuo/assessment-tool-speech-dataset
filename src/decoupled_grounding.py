"""decoupled_grounding.py — token-free 2D grounding head for AQUA-NL.

WHAT THIS IS
------------
A DETR/Slot-Attention/Q-Former-style grounding head that produces one attention
map per scored feature WITHOUT asking the language model to emit any special
section/feature tokens. It holds a fixed set of LEARNED per-feature query
embeddings (`self.queries`, shape (n_features, d_model)) that cross-attend to the
audio's time-frequency patch features (BEATs T*F patches). Each query yields:
  - A   — its attention map over the P=T*F patches (the per-feature 2D grounding).
  - z   — the attention-pooled patch vector A @ V.detach().
  - y   — a shallow readout of z to that feature's scalar (SNR, F0, ...).

WHY DECOUPLE (the degeneration problem this fixes)
--------------------------------------------------
The previous approach made the LM emit <sec_*>/<f_*> vocab tokens to *drive*
per-attribute attention. On hard inputs the LM emits those foreign tokens inside
otherwise-clean prose (degeneration): the grounding signal is entangled with text
generation, so a generation slip corrupts grounding and a grounding slip corrupts
text. Here the queries are PARAMETERS, not vocabulary. They cross-attend to audio
independently of what the LM decodes. The LM is free to generate clean, untagged
prose and never emits a special token. Grounding lives entirely in this head;
generation lives entirely in the LM. They share only the frozen/encoder audio
features, never the decoding channel. This is the standard DETR object-query /
Slot-Attention / Q-Former pattern, and it removes the degeneration at its root.

THE GROUNDING PROPERTY (copied from section_readout.py — V.detach)
------------------------------------------------------------------
The readout is supervised against Praat ground truth. We pool with V DETACHED:

    z = softmax(queries @ K^T / sqrt(d)) @ V.detach()
    y = readout(z)
    loss = masked_huber(y, gt)

`z` is the ONLY downstream product of the attention A, so a loss on `y` is a loss
on A. Crucially `z` has two ways to carry a feature: (a) move A onto the patches
that actually encode the feature, or (b) rewrite the patch encodings V to smuggle
the feature in. Path (b) leaves A ungrounded. We close it by detaching V in the
pooling, so the regression gradient cannot flow into V's projection — the only
remaining descent direction is to reshape A, which means reshaping the queries /
K_proj. The gradient therefore lands on `self.queries` (and K_proj), forcing each
feature's map onto real evidence. The readout is kept SHALLOW (linear or 1-hidden
MLP) so it cannot itself absorb the feature and let A stay flat: the attention map
is the bottleneck.

PER-FEATURE READOUTS (NOT one shared readout)
---------------------------------------------
Each feature gets its OWN readout (its own weight row + bias), batched into a single
parameter and applied with einsum (no Python loop). A single shared Linear(d_model→1)
reused across all features made the readout specialize toward the large-magnitude
features (f0_mean ~165, f0_sd ~46) and DRAG the small-magnitude ones (overlap_ratio
~0.3) the wrong way: in a live run overlap_ratio error rose during training while
f0_mean stayed pinned near 0. Since the readout gradient flows back into the
attention queries, a corrupted shared readout corrupts BOTH the scalar predictions
AND the per-feature 2D maps. Per-feature heads make every feature's readout (and
hence its query/map gradient) independent of the others. An optional
`feature_init_bias` seeds each readout's output bias with that feature's prior mean
so a high-mean feature does not start in a deep hole at 0.

(`V_proj` still receives gradient from any *separate* main task that consumes a
non-detached z, but NOT from this grounding loss — exactly as section_readout.py.)

MAP COLLAPSE (a flagged risk) AND THE DIVERSITY PENALTY
-------------------------------------------------------
With independent queries there is a failure mode where two features' maps collapse
to the same distribution (the head reads both scalars from one shared region).
`query_orthogonality_penalty` is an optional regularizer on the unit-normalized
query table: it penalizes off-diagonal cosine similarity so distinct features keep
distinct query directions (hence distinct maps). It is ~0 when the queries are
mutually orthogonal and >0 when they are parallel/identical.

SCOPE
-----
Training-only, like section_readout.py. At inference the maps are extracted for
figures; nothing here touches generation. Pure torch — no transformers/LM deps —
so it is unit-testable on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_set import N_FEATURES, SUPERVISED_FEATURES, FEATURE_SCALES


# BEATs patch feature dim. The encoder emits 768-d patch embeddings; kept as a
# module default so callers don't have to thread the magic number through.
DEFAULT_D_PATCH: int = 768


class DecoupledGroundingHead(nn.Module):
    """Learned per-feature queries cross-attending to T*F audio patches.

    Token-free: the queries are `nn.Parameter`s, NOT tied to any vocab token, so
    grounding is decoupled from the LM's text generation.

    Args:
        n_features:  number of scored features = number of queries (one each).
                     Defaults to feature_set.N_FEATURES so the head stays synced
                     with the canonical catalog.
        d_model:     internal attention dim (query / key / value width).
        d_patch:     incoming BEATs patch feature dim (e.g. 768).
        n_heads:     attention heads. Default 1 (single-head) so each feature's
                     map is a single, directly-interpretable (T, F) distribution
                     for the figures; the returned A is always (B, n_features, P)
                     regardless of head count (heads are averaged into one map).
        readout_hidden: None → a single PER-FEATURE Linear(d_model→1) readout (true
                     bottleneck, matches "keep it shallow"); an int → one per-feature
                     GELU hidden layer.
        feature_init_bias: optional length-n_features tensor / list of the per-feature
                     TARGET means used to initialize each readout's output bias, so a
                     feature starts predicting its prior mean instead of 0. UNITS: the
                     same units `grounding_loss` regresses, i.e. the RAW scalar units
                     of `gt_scalars` (the head predicts raw scalars; the loss
                     scale-normalizes only the *error*, never the prediction). So pass
                     raw per-feature means, e.g. [snr≈15, srmr≈5, f0_mean≈165, ...] in
                     SUPERVISED_FEATURES order. Default None → zeros (prior behavior).
        huber_delta: smooth-L1 / Huber transition point on the scale-normalized
                     error (same convention as section_readout.py).
    """

    def __init__(
        self,
        d_model: int,
        d_patch: int = DEFAULT_D_PATCH,
        n_features: int = N_FEATURES,
        n_heads: int = 1,
        readout_hidden: int | None = None,
        huber_delta: float = 1.0,
        feature_init_bias: "torch.Tensor | list[float] | tuple[float, ...] | None" = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.d_patch = int(d_patch)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.huber_delta = float(huber_delta)

        # ── learned per-feature query table — ONE query per scored feature ──────
        # nn.Parameter (NOT nn.Embedding tied to a tokenizer / vocab row): these
        # are free vectors, decoupled from the LM's token space. Initialized with
        # an orthogonal-ish spread (normalized normal) so different features start
        # from different directions and don't immediately collapse to one map.
        self.queries = nn.Parameter(torch.empty(self.n_features, self.d_model))
        nn.init.normal_(self.queries, mean=0.0, std=1.0 / math.sqrt(self.d_model))

        # ── key / value projections from patch space → attention space ─────────
        self.K_proj = nn.Linear(self.d_patch, self.d_model)
        self.V_proj = nn.Linear(self.d_patch, self.d_model)

        # ── SHALLOW *PER-FEATURE* readout: each feature's pooled z_i → 1 scalar ──
        # PER-FEATURE, NOT shared. A single shared Linear(d_model→1) reused across
        # every feature's z made the readout specialize toward the large-magnitude
        # features (f0_mean ~165, f0_sd ~46) and DRAG the small ones (overlap_ratio
        # ~0.3) the wrong way — overlap_ratio error climbed during training while
        # f0_mean stayed pinned near 0. Because the readout's gradient flows back
        # into the attention queries, a corrupted shared readout corrupts BOTH the
        # scalars AND every per-feature 2D map. Here each feature i owns its own
        # (W[i], b[i]) (and its own hidden layer when readout_hidden is set), so no
        # feature can drag another's parameters or its query/map. Implemented as a
        # single batched per-feature parameter (shape (n_features, ...)) applied with
        # einsum — no Python loop, one tensor op for all features.
        #
        # Parameterized form (NOT an nn.Linear/nn.Sequential) so each feature row is
        # an isolated parameter group whose gradient is independent by construction.
        self._readout_hidden = None if readout_hidden is None else int(readout_hidden)
        if self._readout_hidden is None:
            # Per-feature linear: pred[:, i] = z[:, i] @ W[i] + b[i].
            #   W: (n_features, d_model), b: (n_features,)
            self.readout_weight = nn.Parameter(torch.empty(self.n_features, self.d_model))
            self.readout_bias = nn.Parameter(torch.zeros(self.n_features))
            nn.init.normal_(self.readout_weight, mean=0.0, std=0.01)
        else:
            h = self._readout_hidden
            # Per-feature 1-hidden GELU MLP, all features batched:
            #   hidden[:, i] = GELU(z[:, i] @ W1[i] + b1[i])     W1: (Nf, d_model, h)
            #   pred[:, i]   = hidden[:, i] @ W2[i] + b2[i]      W2: (Nf, h, 1)
            self.readout_w1 = nn.Parameter(torch.empty(self.n_features, self.d_model, h))
            self.readout_b1 = nn.Parameter(torch.zeros(self.n_features, h))
            self.readout_w2 = nn.Parameter(torch.empty(self.n_features, h, 1))
            self.readout_bias = nn.Parameter(torch.zeros(self.n_features))
            # Kaiming-flavored small init on the hidden layer, near-zero output layer
            # so the head still starts ~at the bias (small-init philosophy).
            nn.init.normal_(self.readout_w1, mean=0.0, std=1.0 / math.sqrt(self.d_model))
            nn.init.normal_(self.readout_w2, mean=0.0, std=0.01)

        # Per-feature bias init: each feature's output bias → its prior TARGET mean
        # (raw scalar units; see __init__ docstring). With near-zero output weights
        # the head therefore starts predicting each feature's mean instead of 0, so a
        # large-mean feature (f0_mean ~165) doesn't sit in a ~165 Hz hole at step 0.
        if feature_init_bias is not None:
            bias_t = torch.as_tensor(feature_init_bias, dtype=self.readout_bias.dtype)
            if bias_t.shape != (self.n_features,):
                raise ValueError(
                    f"feature_init_bias must have shape ({self.n_features},), "
                    f"got {tuple(bias_t.shape)}"
                )
            with torch.no_grad():
                self.readout_bias.copy_(bias_t)

        self._scale = 1.0 / math.sqrt(self.d_head)

        # Per-feature loss-normalization scales — non-persistent so they never go
        # stale against the catalog on resume. When the head runs the canonical
        # catalog (n_features == N_FEATURES) we use FEATURE_SCALES so F0 (~150)
        # doesn't dominate overlap_ratio (~0.5); for a custom n_features (tests /
        # ablations not on the catalog) we fall back to unit scales.
        if self.n_features == N_FEATURES:
            scales = torch.tensor(FEATURE_SCALES, dtype=torch.float32)
        else:
            scales = torch.ones(self.n_features, dtype=torch.float32)
        self.register_buffer("scales", scales, persistent=False)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-attend the learned queries to the audio patches.

        Args:
            patches:    (B, P, d_patch) flattened T*F patches, OR (B, T, F, d_patch)
                        which is reshaped to (B, T*F, d_patch).
            patch_mask: (B, P) bool/float, True/1 for VALID patches. Optional. May
                        also be given as (B, T, F) when patches are (B, T, F, d).
                        Masked positions get -inf pre-softmax (≈0 attention).

        Returns:
            A:           (B, n_features, P) — per-feature attention map over patches
                         (heads averaged). Rows sum to 1 over valid patches.
            z:           (B, n_features, d_model) — pooled vectors A @ V.detach().
                         V IS DETACHED here: this is the grounding-readout pool, so
                         the grounding gradient lands on the attention/queries, not
                         on V's encoding.
            pred_scalars:(B, n_features) — shallow readout of z.
        """
        patches, patch_mask = self._flatten_inputs(patches, patch_mask)
        B, P, _ = patches.shape
        H, Dh = self.n_heads, self.d_head

        # K, V : (B, P, d_model) → (B, H, P, Dh)
        K = self.K_proj(patches).view(B, P, H, Dh).transpose(1, 2)   # (B, H, P, Dh)
        V = self.V_proj(patches).view(B, P, H, Dh).transpose(1, 2)   # (B, H, P, Dh)

        # queries : (n_features, d_model) → (1, H, n_features, Dh)
        q = self.queries.view(self.n_features, H, Dh).transpose(0, 1).unsqueeze(0)  # (1,H,Nf,Dh)

        # scores : (B, H, n_features, P)
        scores = torch.matmul(q, K.transpose(-1, -2)) * self._scale

        if patch_mask is not None:
            # mask: (B, P) → (B, 1, 1, P); set invalid positions to -inf.
            neg = torch.finfo(scores.dtype).min
            mexp = patch_mask.view(B, 1, 1, P)
            scores = scores.masked_fill(~mexp, neg)

        A_heads = torch.softmax(scores, dim=-1)                      # (B, H, n_features, P)

        # Pooled z with V DETACHED — the grounding property.
        z_heads = torch.matmul(A_heads, V.detach())                 # (B, H, n_features, Dh)
        # Recombine heads: concat along feature dim → (B, n_features, d_model).
        z = z_heads.transpose(1, 2).reshape(B, self.n_features, self.d_model)

        # Returned attention map: average over heads to ONE (B, n_features, P) map.
        A = A_heads.mean(dim=1)                                      # (B, n_features, P)

        pred_scalars = self._readout(z)                             # (B, n_features)
        return A, z, pred_scalars

    # ── per-feature readout ─────────────────────────────────────────────────────
    def _readout(self, z: torch.Tensor) -> torch.Tensor:
        """Apply each feature's OWN readout to its OWN pooled vector z_i.

        z: (B, n_features, d_model) → pred: (B, n_features). Feature i is computed
        from z[:, i] and parameters row i ONLY, so its gradient never touches another
        feature's readout (the independence property the per-feature heads buy us).
        Batched with einsum — one op for all features, no Python loop.
        """
        if self._readout_hidden is None:
            # pred[b, i] = sum_d z[b, i, d] * W[i, d] + b[i]
            pred = torch.einsum("bid,id->bi", z, self.readout_weight) + self.readout_bias
            return pred
        # hidden[b, i, h] = sum_d z[b, i, d] * W1[i, d, h] + b1[i, h]
        hidden = torch.einsum("bid,idh->bih", z, self.readout_w1) + self.readout_b1
        hidden = F.gelu(hidden)
        # pred[b, i] = sum_h hidden[b, i, h] * W2[i, h, 0] + bias[i]
        pred = torch.einsum("bih,iho->bio", hidden, self.readout_w2).squeeze(-1) + self.readout_bias
        return pred

    # ── input normalization ────────────────────────────────────────────────────
    def _flatten_inputs(
        self,
        patches: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Accept (B,P,d) or (B,T,F,d); fold T,F into P. Mask folded the same way
        and coerced to bool."""
        if patches.dim() == 4:
            B, T, Fdim, d = patches.shape
            patches = patches.reshape(B, T * Fdim, d)
            if patch_mask is not None and patch_mask.dim() == 3:
                patch_mask = patch_mask.reshape(B, T * Fdim)
        elif patches.dim() != 3:
            raise ValueError(f"patches must be (B,P,d) or (B,T,F,d), got {tuple(patches.shape)}")
        if patch_mask is not None:
            patch_mask = patch_mask.to(torch.bool)
        return patches, patch_mask

    # ── map reshape helper ──────────────────────────────────────────────────────
    @staticmethod
    def reshape_map(A: torch.Tensor, T: int, F_dim: int) -> torch.Tensor:
        """(B, n_features, P) → (B, n_features, T, F) given P == T*F."""
        B, Nf, P = A.shape
        if P != T * F_dim:
            raise ValueError(f"P={P} does not factor as T*F = {T}*{F_dim} = {T * F_dim}")
        return A.reshape(B, Nf, T, F_dim)

    # ── masked grounding loss ───────────────────────────────────────────────────
    def grounding_loss(
        self,
        pred_scalars: torch.Tensor,   # (B, n_features)
        gt_scalars: torch.Tensor,     # (B, n_features)
        gt_mask: torch.Tensor,        # (B, n_features) bool — True where supervised
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale-normalized masked Huber over present features.

        Mirrors SectionReadoutHead._masked_huber: divide the error by each
        feature's typical magnitude so F0 (~150) doesn't dominate overlap_ratio
        (~0.5), Huber it, average over supervised entries only.

        Returns:
            loss: scalar (0.0 if nothing is supervised — zero mask → zero loss).
            mae:  (n_features,) per-feature mean |normalized error| over supervised
                  entries (0 where a feature had no supervision this batch).
        """
        scales = self.scales.to(pred_scalars.device, pred_scalars.dtype)   # (n_features,)
        err = (pred_scalars - gt_scalars.to(pred_scalars.dtype)) / scales  # unit-free
        maskf = gt_mask.to(pred_scalars.dtype)
        per = F.huber_loss(
            err, torch.zeros_like(err), reduction="none", delta=self.huber_delta,
        ) * maskf
        denom = maskf.sum().clamp(min=1.0)
        loss = per.sum() / denom

        with torch.no_grad():
            cnt = maskf.sum(dim=0).clamp(min=1.0)
            mae = (err.abs() * maskf).sum(dim=0) / cnt
        return loss, mae


# ── optional anti-collapse regularizer ──────────────────────────────────────────
def query_orthogonality_penalty(queries: torch.Tensor) -> torch.Tensor:
    """Off-diagonal cosine-similarity penalty on the query table.

    Encourages distinct features to keep distinct query directions (hence distinct
    attention maps) — guards the map-collapse failure mode flagged for independent
    per-feature queries.

    Args:
        queries: (n_features, d_model) — e.g. `head.queries`.

    Returns:
        scalar = mean of squared off-diagonal cosine similarities. It is ~0 when
        the queries are mutually orthogonal and grows toward 1 as they become
        parallel / identical. With a single query the penalty is exactly 0.
    """
    if queries.dim() != 2:
        raise ValueError(f"queries must be (n_features, d_model), got {tuple(queries.shape)}")
    n = queries.shape[0]
    if n < 2:
        return queries.new_zeros(())
    q = F.normalize(queries, dim=-1)            # unit rows
    gram = q @ q.t()                            # (n, n) cosine sims, diag ≈ 1
    off = gram - torch.diag_embed(torch.diagonal(gram))
    # n*(n-1) off-diagonal entries.
    return (off ** 2).sum() / (n * (n - 1))


def feature_names() -> list[str]:
    """Convenience: the ordered short names of the scored features (for naming the
    per-feature MAEs / maps in wandb and figures)."""
    return [name for name, _csv, _fmt in SUPERVISED_FEATURES]


def decoupled_grounding_loss_term(
    head: "DecoupledGroundingHead | None",
    batch: dict,
    lambda_decoupled: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Parallel grounding-loss term off the BEATs patches (the train.py integration).

    Lives HERE (not in train.py) so it is importable and unit-testable WITHOUT
    pulling transformers / peft / wandb — train.py just re-exports it. It runs the
    token-free DecoupledGroundingHead on the batch's precomputed BEATs patches,
    regresses each scored feature's scalar from the attention-pooled
    z = A · V.detach(), and returns lambda_decoupled * masked_huber as a fresh
    loss term. Because the head reads `beats_patches` straight off the batch (the
    SAME field section_readout consumes) and pools over V.detach(), its gradient
    lands on the head's own queries / K_proj / readout — it NEVER flows through the
    LM token CE. It is a fully decoupled branch added to the total loss in
    compute_loss.

    Pulls from the batch (mirrors train.py's _build_section_ctx / section_readout):
        beats_patches:       (B, P, d_patch) precomputed BEATs patch embeddings.
        beats_patches_mask:  (B, P) bool, True at PADDED positions (collate_fn's
                             convention). The head wants True at VALID positions,
                             so it is inverted here.
        gt_scalars:          (B, F) Praat scalar GT (feature_set.extract_scalars).
        gt_mask:             (B, F) bool, True where the scalar was measured.

    Returns:
        (weighted_loss_or_None, metrics). No-op (None, {}) whenever the head is
        absent, lambda is <= 0, the batch carries no BEATs patches (legacy .pt), or
        there are no GT scalars — so it is safe to call unconditionally and is
        zero-overhead when off. metrics has 'loss_decoupled' (the UNWEIGHTED loss)
        and one 'decoupled_mae/<feat>' per scored feature.
    """
    if head is None or lambda_decoupled <= 0.0:
        return None, {}
    patches = batch.get("beats_patches")
    gt_scalars = batch.get("gt_scalars")
    gt_mask = batch.get("gt_mask")
    if patches is None or gt_scalars is None or gt_mask is None:
        return None, {}

    head_dtype = head.K_proj.weight.dtype
    patches = patches.to(device).to(head_dtype)

    # collate_fn emits beats_patches_mask=True at PADDED positions; the head's
    # patch_mask is True at VALID positions, so invert. None → all valid.
    pad_mask = batch.get("beats_patches_mask")
    valid_mask = None
    if pad_mask is not None:
        valid_mask = ~pad_mask.to(device).to(torch.bool)

    gt_scalars = gt_scalars.to(device)
    gt_mask = gt_mask.to(device)

    _A, _z, pred_scalars = head(patches, patch_mask=valid_mask)
    loss, mae = head.grounding_loss(pred_scalars, gt_scalars, gt_mask)
    weighted = lambda_decoupled * loss

    metrics: dict[str, float] = {"loss_decoupled": float(loss.detach().item())}
    for i, fname in enumerate(feature_names()):
        metrics[f"decoupled_mae/{fname}"] = float(mae[i].item())
    return weighted, metrics
