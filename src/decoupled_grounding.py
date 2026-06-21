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


# BEATs frequency-patch count (128 mel / 16 patch). The flat patch map of length
# P factors as P = T_p * F_P, TIME-MAJOR (flat index = t * F_P + f) — the exact
# convention used in src/grounding_metrics.py, scripts/attention_gt_alignment.py,
# and src/spec_encoder.py::PatchGrid. The overlap-map supervision marginalizes the
# overlap query's map over these F_P frequency bins because overlap is a TIME
# property, so the target is a time-mask broadcast across all frequency bins.
F_P_DEFAULT: int = 8

# WavLM frame rate (Hz). duration_sec = n_wavlm_frames / WAVLM_FRAME_RATE_HZ —
# the SAME convention as src/grounding_validate.py::clip_duration_sec and
# src/inference.py::measured_duration_sec.
WAVLM_FRAME_RATE_HZ: float = 50.0

# Fallback only: WavLM frames per BEATs time-patch, used to recover a clip's VALID
# time-patch count from its duration when overlap_time_target is called WITHOUT an
# explicit valid_t_p (i.e. the standalone/unit path). The training loss term ALWAYS
# passes the exact valid_t_p it reads off the patch-valid mask, so this constant never
# enters the training objective — it just lets a direct overlap_time_target(segs, dur,
# padded_t_p) recover the valid region instead of spreading the clip across padding.
# 20 WavLM frames / patch → a 4 s clip (200 frames) has 10 valid time-patches.
WAVLM_FRAMES_PER_TIME_PATCH: int = 20


# BEATs patch feature dim. The encoder emits 768-d patch embeddings; kept as a
# module default so callers don't have to thread the magic number through.
DEFAULT_D_PATCH: int = 768

# ── hard-concrete / L0 gate constants (Louizos, Welling, Kingma — arXiv:1712.01312) ──
# The binary-concrete sample is "stretched" from (0,1) to (HC_GAMMA, HC_ZETA) and
# hard-clamped to [0,1], so a gate is EXACTLY 0 when the stretched sample is < 0 and
# EXACTLY 1 when it is > 1 — true deletion at the tails — while staying differentiable
# in the interior via the reparameterized concrete relaxation.
HC_GAMMA: float = -0.1   # lower stretch bound (< 0 → gives true-0 mass)
HC_ZETA: float = 1.1     # upper stretch bound (> 1 → gives true-1 mass)
HC_EPS: float = 1e-6     # numerical floor for the inverse-sigmoid noise term


# ── grounding modes ──────────────────────────────────────────────────────────
# "softmax"    — v17 head: A = softmax(qKᵀ), z = A·V.detach(), readout(z). Default.
# "bottleneck" — v18 head: per-feature hard-concrete keep-mask over patches, un-kept
#                patches substituted by the DETACHED per-patch marginal mean (the
#                IBA noise baseline), scalar read by mean-pool of the substituted
#                representation. Deletion-faithful by construction + a closed-form
#                bits penalty (β·meanbits). See src/.../2dmap-faithfulness-design.md.
GROUNDING_MODES = ("softmax", "bottleneck")


# ── per-feature bits-penalty β defaults (design doc §4.2) ────────────────────
# Length-N_FEATURES, in SUPERVISED_FEATURES order
#   [snr, srmr, f0_mean, f0_sd, speaking_rate, pause_count, pause_rate, overlap_ratio]
# GLOBAL features (snr, srmr, f0_mean, f0_sd, speaking_rate) get β=0: a diffuse mask
# is the FAITHFUL answer for a clip-global attribute, so it is never penalized. Only
# the localizable features pay bits: overlap_ratio (the one feature with oracle GT
# regions, 0.05) and pause_count/pause_rate (0.02 each). Keyed by short name so the
# constructor arg can be a partial dict; missing keys fall back to these.
DEFAULT_BITS_BETA: dict[str, float] = {
    "snr": 0.0,
    "srmr": 0.0,
    "f0_mean": 0.0,
    "f0_sd": 0.0,
    "speaking_rate": 0.0,
    "pause_count": 0.02,
    "pause_rate": 0.02,
    "overlap_ratio": 0.05,
}


def hard_concrete_sample(
    logits: torch.Tensor,
    temp: float,
    training: bool,
) -> torch.Tensor:
    """Hard-concrete keep-mask λ ∈ [0,1] per element (Louizos 1712.01312, eq. 10-12).

    TRAIN (training=True): draws u~Uniform(0,1) per call, forms the binary-concrete
    sample s = sigmoid((logits + log u − log(1−u)) / temp), STRETCHES it to
    (HC_GAMMA, HC_ZETA), then hard-clamps to [0,1]. The noise makes the gate
    stochastic and the clamp gives genuine 0/1 mass in the tails while the
    reparameterization keeps gradients flowing to `logits`.

    EVAL (training=False): deterministic, no noise — s = sigmoid(logits/temp),
    stretched and clamped the same way, so the returned mask is the expected
    deterministic keep-map λ̄ used for figures / extraction (and is ~binary once
    the logits are confident, by the same clamp).

    Args:
        logits:   (...,) keep-logit per element (q·Kᵀ/√d in the head).
        temp:     concrete temperature; lower → sharper toward 0/1.
        training: stochastic (True) vs deterministic (False).
    Returns:
        λ of the same shape as `logits`, every entry in [0,1].
    """
    if training:
        u = torch.rand_like(logits).clamp_(HC_EPS, 1.0 - HC_EPS)
        s = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / temp)
    else:
        s = torch.sigmoid(logits / temp)
    s_stretched = s * (HC_ZETA - HC_GAMMA) + HC_GAMMA
    return s_stretched.clamp(0.0, 1.0)


def hard_concrete_keepprob(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Closed-form P(gate > 0) for the hard-concrete gate (the analytic bits term).

    From Louizos 1712.01312: the probability the stretched gate is strictly positive
    (i.e. the patch is kept) is sigmoid(logits − temp·log(−HC_GAMMA/HC_ZETA)). No
    Monte-Carlo over masks — one cheap expression per element. Summed/averaged over
    patches it is the EXPECTED fraction of patches kept = meanbits.

    Args:
        logits: (...,) keep-logit per element.
        temp:   concrete temperature (same value used in the sample).
    Returns:
        keep-probability in (0,1), same shape as `logits`.
    """
    return torch.sigmoid(logits - temp * math.log(-HC_GAMMA / HC_ZETA))


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
        grounding_mode: str = "softmax",
        bits_beta_per_feature: "dict[str, float] | list[float] | tuple[float, ...] | None" = None,
        concrete_temp: float = 1.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if grounding_mode not in GROUNDING_MODES:
            raise ValueError(
                f"grounding_mode must be one of {GROUNDING_MODES}, got {grounding_mode!r}"
            )
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.d_patch = int(d_patch)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.huber_delta = float(huber_delta)
        self.grounding_mode = grounding_mode

        # ── bottleneck-mode state (hard-concrete keep-mask) ────────────────────
        # `concrete_temp` is an annealable BUFFER (rides into checkpoints, set per
        # epoch by train.py). β is a length-n_features tensor BUFFER (non-persistent
        # — re-resolved from config each run so it never goes stale).
        self.register_buffer(
            "concrete_temp", torch.tensor(float(concrete_temp)), persistent=True,
        )
        beta_vec = self._resolve_beta(bits_beta_per_feature)
        self.register_buffer("bits_beta", beta_vec, persistent=False)

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

    # ── per-feature β resolution (bottleneck mode) ────────────────────────────
    def _resolve_beta(
        self,
        bits_beta_per_feature: "dict[str, float] | list[float] | tuple[float, ...] | None",
    ) -> torch.Tensor:
        """Resolve the per-feature bits-penalty β into a length-n_features tensor.

        Accepts a dict keyed by short feature name (partial dicts allowed — missing
        keys fall back to DEFAULT_BITS_BETA when on the canonical catalog, else 0),
        a length-n_features list/tuple (positional), or None (→ DEFAULT_BITS_BETA on
        the catalog, all-zeros off it). GLOBAL features default to β=0 so a diffuse
        mask is never penalized.
        """
        if isinstance(bits_beta_per_feature, dict):
            names = (feature_names() if self.n_features == N_FEATURES
                     else [f"f{i}" for i in range(self.n_features)])
            beta = []
            for nm in names:
                if nm in bits_beta_per_feature:
                    beta.append(float(bits_beta_per_feature[nm]))
                else:
                    beta.append(float(DEFAULT_BITS_BETA.get(nm, 0.0)))
            return torch.tensor(beta, dtype=torch.float32)
        if isinstance(bits_beta_per_feature, (list, tuple)):
            if len(bits_beta_per_feature) != self.n_features:
                raise ValueError(
                    f"bits_beta_per_feature list must have {self.n_features} entries "
                    f"(SUPERVISED_FEATURES order), got {len(bits_beta_per_feature)}"
                )
            return torch.tensor([float(x) for x in bits_beta_per_feature], dtype=torch.float32)
        # None → catalog defaults (DEFAULT_BITS_BETA) on the canonical catalog, else 0.
        if self.n_features == N_FEATURES:
            return torch.tensor(
                [DEFAULT_BITS_BETA.get(nm, 0.0) for nm in feature_names()],
                dtype=torch.float32,
            )
        return torch.zeros(self.n_features, dtype=torch.float32)

    def set_concrete_temp(self, temp: float) -> None:
        """Set the concrete temperature in-place (called per-epoch for annealing)."""
        with torch.no_grad():
            self.concrete_temp.fill_(float(temp))

    def set_bits_beta(self, beta: "torch.Tensor | list[float] | tuple[float, ...]") -> None:
        """Overwrite the per-feature β in-place (called per-epoch for bits warmup)."""
        bt = torch.as_tensor(beta, dtype=self.bits_beta.dtype, device=self.bits_beta.device)
        if bt.shape != (self.n_features,):
            raise ValueError(
                f"beta must have shape ({self.n_features},), got {tuple(bt.shape)}"
            )
        with torch.no_grad():
            self.bits_beta.copy_(bt)

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

        Returns (SOFTMAX mode — v17, unchanged):
            A:           (B, n_features, P) — per-feature attention map over patches
                         (heads averaged). Rows sum to 1 over valid patches.
            z:           (B, n_features, d_model) — pooled vectors A @ V.detach().
                         V IS DETACHED here: this is the grounding-readout pool, so
                         the grounding gradient lands on the attention/queries, not
                         on V's encoding.
            pred_scalars:(B, n_features) — shallow readout of z.

        Returns (BOTTLENECK mode — v18):
            mask:        (B, n_features, P) — the hard-concrete KEEP-mask λ̄ ∈ [0,1]
                         per patch (does NOT sum to 1; it is a per-patch keep
                         probability, exactly the map IoU/RISE/pointing expect).
            z:           (B, n_features, d_model) — mean-pool of the noise-substituted
                         Z over valid patches (V still detached → grounding property).
            pred_scalars:(B, n_features).
        Side effect (bottleneck): the per-(b,f,p) keep-LOGIT and the valid mask are
        stashed on `self._last_logit_lambda` / `self._last_valid_mask` so the loss
        term can add the closed-form bits penalty without a second forward.
        """
        patches, patch_mask = self._flatten_inputs(patches, patch_mask)
        B, P, _ = patches.shape
        H, Dh = self.n_heads, self.d_head

        if self.grounding_mode == "bottleneck":
            return self._forward_bottleneck(patches, patch_mask, B, P, H, Dh)

        # ── SOFTMAX path (v17 — verbatim) ─────────────────────────────────────
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

    # ── bottleneck forward (v18 — hard-concrete keep-mask + noise substitution) ──
    def _forward_bottleneck(
        self,
        patches: torch.Tensor,    # (B, P, d_patch) already flattened
        patch_mask: torch.Tensor | None,   # (B, P) bool, True = VALID
        B: int, P: int, H: int, Dh: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The faithfulness-by-construction path. See design doc §3.2.

        Per feature f, per patch p:
          logit_λ = (q_f · K_p)/√d            (heads averaged → one logit per patch)
          λ       = hard_concrete(logit_λ)    ∈ [0,1], true 0/1 in the tails
          R       = V_proj(patches)           (kept patch representation; V detached)
          R̄       = masked_mean_p(R.detach()) (the IBA noise baseline)
          Z       = λ·R + (1−λ)·R̄.detach()    (un-kept patches replaced by noise)
          z       = masked_mean_p(Z)          (pool over valid patches)
          pred    = readout(z)
        """
        # K, V projections (multi-head folded back so the keep-logit is single per patch).
        K = self.K_proj(patches).view(B, P, H, Dh).transpose(1, 2)   # (B, H, P, Dh)
        V = self.V_proj(patches)                                     # (B, P, d_model) — "R"
        q = self.queries.view(self.n_features, H, Dh).transpose(0, 1).unsqueeze(0)  # (1,H,Nf,Dh)

        # per-head scores (B,H,Nf,P) → average heads → keep-logit (B,Nf,P).
        scores = torch.matmul(q, K.transpose(-1, -2)) * self._scale  # (B, H, Nf, P)
        logit_lambda = scores.mean(dim=1)                            # (B, Nf, P)

        # valid mask (B,P) → (B,1,P). Invalid patches are forced un-kept (λ=0) by
        # driving their logit very negative; they also never enter the pool denom.
        if patch_mask is not None:
            valid = patch_mask.view(B, 1, P)                         # (B,1,P) bool
            neg = torch.finfo(logit_lambda.dtype).min
            logit_lambda = logit_lambda.masked_fill(~valid, neg)
        else:
            valid = torch.ones(B, 1, P, dtype=torch.bool, device=patches.device)

        # hard-concrete keep-mask λ ∈ [0,1] (stochastic in train, deterministic eval).
        lam = hard_concrete_sample(logit_lambda, float(self.concrete_temp), self.training)  # (B,Nf,P)
        lam = lam * valid.to(lam.dtype)   # belt-and-braces: invalid → exactly 0

        # R = kept patch representation; R̄ = DETACHED per-patch marginal mean (noise).
        R = V                                                        # (B, P, d_model)
        validf = valid.squeeze(1).to(R.dtype).unsqueeze(-1)         # (B, P, 1)
        denom_patches = validf.sum(dim=1).clamp(min=1.0)           # (B, 1)
        R_bar = (R.detach() * validf).sum(dim=1) / denom_patches   # (B, d_model) — IBA baseline
        R_bar = R_bar.detach().unsqueeze(1).unsqueeze(2)           # (B,1,1,d_model)

        # noise substitution — THE faithfulness step. V detached (grounding property),
        # baseline detached (can't be made informative).
        R_det = R.detach().unsqueeze(1)                            # (B,1,P,d_model)
        lam_e = lam.unsqueeze(-1)                                  # (B,Nf,P,1)
        Z = lam_e * R_det + (1.0 - lam_e) * R_bar                  # (B,Nf,P,d_model)

        # mean-pool over VALID patches → z (B,Nf,d_model).
        valid_e = valid.unsqueeze(-1).to(Z.dtype)                 # (B,1,P,1)
        z = (Z * valid_e).sum(dim=2) / denom_patches.unsqueeze(1) # (B,Nf,d_model)

        pred_scalars = self._readout(z)                           # (B, Nf)

        # the returned "map" is the DETERMINISTIC expected keep-mask λ̄ (figures /
        # extraction / metrics). In eval `lam` is already deterministic; in train we
        # recompute the deterministic value so the returned map is stable to look at.
        if self.training:
            with torch.no_grad():
                lam_bar = hard_concrete_sample(logit_lambda, float(self.concrete_temp), training=False)
                lam_bar = lam_bar * valid.to(lam_bar.dtype)
        else:
            lam_bar = lam

        # stash for the bits penalty (read by decoupled_grounding_loss_term).
        self._last_logit_lambda = logit_lambda
        self._last_valid_mask = valid
        return lam_bar, z, pred_scalars

    # ── bits penalty (closed-form, bottleneck mode) ──────────────────────────────
    def bits_penalty(
        self,
        logit_lambda: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-feature analytic bits term Σ_f β_f · meanbits(λ_f) (design doc §4.1).

        meanbits(λ_f) = mean over valid patches of P(gate>0) = closed-form keep-prob.
        Uses the stashed logit/valid from the last bottleneck forward when not given.

        Returns:
            (weighted_bits_scalar, meanbits_dict). The scalar is Σ_f β_f·meanbits_f
            (already β-weighted, NOT yet times any global bits_lambda). The dict has
            one 'meanbits/<feat>' (the UNWEIGHTED per-feature keep fraction) for wandb.
        """
        if logit_lambda is None:
            logit_lambda = getattr(self, "_last_logit_lambda", None)
        if valid_mask is None:
            valid_mask = getattr(self, "_last_valid_mask", None)
        if logit_lambda is None:
            zero = (self.bits_beta.sum() * 0.0)
            return zero, {}

        B, Nf, P = logit_lambda.shape
        if valid_mask is None:
            valid = torch.ones(B, 1, P, dtype=torch.bool, device=logit_lambda.device)
        else:
            valid = valid_mask if valid_mask.dim() == 3 else valid_mask.view(B, 1, P)

        keep = hard_concrete_keepprob(logit_lambda, float(self.concrete_temp))  # (B,Nf,P)
        validf = valid.to(keep.dtype)                                           # (B,1,P)
        denom = validf.sum(dim=-1).clamp(min=1.0)                              # (B,1)
        # mean keep-prob over valid patches, per (b,f) → mean over batch → (Nf,)
        meanbits_bf = (keep * validf).sum(dim=-1) / denom                       # (B,Nf)
        meanbits = meanbits_bf.mean(dim=0)                                      # (Nf,)

        beta = self.bits_beta.to(meanbits.device, meanbits.dtype)              # (Nf,)
        weighted = (beta * meanbits).sum()

        metrics: dict[str, float] = {}
        names = (feature_names() if self.n_features == N_FEATURES
                 else [f"f{i}" for i in range(self.n_features)])
        for i, nm in enumerate(names):
            metrics[f"meanbits/{nm}"] = float(meanbits[i].detach().item())
        return weighted, metrics

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


def overlap_ratio_index() -> int:
    """Index of `overlap_ratio` in SUPERVISED_FEATURES (the one feature with oracle
    GT regions). Resolved from the catalog so it tracks any reordering."""
    return feature_names().index("overlap_ratio")


# ── DIRECT overlap-map supervision (the strongest grounding claim) ───────────────
# The decoupled grounding loss only WEAKLY supervises the 2D map: it regresses the
# overlap_ratio SCALAR from the pooled z, and a diffuse map predicts that scalar as
# well as a sharp one does. Here we add a SEGMENTATION-style target: the overlap
# query's map, marginalized over frequency to a per-time-bin activation, must land
# on the time region the oracle overlap_segments cover. This is the only feature
# with frame-level ground truth, so it is the strongest available grounding signal.


def overlap_time_target(
    overlap_segments: "list[tuple[float, float]] | list[list[float]]",
    duration_sec: float,
    t_p: int,
    soft: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    valid_t_p: int | None = None,
) -> torch.Tensor:
    """Build the per-clip (T_p,) TARGET time-mask from oracle overlap_segments.

    TIME MAPPING (documented, exact):
      The clip's REAL (unpadded) span is divided into `valid_t_p` equal time-patch
      bins; bin i ∈ [0, valid_t_p) covers the half-open wall-clock interval
          [ i * bin_dur ,  (i+1) * bin_dur ),    bin_dur = duration_sec / valid_t_p
      i.e. every VALID bin spans `duration_sec / valid_t_p` seconds. duration_sec comes
      from the WavLM frame count (n_frames / 50 Hz) — the same convention as
      grounding_validate.clip_duration_sec and inference.measured_duration_sec — and
      `valid_t_p` is the clip's UNPADDED time-patch count (its valid BEATs patch count
      // F_P). A bin is POSITIVE iff its interval intersects ANY overlap span, using
      the half-open intersection `t0 < seg_end and t1 > seg_start` (identical to
      grounding_metrics._time_bin_in_windows and attention_gt_alignment), so a clip
      with NO overlap (empty segments) yields an ALL-ZERO target (the map should be
      empty) and a MULTI-segment clip lights every bin any span touches.

    PADDED-BATCH CORRECTNESS (the fix for the variable-length-batch bug): the returned
    vector has length `t_p` (the BATCH-PADDED time-patch count), but bins are only ever
    SET over the VALID region [0, valid_t_p); the trailing [valid_t_p, t_p) bins (which
    correspond to PADDING patches whose map activation is forced to ~0) stay 0. Using
    `duration_sec / valid_t_p` (NOT duration_sec / t_p) keeps each bin's wall-clock
    width matched to the eval builder (grounding_metrics.iou_time, which runs the head
    on the UNPADDED clip with its own valid t_p), so the train target and the eval
    metric score the SAME time region.

    valid_t_p resolution: when given (the training loss term ALWAYS passes the exact
    valid time-patch count it reads off the patch-valid mask) it is used directly,
    clamped to ≤ t_p. When None (standalone / unit-test path with no mask), it is
    recovered from the duration as round(duration_sec * 50 / WAVLM_FRAMES_PER_TIME_PATCH),
    clamped to [1, t_p]; this fallback never enters the training objective.

    soft=True returns the FRACTION of each bin covered by overlap (∈ [0,1]) — a soft
    target that rewards partial-bin coverage and is smoother for Dice/gradient. With
    soft=False it returns the hard {0,1} intersection indicator. Either way an empty
    segment list → all zeros, and a clip with no measurable duration → all zeros.

    Args:
        overlap_segments: [(start_s, end_s), ...] in SECONDS (the .pt oracle field).
        duration_sec:     clip duration in seconds (n_wavlm_frames / 50).
        t_p:              OUTPUT length = the batch-padded time-patch count (P // F_P).
        soft:             True → per-bin covered fraction; False → hard 0/1 indicator.
        valid_t_p:        the clip's UNPADDED time-patch count (valid patches // F_P).
                          None → recover from duration via WAVLM_FRAMES_PER_TIME_PATCH.
    Returns:
        (t_p,) float target in [0,1], nonzero only in [0, valid_t_p); all-zero when
        there is no overlap / no duration.
    """
    target = torch.zeros(int(t_p), device=device, dtype=dtype)
    if t_p <= 0 or duration_sec <= 0 or not overlap_segments:
        return target
    # Resolve the clip's VALID time-patch count (the divisor for bin_dur). The loss
    # term passes it exactly; the standalone path recovers it from the duration so a
    # padded t_p doesn't spread the clip across padding bins.
    if valid_t_p is None:
        vtp = int(round(float(duration_sec) * WAVLM_FRAME_RATE_HZ
                        / float(WAVLM_FRAMES_PER_TIME_PATCH)))
    else:
        vtp = int(valid_t_p)
    vtp = max(1, min(vtp, int(t_p)))
    bin_dur = float(duration_sec) / float(vtp)
    for seg in overlap_segments:
        s0 = float(seg[0])
        s1 = float(seg[1])
        if s1 <= s0:
            continue
        # Only ever fill the VALID region [0, vtp); trailing padding bins stay 0.
        for i in range(vtp):
            t0 = i * bin_dur
            t1 = (i + 1) * bin_dur
            # half-open intersection of [t0,t1) with [s0,s1)
            lo = max(t0, s0)
            hi = min(t1, s1)
            if hi > lo:
                if soft:
                    # fraction of this bin covered by the span (accumulate across
                    # segments, clamp so multiple touching segments can't exceed 1).
                    target[i] = min(1.0, float(target[i]) + (hi - lo) / bin_dur)
                else:
                    target[i] = 1.0
    return target


def overlap_time_activation(
    overlap_map: torch.Tensor,
    f_p: int = F_P_DEFAULT,
    reduce: str = "max",
) -> torch.Tensor:
    """Marginalize the overlap feature's flat map over frequency → per-time activation.

    reduce='max' is the DEFAULT (not 'mean'). For a SOFTMAX map the row sums to 1 over
    P, so mean-over-frequency gives a per-time activation that sums to exactly 1/F_P
    for EVERY clip irrespective of shape (mass conservation) — a concentrated map and a
    diffuse one are then indistinguishable, killing the segmentation gradient. Max is
    not mass-conserving: a peaked row → a high per-time max on its bin, a diffuse row →
    a low max everywhere. For a BOTTLENECK keep-prob map (already unnormalized, true 1
    in the keep tail) max-over-frequency → 1 on a kept time bin, the correct reading.
    'mean' remains available for callers that want the conserved marginal explicitly.

    Args:
        overlap_map: (B, P) the overlap query's map (softmax A[:, ovl, :] rows, OR the
                     bottleneck keep-mask λ[:, ovl, :] keep-probs). P = T_p * F_P,
                     TIME-MAJOR (index = t*F_P + f).
        f_p:         frequency-patch count (default 8). T_p = P // f_p.
        reduce:      'max' (default) or 'mean' over the F_P frequency bins per time.
    Returns:
        (B, T_p) per-time-bin activation in the same numeric range as the input map.
    """
    if overlap_map.dim() != 2:
        raise ValueError(f"overlap_map must be (B, P), got {tuple(overlap_map.shape)}")
    B, P = overlap_map.shape
    t_p = P // f_p
    if t_p == 0:
        raise ValueError(f"P={P} too short for f_p={f_p}")
    grid = overlap_map[:, : t_p * f_p].reshape(B, t_p, f_p)   # (B, T_p, F_P)
    if reduce == "max":
        return grid.max(dim=-1).values
    if reduce == "mean":
        return grid.mean(dim=-1)
    raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")


def soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft-Dice loss between a per-time activation and a per-time target.

    WHY DICE (not BCE): overlap spans are SHORT — on a clip with one 1 s overlap in
    4 s, only ~25% of the time-bins are positive, and on clean clips 0%. A per-bin
    BCE is dominated by the majority NEGATIVE bins, so the trivial all-zero map gets
    a low BCE and the positive region is under-weighted. Soft-Dice is the harmonic
    overlap of the two masses, normalized by their sizes, so it is INVARIANT to the
    positive/negative ratio and directly maximizes intersection-over-(pred+gt) — the
    same quantity the IoU figure reports. It also gives a clean, non-vanishing
    gradient onto the positive bins even when they are a tiny fraction.

    Args:
        pred:   (B, T_p) activation in [0,1] (softmax mass per bin or keep-prob).
        target: (B, T_p) soft/hard target in [0,1].
        valid:  optional (B,) bool/float — rows to include (e.g. only overlap clips
                for the positive term). None → all rows.
        eps:    Laplace smoothing on numerator+denominator; also makes an all-zero
                pred vs all-zero target give loss 0 (the clean-clip ideal).
    Returns:
        scalar mean Dice loss over the included rows (1 - dice). 0 when no rows.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    inter = (pred * target).sum(dim=-1)                      # (B,)
    denom = pred.sum(dim=-1) + target.sum(dim=-1)            # (B,)
    dice = (2.0 * inter + eps) / (denom + eps)               # (B,) in (0,1]
    per_clip = 1.0 - dice                                    # (B,)
    if valid is not None:
        v = valid.to(per_clip.dtype)
        denom_v = v.sum().clamp(min=1.0)
        return (per_clip * v).sum() / denom_v
    return per_clip.mean()


def overlap_map_loss_from_map(
    overlap_map: torch.Tensor,
    overlap_targets: torch.Tensor,
    has_overlap: torch.Tensor,
    f_p: int = F_P_DEFAULT,
    reduce: str = "max",
    empty_weight: float = 1.0,
    iou_thresh: str | float = "median",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Segmentation-style supervision of the OVERLAP query's 2D map.

    Marginalizes the overlap map over frequency to a (B, T_p) per-time activation and
    compares it to the (B, T_p) oracle time-target with:

      - a soft-DICE positive term on OVERLAP-bearing clips (push the map ONTO the
        overlapped region), masked to has_overlap so non-overlap clips never
        contribute a (meaningless) positive Dice term, AND
      - a low-activation term on NON-overlap (clean) clips: the activation is driven
        toward 0 so a clean clip gets a near-EMPTY overlap map. This is what supports
        the hedging story (no overlap → the model should not light the overlap map).
        Implemented as the same Dice against the all-zero target, which for a clean
        clip reduces to pushing total activation down.

    FREQUENCY REDUCE = 'max' (default, NOT 'mean'). In SOFTMAX mode the attention row
    sums to 1 over P, so a MEAN-over-frequency activation sums to exactly 1/F_P for
    EVERY clip regardless of map shape (mass conservation). Under that reduction the
    clean-clip 'empty' Dice term is a constant with no gradient, and a concentrated map
    is indistinguishable from a diffuse one. A MAX-over-frequency reduction is NOT
    mass-conserving: a peaked softmax row → a high per-time max on its peak bin, a
    diffuse row → a low max everywhere, so the positive Dice gets gradient toward the
    target region and the clean term gets gradient toward empty. In BOTTLENECK mode the
    activation is the per-patch keep-PROBABILITY (unnormalized, true 1 in the keep
    tail), so max-over-frequency → 1 on a kept time bin, 0 on a dropped one — already
    the correct, non-conserved reading. Max makes the supervision HONEST in BOTH modes.

    Args:
        overlap_map:     (B, P) the overlap feature's flat map (softmax row or λ).
        overlap_targets: (B, T_p) per-clip time targets from overlap_time_target.
        has_overlap:     (B,) bool/float — True where the clip HAS overlap.
        f_p:             frequency-patch count (default 8).
        reduce:          'max' (default) | 'mean' frequency marginalization. 'max' is
                         scale-appropriate for BOTH the softmax row and the keep-prob.
        empty_weight:    weight on the clean-clip low-activation term.
        iou_thresh:      threshold for the logged train IoU. 'median' (default) →
                         per-clip median of the activation, MATCHING the eval metric
                         grounding_metrics.iou_time(thresh='median') so train/eval
                         agree; 'mean' → per-clip mean; a float → an absolute threshold.
                         (An absolute 0.5 was identically-0 IoU in softmax mode because
                         a mass-conserved activation never exceeds 1/F_P.)
    Returns:
        (loss, metrics). loss is the (positive-Dice + empty_weight*clean-Dice) blend,
        averaged over whichever group is present. metrics has 'overlap_map_dice_pos',
        'overlap_map_empty', and 'overlap_map_iou' (mean IoU on overlap clips).
    """
    act = overlap_time_activation(overlap_map, f_p=f_p, reduce=reduce)   # (B, T_p)
    # Align target T_p to the activation's T_p (defensive — both come from the same P).
    Tp = act.shape[1]
    if overlap_targets.shape[1] != Tp:
        tgt = overlap_targets[:, :Tp] if overlap_targets.shape[1] > Tp else F.pad(
            overlap_targets, (0, Tp - overlap_targets.shape[1])
        )
    else:
        tgt = overlap_targets
    tgt = tgt.to(act.dtype)
    has = has_overlap.to(act.dtype)                                      # (B,)

    # POSITIVE term — Dice only on overlap clips (push map ONTO the region).
    pos_loss = soft_dice_loss(act, tgt, valid=has)
    # CLEAN term — Dice on non-overlap clips against the all-zero target (→ empty map).
    clean = 1.0 - has
    zero_tgt = torch.zeros_like(tgt)
    clean_loss = soft_dice_loss(act, zero_tgt, valid=clean)

    n_pos = float(has.sum().item())
    n_clean = float(clean.sum().item())
    # Blend: average the two groups, weighting present groups only.
    total = act.new_zeros(())
    wsum = 0.0
    if n_pos > 0:
        total = total + pos_loss
        wsum += 1.0
    if n_clean > 0:
        total = total + empty_weight * clean_loss
        wsum += empty_weight
    loss = total / wsum if wsum > 0 else total

    # ── train IoU on overlap clips (logging only, no grad) ──
    # Threshold is RELATIVE by default ('median' → per-clip median of the activation
    # over its time bins), EXACTLY matching the eval metric
    # grounding_metrics.iou_time(thresh='median'), so the logged train IoU and the
    # validation IoU are on the same footing. (An absolute 0.5 read 0 in softmax mode,
    # where a mass-conserved activation maxes near 1/F_P; the relative threshold is
    # meaningful in BOTH the softmax row and the bottleneck keep-prob.)
    with torch.no_grad():
        if isinstance(iou_thresh, str):
            if iou_thresh == "median":
                thr = act.median(dim=-1, keepdim=True).values         # (B,1)
            elif iou_thresh == "mean":
                thr = act.mean(dim=-1, keepdim=True)                  # (B,1)
            else:
                raise ValueError(f"iou_thresh str must be 'median'|'mean', got {iou_thresh!r}")
            pred_bin = act > thr
        else:
            pred_bin = act > float(iou_thresh)
        gt_bin = (tgt > 0.5)
        inter = (pred_bin & gt_bin).sum(dim=-1).to(act.dtype)
        union = (pred_bin | gt_bin).sum(dim=-1).clamp(min=1).to(act.dtype)
        iou_per = inter / union
        denom_pos = has.sum().clamp(min=1.0)
        iou = float((iou_per * has).sum() / denom_pos)

    metrics = {
        "overlap_map_dice_pos": float(pos_loss.detach().item()) if n_pos > 0 else 0.0,
        "overlap_map_empty": float(clean_loss.detach().item()) if n_clean > 0 else 0.0,
        "overlap_map_iou": iou,
    }
    return loss, metrics


def _grad_overlap_map(
    head: "DecoupledGroundingHead",
    returned_map: torch.Tensor,
    overlap_idx: int,
) -> torch.Tensor:
    """Gradient-carrying (B, P) overlap-feature map for the segmentation loss.

    SOFTMAX mode: the head's returned A IS the grad-carrying attention (rows sum to 1
    over valid patches), so slice row `overlap_idx` directly.

    BOTTLENECK mode: the head's returned λ̄ is computed under no_grad (a stable map for
    figures), so it carries NO gradient. We instead rebuild the DIFFERENTIABLE
    keep-PROBABILITY P(gate>0) from the stashed keep-logit via hard_concrete_keepprob —
    the same closed form the bits penalty uses — which has a clean gradient onto the
    queries / K_proj. Invalid (padded) patches were driven to a very negative logit, so
    their keep-prob is ≈0 and they contribute nothing to the time activation.
    """
    if getattr(head, "grounding_mode", "softmax") == "bottleneck":
        logit = getattr(head, "_last_logit_lambda", None)   # (B, Nf, P)
        if logit is not None:
            keep = hard_concrete_keepprob(logit, float(head.concrete_temp))  # (B,Nf,P)
            return keep[:, overlap_idx, :]
    # softmax (or bottleneck without a stashed logit, e.g. eval) → returned map row.
    return returned_map[:, overlap_idx, :]


def decoupled_grounding_loss_term(
    head: "DecoupledGroundingHead | None",
    batch: dict,
    lambda_decoupled: float,
    device: torch.device | str = "cpu",
    bits_lambda: float = 0.0,
    lambda_overlap_map: float = 0.0,
    overlap_map_reduce: str = "max",
    overlap_map_empty_weight: float = 1.0,
    overlap_target_soft: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Parallel grounding-loss term off the BEATs patches (the train.py integration).

    Lives HERE (not in train.py) so it is importable and unit-testable WITHOUT
    pulling transformers / peft / wandb — train.py just re-exports it. It runs the
    token-free DecoupledGroundingHead on the batch's precomputed BEATs patches,
    regresses each scored feature's scalar from the pooled z (softmax: A·V.detach();
    bottleneck: mean-pool of the noise-substituted Z), and returns
    lambda_decoupled · masked_huber as a fresh loss term. Because the head reads
    `beats_patches` straight off the batch (the SAME field section_readout consumes)
    and pools over V.detach(), its gradient lands on the head's own queries / K_proj
    / readout — it NEVER flows through the LM token CE. It is a fully decoupled branch
    added to the total loss in compute_loss.

    In BOTTLENECK mode (head.grounding_mode == "bottleneck") the closed-form bits
    penalty Σ_f β_f·meanbits(λ_f) is added, scaled by `bits_lambda` (the per-epoch
    warmed global bits weight; 0 during bits warmup or for the softmax head).

    DIRECT OVERLAP-MAP SUPERVISION (lambda_overlap_map > 0): a SEGMENTATION-style loss
    on the OVERLAP query's map only. Builds a (B, T_p) oracle time-target from the
    batch's `overlap_segments` + per-clip duration (`audio_lens` / 50 Hz), marginalizes
    the overlap map over frequency, and adds lambda_overlap_map · (soft-Dice positive
    on overlap clips + low-activation on clean clips). Works in BOTH grounding modes
    (softmax row or differentiable bottleneck keep-prob). No-op when the batch carries
    no overlap_segments/audio_lens (back-compat). The overlap_ratio scalar regression
    above is UNCHANGED.

    Pulls from the batch (mirrors train.py's _build_section_ctx / section_readout):
        beats_patches:       (B, P, d_patch) precomputed BEATs patch embeddings.
        beats_patches_mask:  (B, P) bool, True at PADDED positions (collate_fn's
                             convention). The head wants True at VALID positions,
                             so it is inverted here.
        gt_scalars:          (B, F) Praat scalar GT (feature_set.extract_scalars).
        gt_mask:             (B, F) bool, True where the scalar was measured.
        overlap_segments:    (overlap-map supervision) list[B] of [(start_s,end_s),...].
        audio_lens:          (overlap-map supervision) (B,) int WavLM frame counts.

    Returns:
        (weighted_loss_or_None, metrics). No-op (None, {}) whenever the head is
        absent, lambda is <= 0, the batch carries no BEATs patches (legacy .pt), or
        there are no GT scalars — so it is safe to call unconditionally and is
        zero-overhead when off. metrics has 'loss_decoupled' (the UNWEIGHTED Huber),
        one 'decoupled_mae/<feat>' per scored feature, — in bottleneck mode —
        'loss_bits' (the UNWEIGHTED Σβ·meanbits) plus one 'meanbits/<feat>' each, and
        — when overlap-map supervision is on — 'loss_overlap_map' (UNWEIGHTED),
        'overlap_map_iou', 'overlap_map_dice_pos', 'overlap_map_empty'.
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

    _map, _z, pred_scalars = head(patches, patch_mask=valid_mask)
    loss, mae = head.grounding_loss(pred_scalars, gt_scalars, gt_mask)
    weighted = lambda_decoupled * loss

    metrics: dict[str, float] = {"loss_decoupled": float(loss.detach().item())}
    for i, fname in enumerate(feature_names()):
        metrics[f"decoupled_mae/{fname}"] = float(mae[i].item())

    # [bottleneck] add the closed-form bits penalty (β-weighted per feature). The
    # stashed logit/valid from the forward above feed bits_penalty (no 2nd forward).
    if getattr(head, "grounding_mode", "softmax") == "bottleneck":
        bits_weighted, bits_metrics = head.bits_penalty()
        metrics["loss_bits"] = float(bits_weighted.detach().item())
        metrics.update(bits_metrics)
        if bits_lambda > 0.0:
            weighted = weighted + bits_lambda * bits_weighted.to(weighted.dtype)

    # [overlap-map supervision] DIRECT segmentation loss on the overlap query's map.
    # Reuses the SINGLE forward above (no second pass): pulls the grad-carrying overlap
    # map (softmax row or bottleneck keep-prob), builds per-clip time targets from the
    # oracle overlap_segments + duration, and adds the soft-Dice term.
    if lambda_overlap_map > 0.0:
        segs_list = batch.get("overlap_segments")
        audio_lens = batch.get("audio_lens")
        if segs_list is not None:
            ovl_idx = overlap_ratio_index()
            ovl_map = _grad_overlap_map(head, _map, ovl_idx)          # (B, P) grad-carrying
            B, P = ovl_map.shape
            t_p = P // F_P_DEFAULT                                    # PADDED time-patch count
            # Per-clip VALID patch count → VALID time-patch count, so the target's
            # bins land on the clip's REAL [0, valid_t_p) region (NOT spread across the
            # batch-padded t_p / into padding bins). Source, IDENTICAL across modes:
            #   softmax   → valid_mask (~beats_patches_mask), shape (B, P).
            #   bottleneck→ head._last_valid_mask, shape (B, 1, P).
            # When no mask is present (all patches valid) the count is the full P → the
            # valid t_p equals the padded t_p (no padding, unchanged behavior).
            valid_patch_counts = None
            if valid_mask is not None:
                valid_patch_counts = valid_mask.reshape(B, -1).sum(dim=1)   # (B,)
            else:
                bn_valid = getattr(head, "_last_valid_mask", None)
                if bn_valid is not None:
                    valid_patch_counts = bn_valid.reshape(B, -1).sum(dim=1)  # (B,)
            if t_p > 0:
                # Per-clip duration: WavLM frame count / 50 Hz when available, else the
                # max overlap-segment end (matches grounding_validate.clip_duration_sec).
                targets = []
                has = []
                for b in range(B):
                    segs = segs_list[b] if b < len(segs_list) else []
                    segs = segs or []
                    if audio_lens is not None and b < len(audio_lens):
                        dur = float(audio_lens[b]) / WAVLM_FRAME_RATE_HZ
                    else:
                        dur = float(max((e for _s, e in segs), default=0.0))
                    if valid_patch_counts is not None:
                        valid_tp_b = int(valid_patch_counts[b].item()) // F_P_DEFAULT
                    else:
                        valid_tp_b = t_p
                    targets.append(
                        overlap_time_target(
                            segs, dur, t_p, soft=overlap_target_soft,
                            device=ovl_map.device, dtype=ovl_map.dtype,
                            valid_t_p=valid_tp_b,
                        )
                    )
                    has.append(1.0 if segs else 0.0)
                target_t = torch.stack(targets, dim=0)               # (B, T_p)
                has_t = torch.tensor(has, device=ovl_map.device, dtype=ovl_map.dtype)
                ovl_loss, ovl_metrics = overlap_map_loss_from_map(
                    ovl_map, target_t, has_t,
                    f_p=F_P_DEFAULT, reduce=overlap_map_reduce,
                    empty_weight=overlap_map_empty_weight,
                )
                metrics["loss_overlap_map"] = float(ovl_loss.detach().item())
                metrics["overlap_map_iou"] = ovl_metrics["overlap_map_iou"]
                metrics["overlap_map_dice_pos"] = ovl_metrics["overlap_map_dice_pos"]
                metrics["overlap_map_empty"] = ovl_metrics["overlap_map_empty"]
                weighted = weighted + lambda_overlap_map * ovl_loss.to(weighted.dtype)

    return weighted, metrics
