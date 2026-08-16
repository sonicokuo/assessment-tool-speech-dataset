"""Reliability-Aware Adapter for Overlap-Aware Speech Quality Description."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# The aux regression head's output dim MUST equal the number of supervised
# features. Read it from feature_set (the single source of truth) so expanding
# SUPERVISED_FEATURES (e.g. 8 → 12) auto-tracks the head, the masked-MSE GT
# tensor, and the presence mask without a second edit here.
from data.feature_set import N_FEATURES as _FS_N_FEATURES

# ── Constants ──────────────────────────────────────────────
MODEL_DIM = 1024
LM_DIM = 4096  # LM hidden dims
AUDIO_DIM = 1024  # WavLM output dims
# Per-frame overlap feature vector written by preprocess.py:
#   col 0: is_overlap                binary; 1 if this frame is inside an overlap segment
#   col 1: segment_duration_s        float;  duration (seconds) of the segment this frame belongs to, 0 outside overlap
#   col 2: frac_through_segment      float;  0–1 position within current segment (0 at start, 1 at end), 0 outside
#   col 3: density_300ms             float;  local overlap density smoothed over a ±150 ms window (0–1)
# NOTE: clip_overlap_ratio (the clip-wide GT scalar) was previously col 3 of a 5-channel layout.
# It was removed because it's also an SFS-evaluated feature; feeding it as input is data leakage —
# the model could trivially copy a side channel to its output and inflate overlap_ratio accuracy.
# Old checkpoints (5-channel) are incompatible with this 4-channel layout; retrain after upgrading.
OVERLAP_FEATURES = 4
OVERLAP_DIM = 32  # output of OverlapEmbedding to learn representation
N_AUX_FEATURES = _FS_N_FEATURES  # = feature_set.N_FEATURES (11); aux regression head output size


# ── Components ──────────────────────────────────────────────
# Overlap Embedding Layer (non-linear)
class OverlapEmbedding(nn.Module):
    def __init__(self, in_features: int = OVERLAP_FEATURES, embed_dim: int = OVERLAP_DIM):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(in_features, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, overlap_info: torch.Tensor) -> torch.Tensor:
        """(B, T, OVERLAP_FEATURES) -> (B, T, OVERLAP_DIM)"""
        return self.embedding(overlap_info)


# Conv Compress Block
class ConvCompressor(nn.Module):
    """Compress WavLM's 50 Hz frames to a prefix token rate.

    `compression` is the TOTAL factor: 8 -> 6.25 tok/s (160 ms/token, the default and what
    every checkpoint before 2026-07-27 used), 4 -> 12.5 tok/s (80 ms/token). At 6.25 Hz a
    token spans 160 ms, which aliases syllables (4-7/s) and short pauses (100-300 ms); the
    two weakest features are exactly speaking_rate and pause_count, so halving the stride is
    the direct test of the resolution hypothesis (Voxtral's 6.25-vs-12.5 Hz ablation).
    Only conv2's stride changes, so compression=8 is byte-identical to the previous code.
    """

    def __init__(self, in_dim: int = AUDIO_DIM, out_dim: int = MODEL_DIM,
                 compression: int = 8):
        super().__init__()
        if compression not in (4, 8):
            raise ValueError(f"compression must be 4 or 8, got {compression}")
        self.compression = compression
        stride2 = compression // 4          # 8 -> 2, 4 -> 1
        # We can also use average pooling. It has similar effect as we want here but conv layers give weights to eachi dim.
        self.conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=4,
            stride=4,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=2,
            stride=stride2,
        )

        self.gelu = nn.GELU()

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) -> (B, T//compression, out_dim)"""
        x = audio_features.transpose(1, 2)  # (B, T, 1024) -> (B, 1024, T)
        x = self.conv1(x)  # (B, 1024, T//4)
        x = self.gelu(x)
        x = self.conv2(x)  # (B, 1024, T//compression)
        x = self.gelu(x)
        x = x.transpose(1, 2)  # (B, T//compression, 1024)

        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for a given input length."""
        stride2 = self.compression // 4
        after_conv1 = (input_length - 4) // 4 + 1
        after_conv2 = (after_conv1 - 2) // stride2 + 1

        return after_conv2


# FiLM Conditioning Module
class FiLMConditioning(nn.Module):
    def __init__(self, lm_dim: int = LM_DIM, overlap_dim: int = OVERLAP_DIM):
        super().__init__()
        self.gamma = nn.Linear(overlap_dim, lm_dim)
        self.beta = nn.Linear(overlap_dim, lm_dim)

        # Residual init at expectation: gamma ≈ 1, beta ≈ 0 → FiLM(x, *) ≈ x at step 0.
        # Gamma weight uses small-random init (was zeros_) so the gradient of FiLM output
        # w.r.t. overlap_embed is NONZERO at step 0; otherwise FiLM-* variants are blind to
        # overlap signal until gamma.weight drifts off zero through random gradient noise.
        # See tests/test_film_init_diagnostic.py for the bug demonstration.
        nn.init.normal_(self.gamma.weight, mean=0.0, std=0.01)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, audio: torch.Tensor, overlap_embed: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(overlap_embed)
        beta = self.beta(overlap_embed)

        return gamma * audio + beta


# Sequential Context Block
class MambaContextBlock(nn.Module):
    """Adds sequential context to compressed audio tokens using Mamba SSM.

    After conv compression, each token only sees its local 160ms window.
    Mamba scans left-to-right, allowing each token to accumulate information
    from all preceding tokens — so token 20 knows about the clean speech
    at tokens 0-12 AND the overlap starting at token 13.

    Uses 1-2 Mamba layers. Each layer:
      - Selective state space: decides what to remember/forget per-step
      - d_state=16: 16-dim hidden state (how much "memory" per step)
      - d_conv=4: local conv within Mamba for fine-grained patterns
      - expand=2: internal expansion factor (2x wider intermediate dim)

    """

    def __init__(self, d_model: int = MODEL_DIM, n_layers: int = 1):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,  # SSM state dimension
                    d_conv=4,  # local convolution width
                    expand=2,  # internal expansion factor
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) — compressed audio tokens
        Returns:
            (B, N, d_model) — contextualized tokens (same shape)
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


# Self-Attention Context Block
class SelfAttentionContextBlock(nn.Module):
    def __init__(
        self,
        d_model: int = MODEL_DIM,
        n_head: int = 8,
        n_layers: int = 1,
    ):
        super().__init__()

        # Sinusoidal positional encoding
        # Max 500 tokens covers 4000 frames, 80 seconds, or 80k ms
        self.register_buffer("pos_enc", self._sinusoidal_pe(500, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 2,  # match Mamba's expand=2
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_layers)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-torch.arange(0, d_model, 2) / d_model * torch.log(torch.tensor(10000.0)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        N = x.shape[1]
        # BUGFIX 2026-07-25: `x +=` mutated the caller's tensor in place.
        x = x + self.pos_enc[:, :N, :]
        # collate_fn zero-pads to the batch max. Without a key-padding mask,
        # self-attention mixes padded positions into every real token (batch>1
        # training only), which would corrupt any attention-variant A/B.
        pad_mask = None
        if lengths is not None:
            ar = torch.arange(N, device=x.device).unsqueeze(0)
            pad_mask = ar >= lengths.to(x.device).unsqueeze(1)   # True = ignore
        x = self.layers(x, src_key_padding_mask=pad_mask)

        return x


# Full Assembled Adapter
class ReliabilityAwareAdapter(nn.Module):
    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        overlap_dim: int = OVERLAP_DIM,
        overlap_features: int = OVERLAP_FEATURES,
        d_model: int = MODEL_DIM,
        lm_dim: int = LM_DIM,
        n_layers: int = 1,
        context_type: str = "mamba",
        compression: int = 8,
        conditioning: str = "film",
    ):
        super().__init__()
        self.context_type = context_type

        # ------- Audio Path -------
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=d_model,
                                         compression=compression)

        # Context Block
        if n_layers < 0:
            raise ValueError("n_layers must be >= 0")
        elif n_layers == 0 or context_type == "none":
            self.context = nn.Identity()
        elif context_type == "mamba":
            self.context = MambaContextBlock(d_model=d_model, n_layers=n_layers)
        elif context_type == "attn":
            self.context = SelfAttentionContextBlock(d_model=d_model, n_layers=n_layers)
        else:
            raise ValueError(f"Unknown context_type '{context_type}'. Choose from: mamba, attn, none")

        self.proj_up = nn.Linear(d_model, lm_dim)

        # ------- Overlap Path -------
        self.overlap_embed = OverlapEmbedding(
            in_features=overlap_features,
            embed_dim=overlap_dim,
        )

        # ------- Overlap Conditioning -------
        # `conditioning` selects HOW overlap enters, holding the context block fixed.
        # This exists because concat-only -> film only isolates FiLM in the NO-CONTEXT
        # setting, while the shipped model is film-attn. Attention is a far more
        # expressive mixer than an MLP and could learn the modulation itself, so FiLM
        # may be redundant precisely where we use it. Answering that needs an
        # attention + non-FiLM arm (2026-07-29).
        # NOTE param asymmetry: FiLM is 2*(overlap_dim*lm_dim) ~= 0.26M, concat is
        # (lm_dim+overlap_dim)*lm_dim ~= 16.9M. The concat arm is therefore LARGER, which
        # biases against FiLM -- the conservative direction. Report it either way.
        if conditioning not in ("film", "concat"):
            raise ValueError(f"conditioning must be 'film' or 'concat', got {conditioning!r}")
        self.conditioning = conditioning
        if conditioning == "film":
            self.film = FiLMConditioning(lm_dim=lm_dim, overlap_dim=overlap_dim)
        else:
            # Mirrors ConcatOnlyAdapter's fusion so the two concat arms stay comparable.
            self.cond_proj = nn.Linear(lm_dim + overlap_dim, lm_dim)

        # ------- MLP -------
        self.mlp = nn.Sequential(
            nn.Linear(lm_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        overlap_info: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ------- Audio Path -------
        x = self.compressor(audio_features)
        # BUGFIX 2026-07-25: map input-frame lengths to POST-CONV lengths and hand
        # them to the context block, so attention variants get a key-padding mask
        # instead of attending over collate zero-padding (batch>1 training only).
        ctx_lens = None
        if lengths is not None:
            ratio = max(audio_features.shape[1] / max(x.shape[1], 1), 1.0)
            ctx_lens = (lengths.to(x.device).float() / ratio).ceil().long().clamp(1, x.shape[1])
        try:
            x = self.context(x, lengths=ctx_lens)
        except TypeError:      # Mamba block takes no lengths (unidirectional, no mask needed)
            x = self.context(x)
        x = self.proj_up(x)

        N = x.shape[1]

        # ------- Overlap Path -------
        o = self.overlap_embed(overlap_info)  # (B, T, overlap_dim)
        o = o.transpose(1, 2)  # (B, overlap_dim, T)
        o = F.adaptive_avg_pool1d(o, N)
        o = o.transpose(1, 2)  # (B, N, overlap_dim)

        # ------- Conditioning + MLP -------
        if self.conditioning == "film":
            x = self.film(x, o)
        else:
            x = self.cond_proj(torch.cat([x, o], dim=-1))
        x = self.mlp(x)

        return x


# Ablation Variants
class ConcatOnlyAdapter(nn.Module):
    """Baseline: concat overlap embeddings with audio, let MLP sort it out."""

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        overlap_features: int = OVERLAP_FEATURES,
        overlap_dim: int = OVERLAP_DIM,
        mamba_dim: int = MODEL_DIM,
        lm_dim: int = LM_DIM,
        compression: int = 8,
    ):
        super().__init__()
        # train.py passes compression unconditionally, so EVERY variant must accept it
        # or the run dies at adapter construction (caught 2026-07-29 before launch).
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=mamba_dim,
                                         compression=compression)
        self.overlap_embed = OverlapEmbedding(in_features=overlap_features, embed_dim=overlap_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mamba_dim + overlap_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, audio_features: torch.Tensor, overlap_info: torch.Tensor) -> torch.Tensor:
        x = self.compressor(audio_features)
        N = x.shape[1]
        o = self.overlap_embed(overlap_info)
        o = o.transpose(1, 2)
        o = F.adaptive_avg_pool1d(o, N)
        o = o.transpose(1, 2)
        x = torch.cat([x, o], dim=-1)
        x = self.mlp(x)
        return x


# Sigmoid Gating
class SigmoidGateAdapter(nn.Module):
    """Sigmoid gating: gate = sigmoid(f(overlap)), output = gate * audio."""

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        overlap_features: int = OVERLAP_FEATURES,
        overlap_dim: int = OVERLAP_DIM,
        mamba_dim: int = MODEL_DIM,
        lm_dim: int = LM_DIM,
        compression: int = 8,
    ):
        super().__init__()
        # train.py passes compression unconditionally, so EVERY variant must accept it
        # or the run dies at adapter construction (caught 2026-07-29 before launch).
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=mamba_dim,
                                         compression=compression)
        self.overlap_embed = OverlapEmbedding(in_features=overlap_features, embed_dim=overlap_dim)
        self.proj_up = nn.Linear(mamba_dim, lm_dim)
        self.gate = nn.Linear(overlap_dim, lm_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)  # sigmoid(2) ≈ 0.88
        self.mlp = nn.Sequential(
            nn.Linear(lm_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, audio_features: torch.Tensor, overlap_info: torch.Tensor) -> torch.Tensor:
        x = self.compressor(audio_features)
        N = x.shape[1]
        x = self.proj_up(x)
        o = self.overlap_embed(overlap_info)
        o = o.transpose(1, 2)
        o = F.adaptive_avg_pool1d(o, N)
        o = o.transpose(1, 2)
        g = torch.sigmoid(self.gate(o))
        x = g * x
        x = self.mlp(x)
        return x


# Q-Former (tests if temporal locality matters)
class QFormerAdapter(nn.Module):
    """Q-Former baseline: learnable queries cross-attend to all audio frames."""

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        overlap_features: int = OVERLAP_FEATURES,
        overlap_dim: int = OVERLAP_DIM,
        lm_dim: int = LM_DIM,
        compression: int = 8,   # accepted for interface parity; QFormer has no ConvCompressor
        n_queries: int = 32,
        n_heads: int = 8,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, audio_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=audio_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(audio_dim)
        self.overlap_embed = OverlapEmbedding(in_features=overlap_features, embed_dim=overlap_dim)
        self.proj_up = nn.Linear(audio_dim, lm_dim)
        self.film = FiLMConditioning(lm_dim=lm_dim, overlap_dim=overlap_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lm_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, audio_features: torch.Tensor, overlap_info: torch.Tensor) -> torch.Tensor:
        B = audio_features.shape[0]
        N = self.queries.shape[1]
        q = self.queries.expand(B, -1, -1)
        x, _ = self.cross_attn(q, audio_features, audio_features)
        x = self.norm(x + q)
        x = self.proj_up(x)
        o = self.overlap_embed(overlap_info)
        o = o.transpose(1, 2)
        o = F.adaptive_avg_pool1d(o, N)
        o = o.transpose(1, 2)
        x = self.film(x, o)
        x = self.mlp(x)
        return x


# ── Auxiliary regression head ──────────────────────────────────────
class AdapterWithAuxHead(nn.Module):
    """Wraps any inner adapter and adds an auxiliary regression head.

    The aux head mean-pools the prefix tokens and projects to N_AUX_FEATURES scalars.
    Used by B-full multi-task training to give the adapter a direct, undiluted MSE
    gradient on the audio→numerical-feature mapping — bypassing the LM and the noisy
    digit-subword cross-entropy path.

    Two head modes (selected by `reliability_head`):
      - reliability_head=False (default): a plain `nn.Linear(lm_dim, n_features)`
        regress_head predicting per-feature MEANS. forward returns (prefix,
        scalar_pred) and behaviour is byte-identical to the original aux head.
      - reliability_head=True: a HETEROSCEDASTIC `ReliabilityHead` predicting per
        feature a mean AND a log-variance (Linear(lm_dim, 2*n_features)). forward
        returns (prefix, (mean, log_var)); scalar_pred is the 2-tuple. The predicted
        σ = exp(0.5·log_var) is the per-feature ABSTENTION signal consumed by the
        risk-coverage eval. Trained with the heteroscedastic NLL (see compute_loss's
        lambda_nll term).

    At inference the prefix goes into the LM as before; the scalar head output is used
    only for the numbers/abstention path.
    """

    def __init__(
        self,
        inner: nn.Module,
        lm_dim: int = LM_DIM,
        n_features: int = N_AUX_FEATURES,
        reliability_head: bool = False,
        aux_pool: str = "mean",
    ):
        super().__init__()
        self.inner = inner
        self.reliability_head = bool(reliability_head)
        # AUX POOLING (added 2026-08-06). "mean" is the historical default and stays
        # byte-identical. "linear_softmax" exists because MEAN POOLING PROVABLY CANNOT
        # LOCALISE and we measured exactly that:
        #
        #   Exact per-token attributions on 400 test clips (the emitted value decomposes
        #   as v_f = (1/N)*SUM_t W_f.prefix_t, so each token's share is computable in
        #   closed form, not estimated). Normalised entropy of |contribution| over time,
        #   where 1.0 = perfectly uniform:
        #       snr 0.983 · srmr 0.993 · f0_mean 0.996 · jitter 0.996 · shimmer 0.997
        #       speaking_rate 0.997 · pause_rate 0.970 · pause_count 0.932
        #   and clip-SPECIFICITY was NEGATIVE for 8 of 11 features, i.e. a clip's map
        #   matched OTHER clips' structure better than its own -- the population-average
        #   map, the same failure that hollowed out the SRMR map (specificity 0.005-0.012).
        #
        # The cause is structural, not incidental: with y = mean_t(z_t) every frame
        # contributes exactly 1/T regardless of content, so nothing in the objective
        # rewards concentrating evidence. Wang et al., ICASSP 2019 (arXiv 1810.09050)
        # compare five MIL pooling functions for sound-event localisation and find mean
        # pooling the WORST, precisely because it rewards flat maps.
        #
        # linear_softmax weights each frame by its own magnitude, w_t = |z_t| / SUM|z_t|,
        # so a frame only matters if it commits. Signed variant of Wang's y = SUM z^2/SUM z
        # (theirs assumes non-negative detections; our z is signed, e.g. negative SNR dB).
        # Crucially it PRESERVES exact attribution: contribution_t = w_t * z_t still sums
        # to the emitted value, so the explainability map stays faithful by construction.
        if aux_pool not in ("mean", "linear_softmax"):
            raise ValueError(f"aux_pool must be 'mean' or 'linear_softmax', got {aux_pool!r}")
        self.aux_pool = aux_pool
        if self.reliability_head:
            # Imported here (not at module top) so adapter.py stays importable in
            # environments that only need the plain adapter. ReliabilityHead is a
            # thin Linear(lm_dim, 2*n_features) wrapper splitting mean / log-var.
            from model.reliability_head import ReliabilityHead
            self.regress_head = ReliabilityHead(lm_dim, n_features=n_features)
        else:
            self.regress_head = nn.Linear(lm_dim, n_features)

    def forward(
        self,
        audio_features: torch.Tensor,
        overlap_info: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object]:
        """`lengths` = per-clip UNPADDED WavLM frame counts (batch["audio_lens"]).

        BUGFIX 2026-07-25: collate_fn zero-pads to the batch max but nothing was
        threaded through, so the pooled aux/reliability input averaged PADDING
        during training (batch>1) while inference (batch=1) has none — a
        train/inference mismatch on exactly the head that drives abstention.
        None keeps the legacy unmasked mean (byte-identical at batch=1).
        """
        try:
            prefix = self.inner(audio_features, overlap_info, lengths=lengths)
        except TypeError:      # legacy variants without the lengths kwarg
            prefix = self.inner(audio_features, overlap_info)   # (B, N, lm_dim)
        # Valid-frame mask (1 = real, 0 = collate padding); shared by both poolings.
        if lengths is None:
            mask = None
        else:
            # Map input-frame lengths to post-conv prefix lengths via the actual
            # compression ratio of this forward (variant-agnostic).
            n_in, n_out = audio_features.shape[1], prefix.shape[1]
            L = lengths.to(prefix.device).long()
            # EXACT valid-output length, not ceil(L/ratio). The rounding matters:
            # ceil(250/8) = 32 but the true conv output for 250 frames is 31, so the
            # ceil version marks one padding-contaminated position as valid. Under mean
            # pooling that leaks 1/32 of one frame (measured output drift 0.079 when the
            # padded region is filled with garbage); under linear_softmax the same
            # contaminated frame can carry a large |z| and DOMINATE the magnitude
            # weighting (drift 1.82, a 23x amplification). Fixed 2026-08-06.
            comp = getattr(getattr(self.inner, "compressor", None), "get_output_length", None)
            if comp is not None:
                s2 = max(int(getattr(self.inner.compressor, "compression", 8)) // 4, 1)
                a1 = torch.div(L - 4, 4, rounding_mode="floor") + 1      # after conv1
                plen = torch.div(a1 - 2, s2, rounding_mode="floor") + 1  # after conv2
            else:                                    # variants without a ConvCompressor
                ratio = max(n_in / max(n_out, 1), 1.0)
                plen = torch.div(L.float(), ratio, rounding_mode="floor").long()
            plen = plen.clamp(1, n_out)
            mask = (torch.arange(n_out, device=prefix.device).unsqueeze(0)
                    < plen.unsqueeze(1)).to(prefix.dtype)    # (B, N) 1 = real

        if self.aux_pool == "mean":
            if mask is None:
                pooled = prefix.mean(dim=1)                 # (B, lm_dim)
            else:
                pooled = (prefix * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1.0).unsqueeze(-1)
            if self.reliability_head:
                mean, log_var = self.regress_head(pooled)   # each (B, n_features)
                return prefix, (mean, log_var)
            return prefix, self.regress_head(pooled)        # (B, n_features)

        # ── linear_softmax ────────────────────────────────────────────────────────
        # Apply the head PER FRAME, then pool the OUTPUTS weighted by their own
        # magnitude. Projecting first is what creates the localisation incentive:
        # pooling features and projecting afterwards (the "mean" branch) is linear, so
        # any weighting collapses back into a single averaged vector and the model can
        # smear evidence for free. Pooling per-frame PREDICTIONS cannot be collapsed.
        z = self.regress_head(prefix)                       # (B, N, F) or (B, N, 2F)
        if self.reliability_head:
            mean_t, logvar_t = z                            # each (B, N, F)
        else:
            mean_t, logvar_t = z, None
        w = mean_t.abs()
        if mask is not None:
            w = w * mask.unsqueeze(-1)                      # padded frames get zero weight
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, N, F), sums to 1 over time
        mean = (w * mean_t).sum(dim=1)                      # (B, F)
        if logvar_t is None:
            return prefix, mean
        # log-variance pooled with the SAME weights, so the uncertainty is read from the
        # same frames the value came from rather than from an unrelated average.
        log_var = (w * logvar_t).sum(dim=1)
        return prefix, (mean, log_var)


# Factory function: build any variant by name
def build_adapter(
    variant: str = "film-mamba",
    with_aux_head: bool = True,
    n_aux_features: int = N_AUX_FEATURES,
    reliability_head: bool = False,
    aux_pool: str = "mean",
    **kwargs,
) -> nn.Module:
    """Build an adapter variant by name, optionally wrapped with an aux regression head.

    Args:
        variant: one of concat-only / sigmoid-gate / film / film-attn / film-attn-2L /
                 film-mamba / film-mamba-2L / qformer.
        with_aux_head: if True (default), wraps the variant with AdapterWithAuxHead so
                       forward returns (prefix, scalar_pred). Set False to retain the
                       legacy single-tensor return signature (e.g. for old checkpoints).
        n_aux_features: number of scalar features the aux head regresses to (default
                       N_AUX_FEATURES = feature_set.N_FEATURES = 12).
        reliability_head: if True, the aux head is the HETEROSCEDASTIC ReliabilityHead
                       (predicts per-feature mean AND log-variance), and forward returns
                       (prefix, (mean, log_var)). Default False → plain Linear mean head,
                       byte-identical to before. Only consulted when with_aux_head=True.

    Returns:
        nn.Module whose forward(audio, overlap) returns:
          - with_aux_head=True, reliability_head=False:
              (prefix: (B,N,lm_dim), scalar_pred: (B, n_aux_features))
          - with_aux_head=True, reliability_head=True:
              (prefix: (B,N,lm_dim), (mean: (B, n_aux_features), log_var: (B, n_aux_features)))
          - with_aux_head=False: prefix only (legacy)
    """
    variants = {
        "concat-only": lambda **kw: ConcatOnlyAdapter(**kw),
        "sigmoid-gate": lambda **kw: SigmoidGateAdapter(**kw),
        "film": lambda **kw: ReliabilityAwareAdapter(context_type="none", **kw),
        "film-attn": lambda **kw: ReliabilityAwareAdapter(context_type="attn", n_layers=1, **kw),
        # FiLM ablation HOLDING THE CONTEXT BLOCK FIXED: same attention stack as
        # film-attn, overlap fused by concat+Linear instead of FiLM. film-attn minus
        # attn-concat is the marginal value of FiLM in the configuration we ship.
        "attn-concat": lambda **kw: ReliabilityAwareAdapter(
            context_type="attn", n_layers=1, conditioning="concat", **kw),
        "mamba-concat": lambda **kw: ReliabilityAwareAdapter(
            context_type="mamba", n_layers=1, conditioning="concat", **kw),
        "film-attn-2L": lambda **kw: ReliabilityAwareAdapter(context_type="attn", n_layers=2, **kw),
        "film-mamba": lambda **kw: ReliabilityAwareAdapter(context_type="mamba", n_layers=1, **kw),
        "film-mamba-2L": lambda **kw: ReliabilityAwareAdapter(context_type="mamba", n_layers=2, **kw),
        "qformer": lambda **kw: QFormerAdapter(**kw),
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variants.keys())}")

    inner = variants[variant](**kwargs)
    lm_dim = kwargs.get("lm_dim", LM_DIM)
    if with_aux_head:
        return AdapterWithAuxHead(
            inner, lm_dim=lm_dim, n_features=n_aux_features,
            reliability_head=reliability_head, aux_pool=aux_pool,
        )
    return inner
