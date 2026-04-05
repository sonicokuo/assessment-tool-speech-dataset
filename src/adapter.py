"""Reliability-Aware Adapter for Overlap-Aware Speech Quality Description."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ── Constants ──────────────────────────────────────────────
MODEL_DIM = 1024
LM_DIM = 4096  # LM hidden dims
AUDIO_DIM = 1024  # WavLM output dims
OVERLAP_FEATURES = 2  # per frame: (is_overlap, overlap_ratio)
OVERLAP_DIM = 32  # output of OverlapEmbedding to learn representation


# ── Components ──────────────────────────────────────────────
# Overlap Embedding Layer (non-linear)
class OverlapEmbedding(nn.Module):
    def __init__(self, in_features: int = OVERLAP_FEATURES, embed_dim: int = OVERLAP_DIM):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(in_features, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, overlap_info: torch.Tensor) -> torch.Tensor:
        """(B, T, 2) -> (B, T, OVERLAP_DIM)"""
        return self.embedding(overlap_info)


# Conv Compress Block
class ConvCompressor(nn.Module):
    def __init__(self, in_dim: int = AUDIO_DIM, out_dim: int = MODEL_DIM):
        super().__init__()
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
            stride=2,
        )

        self.gelu = nn.GELU()

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) -> (B, T//8, out_dim)"""
        x = audio_features.transpose(1, 2)  # (B, T, 1024) -> (B, 1024, T)
        x = self.conv1(x)  # (B, 1024, T//4)
        x = self.gelu(x)
        x = self.conv2(x)  # (B, 1024, T//8)
        x = self.gelu(x)
        x = x.transpose(1, 2)  # (B, T//8, 1024)

        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for a given input length."""
        after_conv1 = (input_length - 4) // 4 + 1
        after_conv2 = (after_conv1 - 2) // 2 + 1

        return after_conv2


# FiLM Conditioning Module
class FiLMConditioning(nn.Module):
    def __init__(self, lm_dim: int = LM_DIM, overlap_dim: int = OVERLAP_DIM):
        super().__init__()
        self.gamma = nn.Linear(overlap_dim, lm_dim)
        self.beta = nn.Linear(overlap_dim, lm_dim)

        nn.init.zeros_(self.gamma.weight)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[1]
        x += self.pos_enc[:, :N, :]
        x = self.layers(x)

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
    ):
        super().__init__()
        self.context_type = context_type

        # ------- Audio Path -------
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=d_model)

        # Context Block
        if n_layers < 0:
            raise ValueError(f"n_layers should > 0")
        elif context_type == "mamba" and n_layers > 0:
            self.context = MambaContextBlock(d_model=d_model, n_layers=n_layers)
        elif context_type == "attn" and n_layers > 0:
            self.context = SelfAttentionContextBlock(d_model=d_model, n_layers=n_layers)
        else:
            self.context = nn.Identity()

        self.proj_up = nn.Linear(d_model, lm_dim)

        # ------- Overlap Path -------
        self.overlap_embed = OverlapEmbedding(
            in_features=overlap_features,
            embed_dim=overlap_dim,
        )

        # ------- FiLM Conditioning -------
        self.film = FiLMConditioning(lm_dim=lm_dim, overlap_dim=overlap_dim)

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
    ) -> torch.Tensor:
        # ------- Audio Path -------
        x = self.compressor(audio_features)
        x = self.context(x)
        x = self.proj_up(x)

        N = x.shape[1]

        # ------- Overlap Path -------
        o = self.overlap_embed(overlap_info)  # (B, T, overlap_dim)
        o = o.transpose(1, 2)  # (B, overlap_dim, T)
        o = F.adaptive_avg_pool1d(o, N)
        o = o.transpose(1, 2)  # (B, N, overlap_dim)

        # ------- MLP -------
        x = self.film(x, o)
        x = self.mlp(x)

        return x


# Ablation Variants
class ConcatOnlyAdapter(nn.Module):
    """Baseline: concat overlap embeddings with audio, let MLP sort it out."""

    def __init__(
        self,
        audio_dim=AUDIO_DIM,
        overlap_features=OVERLAP_FEATURES,
        overlap_dim=OVERLAP_DIM,
        mamba_dim=MODEL_DIM,
        lm_dim=LM_DIM,
    ):
        super().__init__()
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=mamba_dim)
        self.overlap_embed = OverlapEmbedding(in_features=overlap_features, embed_dim=overlap_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mamba_dim + overlap_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, audio_features, overlap_info):
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
        audio_dim=AUDIO_DIM,
        overlap_features=OVERLAP_FEATURES,
        overlap_dim=OVERLAP_DIM,
        mamba_dim=MODEL_DIM,
        lm_dim=LM_DIM,
    ):
        super().__init__()
        self.compressor = ConvCompressor(in_dim=audio_dim, out_dim=mamba_dim)
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

    def forward(self, audio_features, overlap_info):
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
        audio_dim=AUDIO_DIM,
        overlap_features=OVERLAP_FEATURES,
        overlap_dim=OVERLAP_DIM,
        lm_dim=LM_DIM,
        n_queries=32,
        n_heads=8,
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

    def forward(self, audio_features, overlap_info):
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


# Factory function: build any variant by name
def build_adapter(variant: str = "film-mamba", **kwargs) -> nn.Module:
    """Build an adapter variant by name. Use this for wandb sweeps.

    All variants have the same interface:
        forward(audio_features: (B,T,1024), overlap_info: (B,T,3)) → (B,N,lm_dim)
    """
    variants = {
        "concat-only": lambda **kw: ConcatOnlyAdapter(**kw),
        "sigmoid-gate": lambda **kw: SigmoidGateAdapter(**kw),
        "film": lambda **kw: ReliabilityAwareAdapter(context_type="none", **kw),
        "film-attn": lambda **kw: ReliabilityAwareAdapter(context_type="attn", n_layers=1, **kw),
        "film-attn-2L": lambda **kw: ReliabilityAwareAdapter(context_type="attn", n_layers=2, **kw),
        "film-mamba": lambda **kw: ReliabilityAwareAdapter(context_type="mamba", n_layers=1, **kw),
        "film-mamba-2L": lambda **kw: ReliabilityAwareAdapter(context_type="mamba", n_layers=2, **kw),
        "qformer": lambda **kw: QFormerAdapter(**kw),
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variants.keys())}")

    return variants[variant](**kwargs)
