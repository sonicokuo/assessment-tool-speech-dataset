"""Pretrained spectrogram encoder for AQUA-NL's section-query cross-attention.

Default backend: BEATs (Microsoft, 2022) — self-supervised on AudioSet, vendored
locally because the official release is not on HF Hub. See src/beats/__init__.py
for the vendor manifest and pretrained checkpoint download path.

Fallback backend: AST (MIT, 2021) — supervised on AudioSet, plug-and-play via
HF Hub. Useful if you want a quick A/B comparison or if BEATs's checkpoint isn't
available in the environment.

Why pretrained: trained from scratch on Libri2Mix the encoder underfits — audio
representations aren't learnable from ~13k clips with only the downstream LM-CE
gradient. Pretrained on AudioSet's 2M clips, the patches already encode the
acoustic events (speech, noise, reverb, music) we want to localise attention over.

The output is a flat sequence of patches (B, n_patches, d) which can be reshaped
to a 2D (T_p, F_p) grid via the `time_dim` and `freq_dim` fields on PatchGrid.
For BEATs on a 5 s clip: T_p ≈ 31, F_p = 8, n_patches = 248, d = 768.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PatchGrid:
    """Shape metadata for a 2D patch grid output.

    Attributes:
        n_patches:  total patches per clip = time_dim * freq_dim.
        d_patch:    embedding dimension per patch.
        time_dim:   number of time-axis bins (T_p).
        freq_dim:   number of frequency-axis bins (F_p).
        backend:    "beats" or "ast" — affects how patches are reshaped to 2D.
    """

    n_patches: int
    d_patch: int
    time_dim: int
    freq_dim: int
    backend: str

    def reshape_attention(self, alpha: torch.Tensor) -> torch.Tensor:
        """Reshape a (..., n_patches) attention vector to a 2D (..., T_p, F_p) grid.

        BEATs's flattening order is row-major over (T_p, F_p), so patches[t * F_p + f]
        corresponds to time-bin t, frequency-bin f. Reshape preserves this convention.
        """
        return alpha.reshape(*alpha.shape[:-1], self.time_dim, self.freq_dim)


class SpecEncoder(nn.Module):
    """Pretrained audio encoder wrapper.

    Supports two backends:
        - "beats" (default): vendored Microsoft BEATs, loaded from a local checkpoint
          path or downloaded from HuggingFace Hub (lpepino/beats_ckpts).
        - "ast": HuggingFace `AutoModel` for AST (MIT/ast-finetuned-audioset-10-10).

    Args:
        model_name:        "beats" | "ast" | a HF model id for AST.
        checkpoint_name:   for BEATs, which checkpoint file to load. Default is the
                           iter3+AS2M model. Other options on lpepino/beats_ckpts:
                           "BEATs_iter3.pt", "BEATs_iter3_plus_AS20K.pt".
        checkpoint_path:   if set, load BEATs from this local path instead of HF Hub.
        freeze:            if True (default), no gradient on encoder params.
        sample_rate:       expected input sample rate (Hz). Libri2Mix is 16 kHz.
    """

    def __init__(
        self,
        model_name: str = "beats",
        checkpoint_name: str = "BEATs_iter3_plus_AS2M.pt",
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.backend = "beats" if model_name.lower().startswith("beats") else "ast"

        if self.backend == "beats":
            self._init_beats(checkpoint_name, checkpoint_path)
        elif self.backend == "ast":
            self._init_ast(model_name)
        else:
            raise ValueError(f"Unknown spec encoder backend '{self.backend}'")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.freeze = freeze

    # ── Backend: BEATs ──────────────────────────────────────────
    def _init_beats(self, checkpoint_name: str, checkpoint_path: Optional[str]) -> None:
        # Vendored under src/beats/ — see src/beats/__init__.py.
        # Both BEATs and BEATsConfig are exported by the package.
        from beats import BEATs, BEATsConfig

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download("lpepino/beats_ckpts", checkpoint_name)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            # BEATs ckpts can include the SSL predictor head we don't use; that's fine.
            # Anything else is suspicious — surface it loudly.
            if any(not k.startswith("predictor") for k in unexpected):
                print(f"[BEATs] unexpected keys (other than predictor): "
                      f"{[k for k in unexpected if not k.startswith('predictor')][:5]}")
            if missing:
                print(f"[BEATs] missing keys: {missing[:5]}")

        self.cfg = cfg
        self.model = model
        self.d_patch = cfg.encoder_embed_dim   # 768

        # BEATs's patch grid for a fixed-length clip:
        #   ta_kaldi.fbank with frame_length=25ms, frame_shift=10ms:
        #     n_frames(T) = (n_samples - 400) // 160 + 1
        #   patch_embedding kernel=16, stride=16:
        #     T_p = n_frames // 16
        #     F_p = 128 // 16 = 8
        self._freq_dim = 128 // cfg.input_patch_size   # 8

    # ── Backend: AST ────────────────────────────────────────────
    def _init_ast(self, model_name: str) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        if model_name == "ast":
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        cfg = self.model.config
        self.d_patch = cfg.hidden_size
        f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        self._freq_dim = f_dim
        self._ast_time_dim = t_dim

    # ── Forward ──────────────────────────────────────────
    @property
    def d_out(self) -> int:
        return self.d_patch

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, PatchGrid]:
        """Encode a batch of waveforms to patch embeddings.

        Args:
            waveform: (B, n_samples) float, mono, at self.sample_rate.

        Returns:
            patches: (B, n_patches, d_patch) — flattened over (time, frequency).
            grid:    PatchGrid with time_dim, freq_dim so the caller can reshape
                     attention vectors back to a 2D heatmap.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Match encoder device.
        device = next(self.model.parameters()).device
        waveform = waveform.to(device).to(torch.float32)

        if self.backend == "beats":
            return self._forward_beats(waveform)
        return self._forward_ast(waveform)

    def _forward_beats(self, waveform: torch.Tensor) -> tuple[torch.Tensor, PatchGrid]:
        # Optional grad management — freeze flag governs requires_grad on params
        # but we still want torch.no_grad to skip building the graph when frozen.
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            features, _ = self.model.extract_features(waveform, padding_mask=None)
        n_patches = features.shape[1]
        # BEATs row-major reshape order: (T_p, F_p) → flatten last 2 dims to T_p*F_p.
        time_dim = n_patches // self._freq_dim
        if time_dim * self._freq_dim != n_patches:
            # Fbank time dim wasn't a multiple of 16; the patch_embedding drops the
            # remainder. The reshape stays valid for the common case (fixed clip length).
            raise RuntimeError(
                f"BEATs n_patches={n_patches} is not divisible by freq_dim={self._freq_dim}; "
                f"check that input waveform length yields fbank.T divisible by patch size."
            )
        grid = PatchGrid(
            n_patches=n_patches, d_patch=self.d_patch,
            time_dim=time_dim, freq_dim=self._freq_dim, backend="beats",
        )
        return features, grid

    def _forward_ast(self, waveform: torch.Tensor) -> tuple[torch.Tensor, PatchGrid]:
        wav_list = [w.detach().cpu().numpy() for w in waveform]
        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(**inputs)
        # AST: hidden = (B, 1 + 1 + n_patches, d) — [CLS, DIST, patches].
        patches = out.last_hidden_state[:, 2:, :]
        n_patches = patches.shape[1]
        grid = PatchGrid(
            n_patches=n_patches, d_patch=self.d_patch,
            time_dim=self._ast_time_dim, freq_dim=self._freq_dim, backend="ast",
        )
        return patches, grid

    def cached_forward(
        self,
        cached_patches: torch.Tensor,
        cached_grid_meta: dict | None = None,
    ) -> tuple[torch.Tensor, PatchGrid]:
        """Reconstruct (patches, PatchGrid) from preprocessed-and-cached patches.

        For the production path we run BEATs once per clip during preprocessing
        and save the patches into the per-clip .pt files. At training time we
        read them directly and skip the encoder forward, paying the BEATs cost
        only once per clip globally.
        """
        if cached_grid_meta is None:
            n_patches = cached_patches.shape[-2]
            d_patch = cached_patches.shape[-1]
            t = n_patches // self._freq_dim
            cached_grid_meta = {
                "n_patches": n_patches, "d_patch": d_patch,
                "time_dim": t, "freq_dim": self._freq_dim,
                "backend": self.backend,
            }
        grid = PatchGrid(**cached_grid_meta)
        return cached_patches, grid
