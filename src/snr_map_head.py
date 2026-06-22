"""snr_map_head.py — SUPERVISED dense local-SNR map head for AQUA-NL.

WHAT THIS IS
------------
A Concept-Bottleneck-style (Koh et al. 2020, arXiv:2007.04612) head that regresses
the DENSE per-frame local-SNR field of a clip and reads the clip-global SNR scalar
THROUGH that dense concept. The global SNR a quality model reports is the
energy-weighted aggregate of a dense per-time (and optionally per-time-frequency)
local-SNR field that the speech-SEPARATION field has directly supervised for a
decade as the Ideal Ratio Mask

    IRM(t,f) = ( SNR(t,f) / (SNR(t,f) + 1) ) ** 0.5

(Wang, Narayanan & Wang, TASLP 2014; survey arXiv:1708.07524). Because Libri2Mix
ships the clean s1/s2 stems, the dense target is computable EXACTLY (oracle), not
estimated — so this turns the 2D map from a post-hoc saliency guess into a
DIRECTLY-supervised contribution: a per-frame quantity tied to the generated SNR
number, the abstention spine retained alongside.

THE TWO REPRESENTATIONS
-----------------------
  TIMELINE (default, cleanest oracle): per-FRAME instantaneous SNR over the WavLM
  frame grid (50 Hz). WavLM frames are chosen for the timeline because (a) the
  adapter already consumes them, so `audio_features` is ALWAYS present (no BEATs /
  beats_cached dependency), (b) the per-frame instantaneous SNR
  10*log10(sum s1^2 / sum s2^2) is itself a 1-D TIME signal, so a frame-rate
  feature is the natural carrier, and (c) the WavLM 50 Hz grid matches the oracle
  target grid exactly (compute_snr_map_targets pools the stems on the same hop).

  IRM GRID (optional): per-T-F-bin Ideal Ratio Mask over the BEATs (T_p, F_bins)
  patch grid. The patch grid is the natural carrier of a TIME-FREQUENCY field, and
  the BEATs T_p x 8 layout is the same one decoupled_grounding / spec_encoder use.
  Off by default (predict_irm=False); only enabled when IRM targets are supplied.

THE CBM TIE (predict the scalar THROUGH the dense concept)
----------------------------------------------------------
The clip SNR scalar is NOT read by a free parallel head — it is the
ENERGY-WEIGHTED POOL of the predicted per-frame SNR over the active frames, the
same aggregation the oracle scalar uses. So a loss on the scalar is a loss on the
pooled dense field, and a loss on the dense field moves the scalar: the dense map
is the bottleneck the scalar must pass through (the CBM property). The pooled
scalar can optionally be tied into the catalog `snr` feature for a joint signal,
but the dense Huber is the primary supervision.

GROUNDING / FAITHFULNESS
------------------------
Unlike the decoupled_grounding head (which detaches V so the gradient lands on
attention), here the supervision is DIRECT and DENSE: every frame has an oracle
target, so the head is forced onto real per-frame evidence with no detach trick
needed — the target itself is the localization. snr_map_validate.py then checks
the faithfulness the design demands: timeline-vs-oracle correlation/MAE, a
deletion test (zeroing predicted-high-SNR frames must DROP the pooled SNR), and a
model-randomization sanity check.

SCOPE
-----
Training-only like the other grounding heads. Pure torch (no transformers / LM
deps) so it is unit-testable on CPU. Default-OFF: train.py only constructs it when
lambda_snr_map > 0, and the loss term is a no-op (returns None) whenever the head
is absent, the weight is 0, or the batch carries no dense target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# WavLM frame rate (Hz) — the timeline grid. Matches preprocess.py (hop 320 @ 16 kHz
# → 50 Hz) and decoupled_grounding.WAVLM_FRAME_RATE_HZ.
WAVLM_FRAME_RATE_HZ: float = 50.0

# WavLM-Large feature dim (the timeline head's input).
AUDIO_DIM: int = 1024

# BEATs patch feature dim + frequency-patch count (the IRM head's input grid). Same
# convention as spec_encoder.PatchGrid (128 mel / 16 patch = 8) and
# decoupled_grounding.F_P_DEFAULT.
DEFAULT_D_PATCH: int = 768
F_P_DEFAULT: int = 8


class SupervisedSNRMapHead(nn.Module):
    """Regress the dense local-SNR field (timeline + optional IRM) from audio features.

    TIMELINE branch (always present): a light depthwise-then-pointwise temporal conv
    over the WavLM frames followed by a per-frame linear → one SNR value per frame
    (B, T) in dB. The conv gives a small temporal receptive field (instantaneous SNR
    is locally smooth) without a heavy sequence model. The clip SNR scalar is the
    ENERGY-WEIGHTED pool of this timeline over the supervised frames (the CBM tie).

    IRM branch (predict_irm=True): a per-patch linear over BEATs (T_p, F_bins) patches
    → a sigmoid mask in [0,1] (B, T_p, F_bins), supervised by BCE/MSE against the
    oracle IRM. Off by default.

    Args:
        audio_dim:   WavLM feature dim feeding the timeline branch (1024).
        d_patch:     BEATs patch dim feeding the IRM branch (768). Only used when
                     predict_irm=True.
        f_bins:      frequency-patch count of the IRM grid (8). predict_irm only.
        hidden:      timeline conv hidden width.
        kernel_size: temporal conv kernel (odd; symmetric padding keeps length T).
        predict_irm: build the optional IRM branch.
        huber_delta: Huber transition on the dB timeline error.
        snr_bias:    initial per-frame output bias (≈ the dataset SNR mean in dB) so
                     the head starts near the prior instead of 0 dB. Default 0.
    """

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        d_patch: int = DEFAULT_D_PATCH,
        f_bins: int = F_P_DEFAULT,
        hidden: int = 256,
        kernel_size: int = 5,
        predict_irm: bool = False,
        huber_delta: float = 1.0,
        snr_bias: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.audio_dim = int(audio_dim)
        self.d_patch = int(d_patch)
        self.f_bins = int(f_bins)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.predict_irm = bool(predict_irm)
        self.huber_delta = float(huber_delta)

        pad = self.kernel_size // 2
        # Timeline branch: project to hidden, temporal conv (length-preserving), GELU,
        # per-frame linear → 1 SNR value per frame.
        self.in_proj = nn.Linear(self.audio_dim, self.hidden)
        self.temporal_conv = nn.Conv1d(
            self.hidden, self.hidden, kernel_size=self.kernel_size, padding=pad,
        )
        self.out_proj = nn.Linear(self.hidden, 1)
        # Near-zero output weights + prior-mean bias so the head starts ~flat at the
        # SNR prior (small-init philosophy, mirrors decoupled_grounding readout).
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.out_proj.bias.fill_(float(snr_bias))

        # IRM branch (optional): per-patch linear → sigmoid mask in [0,1].
        if self.predict_irm:
            self.irm_proj = nn.Linear(self.d_patch, 1)
            nn.init.normal_(self.irm_proj.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                # bias 0 → sigmoid(0)=0.5, a neutral IRM start.
                self.irm_proj.bias.zero_()

    # ── timeline forward ──────────────────────────────────────────────────────
    def forward_timeline(self, audio_features: torch.Tensor) -> torch.Tensor:
        """(B, T, audio_dim) WavLM frames → (B, T) per-frame SNR in dB.

        Length-preserving (output T == input T) so the prediction aligns frame-for-
        frame with the oracle timeline target and the WavLM `audio_lens` mask.
        """
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be (B, T, audio_dim), got {tuple(audio_features.shape)}"
            )
        h = self.in_proj(audio_features)                 # (B, T, hidden)
        h = h.transpose(1, 2)                            # (B, hidden, T)
        h = self.temporal_conv(h)                        # (B, hidden, T)
        h = F.gelu(h)
        h = h.transpose(1, 2)                            # (B, T, hidden)
        snr_frame = self.out_proj(h).squeeze(-1)         # (B, T)
        return snr_frame

    # ── IRM forward (optional) ────────────────────────────────────────────────
    def forward_irm(
        self,
        patches: torch.Tensor,
        t_p: int | None = None,
    ) -> torch.Tensor:
        """BEATs patches → per-T-F IRM mask in [0,1].

        Args:
            patches: (B, P, d_patch) flat patches OR (B, T_p, F_bins, d_patch). Flat P
                     factors as T_p * f_bins TIME-MAJOR (index = t*f_bins + f), the same
                     convention as spec_encoder.PatchGrid / decoupled_grounding.
            t_p:     time-patch count for the flat case; inferred as P // f_bins if None.
        Returns:
            (B, T_p, f_bins) sigmoid mask in [0,1].
        """
        if not self.predict_irm:
            raise RuntimeError("forward_irm called but predict_irm=False")
        if patches.dim() == 4:
            B, T_p, Fb, _ = patches.shape
            flat = patches.reshape(B, T_p * Fb, self.d_patch)
        elif patches.dim() == 3:
            B, P, _ = patches.shape
            Fb = self.f_bins
            T_p = (P // Fb) if t_p is None else int(t_p)
            flat = patches[:, : T_p * Fb, :]
        else:
            raise ValueError(
                f"patches must be (B,P,d) or (B,T_p,F,d), got {tuple(patches.shape)}"
            )
        logit = self.irm_proj(flat).squeeze(-1)          # (B, T_p*Fb)
        mask = torch.sigmoid(logit).reshape(B, T_p, Fb)  # (B, T_p, Fb)
        return mask

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Convenience: the timeline branch (the primary product)."""
        return self.forward_timeline(audio_features)

    # ── CBM scalar pool (predict the scalar THROUGH the dense concept) ─────────
    @staticmethod
    def pooled_snr_db(
        snr_frame: torch.Tensor,           # (B, T) per-frame SNR in dB
        active_mask: torch.Tensor,         # (B, T) bool/float — frames to pool over
        energy_weight: torch.Tensor | None = None,  # (B, T) optional per-frame weight
    ) -> torch.Tensor:
        """Aggregate the per-frame SNR timeline into ONE clip SNR scalar (B,).

        The clip SNR a quality model reports is the energy-weighted aggregate of the
        local-SNR field over the ACTIVE frames — the SAME aggregation the oracle scalar
        uses (clean_features.clean_snr_db pools s1/s2 energy over s1-active frames). So
        this pool is the CBM bottleneck: a loss on the pooled scalar is a loss on the
        dense timeline, and a loss on the dense timeline moves the scalar.

        Pooling is a masked weighted MEAN in dB. (Averaging dB values is a geometric
        mean in the linear domain; it is the standard, numerically-stable readout for a
        per-frame-dB field and is monotone in every frame, so the deletion test —
        removing high-SNR frames must LOWER the pooled value — holds by construction.)

        Args:
            snr_frame:     (B, T) per-frame SNR (dB).
            active_mask:   (B, T) True/1 on frames to include (e.g. s1-active frames).
            energy_weight: (B, T) optional non-negative per-frame weight. None → uniform.
        Returns:
            (B,) pooled clip SNR in dB. A row with no active frames pools to 0.
        """
        m = active_mask.to(snr_frame.dtype)
        if energy_weight is not None:
            w = energy_weight.to(snr_frame.dtype).clamp(min=0.0) * m
        else:
            w = m
        denom = w.sum(dim=-1).clamp(min=1e-8)            # (B,)
        return (snr_frame * w).sum(dim=-1) / denom        # (B,)

    # ── dense timeline loss (masked Huber to s1-active frames) ────────────────
    def timeline_loss(
        self,
        snr_frame: torch.Tensor,           # (B, T) prediction
        snr_target: torch.Tensor,          # (B, T) oracle per-frame SNR (dB)
        target_mask: torch.Tensor,         # (B, T) bool/float — supervised (active) frames
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Masked Huber on the per-frame SNR, restricted to s1-active frames.

        The mask is the s1-active set: the oracle instantaneous SNR is only defined
        where the TARGET speaker is present (elsewhere it is 0/0), so we supervise the
        timeline ONLY on active frames — the same restriction clean_snr_db uses. Frames
        outside the mask (silence / padding) contribute nothing.

        Returns:
            (loss, metrics). loss is the mean Huber over supervised frames (0.0 when the
            mask is empty). metrics has 'snr_map_mae' (mean |error| dB over supervised
            frames) and 'snr_map_n_frames' (supervised-frame count).
        """
        if snr_frame.shape != snr_target.shape:
            raise ValueError(
                f"pred {tuple(snr_frame.shape)} vs target {tuple(snr_target.shape)}"
            )
        tgt = snr_target.to(snr_frame.dtype)
        mf = target_mask.to(snr_frame.dtype)
        err = snr_frame - tgt
        per = F.huber_loss(
            err, torch.zeros_like(err), reduction="none", delta=self.huber_delta,
        ) * mf
        denom = mf.sum().clamp(min=1.0)
        loss = per.sum() / denom
        with torch.no_grad():
            mae = float(((err.abs() * mf).sum() / denom).item())
            n = float(mf.sum().item())
        return loss, {"snr_map_mae": mae, "snr_map_n_frames": n}

    # ── IRM loss (optional MSE/BCE) ───────────────────────────────────────────
    def irm_loss(
        self,
        irm_pred: torch.Tensor,            # (B, T_p, Fb) in [0,1]
        irm_target: torch.Tensor,          # (B, T_p, Fb) oracle IRM in [0,1]
        target_mask: torch.Tensor | None = None,  # (B, T_p) or (B,T_p,Fb) valid bins
        mode: str = "mse",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Masked MSE (default) or BCE between the predicted and oracle IRM grids.

        The IRM is a [0,1] soft mask, so MSE is a clean regression; BCE treats it as a
        per-bin Bernoulli (the separation-literature default). Masked to valid time
        patches when a mask is given (padding excluded).

        Returns:
            (loss, metrics) with 'snr_irm_mae' over supervised bins.
        """
        if irm_pred.shape != irm_target.shape:
            raise ValueError(
                f"irm pred {tuple(irm_pred.shape)} vs target {tuple(irm_target.shape)}"
            )
        tgt = irm_target.to(irm_pred.dtype).clamp(0.0, 1.0)
        if target_mask is None:
            mf = torch.ones_like(irm_pred)
        elif target_mask.dim() == 2:
            mf = target_mask.to(irm_pred.dtype).unsqueeze(-1).expand_as(irm_pred)
        else:
            mf = target_mask.to(irm_pred.dtype)
        if mode == "mse":
            per = (irm_pred - tgt) ** 2 * mf
        elif mode == "bce":
            per = F.binary_cross_entropy(
                irm_pred.clamp(1e-6, 1.0 - 1e-6), tgt, reduction="none",
            ) * mf
        else:
            raise ValueError(f"mode must be 'mse'|'bce', got {mode!r}")
        denom = mf.sum().clamp(min=1.0)
        loss = per.sum() / denom
        with torch.no_grad():
            mae = float(((irm_pred - tgt).abs() * mf).sum() / denom)
        return loss, {"snr_irm_mae": mae}


def snr_map_loss_term(
    head: "SupervisedSNRMapHead | None",
    batch: dict,
    lambda_snr_map: float,
    device: torch.device | str = "cpu",
    lambda_scalar: float = 0.0,
    lambda_irm: float = 0.0,
    irm_mode: str = "mse",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """train.py integration: dense local-SNR-map supervision off the WavLM frames.

    Lives HERE (not train.py) so it imports without transformers / peft / wandb and is
    unit-testable on CPU. Runs the head on the batch's `audio_features` (the SAME field
    the adapter consumes), regresses the per-frame SNR against the oracle stem-derived
    timeline target, and returns lambda_snr_map · masked-Huber as a fresh loss term.
    The head has its OWN parameters (in_proj / temporal_conv / out_proj), so its
    gradient lands on the head — it never touches the LM CE graph. Fully decoupled,
    added to the total loss in compute_loss.

    CBM scalar tie (lambda_scalar > 0): the per-frame timeline is energy-pooled to ONE
    clip SNR (pooled_snr_db over the supervised frames) and regressed against the
    catalog SNR scalar (gt_scalars[:, snr_idx]); this predicts the scalar THROUGH the
    dense concept. 0.0 (default) → no scalar term (dense Huber only).

    IRM (lambda_irm > 0 and head.predict_irm): per-T-F IRM MSE/BCE off the BEATs
    patches vs the oracle IRM target. 0.0 / no IRM target → skipped.

    Pulls from the batch:
        audio_features:   (B, T, audio_dim) WavLM frames (always present).
        snr_map_target:   (B, T) oracle per-frame SNR timeline (dB). REQUIRED — no-op
                          if absent (legacy .pt / no target lookup).
        snr_map_mask:     (B, T) bool, True on s1-active (supervised) frames. Defaults
                          to "all non-padded frames" via audio_lens when absent.
        audio_lens:       (B,) WavLM frame counts (for the default active mask).
        snr_scalar_idx:   int index of `snr` in gt_scalars (for the CBM tie). Defaults
                          to feature_set lookup when not supplied.
        gt_scalars/gt_mask: catalog scalars (for the CBM scalar tie).
        beats_patches/snr_irm_target/snr_irm_mask: optional IRM branch inputs.

    Returns:
        (weighted_loss_or_None, metrics). No-op (None, {}) when the head is absent,
        lambda <= 0, or the batch carries no snr_map_target — safe to call
        unconditionally, zero-overhead when off. metrics has 'loss_snr_map' (UNWEIGHTED
        dense Huber), 'snr_map_mae', and — when on — 'loss_snr_scalar' / 'snr_pooled_mae'
        and 'loss_snr_irm' / 'snr_irm_mae'.
    """
    if head is None or lambda_snr_map <= 0.0:
        return None, {}
    target = batch.get("snr_map_target")
    audio_features = batch.get("audio_features")
    if target is None or audio_features is None:
        return None, {}

    head_dtype = head.in_proj.weight.dtype
    audio_features = audio_features.to(device).to(head_dtype)
    target = target.to(device).to(head_dtype)
    B, T = target.shape

    # Active/supervised mask: explicit snr_map_mask wins; else "non-padded frames"
    # from audio_lens; else all frames.
    mask = batch.get("snr_map_mask")
    if mask is not None:
        mask = mask.to(device).to(torch.bool)
    else:
        audio_lens = batch.get("audio_lens")
        if audio_lens is not None:
            idx = torch.arange(T, device=device).unsqueeze(0)        # (1, T)
            mask = idx < audio_lens.to(device).unsqueeze(1)          # (B, T) bool
        else:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # Align prediction length to the target length (the head is length-preserving on
    # its input T; the target T may differ if the .pt was trimmed — slice/pad to match).
    snr_frame_full = head.forward_timeline(audio_features)            # (B, T_audio)
    Ta = snr_frame_full.shape[1]
    if Ta == T:
        snr_frame = snr_frame_full
    elif Ta > T:
        snr_frame = snr_frame_full[:, :T]
    else:
        snr_frame = F.pad(snr_frame_full, (0, T - Ta))

    loss, metrics = head.timeline_loss(snr_frame, target, mask)
    weighted = lambda_snr_map * loss
    metrics = {"loss_snr_map": float(loss.detach().item()), **metrics}

    # ── CBM scalar tie (predict the scalar THROUGH the pooled dense field) ──
    if lambda_scalar > 0.0:
        gt_scalars = batch.get("gt_scalars")
        gt_mask = batch.get("gt_mask")
        if gt_scalars is not None:
            snr_idx = batch.get("snr_scalar_idx")
            if snr_idx is None:
                from feature_set import FEATURE_NAMES
                snr_idx = FEATURE_NAMES.index("snr") if "snr" in FEATURE_NAMES else 0
            snr_idx = int(snr_idx)
            gt_scalars = gt_scalars.to(device).to(head_dtype)
            pooled = head.pooled_snr_db(snr_frame, mask)             # (B,)
            gt_snr = gt_scalars[:, snr_idx]                          # (B,)
            if gt_mask is not None:
                sm = gt_mask.to(device)[:, snr_idx].to(head_dtype)   # (B,)
            else:
                sm = torch.ones(B, device=device, dtype=head_dtype)
            err = (pooled - gt_snr)
            per = F.huber_loss(
                err, torch.zeros_like(err), reduction="none", delta=head.huber_delta,
            ) * sm
            denom = sm.sum().clamp(min=1.0)
            scalar_loss = per.sum() / denom
            weighted = weighted + lambda_scalar * scalar_loss
            metrics["loss_snr_scalar"] = float(scalar_loss.detach().item())
            with torch.no_grad():
                metrics["snr_pooled_mae"] = float(((err.abs() * sm).sum() / denom).item())

    # ── IRM branch (optional) ──
    if lambda_irm > 0.0 and head.predict_irm:
        irm_target = batch.get("snr_irm_target")
        patches = batch.get("beats_patches")
        if irm_target is not None and patches is not None:
            patches = patches.to(device).to(head_dtype)
            irm_target = irm_target.to(device).to(head_dtype)
            B2, T_p, Fb = irm_target.shape
            irm_pred = head.forward_irm(patches, t_p=T_p)            # (B, T_p, Fb)
            irm_mask = batch.get("snr_irm_mask")
            if irm_mask is not None:
                irm_mask = irm_mask.to(device)
            irm_l, irm_m = head.irm_loss(irm_pred, irm_target, target_mask=irm_mask, mode=irm_mode)
            weighted = weighted + lambda_irm * irm_l
            metrics["loss_snr_irm"] = float(irm_l.detach().item())
            metrics.update(irm_m)

    return weighted, metrics
