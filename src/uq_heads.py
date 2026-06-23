"""uq_heads.py — CHEAP uncertainty-quantification heads for the per-frame SNR map.

WHY THIS EXISTS (the kill-fast gate)
------------------------------------
The supervised dense local-SNR head (src/snr_map_head.py, Pearson 0.726) is a POINT
estimator: one SNR per frame, no "how sure am I". The paper wants a per-frame
UNCERTAINTY that ranks where the prediction is wrong, so the model can ABSTAIN on the
frames it cannot support. The INCUMBENT uncertainty channel is the heteroscedastic
sigma head (src/reliability_head.py + the NLL term in train.py): one Gaussian per
output, the predicted log-variance is the abstention score.

Before paying for a diffusion model (a heavy generative UQ), we ask: does a CHEAP
multimodal / deep-UQ head beat that incumbent sigma at predicting per-frame SNR-map
ERROR, measured by risk-coverage AURC? This module builds the three cheap challengers,
each a small nn.Module trainable in minutes off the SAME frozen WavLM features the SNR
map head consumes (no 8B LM, no encoder grad):

  MDNSNRMapHead       per-frame K-component Gaussian MIXTURE (Bishop 1994,
                      "Mixture Density Networks"). A multimodal predictive law: when
                      the SNR is ambiguous (overlap) the head can place mass on two
                      modes instead of averaging them. The UQ signal is the
                      LAW-OF-TOTAL-VARIANCE mixture variance E[Var] + Var[E].
  MCDropoutSNRMapHead the regression head with dropout LEFT ON at inference (Gal &
                      Ghahramani 2016, "Dropout as a Bayesian Approximation"). n
                      stochastic forward passes -> predictive mean + variance
                      (epistemic). Zero dropout -> zero variance (a hard test anchor).
  ensemble_uncertainty per-frame mean + variance ACROSS an ensemble of independently-
                      trained regression heads (Lakshminarayanan 2017, "Deep
                      Ensembles"). The disagreement of independently-initialised heads
                      is the epistemic uncertainty.

DECOUPLING / TESTABILITY
------------------------
Every head is pure torch (no transformers / peft / wandb), runs on CPU, and consumes
ONLY frozen `audio_features` (B, T, audio_dim). None of them touch the LM CE graph, so
each trains in minutes against the existing oracle SNR-map targets. The bake-off harness
(src/uq_bakeoff.py) then scores all four uncertainty channels (incumbent sigma + these
three) on the SAME held-out frames with risk-coverage AURC.

MATH NOTES (kept rigorous — this is a TRUSTWORTHY gate)
-------------------------------------------------------
  * MDN NLL is the masked NEGATIVE log of the mixture density, computed with logsumexp
    over the K log-component-densities so it is numerically stable (no exp of a large
    negative). log_sigma (not sigma) is the network output so positivity is free and
    1/sigma never divides by ~0; sigma is clamped to a wide [e^LOGSIG_MIN, e^LOGSIG_MAX].
  * MDN predictive mean = sum_k pi_k mu_k. Predictive variance is the LAW OF TOTAL
    VARIANCE: Var = E_k[sigma_k^2] + Var_k[mu_k]
             = sum_k pi_k (sigma_k^2 + mu_k^2) - (sum_k pi_k mu_k)^2.
    The second form is the numerically-cleaner single-pass identity and is what
    `predict` returns; a unit test pins it to the analytic LoTV on a hand example.
  * MC-dropout / ensemble variance use the POPULATION variance (divide by n, the MLE),
    matching the Gal and Lakshminarayanan predictive-variance definitions (NOT the
    n-1 sample variance), so a 1-sample case is a well-defined 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# WavLM-Large feature dim (the head input). Matches snr_map_head.AUDIO_DIM.
AUDIO_DIM: int = 1024

# Clamp range for predicted per-component log-sigma. Mirrors reliability_head's
# LOGVAR clamp philosophy: wide enough to be a no-op in normal training, tight enough
# that a pathological batch cannot drive sigma to 0 (1/sigma blow-up) or infinity
# (vanishing gradient). sigma in [~0.006, ~150] in dB units.
LOGSIG_MIN: float = -5.0
LOGSIG_MAX: float = 5.0

# logsumexp / log floor — guards log(0) when a mixture density underflows.
_LOG_EPS: float = 1e-12

# 0.5 * log(2 pi) — the Gaussian log-normalizer constant.
_HALF_LOG_2PI: float = 0.5 * math.log(2.0 * math.pi)


# ════════════════════════════════════════════════════════════════════════════════
# Mixture Density Network head (per-frame K-component Gaussian mixture over SNR)
# ════════════════════════════════════════════════════════════════════════════════
class MDNSNRMapHead(nn.Module):
    """Per-frame K-component mixture-density head over per-frame SNR (dB).

    A light temporal-conv trunk (same shape as SupervisedSNRMapHead's timeline branch)
    feeds a per-frame head that emits, for EACH of K mixture components, a mean mu_k, a
    log-sigma log_sigma_k, and a mixing logit; a per-frame softmax over the K logits
    gives the mixture weights pi_k. So `forward` returns three (B, T, K) tensors.

    The predictive law per frame is a Gaussian mixture
        p(y) = sum_k pi_k N(y; mu_k, sigma_k^2),
    whose mean is sum_k pi_k mu_k and whose variance is the LAW OF TOTAL VARIANCE
        Var = sum_k pi_k (sigma_k^2 + mu_k^2) - (sum_k pi_k mu_k)^2     (= E[Var]+Var[E]).
    `predict` returns (mean, total_variance); total_variance is the UQ signal the
    bake-off scores. `nll` is the masked negative log mixture-density, stable via
    logsumexp over the per-component log-densities.

    Args:
        audio_dim:   WavLM feature dim feeding the trunk (1024).
        n_components:K, number of mixture components (default 3).
        hidden:      trunk conv hidden width.
        kernel_size: temporal conv kernel (odd; symmetric padding keeps length T).
        snr_bias:    initial per-component mean bias (≈ dataset SNR mean in dB) so the
                     head starts near the prior instead of 0 dB.
    """

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        n_components: int = 3,
        hidden: int = 256,
        kernel_size: int = 5,
        snr_bias: float = 0.0,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.audio_dim = int(audio_dim)
        self.n_components = int(n_components)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)

        pad = self.kernel_size // 2
        self.in_proj = nn.Linear(self.audio_dim, self.hidden)
        self.temporal_conv = nn.Conv1d(
            self.hidden, self.hidden, kernel_size=self.kernel_size, padding=pad,
        )
        # Per-frame head -> 3*K outputs: [mu_1..mu_K, log_sigma_1..log_sigma_K,
        # logit_1..logit_K].
        self.head = nn.Linear(self.hidden, 3 * self.n_components)
        # Small-init means + prior bias (mirrors the SNR map readout small-init), and a
        # log_sigma bias of 0 (sigma=1 dB) so the mixture starts at a sane spread. The
        # mixing logits start at 0 -> uniform pi.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[: self.n_components].fill_(float(snr_bias))  # mu bias

    def forward(
        self, audio_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, audio_dim) WavLM frames -> per-frame mixture params.

        Returns:
            means      (B, T, K) component means mu_k (dB).
            log_sigmas (B, T, K) component log-sigmas (clamped to [LOGSIG_MIN, MAX]).
            logits     (B, T, K) raw mixing logits (softmax over K -> pi).
        Length-preserving (output T == input T) so it aligns frame-for-frame with the
        oracle timeline target and the WavLM mask.
        """
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be (B, T, audio_dim), got {tuple(audio_features.shape)}"
            )
        h = self.in_proj(audio_features)             # (B, T, hidden)
        h = h.transpose(1, 2)                        # (B, hidden, T)
        h = self.temporal_conv(h)                    # (B, hidden, T)
        h = F.gelu(h)
        h = h.transpose(1, 2)                        # (B, T, hidden)
        out = self.head(h)                           # (B, T, 3K)
        K = self.n_components
        means = out[..., :K]
        log_sigmas = out[..., K : 2 * K].clamp(LOGSIG_MIN, LOGSIG_MAX)
        logits = out[..., 2 * K :]
        return means, log_sigmas, logits

    # ── predictive mean + law-of-total-variance ──────────────────────────────
    @staticmethod
    def predict(
        params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixture params -> (mean, total_variance), both (B, T).

        mean  = sum_k pi_k mu_k
        total_variance = E_k[sigma_k^2] + Var_k[mu_k]                (law of total var)
                       = sum_k pi_k (sigma_k^2 + mu_k^2) - mean^2.
        The second (single-pass) identity is what we compute; it equals E[Var]+Var[E]
        exactly and is the per-frame UQ signal the bake-off ranks errors with.
        """
        means, log_sigmas, logits = params
        pi = torch.softmax(logits, dim=-1)                       # (B, T, K)
        var_k = torch.exp(2.0 * log_sigmas)                      # sigma_k^2  (B, T, K)
        mean = (pi * means).sum(dim=-1)                          # (B, T)
        # E[X^2 | k] = sigma_k^2 + mu_k^2; total E[X^2] = sum_k pi_k (sigma_k^2+mu_k^2).
        e_x2 = (pi * (var_k + means.pow(2))).sum(dim=-1)         # (B, T)
        total_var = e_x2 - mean.pow(2)                           # (B, T)
        # numerical floor: variance is >= 0 analytically; clamp tiny negatives from
        # float round-off.
        total_var = total_var.clamp(min=0.0)
        return mean, total_var

    # ── masked mixture NLL (stable via logsumexp) ─────────────────────────────
    def nll(
        self,
        params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,           # (B, T) oracle per-frame SNR (dB)
        mask: torch.Tensor | None = None,  # (B, T) bool/float, supervised frames
    ) -> torch.Tensor:
        """Masked negative log mixture-density, mean over supervised frames.

        Per frame the loss is
            -log sum_k pi_k N(y; mu_k, sigma_k^2)
          = -logsumexp_k [ log pi_k + log N(y; mu_k, sigma_k^2) ]
        with the Gaussian log-density
            log N = -log_sigma_k - 0.5*log(2 pi) - 0.5*((y - mu_k)/sigma_k)^2.
        Using log_pi = log_softmax(logits) and logsumexp keeps the whole thing in log
        space (no exp of a large-magnitude number), so it is finite even when one
        component is far from y. Reduced to the MEAN over mask=True frames (0.0 when the
        mask is empty), matching timeline_loss's masking convention.
        """
        means, log_sigmas, logits = params
        if means.shape[:2] != target.shape:
            raise ValueError(
                f"params {tuple(means.shape[:2])} vs target {tuple(target.shape)}"
            )
        y = target.to(means.dtype).unsqueeze(-1)                 # (B, T, 1)
        log_pi = torch.log_softmax(logits, dim=-1)               # (B, T, K)
        inv_sigma = torch.exp(-log_sigmas)                       # 1/sigma_k
        z = (y - means) * inv_sigma                              # (y-mu)/sigma  (B,T,K)
        log_norm = -log_sigmas - _HALF_LOG_2PI                   # -log sigma - 0.5 log2pi
        log_comp = log_pi + log_norm - 0.5 * z.pow(2)            # (B, T, K)
        log_prob = torch.logsumexp(log_comp, dim=-1)             # (B, T)
        per_frame = -log_prob                                    # (B, T)

        if mask is None:
            return per_frame.mean()
        mf = mask.to(per_frame.dtype)
        denom = mf.sum().clamp(min=1.0)
        return (per_frame * mf).sum() / denom


# ════════════════════════════════════════════════════════════════════════════════
# MC-dropout head (dropout ACTIVE at inference -> predictive mean + variance)
# ════════════════════════════════════════════════════════════════════════════════
class MCDropoutSNRMapHead(nn.Module):
    """Per-frame SNR regression head with dropout kept ON at inference.

    Architecturally identical to SupervisedSNRMapHead's timeline branch (in_proj ->
    temporal conv -> per-frame linear) but with a dropout layer after the trunk that is
    forced ACTIVE during `sample` regardless of self.training (Gal & Ghahramani 2016:
    test-time dropout is approximate Bayesian inference). `sample(audio, n)` runs n
    stochastic forward passes and returns the per-frame predictive (mean, variance)
    across them. With p=0 the dropout is the identity so all n passes coincide and the
    variance is exactly 0 (a hard test anchor).

    Args:
        audio_dim:   WavLM feature dim (1024).
        hidden:      trunk conv hidden width.
        kernel_size: temporal conv kernel (odd).
        p:           dropout probability used at BOTH train and sample time.
        snr_bias:    initial per-frame output bias (≈ dataset SNR mean in dB).
    """

    def __init__(
        self,
        audio_dim: int = AUDIO_DIM,
        hidden: int = 256,
        kernel_size: int = 5,
        p: float = 0.1,
        snr_bias: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout p must be in [0,1), got {p}")
        self.audio_dim = int(audio_dim)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.p = float(p)

        pad = self.kernel_size // 2
        self.in_proj = nn.Linear(self.audio_dim, self.hidden)
        self.temporal_conv = nn.Conv1d(
            self.hidden, self.hidden, kernel_size=self.kernel_size, padding=pad,
        )
        self.dropout = nn.Dropout(self.p)
        self.out_proj = nn.Linear(self.hidden, 1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.out_proj.bias.fill_(float(snr_bias))

    def _trunk(self, audio_features: torch.Tensor, drop: bool) -> torch.Tensor:
        """(B, T, audio_dim) -> (B, T) per-frame SNR, with dropout optionally forced on.

        `drop=True` applies dropout in functional form with training=True so it samples
        a mask even when the module is in eval() — the MC-dropout trick.
        """
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be (B, T, audio_dim), got {tuple(audio_features.shape)}"
            )
        h = self.in_proj(audio_features)             # (B, T, hidden)
        h = h.transpose(1, 2)                        # (B, hidden, T)
        h = self.temporal_conv(h)                    # (B, hidden, T)
        h = F.gelu(h)
        h = h.transpose(1, 2)                        # (B, T, hidden)
        if drop:
            h = F.dropout(h, p=self.p, training=True)
        else:
            h = self.dropout(h)                      # honors self.training
        return self.out_proj(h).squeeze(-1)          # (B, T)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Single forward pass (dropout follows self.training) -> (B, T)."""
        return self._trunk(audio_features, drop=False)

    def sample(
        self, audio_features: torch.Tensor, n: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """n MC-dropout forward passes -> per-frame predictive (mean, variance).

        Dropout is FORCED active for every pass (drop=True), so the n predictions differ.
        Returns the population mean and population variance (divide by n, the predictive
        definition) over the n passes, each (B, T). With p=0 the passes are identical and
        the variance is exactly 0.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        preds = torch.stack(
            [self._trunk(audio_features, drop=True) for _ in range(n)], dim=0
        )                                            # (n, B, T)
        mean = preds.mean(dim=0)                     # (B, T)
        var = preds.var(dim=0, unbiased=False)       # population variance (B, T)
        return mean, var


# ════════════════════════════════════════════════════════════════════════════════
# Deep-ensemble uncertainty (variance ACROSS independently-trained heads)
# ════════════════════════════════════════════════════════════════════════════════
def ensemble_uncertainty(
    preds_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame ensemble mean + variance across a list of regression-head predictions.

    Each element of `preds_list` is one ensemble member's per-frame SNR prediction; all
    members must share the same shape (e.g. (B, T) or (T,)). The deep-ensemble predictive
    uncertainty (Lakshminarayanan 2017) is the disagreement of the independently-trained
    members: the POPULATION variance over members (divide by M, the predictive
    definition), so a single-member ensemble has variance 0. Returns (mean, variance),
    each the shared member shape.
    """
    if len(preds_list) == 0:
        raise ValueError("preds_list must be non-empty")
    stacked = torch.stack([p.float() for p in preds_list], dim=0)   # (M, ...)
    mean = stacked.mean(dim=0)
    var = stacked.var(dim=0, unbiased=False)                        # population (M)
    return mean, var
