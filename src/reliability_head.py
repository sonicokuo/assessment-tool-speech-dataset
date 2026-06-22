"""Heteroscedastic per-feature reliability / abstention head.

Paper pivot ("Observability-Aware Speech-Feature Description"): the model should
report a number for a feature only where the signal can physically support one, and
abstain (band/hedge) on ill-posed features under overlap (single-speaker F0 from a
2-speaker mix). To get an abstention signal we need a *learned* per-feature
uncertainty, not a hand-set rule.

This head replaces the plain aux regression head (Linear(d -> 8) predicting a mean)
with a HETEROSCEDASTIC head: Linear(d -> 2*8) predicting, per feature, a mean AND a
log-variance (log σ²). The per-feature predicted σ is the model's own "this feature
is unreliable here" signal; at eval we abstain on a feature when its σ exceeds a
threshold and sweep the threshold to draw a risk-coverage curve.

Training objective is the heteroscedastic Gaussian negative log-likelihood (Kendall &
Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?",
arXiv:1703.04977, eq. 5):

    NLL_i = 0.5 * exp(-s_i) * (y_i - μ_i)^2 + 0.5 * s_i        (s_i = log σ_i²)

The model is free to inflate s_i (raise σ) to down-weight the squared-error term on
features it cannot predict, paying only the 0.5*s_i log-term — exactly the
"learned loss attenuation" that makes σ a calibrated reliability score. We predict
s = log σ² (not σ) for numerical stability: it is unconstrained, so no positivity
clamp is needed and exp(-s) never divides by ~0.

Deep evidential regression (Amini et al., arXiv:1910.02600) is a richer alternative
that also separates aleatoric from epistemic uncertainty, but it is heavier and less
battle-tested; we pick heteroscedastic-NLL (simpler, proven) and note evidential
regression as future work.

Everything here is additive and default-off: AdapterWithAuxHead only builds the
reliability head when reliability_head=True, and compute_loss only adds the NLL term
when lambda_nll > 0. With the defaults the plain MSE aux head path is byte-identical
to before.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Per-feature error normalization, shared with the plain-MSE aux head so the NLL's
# squared-error term ((y-μ)/scale)^2 is unit-free across features (F0 ~150 Hz must
# not dominate overlap_ratio ~0.5). Imported lazily inside functions to keep this
# module importable without the rest of the package on hand.
from feature_set import N_FEATURES


# Clamp range for the predicted log-variance s = log σ². Without a clamp a single bad
# batch can drive s to ±large and either explode exp(-s)*(error)² (s very negative,
# σ→0) or vanish the squared-error gradient entirely (s very positive, σ→∞). The
# bounds below correspond to σ ∈ [~0.006, ~150] in *normalized* (per-scale) units,
# which is far wider than any real per-feature spread, so the clamp only catches
# pathological excursions and is a no-op in normal training.
LOGVAR_MIN: float = -10.0
LOGVAR_MAX: float = 10.0


class ReliabilityHead(nn.Module):
    """Heteroscedastic per-feature head: pooled prefix -> (mean, log_var) per feature.

    A single Linear(d_in -> 2*n_features) projects the mean-pooled adapter prefix to
    2*n_features outputs; the first n_features are the per-feature means μ, the second
    n_features are the per-feature log-variances s = log σ². Splitting one matrix
    (rather than two heads) keeps the parameter count and the gradient path identical
    to "two parallel Linear(d, n_features)" while sharing the input projection's bias
    bookkeeping.

    forward(pooled) -> (mean (B, n_features), log_var (B, n_features)).
    Use `.sigma(log_var)` to get σ = exp(0.5*s) > 0 for the abstention threshold.
    """

    def __init__(self, d_in: int, n_features: int = N_FEATURES):
        super().__init__()
        self.n_features = n_features
        # 2*n_features: [μ_0..μ_{F-1}, s_0..s_{F-1}].
        self.proj = nn.Linear(d_in, 2 * n_features)
        # Init the log-var rows' bias to 0 (σ²=1 in normalized units) so the head
        # starts at the plain-MSE operating point and *learns* to raise σ on the
        # features it can't predict. Mean rows keep default init.
        with torch.no_grad():
            self.proj.bias[n_features:].zero_()

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(pooled)                       # (B, 2*n_features)
        mean = out[..., : self.n_features]            # (B, n_features)
        log_var = out[..., self.n_features :]         # (B, n_features)
        log_var = log_var.clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mean, log_var

    @staticmethod
    def sigma(log_var: torch.Tensor) -> torch.Tensor:
        """σ = exp(0.5 * log σ²) — strictly positive. The abstention score."""
        return torch.exp(0.5 * log_var)


def heteroscedastic_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked heteroscedastic Gaussian NLL (Kendall & Gal 2017, eq. 5).

    Per feature i:
        L_i = 0.5 * exp(-s_i) * ((y_i - μ_i)/scale_i)^2 + 0.5 * s_i
    with s_i = log σ_i². The squared-error term is normalized by `scale_i` (typical
    feature magnitude) so it is unit-free and comparable across features, mirroring
    the plain-MSE aux head. The result is the mean over the PRESENT (mask=True)
    feature slots, so a clip with no F0 measurement contributes nothing for that slot.

    Args:
        mean:     (B, F) predicted means μ.
        log_var:  (B, F) predicted log-variances s = log σ².
        target:   (B, F) ground-truth scalars y (0-filled where mask is False).
        mask:     (B, F) bool/float, True/1 where the scalar was actually measured.
                  None -> all present.
        scales:   (F,) per-feature normalization. None -> 1.0 (no normalization).

    Returns:
        Scalar tensor: mean NLL over present slots (0.0 if nothing is present).
        Finite and differentiable w.r.t. both mean and log_var.
    """
    if scales is None:
        scales_t = torch.ones(mean.shape[-1], device=mean.device, dtype=mean.dtype)
    else:
        scales_t = scales.to(device=mean.device, dtype=mean.dtype)

    err = (target.to(mean.dtype) - mean) / scales_t            # (B, F), unit-free
    # 0.5 * exp(-s) * err^2 + 0.5 * s
    per_feat = 0.5 * torch.exp(-log_var) * err.pow(2) + 0.5 * log_var   # (B, F)

    if mask is None:
        return per_feat.mean()

    mask_f = mask.to(per_feat.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (per_feat * mask_f).sum() / denom
