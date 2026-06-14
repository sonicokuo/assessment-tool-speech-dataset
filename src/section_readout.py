"""section_readout.py — auxiliary regression readout that GROUNDS section attention.

WHAT THIS IS
------------
A small, training-only regression head bolted onto the SectionQueryHead's
attention output `z`. For each <sec_X> tag, `z` is the attention-weighted sum of
spectrogram-patch values; this head reads the section's acoustic scalar (SNR for
<sec_noise>, F0 for <sec_pitch>, ...) straight out of `z` and is supervised
against Praat ground truth with a masked Huber loss. It is discarded at
inference — it changes nothing about generation. Its sole job is to apply a
gradient that forces each section's attention map `alpha` to land on the part of
the waveform that actually carries that section's evidence.

WHY IT EXISTS (the bypass problem)
----------------------------------
With only the LM cross-entropy loss, "<sec_pitch> should attend to voiced
frames" is a LATENT variable — nothing supervises it directly. Worse, the LM can
read every feature it needs straight out of the audio *prefix* (the adapter
tokens prepended to the context) and simply ignore the injected cross-attention
summary `e_t`. When that happens the gradient pressure on `alpha` vanishes and
the attention map degrades into "wherever the randomly-initialised query happens
to point" — different per section (so the figures *look* discriminative) but not
grounded in real evidence.

THE FIX (and why `z`, and why detach V)
---------------------------------------
We regress the section scalar from `z`. Because `z = sum_p alpha_p * v_p` is the
ONLY downstream product of `alpha`, putting a loss on `z` is putting a loss on
`alpha`. And because this loss is computed entirely inside the head
(patches -> K,V -> alpha -> z -> g), it never touches the LM, so the prefix
bypass cannot satisfy it.

    loss -> g -> z --+--> alpha -> scores -> q, K     ("where to look")
                     |
                     +--> V                            ("what each patch encodes")

`z` has two ways to carry the feature: move `alpha` onto evidence-bearing
patches, OR rewrite the patch encodings `V` to smuggle the feature in. The
second is a shortcut that leaves `alpha` ungrounded — and it's plausible because
BEATs is a self-attention encoder, so each patch embedding is *contextualised*
and may already carry globally-diffused information. We close the shortcut by
detaching V in the regression branch:

    z_reg = einsum(alpha, V.detach())

After detach, the regression gradient CANNOT flow into V's encoding; the only
remaining descent direction is to move `alpha`. V is shaped solely by the main
task (CE -> e_t). This is the crux of the grounding argument.

THE EOS ANALOGY
---------------
This is the same shape of supervision that makes EOS "special": EOS is a plain
vocab token that CE pins to a fixed position (end of sequence) on every example.
Here we pin, on every example, what each <sec_X> tag must ATTEND TO. EOS is
supervised on *what token to emit*; this head is supervised on *where to look*.

SCOPE / EXPECTATIONS
--------------------
Localisation strength is set by the physics of each feature, not by this head:
overlap & pauses are strongly time-localised (sharp maps); pitch & tempo are
medium (voiced/speech regions); noise & reverb are global acoustic properties
(diffuse maps are the *faithful* answer for them). A diffuse SNR map is not a
failure — it says the evidence is everywhere, which is true. The sharp-vs-diffuse
contrast across sections is itself evidence that attention tracks the real
distribution of evidence rather than decoration. Final causal proof lives in
scripts/faithfulness_study.py; correlational alignment in
scripts/attention_gt_alignment.py.

INTEGRATION (see train.py wiring, grep "[section_readout]")
-----------------------------------------------------------
- STATIC  mode: SectionQueryHead.forward_all_sections returns alpha (B, S, P)
  for all sections every step. We recompute z per (clip, section) and supervise
  each section's features densely — no dependence on the tags appearing in the
  target text.
- DYNAMIC mode: the query fires at each <sec_X> (and each <r>) open position in
  the target. We recompute z per fired query and supervise the corresponding
  section's features; <r> opens (range markers, no scalar) are excluded.

Both modes flow through `section_readout_loss(section_ctx, gt_scalars, gt_mask)`,
the single entry point compute_loss calls. Ground-truth scalars + presence mask
are exactly the (B, 8) tensors src/dataset.py already loads from the features CSV
(feature_set.extract_scalars); per-feature magnitude normalisation reuses
feature_set.FEATURE_SCALES so F0 (~150 Hz) doesn't dominate overlap_ratio (~0.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from section_tags import SECTION_TAGS, N_SECTIONS
from feature_set import SUPERVISED_FEATURES, N_FEATURES, FEATURE_SCALES


# ── Section → feature routing ────────────────────────────────────────────────
def build_section_feature_mask() -> torch.Tensor:
    """(N_SECTIONS, N_FEATURES) bool routing matrix.

    R[s, f] is True iff scalar feature f is supervised under section s. Derived
    from each SectionTag.feature_names crossed with SUPERVISED_FEATURES, so it
    stays correct automatically if the catalog changes. The overlap_segments
    span set has no scalar entry in SUPERVISED_FEATURES and is skipped — only
    overlap_ratio is supervised under <sec_overlap>.

    With the EMNLP catalog this is (rows = sections, cols = the 8 scalars):

                  snr srmr f0m f0sd rate pcnt prate ovr
        noise      1   .   .    .    .    .    .    .
        reverb     .   1   .    .    .    .    .    .
        pitch      .   .   1    1    .    .    .    .
        tempo      .   .   .    .    1    .    .    .
        pauses     .   .   .    .    .    1    1    .
        overlap    .   .   .    .    .    .    .    1
    """
    feat_idx = {name: i for i, (name, _csv, _fmt) in enumerate(SUPERVISED_FEATURES)}
    R = torch.zeros(N_SECTIONS, N_FEATURES, dtype=torch.bool)
    for s_idx, sec in enumerate(SECTION_TAGS):
        for fname in sec.feature_names:
            j = feat_idx.get(fname)
            if j is not None:  # span sets (overlap_segments) have no scalar → skip
                R[s_idx, j] = True
    return R


# ── The readout head ─────────────────────────────────────────────────────────
class SectionReadoutHead(nn.Module):
    """Shared MLP g: z (d_v) → 8 scalars, routed per section by a boolean mask.

    ONE shared trunk, not one head per section, is deliberate: the six section
    z-vectors are produced by six different queries but pass through the *same*
    function, so g can only read SNR from <sec_noise>'s z and F0 from
    <sec_pitch>'s z if those z's differ — which forces the six attention maps to
    differentiate. Independent per-section heads would relax that constraint.

    Kept in float32 (do NOT .to(bfloat16) it): the cross-attention runs in bf16,
    but the regression targets span ~0.5 (overlap_ratio) to ~150 (f0_mean) and a
    bf16 readout would lose the low bits. `forward` up-casts the incoming bf16 z.
    """

    def __init__(
        self,
        d_v: int,
        hidden: int | None = None,
        n_features: int = N_FEATURES,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        hidden = hidden or d_v
        self.huber_delta = float(huber_delta)
        self.mlp = nn.Sequential(
            nn.Linear(d_v, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features),
        )
        # Near-zero init on the output layer so the readout starts as ~0 and
        # ramps up with training — mirrors the small-init philosophy used for
        # W_o / injection_gate in SectionQueryHead.
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        # Non-persistent buffers: derived constants, not learned. Excluded from
        # the state_dict so they never go stale against the catalog on resume.
        self.register_buffer("route", build_section_feature_mask(), persistent=False)   # (S, F) bool
        self.register_buffer(
            "scales", torch.tensor(FEATURE_SCALES, dtype=torch.float32), persistent=False  # (F,)
        )

    # The MLP itself — up-casts bf16 z to the head's (float32) param dtype.
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z.to(self.mlp[0].weight.dtype))

    # ── shared scoring core ──────────────────────────────────────────────────
    def _masked_huber(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pred/gt/mask are (M, F). Normalise error by per-feature scale, Huber,
        average over the supervised entries only.

        Returns:
            loss: scalar.
            mae:  (F,) per-feature mean |normalised error| over supervised
                  entries (0 where a feature had no supervision this batch).
                  For wandb — lets you watch "is pitch's z learning F0".
        """
        scales = self.scales.to(pred.device, pred.dtype)             # (F,)
        err = (pred - gt.to(pred.dtype)) / scales                    # (M, F) — unit-free
        maskf = mask.to(pred.dtype)
        per = F.huber_loss(
            err, torch.zeros_like(err), reduction="none", delta=self.huber_delta,
        ) * maskf                                                    # (M, F)
        denom = maskf.sum().clamp(min=1.0)
        loss = per.sum() / denom

        with torch.no_grad():
            cnt = maskf.sum(dim=0).clamp(min=1.0)                    # (F,)
            mae = (err.abs() * maskf).sum(dim=0) / cnt               # (F,)
        return loss, mae

    # ── STATIC mode ──────────────────────────────────────────────────────────
    def loss_static(
        self,
        alpha_all: torch.Tensor,   # (B, S, P) — all sections, every clip
        V: torch.Tensor,           # (B, P, d_v)
        gt_scalars: torch.Tensor,  # (B, F)
        gt_mask: torch.Tensor,     # (B, F) bool — True where CSV had a value
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Recompute z with V DETACHED so the gradient can only move alpha.
        z = torch.einsum("bsp,bpd->bsd", alpha_all, V.detach())     # (B, S, d_v)
        pred = self.forward(z)                                      # (B, S, F)
        B, S, Fdim = pred.shape

        route = self.route.to(pred.device)                         # (S, F)
        gtm = gt_mask.to(pred.device, torch.bool)                  # (B, F)
        # supervise (b, s, f) iff feature f belongs to section s AND clip b has it
        m = route.unsqueeze(0) & gtm.unsqueeze(1)                  # (B, S, F)
        gt = gt_scalars.to(pred.device).unsqueeze(1).expand(B, S, Fdim)

        return self._masked_huber(
            pred.reshape(-1, Fdim), gt.reshape(-1, Fdim), m.reshape(-1, Fdim),
        )

    # ── DYNAMIC mode ─────────────────────────────────────────────────────────
    def loss_dynamic(
        self,
        alpha: torch.Tensor,             # (Nq, P) — one row per fired query
        V: torch.Tensor,                 # (B, P, d_v)
        batch_idx: torch.Tensor,         # (Nq,) long — clip each query attends to
        query_section_idx: torch.Tensor, # (Nq,) long — section idx, or -1 to skip (<r>)
        gt_scalars: torch.Tensor,        # (B, F)
        gt_mask: torch.Tensor,           # (B, F) bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        keep = query_section_idx >= 0    # drop <r> / non-section queries
        if not torch.any(keep):
            return None, None
        alpha = alpha[keep]
        bidx = batch_idx[keep]
        sidx = query_section_idx[keep]

        V_per = V[bidx].detach()                                   # (n, P, d_v) — detached
        z = torch.einsum("np,npd->nd", alpha, V_per)              # (n, d_v)
        pred = self.forward(z)                                     # (n, F)

        route = self.route.to(pred.device)[sidx]                  # (n, F)
        gtm = gt_mask.to(pred.device, torch.bool)[bidx]           # (n, F)
        m = route & gtm
        gt = gt_scalars.to(pred.device)[bidx]                     # (n, F)
        return self._masked_huber(pred, gt, m)


# ── Single entry point used by compute_loss ──────────────────────────────────
def section_readout_loss(
    section_ctx: dict | None,
    gt_scalars: torch.Tensor | None,
    gt_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Dispatch on section_ctx['mode'] and return (loss_or_None, per_feature_metrics).

    Returns (None, {}) — a no-op — whenever anything required is missing: no
    readout head configured, no GT scalars (features CSV not provided), or the
    attention tensors weren't stashed (e.g. legacy .pt files with no BEATs
    patches, so nothing was injected). This keeps the call safe to make
    unconditionally and zero-overhead when lambda_readout is 0.

    section_ctx keys consumed (populated by train.py's injection helpers):
        readout_head:                the SectionReadoutHead.
        mode:                        "static" | "dynamic".
        V:                           (B, P, d_v) patch values.
        readout_alpha:               static (B, S, P) | dynamic (Nq, P).
        readout_batch_idx:           dynamic only (Nq,).
        readout_query_section_idx:   dynamic only (Nq,), -1 to skip.
    """
    if section_ctx is None or gt_scalars is None or gt_mask is None:
        return None, {}
    head: SectionReadoutHead | None = section_ctx.get("readout_head")
    alpha = section_ctx.get("readout_alpha")
    V = section_ctx.get("V")
    if head is None or alpha is None or V is None:
        return None, {}

    mode = section_ctx.get("mode")
    if mode == "static":
        loss, mae = head.loss_static(alpha, V, gt_scalars, gt_mask)
    elif mode == "dynamic":
        batch_idx = section_ctx.get("readout_batch_idx")
        qsi = section_ctx.get("readout_query_section_idx")
        if batch_idx is None or qsi is None:
            return None, {}
        loss, mae = head.loss_dynamic(alpha, V, batch_idx, qsi, gt_scalars, gt_mask)
    else:
        return None, {}

    if loss is None:
        return None, {}

    # Name the per-feature MAEs for wandb (readout_mae/snr, .../f0_mean, ...).
    metrics: dict[str, float] = {}
    for i, (name, _csv, _fmt) in enumerate(SUPERVISED_FEATURES):
        metrics[name] = float(mae[i].item())
    return loss, metrics


def query_section_indices(
    fired_token_ids: torch.Tensor,
    section_id_to_idx: dict[int, int],
) -> torch.Tensor:
    """Map each fired-open token id → its section idx, or -1 if it isn't a
    section open (e.g. the <r> range marker, which fires a query but has no
    scalar to supervise). Used by the dynamic-mode wiring in train.py.
    """
    out = torch.full_like(fired_token_ids, -1)
    for tok_id, s_idx in section_id_to_idx.items():
        out[fired_token_ids == tok_id] = s_idx
    return out
