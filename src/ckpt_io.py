"""Slim-checkpoint save/load helpers (shared by train.py and inference.py).

PROBLEM these solve
-------------------
The old `_ckpt_payload` saved `llm.state_dict()` — the ENTIRE 8B model (frozen
base + LoRA adapters) — plus the full Adam optimizer state, into EVERY
checkpoint. A LoRA-r16 run dumped ~17 GB per `best.pt`/`last.pt` even though the
only thing that actually changed is the LoRA adapters (and, in tagged-mode, a
handful of new-token embed/lm_head rows). The frozen 8B base is identical to the
HF base weights that `from_pretrained` reloads anyway, so saving it is pure waste.

WHAT "slim" means here
----------------------
`slim_llm_state_dict(llm)` returns ONLY the parameters that are not part of the
frozen pretrained base:
  * every `lora_*` tensor (the adapters), and
  * every tensor whose parameter has `requires_grad == True`
    — this captures the tagged-mode embed_tokens/lm_head weights that train.py
      manually unfreezes (it does NOT use peft `modules_to_save`, so
      `get_peft_model_state_dict` alone would miss them).
The frozen base is dropped. On Qwen3-8B + LoRA-r16 this shrinks the LLM portion
from ~16 GB to a few tens of MB (see SLIM SIZE note at the bottom of this file).

How the load reconstructs the SAME model
-----------------------------------------
The base weights are restored by `from_pretrained` (+ `get_peft_model`) BEFORE
the state_dict is loaded, exactly as before. We then overlay the slim dict with
`load_state_dict(sd, strict=False)`: the LoRA adapters and unfrozen rows are
overwritten, the frozen base shows up as "missing keys" (expected — already
correct), and there are zero "unexpected keys". Verified to reproduce an
IDENTICAL forward pass in tests/test_checkpoint_slim.py.

Back-compat
-----------
Old "fat" checkpoints embed the full base. `is_slim_state_dict()` /
`load_llm_state_dict()` detect format by looking for any non-lora,
non-requires_grad base key in the saved dict (or the explicit
`ckpt_format == "peft_slim"` tag we now write). Fat dicts load via the original
`load_state_dict(sd)` (implicitly strict) path; slim dicts load via
`strict=False`. Both paths are exercised by the test suite.
"""
from __future__ import annotations

import os
from typing import Any

CKPT_FORMAT_SLIM = "peft_slim"


# ─── VAL-SUBSET STRATIFICATION ────────────────────────────────────────────────
# Default overlap-ratio bin edges for stratifying the per-epoch val SFS subset.
# low ≤ 0.15 < med ≤ 0.45 < high. Chosen around Libri2Mix's overlap distribution
# (median ~0.4); the exact edges matter less than guaranteeing each regime is
# represented so the subset's SFS isn't dominated by whichever overlap level
# happens to be over-sampled by a uniform draw.
OVERLAP_BIN_EDGES = (0.15, 0.45)


def overlap_bin(ratio: float, edges: tuple = OVERLAP_BIN_EDGES) -> str:
    """Map an overlap_ratio in [0,1] to a coarse bin label ('low'/'med'/'high')."""
    lo, hi = edges
    if ratio <= lo:
        return "low"
    if ratio <= hi:
        return "med"
    return "high"


def overlap_strata_from_csv_map(
    files: list[str],
    feature_csv_map: dict[str, dict],
    edges: tuple = OVERLAP_BIN_EDGES,
) -> list | None:
    """Per-item overlap-ratio stratum labels, or None if overlap is unavailable.

    `files` is the dataset's ordered `.pt` filename list (stems map to CSV
    filenames); `feature_csv_map` is `PreprocessedDataset.feature_csv_map`
    (filename → CSV row dict, has an `overlap_ratio` column). Reading the ratio
    from the CSV is O(1) per clip with no `.pt` load. Returns a list aligned to
    `files` (one label each) for `seeded_val_indices(strata=...)`, or None when
    no CSV / no overlap_ratio column so the caller falls back to seeded uniform.
    """
    if not feature_csv_map:
        return None
    # The dataset's `files` are "<stem>.pt", but the real val CSV keys the map by
    # the source audio name ("<stem>.wav" on Libri2Mix, "<stem>.flac" on clean
    # LibriSpeech); bare-stem keys also occur in older/toy maps. None of those
    # share an extension with ".pt", so matching on the raw/.pt-or-bare name alone
    # silently misses EVERY row (DEFECT 1: every clip → "unknown" → None → the
    # advertised stratification never engaged). Normalize both sides by stripping
    # the extension so ".pt", ".wav", ".flac", and bare-stem all collapse to the
    # same key. Build the stem-keyed index once (O(N)); look up O(1) per clip.
    by_stem: dict[str, dict] = {}
    for csv_key, csv_row in feature_csv_map.items():
        by_stem.setdefault(os.path.splitext(csv_key)[0], csv_row)
    strata: list = []
    saw_ratio = False
    for fn in files:
        row = by_stem.get(os.path.splitext(fn)[0])
        if row is not None and row.get("overlap_ratio") not in (None, ""):
            try:
                strata.append(overlap_bin(float(row["overlap_ratio"]), edges))
                saw_ratio = True
                continue
            except (TypeError, ValueError):
                pass
        strata.append("unknown")
    return strata if saw_ratio else None


def _slim_key_filter(state_keys, trainable_names: set) -> set:
    """The exact subset of `state_keys` that `slim_llm_state_dict` keeps.

    Single source of truth for "which keys are the trainable/adapter set": a key
    is kept iff it is a lora tensor OR its parameter is `requires_grad`. Used by
    `slim_llm_state_dict` (to build the saved dict) AND by `load_llm_state_dict`
    (to compute the set of trainable keys the model EXPECTS, so it can detect a
    slim dict that silently omits one — DEFECT 2).
    """
    return {k for k in state_keys if ("lora_" in k) or (k in trainable_names)}


def expected_slim_keys(llm) -> set:
    """The set of keys `slim_llm_state_dict(llm)` would produce for this model.

    These are the trainable/adapter keys a slim checkpoint MUST cover. Computed
    from the live model so it reflects the actual LoRA targets + unfrozen rows.
    """
    trainable_names = {n for n, p in llm.named_parameters() if p.requires_grad}
    return _slim_key_filter(llm.state_dict().keys(), trainable_names)


def slim_llm_state_dict(llm) -> dict:
    """Adapter-only LLM state dict: LoRA tensors + any trainable (requires_grad) rows.

    Drops the frozen pretrained base. For a full-FT model (nothing frozen) this
    returns the full state dict, which is correct — full-FT has no base to drop.
    """
    full = llm.state_dict()
    trainable_names = {n for n, p in llm.named_parameters() if p.requires_grad}
    keep = _slim_key_filter(full.keys(), trainable_names)
    return {k: full[k] for k in keep}


# Frozen transformer base projections. A FAT dict carries these (one per layer);
# a SLIM dict never does — it only carries the lora_A/lora_B decompositions of
# them (plus any genuinely trainable rows: embed/lm_head, LayerNorms, …). The
# classification rule is "slim iff NONE of these frozen-base keys are present",
# NOT "every key is lora/embed" — the latter wrongly excludes a legitimately
# trainable non-LoRA, non-embed param (e.g. an unfrozen LayerNorm) that
# `slim_llm_state_dict` keeps (DEFECT 3).
_FROZEN_BASE_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def _is_frozen_base_key(k: str) -> bool:
    """True for a standard transformer frozen-base projection weight.

    These are present ONLY in fat dicts. Matched via `base_layer.weight` (peft's
    name for the wrapped frozen proj) and the bare `*.q_proj.weight` style suffix
    (full state_dict of a non-peft / merged base).
    """
    if "lora_" in k:
        return False
    if k.endswith(".base_layer.weight") or k.endswith(".base_layer.bias"):
        return True
    return any(k.endswith(sfx) for sfx in _FROZEN_BASE_SUFFIXES)


def is_slim_state_dict(llm_sd: dict) -> bool:
    """True if `llm_sd` is a slim (adapter-only) dict, False if it is a fat dict.

    A slim dict carries the trainable subset only (lora tensors + any
    `requires_grad` rows such as unfrozen embed/lm_head or LayerNorms); a fat dict
    additionally carries the frozen pretrained base (every transformer block's
    q/k/v/o/gate/up/down projection weights).

    Classification rule: **slim iff the dict LACKS the frozen base weights** —
    i.e. it contains no `*.self_attn.q_proj.weight`-style (or peft
    `*.base_layer.weight`) key. This is the complement of `slim_llm_state_dict`'s
    keep-set: any param it legitimately keeps (LoRA, embed, lm_head, a trainable
    LayerNorm) is correctly classified slim, because none of those are frozen-base
    projection weights. The previous "every key must be lora/embed" rule
    misclassified a trainable LayerNorm as fat (DEFECT 3).
    """
    return not any(_is_frozen_base_key(k) for k in llm_sd)


class SlimLoadError(RuntimeError):
    """A slim checkpoint failed to fully apply the model's trainable adapter."""


def load_llm_state_dict(llm, llm_sd: dict, ckpt_format: str | None = None):
    """Load an LLM state dict, auto-handling both slim and fat formats.

    The frozen base must already be present on `llm` (from `from_pretrained` +
    `get_peft_model`) when loading a slim dict. Returns the
    (missing_keys, unexpected_keys) tuple from `load_state_dict` for the caller
    to log/assert.

    Slim path correctness guard (DEFECT 2)
    --------------------------------------
    `strict=False` makes the load tolerate the absent frozen base, but it ALSO
    silently tolerates a slim dict that is missing a *trainable* adapter key — a
    truncated write, a peft rename, or a filter regression would leave that LoRA
    tensor at fresh zero-init with no error, a silent no-op. After the load we
    therefore POSITIVELY assert:
      (a) every key the slim dict carried was consumed (none `unexpected`), and
      (b) the model's full expected trainable/LoRA set (what
          `slim_llm_state_dict` would emit for THIS model) is entirely present in
          the slim dict, and there is at least one LoRA key.
    `_missing` from `load_state_dict` legitimately lists the entire frozen base,
    so we do NOT raise on `_missing` — only on a missing *trainable* key, computed
    via `expected_slim_keys`.
    """
    slim = (ckpt_format == CKPT_FORMAT_SLIM) or is_slim_state_dict(llm_sd)
    if not slim:
        # Fat (legacy) checkpoint: full base is present → original strict load.
        return llm.load_state_dict(llm_sd)

    # strict=False: frozen base shows up as "missing" (already correct);
    # the lora + unfrozen rows in llm_sd overwrite their counterparts.
    missing, unexpected = llm.load_state_dict(llm_sd, strict=False)

    # (a) Nothing in the slim dict went unused. An unexpected key means the saved
    # adapter does NOT line up with this model (wrong rank/targets/rename); the
    # base-only `_missing` list would otherwise hide that the adapter never landed.
    if unexpected:
        raise SlimLoadError(
            "slim checkpoint has keys this model does not expect "
            f"(adapter mismatch): {list(unexpected)[:5]} ..."
        )

    # (b) Every trainable/adapter key the model expects is actually in the slim
    # dict. `_missing` is dominated by the frozen base (legitimate), so intersect
    # the model's expected-trainable set against what the dict supplied instead of
    # inspecting `_missing` directly.
    expected = expected_slim_keys(llm)
    supplied = set(llm_sd.keys())
    missing_trainable = expected - supplied
    if missing_trainable:
        raise SlimLoadError(
            "slim checkpoint is missing trainable adapter keys the model expects "
            f"(would load at zero-init, silent no-op): {sorted(missing_trainable)[:5]} ... "
            f"[{len(missing_trainable)} missing of {len(expected)} expected]"
        )
    if not any("lora_" in k for k in supplied):
        raise SlimLoadError(
            "slim checkpoint contains no LoRA keys — adapter would be entirely "
            "absent (truncated/empty slim dict)."
        )

    return missing, unexpected


# ─── EXPECTED SLIM SIZE for v17 (Qwen3-8B, LoRA r=16) ──────────────────────────
# LoRA targets = q,k,v,o,gate,up,down across 36 layers. Qwen3-8B dims:
#   hidden=4096, head_dim=128, n_q_heads=32 (q_proj out=4096), n_kv_heads=8
#   (k/v_proj out=1024), intermediate=12288.
# Per matrix LoRA params (r=16) = r*(in+out):
#   q_proj: 16*(4096+4096)=131072 ; o_proj: 16*(4096+4096)=131072
#   k_proj: 16*(4096+1024)= 81920 ; v_proj: 16*(4096+1024)= 81920
#   gate  : 16*(4096+12288)=262144; up    : 16*(4096+12288)=262144
#   down  : 16*(12288+4096)=262144
#   per-layer sum ≈ 1,212,416 params × 36 layers ≈ 43.6M LoRA params.
# In bf16 (2 bytes) ≈ 87 MB. Plus the adapter (~few M params) + decoupled_head
# (~few M). tagged_mode=false in v17 ⇒ no embed/lm_head rows saved.
# best.pt (no optimizer/scheduler) ≈ 0.1–0.2 GB. last.pt adds Adam moments for
# the trainable params only (LoRA+adapter+heads, NOT the 8B base) ≈ 2–3× the
# slim weights ⇒ still well under ~0.5 GB. Either way: 17 GB → ~0.2 GB. The
# headline "~2 GB" upper bound in the task spec is conservative; actual is
# smaller because the optimizer now only tracks trainable params.
