"""
Training script for Overlap-Aware Speech Quality Description.
Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --adapter_variant concat-only --epochs 3
    python src/train.py --config configs/config.yaml --resume_from ./checkpoints/last.pt
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from adapter import build_adapter
from sfs import HybridClaimParser, SFSScorer
from dataset import PreprocessedDataset, collate_fn
from text_metrics import compute_generation_metrics
from feature_set import FEATURE_SCALES, N_FEATURES
from ntl import digit_token_ids, number_token_loss
from section_tags import (
    SPECIAL_TOKENS as TAG_SPECIAL_TOKENS,
    SECTION_TAGS,
    N_SECTIONS,
    RANGE_OPEN_TAG,
    section_open_token_ids,
)
from section_query import SectionQueryHead
# [section_readout] training-only regression head that grounds section attention.
from section_readout import (
    SectionReadoutHead,
    section_readout_loss,
    query_section_indices,
    warmup_lambda,
)
# [decoupled_grounding] token-free 2D grounding head. Parallel branch off the BEATs
# patches — produces per-feature attention maps WITHOUT the LM emitting any special
# token, so the LM generates clean untagged prose (0% degeneration). See
# src/decoupled_grounding.py. Off by default (decoupled_grounding: false).
from decoupled_grounding import (
    DecoupledGroundingHead,
    decoupled_grounding_loss_term,
    feature_names,
    overlap_ratio_index,
)
# [snr_map] supervised DENSE local-SNR-map head (CBM-style; Koh 2007.04612). Regresses
# the per-frame SNR timeline off the WavLM frames against the stem-derived oracle
# target (the directly-supervised grounding the project lacked). Off by default
# (lambda_snr_map: 0 → head not built, loss term no-ops).
from snr_map_head import SupervisedSNRMapHead, snr_map_loss_term
# [srmr_map] supervised TRUE-2D SRMR (acoustic x modulation) modulation-energy map head.
# Regresses the 23x8 oracle log-energy tensor from WavLM frames (lambda_srmr_map: 0 →
# head not built, loss term no-ops). The frequency axis here is non-vacuous (unlike SNR).
from snr_map_head import SupervisedSRMRMapHead, srmr_map_loss_term
# Degeneration-aware, lower-variance best-checkpoint selection.
from ckpt_selection import (
    seeded_val_indices,
    should_save_best,
    passes_degeneration_guard,
    degeneration_stats,
)
# Band-free, lower-variance checkpoint selection (research Q1 protocol):
# continuous SRCC/nMAE composite with a hard BLEU fluency floor + EMA smoothing,
# replacing the saturated small-val SFS-F1 argmax. See src/selection_metric.py.
from selection_metric import (
    band_free_val_scores,
    composite_score,
    ema,
    SELECTION_FEATURES,
)
from feature_set import RECOVERABLE_FEATURES
from ckpt_io import (
    CKPT_FORMAT_SLIM,
    slim_llm_state_dict,
    load_llm_state_dict,
    overlap_strata_from_csv_map,
)
from spec_encoder import SpecEncoder


# ── Tokenizer + LM setup helpers ──────────────────────────────────────────────
def _use_lora(config: dict) -> bool:
    """Whether to wrap the LM with PEFT LoRA. False = full fine-tuning."""
    r = config.get("lora_rank")
    return bool(r)  # 0 / None / False → full FT


def _register_feature_tags(tokenizer: PreTrainedTokenizerBase, llm: nn.Module) -> int:
    """Add per-feature `<f_*>` and `</f>` tokens, resize embeddings, mean-init new rows.

    Uses `add_tokens(..., special_tokens=False)` so the tags get the 1-token
    tokenization guarantee but are NOT stripped by `tokenizer.decode(
    skip_special_tokens=True)` — which is the default everywhere in train.py
    and inference.py. If we registered them as `additional_special_tokens`
    instead, the decoded text would lose every `<f_*>` and `</f>` and SFS
    would parse an empty string.

    Returns the number of tokens added (0 if all tags were already in vocab —
    e.g., on resume).
    """
    added = tokenizer.add_tokens(TAG_SPECIAL_TOKENS, special_tokens=False)
    if added == 0:
        return 0

    old_vocab_size = llm.get_input_embeddings().weight.shape[0]
    llm.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        # Mean-init the new embedding rows. Tied embeddings (input == output)
        # are handled by resize_token_embeddings; for untied LMs we also touch
        # lm_head. Both branches are no-ops if the new rows fall outside the
        # corresponding matrix's range.
        in_emb = llm.get_input_embeddings().weight
        mean_in = in_emb[:old_vocab_size].mean(dim=0)
        in_emb[old_vocab_size:].copy_(mean_in.to(in_emb.dtype))

        out_emb = llm.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] > old_vocab_size:
            mean_out = out_emb.weight[:old_vocab_size].mean(dim=0)
            out_emb.weight[old_vocab_size:].copy_(mean_out.to(out_emb.weight.dtype))

        # [R10] Semantic warm-start: overwrite the just-mean-filled OPEN <sec_*>/
        # <f_*> rows with the mean of their descriptive-word subwords, so the
        # tokens start differentiated (mean-init makes all 19 identical, which
        # the small/section path struggles to break apart -> degeneration).
        # Closing/range markers (</sec>, </f>, <r>, </r>) have no phrase and keep
        # the mean-init above. Safe: falls back to mean-init on any error.
        try:
            if not config.get("semantic_tag_init", True):
                print("[tagged-mode] semantic warm-start DISABLED "
                      "(semantic_tag_init=false); keeping identical mean-init (v13 recipe)")
            else:
                from token_init import build_semantic_tag_init, semantic_init_new_rows
                # boundary = tokenizer length BEFORE the add (the id of the first
                # new token). Qwen3-8B pads the embedding matrix past len(tokenizer),
                # so old_vocab_size would reject every new tag (the R10 bug).
                new_token_start = len(tokenizer) - added
                _out_w = (out_emb.weight if (out_emb is not None
                          and out_emb.weight.shape[0] > old_vocab_size) else None)
                _n_sem = semantic_init_new_rows(
                    in_emb, _out_w, new_token_start,
                    build_semantic_tag_init(tokenizer, new_token_start),
                )
                print(f"[tagged-mode] semantic warm-start: {_n_sem} open tags "
                      f"initialized from their display names")
        except Exception as e:
            print(f"[tagged-mode] semantic init skipped ({type(e).__name__}: {e}); "
                  f"keeping mean-init")

    return added


# ── Loss ──────────────────────────────────────────────
def _tokenize_with_eos(
    tokenizer: PreTrainedTokenizerBase,
    target_text: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize each target, append the EOS token, pad to batch-max.

    Two bugs in the original tokenize-then-pad path made the model never learn
    to stop generating:

      1. `tokenizer(text)` does not append EOS by default — so the training
         target ended with whatever the last natural-language token was, never
         with the EOS token. The LM never saw EOS at a "this is the end"
         position and at inference would keep generating until max_new_tokens.

      2. `train.py` sets `pad_token = eos_token` when the tokenizer has no
         pad_token (common with Qwen). The original label-masking step
         `labels[labels == pad_token_id] = -100` then masks EVERY eos_token
         out of the loss — even the genuine end-of-target EOS. So even if (1)
         were fixed, EOS prediction would never be supervised.

    Fix: tokenize each row individually, truncate to max_length - 1 (room for
    EOS), append eos_token_id, pad with pad_token_id, and return both
    target_ids and attention_mask. The caller uses attention_mask (not
    pad_id == ...) to mask the loss — so a genuine EOS at content end is
    supervised regardless of whether pad_id == eos_id.

    Returns:
        target_ids: (B, L) long. Each row is the tokenized target + one EOS,
                    padded with pad_token_id up to L (the batch maximum).
        attn_mask:  (B, L) long. 1 at content positions (including EOS),
                    0 at padding positions.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(
            "tokenizer.eos_token_id is None; can't terminate generation. "
            "Set tokenizer.eos_token before calling _tokenize_with_eos."
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Fall back to eos_id but we still need to distinguish content from
        # padding for label masking — attention_mask handles that downstream.
        pad_id = eos_id

    truncated_max = max(1, max_length - 1)
    ids_per_row: list[list[int]] = []
    for t in target_text:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=truncated_max,
            add_special_tokens=False,
        )
        ids = list(enc.input_ids)
        ids.append(eos_id)
        ids_per_row.append(ids)

    lengths = [len(ids) for ids in ids_per_row]
    L = max(lengths) if lengths else 1
    B = len(ids_per_row)

    target_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, ids in enumerate(ids_per_row):
        target_ids[i, : lengths[i]] = torch.tensor(ids, dtype=torch.long, device=device)
        attn_mask[i, : lengths[i]] = 1

    return target_ids, attn_mask


def _build_section_ctx(
    section_head: "SectionQueryHead | None",
    spec_encoder: "SpecEncoder | None",
    section_id_to_idx: dict[int, int],
    batch: dict,
    device: torch.device,
    mode: str = "static",
    range_open_id: int | None = None,
    readout_head: "SectionReadoutHead | None" = None,  # [section_readout]
) -> dict | None:
    """Compute the per-batch section context: K/V (+ e_all in static mode).

    BEATs patches are read from the batch (cached during preprocessing); the
    online encoder path isn't supported in v1 because waveforms aren't in the
    .pt files.

    Static mode: computes per-section audio summaries e_all upfront from
    per-section static queries. Single-pass training in _ce_against_target.

    Dynamic mode: defers e_t computation to pass 1 inside _ce_against_target,
    where the LM's hidden states at <sec_X> positions are used as the query
    source (q_t = W_q · h_t). Two-pass training.

    Returns a dict consumed by _ce_against_target. Returns None when there's
    nothing to inject (legacy .pt files without beats_patches → fallback to
    plain LM-CE on the section tags treated as ordinary tokens).
    """
    patches = batch.get("beats_patches")
    if patches is None:
        if spec_encoder is None:
            return None
        raise RuntimeError(
            "Online SpecEncoder path requested but waveforms are not in the dataset. "
            "Run scripts/preprocess_beats.py to cache patches into the .pt files."
        )

    patches = patches.to(device).to(torch.bfloat16)
    # (B, P_max) bool — True at padded positions. Cross-attention sets those
    # to -inf before the softmax so attention never lands on padding. Passed
    # through to SectionQueryHead.forward_dynamic / forward_all_sections.
    key_padding_mask = batch.get("beats_patches_mask")
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device)
    K, V = section_head.precompute_kv(patches)

    if mode == "dynamic":
        # Defer e_t — needs pass-1 hidden states which _ce_against_target produces.
        # range_open_id (optional): when set, _inject_section_summaries_dynamic
        # also fires the cross-attention query at every <r> open in the target,
        # so multi-value spans get per-range training signal too.
        return {
            "mode": "dynamic",
            "head": section_head,
            "K": K, "V": V,
            "key_padding_mask": key_padding_mask,
            "section_id_to_idx": section_id_to_idx,
            "range_open_id": range_open_id,
            # [section_readout] head + per-fired-query alpha get stashed in by
            # _inject_section_summaries_dynamic during pass 1.
            "readout_head": readout_head,
        }

    # Static mode: compute per-section summaries up front.
    e_all, alpha = section_head.forward_all_sections(K, V, key_padding_mask=key_padding_mask)
    return {
        "mode": "static",
        "e_all": e_all,
        "section_id_to_idx": section_id_to_idx,
        # [section_readout] keep alpha (B, S, P) + V (B, P, d_v) so the grounding
        # loss can recompute z = alpha · V.detach() for every section every step.
        "readout_head": readout_head,
        "readout_alpha": alpha,
        "V": V,
    }


def _inject_section_summaries_dynamic(
    llm: nn.Module,
    prefix_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    section_ctx: dict,
) -> torch.Tensor:
    """Dynamic-mode injection — pass 1 of the two-pass training.

    Runs the LM once with un-injected target embeddings, extracts the hidden
    state at each <sec_X> position, computes a per-position audio summary via
    `head.forward_dynamic(h_t, K, V, batch_idx)`, and returns target_embeds
    with those summaries residually added at the section positions.

    Gradient flow: loss → pass-2 LM weights → target_embeds → e_t → W_q → h_t
    → pass-1 LM weights, so the LM learns to produce query-friendly hidden
    states at section opens. The section_head and the LM co-adapt.

    Memory: pass 1's activations are retained for backprop. On Qwen3-1.7B
    full-FT at batch 6 with grad checkpointing this peaks around 45-50 GB —
    fits H100 80 GB with headroom; tight on A100 40 GB.
    """
    head = section_ctx["head"]
    K, V = section_ctx["K"], section_ctx["V"]
    id_to_idx = section_ctx["section_id_to_idx"]

    # Build the full set of "query positions": every <sec_X> open + every <r>
    # open. Both kinds produce dynamic queries through the same forward path;
    # the LM learns to make all of them query-friendly hidden states.
    open_ids: list[int] = list(id_to_idx.keys())
    range_open_id = section_ctx.get("range_open_id")
    if range_open_id is not None:
        open_ids.append(range_open_id)
    open_ids_t = torch.tensor(open_ids, device=target_ids.device, dtype=target_ids.dtype)

    # Locate (b, l) coordinates of every query-firing open in the target.
    is_section = (target_ids.unsqueeze(-1) == open_ids_t).any(-1)           # (B, L) bool
    if not is_section.any():
        # Nothing to inject. Skip the pass-1 forward.
        return target_embeds

    bl = is_section.nonzero(as_tuple=False)                                  # (N, 2)
    batch_idx = bl[:, 0]
    local_pos = bl[:, 1]

    # Pass 1: LM forward with un-injected target embeddings.
    prefix_len = prefix_embeds.shape[1]
    prompt_len = prompt_embeds.shape[1]
    inputs_embeds_pass1 = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
    out1 = llm(inputs_embeds=inputs_embeds_pass1, output_hidden_states=True)
    hidden = out1.hidden_states[-1]                                          # (B, T, d_lm)

    # Hidden state at each section-open position. abs_pos in [prefix_len + prompt_len, ...).
    abs_pos = prefix_len + prompt_len + local_pos
    h_t = hidden[batch_idx, abs_pos]                                         # (N, d_lm)

    # Cross-attend with per-position queries. Pass the padding mask so the
    # softmax over patches ignores padded BEATs slots.
    key_padding_mask = section_ctx.get("key_padding_mask")
    e_t, alpha = head.forward_dynamic(
        h_t, K, V,
        batch_idx=batch_idx,
        key_padding_mask=key_padding_mask,
    )       # (N, d_lm), alpha (N, P)

    # [section_readout] Stash the per-fired-query attention + routing so the
    # grounding loss (computed back in compute_loss) can recompute
    # z = alpha · V.detach(). <r> opens fire a query too but have no scalar →
    # query_section_indices maps them to -1 so loss_dynamic drops them.
    if section_ctx.get("readout_head") is not None:
        fired_ids = target_ids[batch_idx, local_pos]                # (N,)
        section_ctx["readout_alpha"] = alpha
        section_ctx["readout_batch_idx"] = batch_idx
        section_ctx["readout_query_section_idx"] = query_section_indices(
            fired_ids, id_to_idx,
        )

    # Inject e_t at the corresponding (b, l) in target_embeds.
    # Build a sparse additive: scatter e_t into a (B, L, d_lm) zero tensor.
    additive = torch.zeros_like(target_embeds)
    additive[batch_idx, local_pos] = e_t.to(additive.dtype)
    return target_embeds + additive


def _inject_section_summaries(
    target_ids: torch.Tensor,
    target_embeds: torch.Tensor,
    section_ctx: dict,
) -> torch.Tensor:
    """Add per-section audio summaries to target_embeds at <sec_X> open positions.

    For each <sec_X> open token in target_ids, look up its precomputed section
    summary e_all[b, section_idx] and add it residually to target_embeds[b, l]
    so the LM's hidden state at that position is informed by the cross-attention
    over the spectrogram. Vectorised — no Python loops over batch or positions.

    Args:
        target_ids:    (B, L) tokenized target.
        target_embeds: (B, L, d_lm) embeddings of target_ids — modified residually.
        section_ctx:   dict with keys:
            "e_all":              (B, n_sections, d_lm) — precomputed summaries.
            "section_id_to_idx":  dict {token_id: section_idx} for the section opens.

    Returns:
        Modified target_embeds tensor with the residual additions.
    """
    e_all = section_ctx["e_all"]
    id_to_idx = section_ctx["section_id_to_idx"]
    B, L = target_ids.shape

    # Build a (B, L, n_sections) one-hot mask: mask[b, l, s] = 1 iff
    # target_ids[b, l] is the open token for section s. Then a single einsum
    # produces the additive embedding to add to target_embeds.
    section_ids = torch.tensor(
        list(id_to_idx.keys()), device=target_ids.device, dtype=target_ids.dtype,
    )  # (n_sections,)
    section_idx_lookup = torch.tensor(
        list(id_to_idx.values()), device=target_ids.device, dtype=torch.long,
    )  # (n_sections,)

    # For each (b, l), find which (if any) section it matches.
    # target_ids[..., None]: (B, L, 1); section_ids: (n_sections,). Compare → (B, L, n_sections).
    is_match = target_ids.unsqueeze(-1) == section_ids       # (B, L, n_sections)
    if not is_match.any():
        return target_embeds  # nothing to inject

    # Permute to the section_head's section_idx order:
    # mask[b, l, section_idx_lookup[s]] = is_match[b, l, s]
    n_sec = e_all.shape[1]
    mask = torch.zeros(B, L, n_sec, dtype=target_embeds.dtype, device=target_embeds.device)
    mask[..., section_idx_lookup] = is_match.to(target_embeds.dtype)

    # additive: (B, L, d_lm) = einsum (B, L, n_sec) (B, n_sec, d_lm)
    additive = torch.einsum("bls,bsd->bld", mask, e_all)
    return target_embeds + additive


# NTL needs the 10 single-digit token ids for the active tokenizer. They never
# change for a given tokenizer, so resolve them once and cache by tokenizer
# identity (keyed by id(tokenizer)) to avoid re-encoding every step.
_DIGIT_IDS_CACHE: dict[int, torch.Tensor] = {}


def _get_digit_ids(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    key = id(tokenizer)
    cached = _DIGIT_IDS_CACHE.get(key)
    if cached is None:
        cached = digit_token_ids(tokenizer)
        _DIGIT_IDS_CACHE[key] = cached
    return cached


def _ce_against_target(
    llm: nn.Module,
    embed_layer: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_text: list[str],
    max_length: int,
    device: torch.device,
    section_ctx: dict | None = None,
    return_ntl_tensors: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Run one LM forward (prefix + prompt + target) and return CE loss on the target tokens.

    Tokens of the prefix and prompt are masked out via -100 labels so the loss reflects
    only the autoregressive prediction of the target tokens.

    When return_ntl_tensors=True, ALSO returns the tensors needed by the Number Token
    Loss WITHOUT a second LM forward — they are sliced from the same outputs:
        (loss, ntl_logits, ntl_target_ids, ntl_target_mask)
    Here ntl_logits[b, j] is the next-token-shifted distribution that predicts the
    target token ntl_target_ids[b, j] (HF computes CE with an internal shift; we
    reproduce that shift so NTL reads the SAME predicted distribution CE scores).
    ntl_target_mask is the per-position content mask (1 at real target tokens incl.
    EOS, 0 at padding). All three cover only the L target positions.

    Section-query injection (when section_ctx is provided):
      - mode="static":  precomputed per-section e_all is gathered by section_idx
                        and added residually to target_embeds at <sec_X> positions.
                        Single LM forward.
      - mode="dynamic": runs a FIRST LM forward to extract hidden states at <sec_X>
                        positions, computes per-position e_t = W_o(α·V) with
                        α = softmax(W_q · h_t · K^T / √d_k), injects e_t into
                        target_embeds, then runs a SECOND LM forward for the CE loss.
                        Two-pass — ~1.5× compute, ~1.5× memory. Required by the
                        professor's "LM generates a query vector" design.
    """
    target_ids, target_attn = _tokenize_with_eos(tokenizer, target_text, max_length, device)

    prompt_embeds = embed_layer(prompt_ids.expand(prefix_embeds.shape[0], -1))
    target_embeds = embed_layer(target_ids)

    if section_ctx is not None:
        if section_ctx.get("mode") == "dynamic":
            target_embeds = _inject_section_summaries_dynamic(
                llm, prefix_embeds, prompt_embeds, target_embeds, target_ids, section_ctx,
            )
        else:
            target_embeds = _inject_section_summaries(target_ids, target_embeds, section_ctx)

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)

    N = prefix_embeds.shape[1]
    P = prompt_embeds.shape[1]
    ignore_labels = torch.full((prefix_embeds.shape[0], N + P), -100, device=device)
    # Mask labels using attention_mask, NOT pad_token_id equality. With
    # pad_token == eos_token (Qwen fallback in train()), id-based masking
    # would also drop the genuine end-of-target EOS from the loss and the
    # model would never learn to stop. attn_mask is 1 at content (incl. EOS),
    # 0 at padding — exactly what we want.
    target_labels = target_ids.clone()
    target_labels[target_attn == 0] = -100
    labels = torch.cat([ignore_labels, target_labels], dim=1)

    outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
    if not return_ntl_tensors:
        return outputs.loss

    # ── NTL tensor extraction (no extra LM forward) ──────────────────────────
    # outputs.logits is (B, S, V), S = N + P + L. HF's CE shifts internally:
    # logits[:, i] predicts token at position i+1. The L target tokens sit at
    # sequence positions [N+P, N+P+L). So the distribution that predicts the
    # target token at target-index j is logits[:, (N+P) + j - 1]. Sliced over
    # j in [0, L): logits[:, N+P-1 : N+P+L-1]. Aligns ntl_logits[b, j] with
    # target_ids[b, j] exactly as CE does, so NTL reads the same predictions.
    L = target_ids.shape[1]
    start = N + P - 1
    ntl_logits = outputs.logits[:, start:start + L, :]   # (B, L, V)
    ntl_target_ids = target_ids                          # (B, L)
    ntl_target_mask = target_attn                        # (B, L) 1 at content incl. EOS
    return outputs.loss, ntl_logits, ntl_target_ids, ntl_target_mask


def compute_loss(
    adapter: nn.Module,
    llm: nn.Module,
    embed_layer: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    audio_features: torch.Tensor,
    overlap_info: torch.Tensor,
    target_text: list[str],
    prompt_ids: torch.Tensor,
    device: torch.device,
    config: dict,
    target_nums: list[str] | None = None,
    gt_scalars: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
    prompt_nums_ids: torch.Tensor | None = None,
    section_ctx: dict | None = None,
    decoupled_head: "DecoupledGroundingHead | None" = None,
    snr_map_head: "SupervisedSNRMapHead | None" = None,
    srmr_map_head: "SupervisedSRMRMapHead | None" = None,
    batch: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """B-full multi-task forward + auxiliary regression head.

    Computes:
      - lm_loss_prose: CE loss on the natural-language description (prose target).
      - lm_loss_nums:  CE loss on the bare-numbers target (forward A).
      - mse_loss:      MSE on the aux head's scalar predictions, masked by gt_mask.

    Total = lambda_prose * lm_loss_prose + lambda_nums * lm_loss_nums + lambda_mse * mse_loss.

    Args:
        target_nums: per-clip bare-numbers target strings. If None, the nums forward is skipped
                     (legacy single-target training).
        gt_scalars:  (B, n_features) tensor of GT scalars from feature_set.extract_scalars.
                     If None, MSE term is skipped.
        gt_mask:     (B, n_features) bool tensor — True where the scalar was actually measured.

    Returns:
        (total_loss, metrics_dict). metrics_dict has keys 'loss_lm_prose', 'loss_lm_nums',
        'loss_mse', and 'loss_total' for wandb logging.
    """
    audio_features = audio_features.to(device).to(torch.bfloat16)
    overlap_info = overlap_info.to(device).to(torch.bfloat16)

    # AdapterWithAuxHead returns (prefix, scalar_pred); legacy adapters return prefix only.
    # With reliability_head=True the aux head's scalar_pred is itself a (mean, log_var)
    # tuple — split it so the MSE path keeps using the predicted MEAN and the new NLL
    # term (below) gets both mean and log_var. With the plain head, scalar_pred is the
    # mean tensor and reliability_log_var stays None (NLL term is then skipped).
    out = adapter(audio_features, overlap_info)
    if isinstance(out, tuple):
        prefix_embeds, scalar_pred = out
    else:
        prefix_embeds, scalar_pred = out, None

    reliability_log_var = None
    if isinstance(scalar_pred, tuple):
        scalar_pred, reliability_log_var = scalar_pred   # (mean, log_var)

    metrics: dict[str, float] = {}

    # Prose CE loss (always computed). When use_sections=true, section_ctx is
    # threaded through so the per-section audio summaries get residually
    # injected into the target embeddings at <sec_X> open positions.
    #
    # [ntl] Number Token Loss (arXiv 2411.02083, WAS/abs variant). When
    # lambda_ntl > 0 we ask the SAME prose forward to also hand back the
    # next-token-shifted logits + target ids so we can add an ordinal penalty on
    # the prose's digit positions — fixing CE's nominal-digit dilution without a
    # second forward. lambda_ntl <= 0 → byte-identical to before (plain CE,
    # single return value, no NTL tensors built).
    lambda_ntl = float(config.get("lambda_ntl", 0.0))
    want_ntl = lambda_ntl > 0.0
    if want_ntl:
        lm_loss_prose, ntl_logits, ntl_target_ids, ntl_target_mask = _ce_against_target(
            llm, embed_layer, tokenizer,
            prefix_embeds, prompt_ids, target_text,
            max_length=config["max_target_length"],
            device=device,
            section_ctx=section_ctx,
            return_ntl_tensors=True,
        )
    else:
        lm_loss_prose = _ce_against_target(
            llm, embed_layer, tokenizer,
            prefix_embeds, prompt_ids, target_text,
            max_length=config["max_target_length"],
            device=device,
            section_ctx=section_ctx,
        )
    metrics["loss_lm_prose"] = float(lm_loss_prose.detach().item())

    # [ntl] Ordinal digit penalty on the prose target's digit positions. Masked to
    # content tokens (no padding), 0 when there are no digit targets in the batch.
    ntl_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    if want_ntl:
        digit_ids = _get_digit_ids(tokenizer)
        ntl_loss = number_token_loss(
            ntl_logits.float(), ntl_target_ids, digit_ids,
            target_mask=ntl_target_mask,
        ).to(lm_loss_prose.dtype)
        metrics["loss_ntl"] = float(ntl_loss.detach().item())
    else:
        metrics["loss_ntl"] = 0.0

    # [section_readout] Grounding loss. Regresses each section's acoustic scalar
    # out of z = alpha · V.detach() (the attention output), pushing alpha onto
    # evidence-bearing patches. Computed right after the prose forward, which is
    # what stashes alpha/V into section_ctx (the nums forward below carries no
    # section_ctx, so nothing clobbers it). No-op when lambda_readout == 0,
    # there's no readout head, or no GT scalars.
    readout_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    lambda_readout = float(config.get("lambda_readout", 0.0))
    if lambda_readout > 0.0:
        r_loss, r_mae = section_readout_loss(section_ctx, gt_scalars, gt_mask)
        if r_loss is not None:
            readout_loss = r_loss.to(lm_loss_prose.dtype)
            metrics["loss_readout"] = float(r_loss.detach().item())
            for fname, val in r_mae.items():
                metrics[f"readout_mae/{fname}"] = val
        else:
            metrics["loss_readout"] = 0.0
    else:
        metrics["loss_readout"] = 0.0

    # Numbers CE loss (B-full forward A, optional)
    lm_loss_nums = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    has_nums = (target_nums is not None
                and any(t for t in target_nums)
                and float(config.get("lambda_nums", 0.0)) > 0.0)
    if has_nums:
        # Filter out clips with empty nums target (no CSV row available)
        valid_idx = [i for i, t in enumerate(target_nums) if t]
        if valid_idx:
            valid_idx_t = torch.tensor(valid_idx, device=device, dtype=torch.long)
            prefix_subset = prefix_embeds.index_select(0, valid_idx_t)
            nums_subset = [target_nums[i] for i in valid_idx]
            # Bare-numbers targets are short (~80 tokens); cap separately to avoid the
            # prose-target's 384+ length budget.
            nums_max_len = config.get("max_nums_length", 96)
            # Use a dedicated prompt for the numbers-target forward when one is configured.
            # If prompt_nums_ids is None we fall back to prompt_ids (legacy single-prompt setup,
            # which causes both completions to live under the same prompt key).
            nums_prompt = prompt_nums_ids if prompt_nums_ids is not None else prompt_ids
            lm_loss_nums = _ce_against_target(
                llm, embed_layer, tokenizer,
                prefix_subset, nums_prompt, nums_subset,
                max_length=nums_max_len,
                device=device,
            )
        metrics["loss_lm_nums"] = float(lm_loss_nums.detach().item())
    else:
        metrics["loss_lm_nums"] = 0.0

    # Aux regression head MSE (optional, requires scalar_pred + GT)
    mse_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    has_mse = (
        scalar_pred is not None
        and gt_scalars is not None
        and gt_mask is not None
        and float(config.get("lambda_mse", 0.0)) > 0.0
    )
    if has_mse:
        gt_scalars_d = gt_scalars.to(device).to(scalar_pred.dtype)
        gt_mask_d = gt_mask.to(device).to(scalar_pred.dtype)
        # Normalize per-feature squared error by typical magnitude so all 13 features
        # contribute roughly equally. Without this, F0 (~150 Hz) dominates the sum
        # 1000x over overlap_ratio (~0.5) and the adapter optimizes only for F0.
        scales = torch.tensor(FEATURE_SCALES, device=device, dtype=scalar_pred.dtype)
        per_feat_se = ((scalar_pred - gt_scalars_d) / scales) ** 2   # (B, n_feat) — unit-free
        masked = per_feat_se * gt_mask_d                              # (B, n_feat)
        denom = gt_mask_d.sum().clamp(min=1.0)
        mse_loss = masked.sum() / denom
        metrics["loss_mse"] = float(mse_loss.detach().item())
    else:
        metrics["loss_mse"] = 0.0

    # [reliability_head] Heteroscedastic Gaussian NLL (Kendall & Gal 2017). Uses the
    # SAME masked, per-feature-scale-normalized error as the MSE above, but lets the
    # head learn a per-feature log-variance so high predicted σ = "this feature is
    # unreliable here" — the abstention signal the risk-coverage eval consumes. No-op
    # unless the adapter has a reliability head (reliability_log_var is not None),
    # there are GT scalars, and lambda_nll > 0; so plain-MSE runs are unaffected.
    nll_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    has_nll = (
        reliability_log_var is not None
        and scalar_pred is not None
        and gt_scalars is not None
        and gt_mask is not None
        and float(config.get("lambda_nll", 0.0)) > 0.0
    )
    if has_nll:
        from reliability_head import heteroscedastic_nll
        gt_scalars_n = gt_scalars.to(device).to(scalar_pred.dtype)
        gt_mask_n = gt_mask.to(device)
        scales_n = torch.tensor(FEATURE_SCALES, device=device, dtype=scalar_pred.dtype)
        nll_loss = heteroscedastic_nll(
            scalar_pred, reliability_log_var.to(scalar_pred.dtype),
            gt_scalars_n, mask=gt_mask_n, scales=scales_n,
        ).to(lm_loss_prose.dtype)
        metrics["loss_nll"] = float(nll_loss.detach().item())
        # Mean predicted σ over present slots — a quick "is the head using its
        # uncertainty channel" diagnostic. Higher = more abstention headroom.
        with torch.no_grad():
            sigma = torch.exp(0.5 * reliability_log_var.float())
            mask_f = gt_mask_n.to(sigma.dtype)
            denom_s = mask_f.sum().clamp(min=1.0)
            metrics["reliability_sigma_mean"] = float((sigma * mask_f).sum() / denom_s)
    else:
        metrics["loss_nll"] = 0.0

    # [decoupled_grounding] Parallel token-free 2D-grounding term. Runs the head on
    # the batch's BEATs patches (NOT the adapter/LM path) and regresses each scored
    # feature out of A · V.detach(). Its gradient lands on the head's own queries /
    # projections / readout — it never touches the LM CE graph above, so the LM
    # keeps generating clean untagged prose while this head learns the maps in
    # parallel. No-op when off, no head, or no BEATs patches in the batch.
    decoupled_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    lambda_decoupled = float(config.get("lambda_decoupled", 0.0))
    if (
        config.get("decoupled_grounding", False)
        and decoupled_head is not None
        and lambda_decoupled > 0.0
        and batch is not None
    ):
        # [bottleneck] effective bits weight (per-epoch warmed in the train loop;
        # stored back into config as 'bits_lambda_effective'). 0 in softmax mode or
        # during bits warmup → no bits penalty, identical to the pure-Huber term.
        bits_lambda_eff = float(config.get("bits_lambda_effective", 0.0))
        # [overlap-map supervision] direct segmentation loss on the overlap query's
        # map vs the oracle overlap_segments. 0.0 (default) → exact no-op, both modes
        # unchanged; >0 adds lambda_overlap_map · soft-Dice(overlap map, time-target).
        lambda_overlap_map = float(config.get("lambda_overlap_map", 0.0))
        d_loss, d_metrics = decoupled_grounding_loss_term(
            decoupled_head, batch, lambda_decoupled, device,
            bits_lambda=bits_lambda_eff,
            lambda_overlap_map=lambda_overlap_map,
            # 'max' (not 'mean') frequency reduction: a softmax attention row is
            # mass-conserving, so mean-over-freq sums to 1/F_P for every clip → no
            # gradient to shape the map. Max distinguishes a concentrated map from a
            # diffuse one in BOTH softmax and bottleneck modes.
            overlap_map_reduce=str(config.get("overlap_map_reduce", "max")),
            overlap_map_empty_weight=float(config.get("overlap_map_empty_weight", 1.0)),
            overlap_target_soft=bool(config.get("overlap_target_soft", True)),
        )
        if d_loss is not None:
            decoupled_loss = d_loss.to(lm_loss_prose.dtype)
            metrics.update(d_metrics)
        else:
            metrics["loss_decoupled"] = 0.0
    else:
        metrics["loss_decoupled"] = 0.0

    # [snr_map] SUPERVISED dense local-SNR-map term. Runs the head on the batch's WavLM
    # `audio_features` and regresses the per-frame SNR timeline against the stem-derived
    # oracle target (masked to s1-active frames). Optional CBM scalar tie (lambda_snr_scalar
    # pools the timeline → clip SNR, regressed against gt_scalars[snr]) and optional IRM
    # branch (lambda_snr_irm). Its gradient lands on the head's own params — never the LM
    # CE graph. No-op when the head is absent, lambda <= 0, or the batch carries no
    # snr_map_target (already weighted by lambda_snr_map inside snr_map_loss_term).
    snr_map_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    lambda_snr_map = float(config.get("lambda_snr_map", 0.0))
    if snr_map_head is not None and lambda_snr_map > 0.0 and batch is not None:
        s_loss, s_metrics = snr_map_loss_term(
            snr_map_head, batch, lambda_snr_map, device,
            lambda_scalar=float(config.get("lambda_snr_scalar", 0.0)),
            lambda_irm=float(config.get("lambda_snr_irm", 0.0)),
            irm_mode=str(config.get("snr_irm_mode", "mse")),
        )
        if s_loss is not None:
            snr_map_loss = s_loss.to(lm_loss_prose.dtype)
            metrics.update(s_metrics)
        else:
            metrics["loss_snr_map"] = 0.0
    else:
        metrics["loss_snr_map"] = 0.0

    # [srmr_map] SUPERVISED 2D SRMR modulation-energy-map term. Runs the SRMR head on the
    # batch's WavLM `audio_features` and regresses the 23x8 (acoustic x modulation)
    # log-energy map against the oracle clean-s1 SRMR map (masked Huber). Its gradient
    # lands on the head's own params — never the LM CE graph. No-op when the head is
    # absent, lambda <= 0, or the batch carries no srmr_map_target (already weighted by
    # lambda_srmr_map inside srmr_map_loss_term).
    srmr_map_loss = torch.tensor(0.0, device=device, dtype=lm_loss_prose.dtype)
    lambda_srmr_map = float(config.get("lambda_srmr_map", 0.0))
    if srmr_map_head is not None and lambda_srmr_map > 0.0 and batch is not None:
        sr_loss, sr_metrics = srmr_map_loss_term(
            srmr_map_head, batch, lambda_srmr_map, device,
        )
        if sr_loss is not None:
            srmr_map_loss = sr_loss.to(lm_loss_prose.dtype)
            metrics.update(sr_metrics)
        else:
            metrics["loss_srmr_map"] = 0.0
    else:
        metrics["loss_srmr_map"] = 0.0

    lambda_prose = float(config.get("lambda_prose", 1.0))
    lambda_nums = float(config.get("lambda_nums", 0.0))
    lambda_mse = float(config.get("lambda_mse", 0.0))
    lambda_nll = float(config.get("lambda_nll", 0.0))

    total = lambda_prose * lm_loss_prose + lambda_nums * lm_loss_nums + lambda_mse * mse_loss
    # [ntl] add the Number Token Loss on prose digit positions (ntl_loss is 0 when
    # lambda_ntl <= 0 or the batch has no digit targets). lambda_ntl default 0.3
    # (paper default); it is the ordinal complement to the prose CE, so it scales
    # the same prose forward's digit gradient — keep it modest to preserve fluency.
    total = total + lambda_ntl * ntl_loss
    # [reliability_head] add the heteroscedastic NLL term (nll_loss is 0 when the head
    # is absent or lambda_nll == 0). Note: NLL and plain MSE can be used together (NLL
    # trains the variance, MSE keeps a clean mean gradient) or NLL alone — set
    # lambda_mse=0 for NLL-only on the same predicted mean.
    total = total + lambda_nll * nll_loss
    # [section_readout] add the grounding term (readout_loss is 0 when disabled).
    total = total + lambda_readout * readout_loss
    # [decoupled_grounding] add the parallel grounding term (already scaled by
    # lambda_decoupled inside decoupled_grounding_loss_term; 0 when disabled).
    total = total + decoupled_loss
    # [snr_map] add the dense local-SNR-map term (already scaled by lambda_snr_map inside
    # snr_map_loss_term; 0 when disabled).
    total = total + snr_map_loss
    # [srmr_map] add the 2D SRMR-modulation-map term (already scaled by lambda_srmr_map
    # inside srmr_map_loss_term; 0 when disabled).
    total = total + srmr_map_loss
    metrics["loss_total"] = float(total.detach().item())
    return total, metrics


# ── Training ──────────────────────────────────────────────
def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(config["seed"])

    # Tokenizer + LLM
    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    # Register per-feature special tokens BEFORE LoRA wrap so the LoRA matrices
    # see the resized embedding. Only runs when tagged_mode is on (the EMNLP
    # rework path); legacy untagged training leaves the vocab unchanged.
    n_added_tokens = 0
    if config.get("tagged_mode"):
        n_added_tokens = _register_feature_tags(tokenizer, llm)
        print(f"[tagged-mode] Added {n_added_tokens} feature special tokens "
              f"(vocab now {len(tokenizer)})")

    full_ft = not _use_lora(config)
    if full_ft:
        print(f"[full-FT] lora_rank={config.get('lora_rank')!r} → training all LM weights")
    else:
        from peft_config import lora_config_kwargs, uses_pissa
        llm = get_peft_model(llm, LoraConfig(**lora_config_kwargs(config)))
        _extra = []
        if config.get("use_dora"):
            _extra.append("DoRA")
        if uses_pissa(config):
            _extra.append(f"PiSSA({config['init_lora_weights']})")
        print(f"[LoRA] rank={config['lora_rank']} alpha={config['lora_alpha']}"
              + (f" + {'+'.join(_extra)}" if _extra else ""))

    # Row-masked new-token training (tagged_mode only).
    #
    # When tagged_mode adds the <sec_*>/<f_*> tokens, their embed_tokens/lm_head
    # rows are random mean-init. They MUST train or the model can't emit/consume
    # them. The naive fix (LoRA modules_to_save) makes the FULL 151,936-row
    # embedding + output head trainable, and training the whole vocab on narrow
    # speech-prose corrupts unrelated tokens — observed in v10 as Thai-token
    # injection and malformed tags.
    #
    # Fix: unfreeze embed/lm_head but register a backward hook that ZEROES the
    # gradient for every row except the new-token rows. Only the N new rows ever
    # change; all pretrained rows stay byte-identical → no vocab corruption.
    # These params get their own optimizer group (weight_decay=0) so decoupled
    # weight decay can't drift the frozen rows either.
    new_token_params: list = []
    if config.get("tagged_mode") and n_added_tokens > 0:
        old_vocab = len(tokenizer) - n_added_tokens
        seen_ids: set[int] = set()
        for module in (llm.get_input_embeddings(), llm.get_output_embeddings()):
            if module is None or getattr(module, "weight", None) is None:
                continue
            w = module.weight
            if id(w) in seen_ids:   # tied embed/lm_head → mask once
                continue
            seen_ids.add(id(w))
            w.requires_grad_(True)

            def _mask_old_rows(grad, ov=old_vocab):
                g = grad.clone()
                g[:ov] = 0.0
                return g

            w.register_hook(_mask_old_rows)
            new_token_params.append(w)
        print(f"[tagged-mode] row-masked embed/lm_head: only rows "
              f"[{old_vocab}:{len(tokenizer)}] ({n_added_tokens} new tokens) trainable; "
              f"{len(new_token_params)} weight tensor(s) hooked")

    if config["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable()

    embed_layer = llm.get_input_embeddings()

    # Adapter
    # [reliability_head] when reliability_head=true the aux head predicts per-feature
    # (mean, log_var) instead of a bare mean — the heteroscedastic abstention head.
    # Default false → plain Linear mean head, byte-identical to before.
    lm_hidden_size = llm.config.hidden_size
    adapter = build_adapter(
        config["adapter_variant"], lm_dim=lm_hidden_size,
        reliability_head=bool(config.get("reliability_head", False)),
    ).to(device).to(torch.bfloat16)
    if config.get("reliability_head", False):
        print(f"[reliability_head] heteroscedastic aux head ON "
              f"(predicts per-feature mean + log-variance); lambda_nll="
              f"{config.get('lambda_nll', 0.0)}")

    # ── Encoder-unfreeze plumbing (capacity experiment, Q4) ─────────────────
    # By default both SSL frontends are FROZEN and their features are cached, so
    # nothing is loaded here and `encoder_unfreeze_params` is empty — byte-identical
    # to the production frozen path. Setting unfreeze_wavlm_top_n / unfreeze_beats_top_n
    # to N>0 loads (WavLM) or reuses (BEATs/SpecEncoder, built below) the encoder,
    # unfreezes its top-N transformer blocks, and routes those params into their own
    # optimizer group at lr_encoder (a small LR). See src/encoder_unfreeze.py for the
    # GPU-memory implications.
    from encoder_unfreeze import (
        unfreeze_top_n_blocks, encoder_trainable_params, count_blocks,
    )
    wavlm_encoder = None  # only built when unfreeze_wavlm_top_n > 0
    encoder_unfreeze_params: list = []
    _unfreeze_wavlm_n = int(config.get("unfreeze_wavlm_top_n", 0) or 0)
    if _unfreeze_wavlm_n > 0:
        # WavLM is NOT part of the cached-feature training path — audio_features come
        # pre-extracted in the .pt files. Unfreezing it only has an effect if waveforms
        # are fed through it at train time (an end-to-end variant, not the default
        # dataloader). We load + unfreeze it here so the param-group / checkpoint
        # plumbing is exercised and correct; the forward wiring is a separate change.
        from transformers import WavLMModel
        wavlm_name = config.get("wavlm_name", "microsoft/wavlm-large")
        wavlm_encoder = WavLMModel.from_pretrained(
            wavlm_name, torch_dtype=torch.bfloat16,
        ).to(device)
        for p in wavlm_encoder.parameters():
            p.requires_grad_(False)
        _wp = unfreeze_top_n_blocks(wavlm_encoder, _unfreeze_wavlm_n)
        encoder_unfreeze_params.extend(_wp)
        print(f"[encoder-unfreeze] WavLM ({wavlm_name}): unfroze top "
              f"{min(_unfreeze_wavlm_n, count_blocks(wavlm_encoder))}/"
              f"{count_blocks(wavlm_encoder)} blocks "
              f"({sum(p.numel() for p in _wp)/1e6:.1f}M params trainable) at "
              f"lr_encoder={config.get('lr_encoder')}; "
              f"WARNING: waveforms are not in the cached .pt files — this only affects "
              f"training if an end-to-end waveform forward is wired in.")

    # ── Section-query branch (EMNLP rework, Path 3) ────────────────────────
    # When use_sections=true, set up:
    #   - SpecEncoder (BEATs by default) — frozen, runs once per batch (or skipped
    #     if BEATs patches were precomputed and cached in the .pt files)
    #   - SectionQueryHead — learnable per-section queries, cross-attends to spec
    #     patches, produces audio summaries that get injected at <sec_X> positions.
    use_sections = bool(config.get("use_sections"))
    spec_encoder: SpecEncoder | None = None
    section_head: SectionQueryHead | None = None
    readout_head: "SectionReadoutHead | None" = None  # [section_readout]
    section_id_to_idx: dict[int, int] = {}
    if use_sections:
        # Spec encoder is only needed if patches aren't precomputed in the .pt files.
        # `beats_cached: true` (default) → SpecEncoder is None at train time and we
        # read patches from the batch.
        if not config.get("beats_cached", True):
            spec_encoder = SpecEncoder(
                model_name=config.get("spec_encoder_name", "beats"),
                checkpoint_name=config.get("spec_checkpoint_name", "BEATs_iter3_plus_AS2M.pt"),
                checkpoint_path=config.get("spec_checkpoint_path"),
                freeze=config.get("spec_freeze", True),
            ).to(device)
            print(f"[sections] online SpecEncoder: {config.get('spec_encoder_name', 'beats')}")
            # [encoder-unfreeze] Optionally fine-tune the top-N BEATs blocks. The
            # SpecEncoder wraps the backbone at spec_encoder.model; its transformer
            # stack lives at .model.encoder.layers. n=0 (default) is a no-op and the
            # encoder stays frozen exactly as spec_freeze set it.
            _unfreeze_beats_n = int(config.get("unfreeze_beats_top_n", 0) or 0)
            if _unfreeze_beats_n > 0:
                _bp = unfreeze_top_n_blocks(spec_encoder.model, _unfreeze_beats_n)
                encoder_unfreeze_params.extend(_bp)
                spec_encoder.freeze = False  # let SpecEncoder.forward build the graph
                print(f"[encoder-unfreeze] BEATs: unfroze top "
                      f"{min(_unfreeze_beats_n, count_blocks(spec_encoder.model))}/"
                      f"{count_blocks(spec_encoder.model)} blocks "
                      f"({sum(p.numel() for p in _bp)/1e6:.1f}M params trainable) at "
                      f"lr_encoder={config.get('lr_encoder')}.")
        else:
            print(f"[sections] using precomputed BEATs patches from .pt files (beats_cached=true)")
            if int(config.get("unfreeze_beats_top_n", 0) or 0) > 0:
                print("[encoder-unfreeze] WARNING: unfreeze_beats_top_n > 0 but "
                      "beats_cached=true — patches are precomputed, so the BEATs "
                      "encoder is never run at train time and unfreezing is a no-op. "
                      "Set beats_cached: false to run BEATs online.")

        # SectionQueryHead's d_patch must match whatever K/V come from. For BEATs
        # (default) this is 768. Override via config if using AST or a fine-tuned
        # variant.
        d_patch = int(config.get("spec_d_patch", 768))
        section_head = SectionQueryHead(
            n_sections=N_SECTIONS, d_patch=d_patch, d_lm=lm_hidden_size,
            d_k=int(config.get("section_d_k", 256)),
            d_v=int(config.get("section_d_v", 256)),
        ).to(device).to(torch.bfloat16)

        # [section_readout] Optional grounding head. Reads each section's acoustic
        # scalar out of z = alpha · V (V detached) and regresses it against Praat
        # GT, applying a gradient that forces alpha onto evidence-bearing patches.
        # Kept in float32 (NOT bf16) for regression precision; off when
        # lambda_readout == 0 (default), so zero overhead on legacy runs.
        if float(config.get("lambda_readout", 0.0)) > 0.0:
            readout_head = SectionReadoutHead(
                d_v=int(config.get("section_d_v", 256)),
                huber_delta=float(config.get("readout_huber_delta", 1.0)),
            ).to(device)
            print(f"[section_readout] enabled: lambda_readout="
                  f"{config.get('lambda_readout')}, d_v={config.get('section_d_v', 256)}")

        # Build the section_id → section_idx mapping. Must run AFTER the tokenizer
        # has section tokens registered — done in _register_feature_tags above
        # via TAG_SPECIAL_TOKENS which now contains both <f_*> and <sec_*> tags.
        sec_name_to_id = section_open_token_ids(tokenizer)
        section_id_to_idx = {sec_name_to_id[s.name]: i for i, s in enumerate(SECTION_TAGS)}
        sq_mode = config.get("section_query_mode", "static").lower()
        if sq_mode not in {"static", "dynamic"}:
            raise ValueError(f"section_query_mode must be 'static' or 'dynamic', got {sq_mode!r}")
        # <r> open token id — used by dynamic-mode injection so the per-range
        # queries inside multi-value spans also get training signal.
        range_open_id_train = tokenizer.convert_tokens_to_ids(RANGE_OPEN_TAG)
        print(f"[sections] {N_SECTIONS} sections registered; "
              f"query_mode={sq_mode}; ids {list(section_id_to_idx.keys())}; "
              f"range_open_id={range_open_id_train}")

    # [decoupled_grounding] Token-free 2D grounding head — INDEPENDENT of use_sections.
    # Produces per-feature attention maps over the BEATs T*F patches via learned
    # per-feature queries (nn.Parameter, NOT vocab tokens), so the LM generates clean
    # untagged prose and never emits a special token. Needs the precomputed BEATs
    # patches in the .pt files (beats_cached: true) so the dataloader carries
    # beats_patches; off by default (decoupled_grounding: false → zero overhead).
    decoupled_head: "DecoupledGroundingHead | None" = None
    if config.get("decoupled_grounding", False):
        if not config.get("beats_cached", False):
            print("[decoupled_grounding] WARNING: decoupled_grounding=true but "
                  "beats_cached is false — the dataloader will not carry beats_patches "
                  "and the grounding term will be a silent no-op. Set beats_cached: true.")
        # Optional per-feature readout bias init = each feature's prior MEAN, in the
        # SAME raw scalar units the grounding loss regresses (gt_scalars). Seeds each
        # per-feature readout's output bias to that mean so a high-mean feature
        # (f0_mean ~165) doesn't start pinned at 0. Pass via config as a list in
        # SUPERVISED_FEATURES order; None → zeros (prior behavior). Compute it from
        # the train split's per-feature means (over the present/measured entries).
        feature_init_bias = config.get("decoupled_feature_init_bias")  # list[float] | None
        if feature_init_bias is not None and len(feature_init_bias) != N_FEATURES:
            raise ValueError(
                f"decoupled_feature_init_bias must have {N_FEATURES} entries "
                f"(SUPERVISED_FEATURES order), got {len(feature_init_bias)}"
            )
        # [bottleneck] grounding_mode selects v17 softmax (default) vs v18 bottleneck
        # (hard-concrete keep-mask + noise substitution + bits penalty). The v18 head
        # is the SAME size; the bottleneck path is additive and config-gated, so
        # softmax/v17 stays bit-identical when grounding_mode is "softmax" or omitted.
        grounding_mode = str(config.get("grounding_mode", "softmax"))
        # Per-feature bits-penalty β (dict {short_name: float} | list | None). None →
        # the head's catalog defaults (β=0 for the 5 global feats, 0.02 pauses, 0.05
        # overlap_ratio). Only consumed in bottleneck mode.
        bits_beta = config.get("bits_beta_per_feature")
        concrete_temp_start = float(config.get("concrete_temp_start",
                                               config.get("concrete_temp", 1.0)))
        # Kept in float32 (NOT bf16) for regression precision, like SectionReadoutHead.
        decoupled_head = DecoupledGroundingHead(
            d_model=int(config.get("decoupled_d_model", 256)),
            d_patch=int(config.get("spec_d_patch", 768)),
            n_features=N_FEATURES,
            n_heads=int(config.get("decoupled_n_heads", 1)),
            readout_hidden=config.get("decoupled_readout_hidden"),  # None → linear bottleneck
            huber_delta=float(config.get("decoupled_huber_delta", 1.0)),
            feature_init_bias=feature_init_bias,  # None → zeros; else per-feature raw means
            grounding_mode=grounding_mode,
            bits_beta_per_feature=bits_beta,
            concrete_temp=concrete_temp_start,
        ).to(device)
        print(f"[decoupled_grounding] enabled: grounding_mode={grounding_mode}, "
              f"lambda_decoupled={config.get('lambda_decoupled', 0.0)}, "
              f"d_model={config.get('decoupled_d_model', 256)}, "
              f"d_patch={config.get('spec_d_patch', 768)}, n_features={N_FEATURES}, "
              f"feature_init_bias={'set' if feature_init_bias is not None else 'zeros'}")
        if grounding_mode == "bottleneck":
            print(f"[decoupled_grounding] bottleneck: beta={decoupled_head.bits_beta.tolist()}, "
                  f"concrete_temp={concrete_temp_start}→{config.get('concrete_temp_end', 0.3)}, "
                  f"bits_warmup_epochs={config.get('bits_warmup_epochs', 0)}, "
                  f"bits_lambda={config.get('bits_lambda', 1.0)}")
        # [overlap-map supervision] direct segmentation loss on the overlap query's map.
        _lambda_ovl_map = float(config.get("lambda_overlap_map", 0.0))
        if _lambda_ovl_map > 0.0:
            print(f"[decoupled_grounding] overlap-map supervision ON: "
                  f"lambda_overlap_map={_lambda_ovl_map}, "
                  f"reduce={config.get('overlap_map_reduce', 'mean')}, "
                  f"empty_weight={config.get('overlap_map_empty_weight', 1.0)}, "
                  f"soft_target={config.get('overlap_target_soft', True)} "
                  f"(supervises feature '{feature_names()[overlap_ratio_index()]}' "
                  f"idx {overlap_ratio_index()})")

    # [snr_map] Optional SUPERVISED dense local-SNR-map head (CBM-style, Koh 2007.04612).
    # Regresses the per-frame SNR timeline off the WavLM frames against the stem-derived
    # oracle target. Built ONLY when lambda_snr_map > 0, so off-by-default is byte-
    # identical (no head, no params, loss term no-ops). float32 for regression precision.
    snr_map_head: "SupervisedSNRMapHead | None" = None
    if float(config.get("lambda_snr_map", 0.0)) > 0.0:
        snr_map_head = SupervisedSNRMapHead(
            audio_dim=int(config.get("snr_map_audio_dim", 1024)),
            d_patch=int(config.get("spec_d_patch", 768)),
            f_bins=int(config.get("snr_map_f_bins", 8)),
            hidden=int(config.get("snr_map_hidden", 256)),
            kernel_size=int(config.get("snr_map_kernel_size", 5)),
            predict_irm=bool(config.get("snr_map_predict_irm", False)),
            huber_delta=float(config.get("snr_map_huber_delta", 1.0)),
            snr_bias=float(config.get("snr_map_snr_bias", 0.0)),
        ).to(device)
        print(f"[snr_map] enabled: lambda_snr_map={config.get('lambda_snr_map')}, "
              f"hidden={config.get('snr_map_hidden', 256)}, "
              f"kernel={config.get('snr_map_kernel_size', 5)}, "
              f"predict_irm={config.get('snr_map_predict_irm', False)}, "
              f"lambda_snr_scalar={config.get('lambda_snr_scalar', 0.0)}, "
              f"lambda_snr_irm={config.get('lambda_snr_irm', 0.0)}")
        if not config.get("snr_map_dir"):
            print("[snr_map] WARNING: lambda_snr_map>0 but snr_map_dir is unset — the "
                  "dataloader will carry no dense targets and the term will no-op. Set "
                  "snr_map_dir to the compute_snr_map_targets.py output.")

    # [srmr_map] Optional SUPERVISED 2D SRMR modulation-energy-map head (the TRUE-2D
    # grounding branch; Falk et al. TASLP 2010). Regresses the 23x8 (acoustic x
    # modulation) log-energy map off the WavLM frames against the oracle clean-s1 SRMR
    # tensor. Built ONLY when lambda_srmr_map > 0, so off-by-default is byte-identical
    # (no head, no params, loss term no-ops). float32 for regression precision.
    srmr_map_head: "SupervisedSRMRMapHead | None" = None
    if float(config.get("lambda_srmr_map", 0.0)) > 0.0:
        srmr_map_head = SupervisedSRMRMapHead(
            audio_dim=int(config.get("srmr_map_audio_dim", 1024)),
            n_acoustic=int(config.get("srmr_map_n_acoustic", 23)),
            n_modulation=int(config.get("srmr_map_n_modulation", 8)),
            hidden=int(config.get("srmr_map_hidden", 256)),
            huber_delta=float(config.get("srmr_map_huber_delta", 1.0)),
        ).to(device)
        print(f"[srmr_map] enabled: lambda_srmr_map={config.get('lambda_srmr_map')}, "
              f"map={config.get('srmr_map_n_acoustic', 23)}x{config.get('srmr_map_n_modulation', 8)}, "
              f"hidden={config.get('srmr_map_hidden', 256)}")
        if not config.get("srmr_map_dir"):
            print("[srmr_map] WARNING: lambda_srmr_map>0 but srmr_map_dir is unset — the "
                  "dataloader will carry no 2D targets and the term will no-op. Set "
                  "srmr_map_dir to the compute_srmr_maps.py output.")

    # Trainable-parameter summary — printed once at startup and stashed for wandb.run.summary
    # after wandb.init() below. Helps compare adapter vs LoRA/full-FT footprint across runs.
    lm_total = sum(p.numel() for p in llm.parameters())
    lm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    # [encoder-unfreeze] unfrozen top-N SSL-encoder blocks (0 in the default frozen path).
    encoder_trainable = sum(p.numel() for p in encoder_unfreeze_params)
    trainable_total = lm_trainable + adapter_trainable + encoder_trainable
    param_summary = {
        "params/lm_total": lm_total,
        "params/lm_trainable": lm_trainable,         # full-FT: lm_total; LoRA: small subset
        "params/lora_trainable": lm_trainable if not full_ft else 0,  # legacy compat key
        "params/adapter_trainable": adapter_trainable,
        "params/encoder_trainable": encoder_trainable,  # [encoder-unfreeze]
        "params/trainable_total": trainable_total,
        "params/trainable_pct_of_lm": 100.0 * trainable_total / lm_total,
        "params/full_ft": int(full_ft),
        "params/tagged_mode": int(bool(config.get("tagged_mode"))),
        "params/n_added_tokens": n_added_tokens,
    }
    mode = "full-FT" if full_ft else "LoRA"
    print(
        f"Parameters  —  LM total: {lm_total/1e9:.2f}B  |  {mode} trainable: {lm_trainable/1e6:.1f}M  "
        f"|  adapter trainable: {adapter_trainable/1e6:.1f}M  "
        f"|  grand total trainable: {trainable_total/1e6:.1f}M "
        f"({param_summary['params/trainable_pct_of_lm']:.3f}% of LM)"
    )

    # Prompt
    # Prompt for the prose target / inference. Falls back to legacy `prompt` field if
    # `prompt_prose` isn't set.
    prose_prompt_str = config.get("prompt_prose") or config["prompt"]
    prompt_ids = tokenizer(prose_prompt_str, return_tensors="pt").input_ids.to(device)
    print(f"[prompt-prose] {prose_prompt_str!r}")

    # Optional separate prompt for B-full's numbers target. When set, the LM learns
    # bare-numbers as a completion of THIS prompt only — at inference (prose prompt
    # only) the LM produces prose-only output. Without this, both completions share
    # one prompt key and inference outputs a numbers-then-prose mix.
    prompt_nums_ids = None
    if config.get("prompt_nums"):
        prompt_nums_ids = tokenizer(config["prompt_nums"], return_tensors="pt").input_ids.to(device)
        print(f"[prompt-nums]  {config['prompt_nums']!r}")
    else:
        print(f"[prompt-nums]  not set — bare-numbers target will use the prose prompt "
              f"(legacy single-prompt B-full; inference will mix formats).")

    # Dataset + Dataloader
    data_dir = config["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
    assert os.path.isdir(val_dir), f"Val directory not found: {val_dir}"
    assert any(f.endswith(".pt") for f in os.listdir(train_dir)), (
        f"No .pt files in {train_dir}. Run preprocess.py first."
    )
    assert any(f.endswith(".pt") for f in os.listdir(val_dir)), f"No .pt files in {val_dir}. Run preprocess.py first."

    if not os.path.isdir(test_dir):
        print("Warning: no test/ directory found. Run inference.py separately for test evaluation.")

    # B-full needs a features CSV per split for the bare-numbers target + aux-head GT.
    # config["features_csv"] is a {split: path} dict OR a single path used for all splits.
    features_csv = config.get("features_csv")
    if isinstance(features_csv, dict):
        train_csv = features_csv.get("train") or features_csv.get("train-100")
        val_csv = features_csv.get("val") or features_csv.get("dev")
    else:
        train_csv = val_csv = features_csv

    # [snr_map] optional oracle dense local-SNR-map target dirs (build-A), per split.
    # {split: path} dict or a single path; None → no dense targets → snr_map_loss_term
    # no-ops (default-off byte-identical).
    snr_map_dir = config.get("snr_map_dir")
    if isinstance(snr_map_dir, dict):
        train_snr_dir = snr_map_dir.get("train") or snr_map_dir.get("train-100")
        val_snr_dir = snr_map_dir.get("val") or snr_map_dir.get("dev")
    else:
        train_snr_dir = val_snr_dir = snr_map_dir

    # [srmr_map] optional oracle 2D SRMR-modulation-map target dirs (build-A), per split.
    # Same {split: path} dict or single path convention; None → no targets → no-op.
    srmr_map_dir = config.get("srmr_map_dir")
    if isinstance(srmr_map_dir, dict):
        train_srmr_dir = srmr_map_dir.get("train") or srmr_map_dir.get("train-100")
        val_srmr_dir = srmr_map_dir.get("val") or srmr_map_dir.get("dev")
    else:
        train_srmr_dir = val_srmr_dir = srmr_map_dir

    train_set = PreprocessedDataset(
        train_dir, config["descriptions_path"], features_csv=train_csv,
        snr_map_dir=train_snr_dir, srmr_map_dir=train_srmr_dir,
    )
    # The dataset keys descriptions by clip stem and looks up the CURRENT split's
    # stems in whatever JSON it is handed. The legacy single combined JSON (e.g.
    # descriptions_observability_all.json) contains every split's stems, so one
    # `descriptions_path` resolves both train and val. The NEW canonical builder
    # writes SPLIT-LOCAL files (descriptions_canonical_{train,dev,test}.json), so the
    # train file does NOT contain val stems — val target_text would silently never
    # resolve. `descriptions_path_val` (optional) lets a split-local setup point the
    # val dataset at the dev JSON. Absent → falls back to descriptions_path, so every
    # existing combined-JSON config is byte-identical.
    val_descriptions_path = config.get("descriptions_path_val") or config["descriptions_path"]
    val_set = PreprocessedDataset(
        val_dir, val_descriptions_path, features_csv=val_csv,
        snr_map_dir=val_snr_dir, srmr_map_dir=val_srmr_dir,
    )
    assert train_set.descriptions is not None, f"Descriptions not found: {config['descriptions_path']}"
    print(f"Loaded: train={len(train_set)}, val={len(val_set)}")
    if train_csv:
        n_with_csv = len(train_set.feature_csv_map)
        print(f"  features CSV (train): {train_csv} → {n_with_csv} clip rows")
    else:
        print("  features CSV: not provided — B-full forward A and aux-head MSE will be skipped.")

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_fn,
    )

    # Optimizer
    # `lr_lm` is the canonical key going forward (covers both full-FT and LoRA).
    # `lr_lora` stays supported as a fallback so existing PSC configs keep working.
    lr_lm = float(config.get("lr_lm") or config.get("lr_lora"))
    # New-token embed/lm_head rows go in their OWN group with weight_decay=0 so
    # decoupled AdamW decay can't drift the (gradient-masked) frozen rows. They
    # must also be excluded from the main LM group to avoid double-registration.
    nt_ids = {id(p) for p in new_token_params}
    param_groups = [
        {"params": adapter.parameters(), "lr": config["lr_adapter"]},
        {"params": [p for p in llm.parameters()
                    if p.requires_grad and id(p) not in nt_ids], "lr": lr_lm},
    ]
    if new_token_params:
        param_groups.append({
            "params": new_token_params, "lr": lr_lm, "weight_decay": 0.0,
        })
    if section_head is not None:
        # Section-query head uses the adapter LR — it's a small new module similar
        # in size to the adapter and learns from CE gradient.
        param_groups.append({"params": section_head.parameters(), "lr": config["lr_adapter"]})
    if readout_head is not None:
        # [section_readout] grounding head, also at adapter LR.
        param_groups.append({"params": readout_head.parameters(), "lr": config["lr_adapter"]})
    if decoupled_head is not None:
        # [decoupled_grounding] token-free 2D grounding head, also at adapter LR.
        # Its queries / projections / readout train on the parallel grounding loss.
        param_groups.append({"params": decoupled_head.parameters(), "lr": config["lr_adapter"]})
    if snr_map_head is not None:
        # [snr_map] dense local-SNR-map head, also at adapter LR. Its conv/proj train
        # on the parallel dense Huber against the stem-derived oracle target.
        param_groups.append({"params": snr_map_head.parameters(), "lr": config["lr_adapter"]})
    if srmr_map_head is not None:
        # [srmr_map] 2D SRMR-modulation-map head, also at adapter LR. Its MLP trains on
        # the parallel masked Huber against the clean-s1 oracle 23x8 log-energy tensor.
        param_groups.append({"params": srmr_map_head.parameters(), "lr": config["lr_adapter"]})
    if encoder_unfreeze_params:
        # [encoder-unfreeze] the unfrozen top-N SSL-encoder blocks get their OWN group
        # at lr_encoder — a small LR (e.g. 1e-5) distinct from the adapter LR so
        # fine-tuning the pretrained frontend doesn't blow away its features. Falls
        # back to lr_lm if lr_encoder isn't set. Dedupe by id against the LM group is
        # unnecessary (WavLM/BEATs params are disjoint from the LM), but the params
        # are a fresh disjoint set regardless.
        lr_encoder = float(config.get("lr_encoder") or config.get("lr_lm") or lr_lm)
        param_groups.append({"params": encoder_unfreeze_params, "lr": lr_encoder})
        print(f"[encoder-unfreeze] {sum(p.numel() for p in encoder_unfreeze_params)/1e6:.1f}M "
              f"encoder params in their own optimizer group at lr={lr_encoder}")
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config["weight_decay"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
    )

    # Scheduler
    accum_steps = config["gradient_accumulation_steps"]
    steps_per_epoch = -(-len(train_loader) // accum_steps)  # ceil division
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config["min_lr_ratio"] * config["lr_adapter"],
    )
    # start_factor scales all param groups uniformly — the LoRA group
    # (lr=2e-5) starts proportionally lower than the adapter group (lr=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8 / config["lr_adapter"],
        total_iters=warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_steps],
    )

    # Resume from checkpoint
    # Best-checkpoint metric: val_sfs_f1 (higher is better). val_loss is dominated by
    # prose CE at its entropy floor and BLEU/ROUGE plateau ~3-8 epochs before SFS_F1
    # — using val_loss for early stopping cuts off digit grounding prematurely.
    start_epoch = 0
    best_val_sfs_f1 = float("-inf")
    # Running max of clean-epoch BLEU; the degeneration guard uses it as a
    # RELATIVE floor so a fluency collapse (high SFS, low BLEU) can't be selected.
    best_val_bleu = None
    wandb_run_id = None

    # ── Band-free composite selection state (research Q1 protocol) ────────────
    # select_metric chooses the best.pt axis. 'composite' (default) selects on an
    # EMA-smoothed continuous SRCC/nMAE composite with a hard BLEU floor; 'sfs_f1'
    # reproduces the legacy degeneration-gated SFS-F1 argmax BYTE-IDENTICALLY.
    select_metric = config.get("select_metric", "composite")
    lam_nmae = float(config.get("lam_nmae", 0.5))
    bleu_floor = config.get("bleu_floor", 5.0)
    val_select_ema_beta = float(config.get("val_select_ema_beta", 0.7))
    best_val_composite = float("-inf")   # best EMA-smoothed composite seen so far
    composite_ema = None                  # running EMA of the raw composite

    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], weights_only=False)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        # New checkpoints use "llm_state_dict"; pre-2026-05-11 ones used "lora_state_dict".
        # SLIM ckpts (ckpt_format="peft_slim") carry only LoRA + unfrozen rows — loaded
        # with strict=False over the already-restored frozen base. Old FAT ckpts carry
        # the full base — auto-detected and loaded strict. See src/ckpt_io.py.
        llm_sd = checkpoint.get("llm_state_dict") or checkpoint["lora_state_dict"]
        _missing, _unexpected = load_llm_state_dict(
            llm, llm_sd, ckpt_format=checkpoint.get("ckpt_format"),
        )
        if _unexpected:
            raise RuntimeError(f"Unexpected keys loading LLM checkpoint: {_unexpected[:5]} ...")
        if section_head is not None and "section_head_state_dict" in checkpoint:
            section_head.load_state_dict(checkpoint["section_head_state_dict"])
        if readout_head is not None and "readout_head_state_dict" in checkpoint:  # [section_readout]
            readout_head.load_state_dict(checkpoint["readout_head_state_dict"])
        if decoupled_head is not None and "decoupled_head_state_dict" in checkpoint:  # [decoupled_grounding]
            decoupled_head.load_state_dict(checkpoint["decoupled_head_state_dict"])
        if snr_map_head is not None and "snr_map_head_state_dict" in checkpoint:  # [snr_map]
            snr_map_head.load_state_dict(checkpoint["snr_map_head_state_dict"])
        if srmr_map_head is not None and "srmr_map_head_state_dict" in checkpoint:  # [srmr_map]
            srmr_map_head.load_state_dict(checkpoint["srmr_map_head_state_dict"])
        # best.pt no longer carries optimizer/scheduler (inference-only). Resuming is
        # meant to use last.pt, which does. Guard so resuming from a best.pt (or any
        # optimizer-less ckpt) doesn't KeyError — it just starts fresh optimizer state.
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("  [resume] no optimizer_state_dict in checkpoint (best.pt?) — "
                  "starting optimizer from scratch. Resume from last.pt for exact continuation.")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # Backwards-compat: old checkpoints stored "best_val_loss"; new ones store "best_val_sfs_f1".
        if "best_val_sfs_f1" in checkpoint:
            best_val_sfs_f1 = checkpoint["best_val_sfs_f1"]
        # Composite-selection state (absent in pre-band-free ckpts → cold start).
        if "best_val_composite" in checkpoint:
            best_val_composite = checkpoint["best_val_composite"]
        if checkpoint.get("composite_ema") is not None:
            composite_ema = checkpoint["composite_ema"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Resumed from epoch {start_epoch}, best_val_sfs_f1={best_val_sfs_f1:.4f}, "
              f"best_val_composite={best_val_composite:.4f}")

    # Wandb. `wandb_entity` in the YAML pins the team account so contributors
    # don't have to remember to export WANDB_ENTITY every session. Passing
    # `entity=None` lets wandb fall back to its own resolution (env var, then
    # the default entity from `wandb login`'s ~/.netrc, then personal account).
    run_name = config.get("wandb_run_name") or f"{config['adapter_variant']}-seed{config['seed']}"
    wandb_entity = config.get("wandb_entity")
    if wandb_run_id:
        wandb.init(
            project=config["wandb_project"],
            entity=wandb_entity,
            id=wandb_run_id,
            resume="must",
            config=config,
        )
    else:
        wandb.init(
            project=config["wandb_project"],
            entity=wandb_entity,
            name=run_name,
            config=config,
        )
    wandb_run_id = wandb.run.id

    # Push param counts into the wandb run summary so the run overview shows them without scrolling logs.
    for k, v in param_summary.items():
        wandb.run.summary[k] = v

    # Training loop
    os.makedirs(config["save_dir"], exist_ok=True)

    max_steps = int(config["max_steps"]) if config.get("max_steps") else None

    # ── Per-epoch val SFS subset (FIX 2) ──────────────────────────────────────
    # val_subset_size clips are scored every epoch for val SFS. 32 was too noisy
    # to rank epochs / select best.pt (bootstrap CI ±0.06); 256 roughly halves
    # the noise. The subset is FIXED across epochs (same seed, same clips) so
    # epoch-to-epoch SFS deltas are paired/comparable. When the val features CSV
    # exposes overlap_ratio, the subset is STRATIFIED across low/med/high overlap
    # bins so it stays representative; otherwise it's a seeded uniform draw.
    # COST: ~val_subset_size greedy generations per epoch (≈256 × ~0.3-1 s on an
    # 8B LoRA on PSC ⇒ a few minutes/epoch). Backward-compat: if val_subset_size
    # is absent, fall back to the legacy val_sfs_n; if that is ALSO absent, default
    # to 200 — the research Q1 lower bound (150-250 clips) for a stable SRCC
    # (Schonbrodt & Perugini 2013; Bonett & Wright 2000). A config that explicitly
    # set val_sfs_n keeps its value, so this only changes brand-new runs.
    val_subset_size = int(config.get("val_subset_size", config.get("val_sfs_n", 200)))
    val_strata = overlap_strata_from_csv_map(val_set.files, val_set.feature_csv_map)
    if val_strata is not None:
        from collections import Counter as _Counter
        print(f"[val-subset] size={val_subset_size}, stratified by overlap bin: "
              f"{dict(_Counter(val_strata))}")
    else:
        print(f"[val-subset] size={val_subset_size}, seeded uniform "
              f"(no overlap_ratio in val features CSV)")

    # Readout-grounding warmup: the readout gradient destabilizes generation in
    # dynamic mode (v12: lambda 0.5 -> 31% degenerate). Ramp it in over the first
    # few epochs so the LM stabilizes first. Target stored once; the effective
    # lambda_readout is overwritten per-epoch below.
    _readout_target = float(config.get("lambda_readout", 0.0))
    _readout_warmup = int(config.get("lambda_readout_warmup_epochs", 0))

    # [bottleneck] bits-penalty warmup + concrete-temperature anneal. The bits term
    # is β=0 for the first `bits_warmup_epochs` (the readout learns to predict from
    # the FULL representation first), then the GLOBAL bits weight ramps in like
    # lambda_readout; concrete_temp anneals linearly start→end so the keep-mask
    # sharpens toward 0/1 by the end of training. Only active in bottleneck mode.
    _is_bottleneck = (
        decoupled_head is not None
        and getattr(decoupled_head, "grounding_mode", "softmax") == "bottleneck"
    )
    _bits_lambda_target = float(config.get("bits_lambda", 1.0))
    _bits_warmup = int(config.get("bits_warmup_epochs", 0))
    _temp_start = float(config.get("concrete_temp_start", config.get("concrete_temp", 1.0)))
    _temp_end = float(config.get("concrete_temp_end", _temp_start))
    _n_epochs = int(config["epochs"])

    for epoch in range(start_epoch, config["epochs"]):
        if _readout_target > 0.0:
            config["lambda_readout"] = warmup_lambda(_readout_target, epoch, _readout_warmup)
            print(f"[section_readout] epoch {epoch+1}: effective lambda_readout="
                  f"{config['lambda_readout']:.4f} (target {_readout_target}, "
                  f"warmup {_readout_warmup} epochs)")
        if _is_bottleneck:
            # bits weight: 0 for epochs < bits_warmup_epochs, then warmup_lambda from
            # (epoch - bits_warmup_epochs) over the remaining epochs.
            if epoch < _bits_warmup:
                config["bits_lambda_effective"] = 0.0
            else:
                ramp = max(1, _n_epochs - _bits_warmup)
                config["bits_lambda_effective"] = warmup_lambda(
                    _bits_lambda_target, epoch - _bits_warmup, ramp,
                )
            # concrete temperature anneal (linear start→end across all epochs).
            frac = 0.0 if _n_epochs <= 1 else min(1.0, max(0, epoch) / (_n_epochs - 1))
            cur_temp = _temp_start + (_temp_end - _temp_start) * frac
            decoupled_head.set_concrete_temp(cur_temp)
            print(f"[bottleneck] epoch {epoch+1}: bits_lambda_effective="
                  f"{config['bits_lambda_effective']:.4f} (target {_bits_lambda_target}, "
                  f"warmup {_bits_warmup} epochs), concrete_temp={cur_temp:.4f}")
        print("\nEpoch: {}/{}".format(epoch + 1, config["epochs"]))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        # ── Train ──
        adapter.train()
        llm.train()
        train_loss = 0.0
        n_steps = 0

        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for batch_idx, batch in enumerate(train_loader):
            section_ctx = _build_section_ctx(
                section_head, spec_encoder, section_id_to_idx, batch, device,
                mode=sq_mode, range_open_id=range_open_id_train,
                readout_head=readout_head,  # [section_readout]
            ) if section_head is not None else None

            loss, loss_metrics = compute_loss(
                adapter,
                llm,
                embed_layer,
                tokenizer,
                batch["audio_features"],
                batch["overlap_info"],
                batch["target_text"],
                prompt_ids,
                device,
                config,
                target_nums=batch.get("target_nums"),
                gt_scalars=batch.get("gt_scalars"),
                gt_mask=batch.get("gt_mask"),
                prompt_nums_ids=prompt_nums_ids,
                section_ctx=section_ctx,
                decoupled_head=decoupled_head,   # [decoupled_grounding]
                snr_map_head=snr_map_head,       # [snr_map] dense local-SNR-map head
                srmr_map_head=srmr_map_head,     # [srmr_map] 2D SRMR-modulation-map head
                batch=batch,                     # heads read audio_features/beats/gt off it
            )
            loss = loss / accum_steps
            # Defensive NaN/Inf guard. Without this, one bad batch (e.g., a
            # bf16 numerical issue in the section_head cross-attention, an
            # all-padded BEATs row attended to -inf, an exploded W_o output
            # before LayerNorm hits) would propagate NaN through
            # optimizer.step() and silently corrupt every weight thereafter.
            # Symptom would be exactly: train_loss looks fine for one more
            # step then val collapses to 0 permanently.
            if not torch.isfinite(loss):
                print(f"[WARN] non-finite loss at step {n_steps} "
                      f"(batch {batch_idx}, epoch {epoch+1}); "
                      f"skipping this step")
                optimizer.zero_grad()
                continue
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), config["grad_clip"])
                torch.nn.utils.clip_grad_norm_(llm.parameters(), config["grad_clip"])
                # section_head must be clipped too. In dynamic mode it sits at
                # the end of a long backward chain (loss -> pass-2 LM ->
                # target_embeds -> e_t -> W_q -> pass-1 LM); accumulated unclipped
                # gradients let W_q drift to large magnitudes after a few epochs,
                # which collapses the attention softmax and makes the LM
                # memorize a degenerate query distribution. Symptom: train_loss
                # drops smoothly while every val metric regresses by epoch 4.
                if section_head is not None:
                    torch.nn.utils.clip_grad_norm_(section_head.parameters(), config["grad_clip"])
                if readout_head is not None:  # [section_readout]
                    torch.nn.utils.clip_grad_norm_(readout_head.parameters(), config["grad_clip"])
                if decoupled_head is not None:  # [decoupled_grounding] clip the grounding head too
                    torch.nn.utils.clip_grad_norm_(decoupled_head.parameters(), config["grad_clip"])
                if snr_map_head is not None:  # [snr_map] clip the dense-map head too
                    torch.nn.utils.clip_grad_norm_(snr_map_head.parameters(), config["grad_clip"])
                if srmr_map_head is not None:  # [srmr_map] clip the 2D SRMR-map head too
                    torch.nn.utils.clip_grad_norm_(srmr_map_head.parameters(), config["grad_clip"])
                if encoder_unfreeze_params:  # [encoder-unfreeze] clip the unfrozen encoder blocks
                    torch.nn.utils.clip_grad_norm_(encoder_unfreeze_params, config["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            n_steps += 1

            if max_steps is not None and (n_steps + epoch * len(train_loader)) >= max_steps:
                print(f"[max_steps] reached {max_steps} steps — stopping smoke run.")
                sys.exit(0)

            if n_steps % config["log_every"] == 0:
                log_payload = {
                    "train_loss": loss.item() * accum_steps,
                    "train_loss_lm_prose": loss_metrics["loss_lm_prose"],
                    "train_loss_lm_nums": loss_metrics["loss_lm_nums"],
                    "train_loss_mse": loss_metrics["loss_mse"],
                    "lr": scheduler.get_last_lr()[0],
                    "step": n_steps + epoch * len(train_loader),
                }
                # [ntl] surface the Number Token Loss when active (0 / absent when
                # lambda_ntl <= 0). THE signal for whether digit accuracy is being
                # supervised ordinally rather than nominally.
                if loss_metrics.get("loss_ntl", 0.0):
                    log_payload["train_loss_ntl"] = loss_metrics["loss_ntl"]
                    log_payload["lambda_ntl"] = float(config.get("lambda_ntl", 0.0))
                # [section_readout] surface the grounding loss + per-feature MAE
                # (computed in compute_loss but otherwise discarded). These are
                # THE signal for whether attention is becoming grounded.
                if "loss_readout" in loss_metrics:
                    log_payload["train_loss_readout"] = loss_metrics["loss_readout"]
                    log_payload["lambda_readout"] = float(config.get("lambda_readout", 0.0))
                    for k, v in loss_metrics.items():
                        if k.startswith("readout_mae/"):
                            log_payload[f"train_{k}"] = v
                # [decoupled_grounding] surface the parallel grounding loss + per-feature
                # MAE — THE signal for whether the token-free maps are getting grounded.
                if "loss_decoupled" in loss_metrics:
                    log_payload["train_loss_decoupled"] = loss_metrics["loss_decoupled"]
                    log_payload["lambda_decoupled"] = float(config.get("lambda_decoupled", 0.0))
                    for k, v in loss_metrics.items():
                        if k.startswith("decoupled_mae/"):
                            log_payload[f"train_{k}"] = v
                # [bottleneck] surface the bits penalty + per-feature meanbits — THE
                # signal for whether the keep-mask is sparsifying onto evidence.
                if "loss_bits" in loss_metrics:
                    log_payload["train_loss_bits"] = loss_metrics["loss_bits"]
                    log_payload["bits_lambda_effective"] = float(
                        config.get("bits_lambda_effective", 0.0))
                    log_payload["concrete_temp"] = (
                        float(decoupled_head.concrete_temp)
                        if decoupled_head is not None else 0.0)
                    for k, v in loss_metrics.items():
                        if k.startswith("meanbits/"):
                            log_payload[f"train_{k}"] = v
                # [snr_map] surface the dense-map loss + MAE (+ optional scalar/IRM).
                if loss_metrics.get("loss_snr_map", 0.0):
                    log_payload["train_loss_snr_map"] = loss_metrics["loss_snr_map"]
                    log_payload["lambda_snr_map"] = float(config.get("lambda_snr_map", 0.0))
                    for k in ("snr_map_mae", "loss_snr_scalar", "snr_pooled_mae",
                              "loss_snr_irm", "snr_irm_mae"):
                        if k in loss_metrics:
                            log_payload[f"train_{k}"] = loss_metrics[k]
                # [srmr_map] surface the 2D SRMR-map loss + MAE.
                if loss_metrics.get("loss_srmr_map", 0.0):
                    log_payload["train_loss_srmr_map"] = loss_metrics["loss_srmr_map"]
                    log_payload["lambda_srmr_map"] = float(config.get("lambda_srmr_map", 0.0))
                    for k in ("srmr_map_mae", "srmr_map_n_bands"):
                        if k in loss_metrics:
                            log_payload[f"train_{k}"] = loss_metrics[k]
                wandb.log(log_payload)

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(train_loss / n_steps)),
                lr="{:.2e}".format(float(scheduler.get_last_lr()[0])))
            batch_bar.update()

        batch_bar.close()
        avg_train_loss = train_loss / n_steps

        # ── Validate ──
        avg_val_loss = None
        avg_sfs_f1 = None  # in scope for best-checkpoint check below; None on non-eval epochs
        val_gen_texts: list = []   # this epoch's val generations, for the degeneration guard
        val_bleu = None            # BLEU on those generations (None if not computed)
        avg_composite = None       # this epoch's raw band-free composite (None on non-eval epochs)
        if (epoch + 1) % config["eval_every_epoch"] == 0:
            adapter.eval()
            llm.eval()
            val_loss = 0.0
            n_val = 0

            batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

            val_metrics_sum = {"loss_lm_prose": 0.0, "loss_lm_nums": 0.0, "loss_mse": 0.0}
            # [section_readout] held-out grounding signal: loss + per-feature MAE.
            val_readout_loss_sum = 0.0
            val_readout_mae_sum: dict[str, float] = {}
            val_readout_n = 0
            # [decoupled_grounding] held-out parallel grounding signal: loss + MAE.
            val_decoupled_loss_sum = 0.0
            val_decoupled_mae_sum: dict[str, float] = {}
            val_decoupled_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Match the training forward path exactly: pass `mode=sq_mode`
                    # so val_loss is computed under the same section-query mode
                    # the model is being trained with. Without this, val_loss
                    # silently used mode="static" (the _build_section_ctx default)
                    # even when training was using mode="dynamic", making
                    # val_loss numbers an unreliable train/val gap signal.
                    section_ctx = _build_section_ctx(
                        section_head, spec_encoder, section_id_to_idx, batch, device,
                        mode=sq_mode, range_open_id=range_open_id_train,
                        readout_head=readout_head,  # [section_readout]
                    ) if section_head is not None else None

                    loss, loss_metrics = compute_loss(
                        adapter,
                        llm,
                        embed_layer,
                        tokenizer,
                        batch["audio_features"],
                        batch["overlap_info"],
                        batch["target_text"],
                        prompt_ids,
                        device,
                        config,
                        target_nums=batch.get("target_nums"),
                        gt_scalars=batch.get("gt_scalars"),
                        gt_mask=batch.get("gt_mask"),
                        # Without this the val nums-loss falls back to prompt_ids,
                        # silently using the prose prompt for the bare-numbers
                        # target. Aligns val with train.
                        prompt_nums_ids=prompt_nums_ids,
                        section_ctx=section_ctx,
                        decoupled_head=decoupled_head,   # [decoupled_grounding]
                        batch=batch,                     # [decoupled_grounding]
                    )
                    val_loss += loss.item()
                    for k in val_metrics_sum:
                        val_metrics_sum[k] += loss_metrics[k]
                    n_val += 1

                    # [section_readout] accumulate grounding metrics when present.
                    if "loss_readout" in loss_metrics:
                        val_readout_loss_sum += loss_metrics["loss_readout"]
                        for k, v in loss_metrics.items():
                            if k.startswith("readout_mae/"):
                                val_readout_mae_sum[k] = val_readout_mae_sum.get(k, 0.0) + v
                        val_readout_n += 1

                    # [decoupled_grounding] accumulate parallel grounding metrics
                    # when the head actually ran (loss_decoupled present AND nonzero
                    # — present-but-0.0 means the branch was off / had no patches).
                    if loss_metrics.get("loss_decoupled", 0.0) != 0.0 or any(
                        k.startswith("decoupled_mae/") for k in loss_metrics
                    ):
                        val_decoupled_loss_sum += loss_metrics.get("loss_decoupled", 0.0)
                        for k, v in loss_metrics.items():
                            if k.startswith("decoupled_mae/"):
                                val_decoupled_mae_sum[k] = val_decoupled_mae_sum.get(k, 0.0) + v
                        val_decoupled_n += 1

                    batch_bar.set_postfix(
                        loss="{:.04f}".format(float(val_loss / n_val)))
                    batch_bar.update()

            batch_bar.close()
            avg_val_loss = val_loss / n_val

            # ── Qualitative: generate text on a fixed slice of val, SFS-score, log to wandb ──
            # Catches degenerate outputs ("AND THE THE THE...") immediately and tracks SFS F1
            # epoch-by-epoch so you can see the faithfulness curve without waiting for inference.py.
            # HybridClaimParser auto-detects tagged spans (EMNLP rework) and falls back to the
            # legacy regex parser on untagged outputs, so a single trainer handles both modes.
            claim_parser = HybridClaimParser()
            sfs_scorer = SFSScorer()

            sample_rows = []
            sfs_f1s, sfs_precs, sfs_recs = [], [], []
            # ── Band-free selection accumulators (research Q1 protocol) ─────────
            # Full (untruncated) generations + filenames + per-clip clean GT, fed
            # to selection_metric.band_free_val_scores after the loop. clean_gt is
            # parsed from the verbalized target text (the same band-free GT source
            # metrics_calibrated uses), restricted to the canonical 12 features.
            bf_gen_texts: list[str] = []
            bf_filenames: list[str] = []
            bf_clean_gt: dict[str, dict[str, float]] = {}
            # val_subset_size (FIX 2) controls the val-time generation sample count.
            # Default 256 (was 32) — 32's bootstrap CI (±0.06) was too noisy to rank
            # epochs / select best.pt; 256 roughly halves the noise floor. Computed
            # once above (size + overlap strata); the seeded+stratified subset is the
            # SAME clips every epoch so cross-epoch SFS deltas are paired.
            val_idx = seeded_val_indices(
                len(val_set),
                val_subset_size,
                seed=config.get("val_sfs_seed", 1234),
                strata=val_strata,
            )
            n_samples = len(val_idx)

            # If use_sections=true, route val-time generation through inference.generate
            # so the section hook fires and attention maps are produced (they're
            # discarded here — we only need the text for SFS — but firing the hook
            # is what makes the section_head learn signal-consistent queries).
            inference_generate = None
            if section_head is not None:
                from inference import generate as inference_generate

            with torch.no_grad():
                for i in val_idx:
                    sample = val_set[i]

                    if inference_generate is not None and "beats_patches" in sample:
                        # Section path: hand-rolled token-by-token with hook.
                        patches = sample["beats_patches"].unsqueeze(0).to(device).to(torch.bfloat16)
                        K, V = section_head.precompute_kv(patches)
                        sec_name_to_id = section_open_token_ids(tokenizer)
                        clip_section_ctx = {
                            "mode": sq_mode,
                            "head": section_head, "K": K, "V": V,
                            "section_id_to_idx": section_id_to_idx,
                            "id_to_section_name": {sec_name_to_id[s.name]: s.name for s in SECTION_TAGS},
                        }
                        gen_text, _attn_maps = inference_generate(
                            adapter, llm, tokenizer,
                            sample["audio_features"], sample["overlap_info"],
                            prompt_ids, device,
                            max_new_tokens=config.get("max_target_length", 256),
                            temperature=1.0, top_k=0, top_p=1.0,  # greedy
                            section_ctx=clip_section_ctx,
                        )
                    else:
                        # Legacy path (no sections): HF's optimized generate.
                        af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
                        oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
                        out = adapter(af, oi)
                        prefix = out[0] if isinstance(out, tuple) else out
                        prompt_emb = embed_layer(prompt_ids)
                        inputs_embeds = torch.cat([prefix, prompt_emb], dim=1)
                        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
                        gen_ids = llm.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            max_new_tokens=config.get("max_target_length", 256),
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

                    # SFS vs target text (parse both; use target claims as ground truth).
                    # Same approach inference.py uses when features_path isn't provided.
                    target_text = sample.get("target_text", "") or ""
                    target_claims = claim_parser.parse(target_text)
                    ground_truth = {c.feature: c.value for c in target_claims}
                    # Only put overlap spans into the GT when the model is actually
                    # trained to emit them (tagged/section path). Untagged targets are
                    # built with --no-overlap-segments and never mention spans, so
                    # injecting them here would add an "overlap_span" entry to the SFS
                    # recall denominator that the model can never satisfy (caps recall
                    # at 8/9 on every overlap clip). Gated by score_overlap_spans
                    # (default True preserves the tagged-path behavior).
                    if config.get("score_overlap_spans", True) and sample.get("overlap_segments"):
                        ground_truth["overlap_segments"] = sample["overlap_segments"]
                    pred_claims = claim_parser.parse(gen_text)

                    if ground_truth:
                        sfs_result = sfs_scorer.score(pred_claims, ground_truth)
                        p, r, f1 = sfs_result["precision"], sfs_result["recall"], sfs_result["f1"]
                    else:
                        p = r = f1 = 0.0
                    sfs_precs.append(p); sfs_recs.append(r); sfs_f1s.append(f1)

                    sample_rows.append(
                        (epoch + 1, sample.get("filename", "?"),
                         target_text[:400], gen_text[:400],
                         round(p, 3), round(r, 3), round(f1, 3))
                    )

                    # Band-free selection: stash the FULL generation, the filename,
                    # and the per-clip clean GT (target-parsed scalars restricted to
                    # the canonical 12 features). composite_score is computed after
                    # the loop. Skip clips with no parseable GT (nothing to rank).
                    fname_key = sample.get("filename", f"_idx_{i}")
                    clip_gt = {
                        c.feature: c.value for c in target_claims
                        if c.feature in SELECTION_FEATURES
                    }
                    if clip_gt:
                        bf_gen_texts.append(gen_text)
                        bf_filenames.append(fname_key)
                        bf_clean_gt[fname_key] = clip_gt

            # wandb.Table renders as a browseable table in the UI per epoch.
            # GATED behind log_val_samples_table (DEFAULT FALSE): the per-epoch
            # val-samples Artifact/Table is the heavy object whose upload STALLS
            # online wandb sync (each epoch re-uploads a fresh table artifact;
            # the sync thread blocks behind it and the run hangs). With the gate
            # off, the scalar metrics below still stream online — only the table
            # artifact is skipped — so future runs can train ONLINE without the
            # hang. The same rows are still dumped to disk as JSON (below), so no
            # information is lost; inspect them offline or `wandb sync` later.
            log_val_samples_table = config.get("log_val_samples_table", False)
            table = None
            if log_val_samples_table:
                table = wandb.Table(columns=["epoch", "filename", "target", "generated",
                                             "sfs_precision", "sfs_recall", "sfs_f1"])
                for row in sample_rows:
                    table.add_data(*row)

            # Dump JSON to disk for offline inspection (same shape as IDL HW4's text_val_epoch_*.json).
            samples_dir = os.path.join(config["save_dir"], "val_samples")
            os.makedirs(samples_dir, exist_ok=True)
            samples_json_path = os.path.join(samples_dir, f"epoch_{epoch + 1:03d}.json")
            with open(samples_json_path, "w") as f:
                import json as _json
                _json.dump(
                    [
                        {
                            "filename": r[1],
                            "target": r[2],
                            "generated": r[3],
                            "sfs_precision": r[4],
                            "sfs_recall": r[5],
                            "sfs_f1": r[6],
                        }
                        for r in sample_rows
                    ],
                    f, indent=2, ensure_ascii=False,
                )

            # Epoch-average SFS scalars — plottable as curves across epochs.
            avg_sfs_p = sum(sfs_precs) / max(1, len(sfs_precs))
            avg_sfs_r = sum(sfs_recs) / max(1, len(sfs_recs))
            avg_sfs_f1 = sum(sfs_f1s) / max(1, len(sfs_f1s))

            # Generation-quality metrics on the same 8 val samples. BLEU/ROUGE are near-free;
            # BERTScore is opt-in (use_bertscore: true in config) since it loads a ~1 GB model.
            hyps = [r[3] for r in sample_rows]   # generated
            refs = [r[2] for r in sample_rows]   # target
            gen_metrics = compute_generation_metrics(
                hyps, refs,
                use_bertscore=config.get("use_bertscore", False),
            )
            # Hand the generations + BLEU to the degeneration-aware selector below.
            val_gen_texts = list(hyps)
            val_bleu = gen_metrics.get("bleu")

            # ── Band-free composite (research Q1 protocol) ─────────────────────
            # Continuous per-feature SRCC / nMAE / coverage over the canonical 12
            # features, joined to the target-parsed clean GT. The composite =
            # mean_SRCC(reliable, non-degenerate, snr-excluded) - lam_nmae*mean_nMAE
            # with a HARD BLEU floor; EMA-smoothed across epochs to denoise the
            # selection signal. These are logged regardless of select_metric so the
            # curve is always visible; they only DRIVE selection when
            # select_metric=='composite'.
            bf_scores = band_free_val_scores(bf_gen_texts, bf_filenames, bf_clean_gt)
            bf_pf = bf_scores["per_feature"]
            # Raw composite (with BLEU floor) and its EMA. The floor uses val_bleu;
            # if BLEU is unavailable (dep missing) the floor is skipped (None) so a
            # missing fluency signal never silently rejects every epoch.
            raw_composite = composite_score(
                bf_pf, RECOVERABLE_FEATURES,
                lam_nmae=lam_nmae,
                bleu=val_bleu,
                bleu_floor=(bleu_floor if val_bleu is not None else None),
            )
            # -inf (floor failure) must not poison the EMA — treat it as a NaN
            # update (EMA passes prev through), but still log the raw -inf so a
            # collapse is visible. Otherwise smooth normally.
            ema_update = raw_composite if math.isfinite(raw_composite) else float("nan")
            composite_ema = ema(composite_ema, ema_update, beta=val_select_ema_beta)
            avg_composite = raw_composite

            # Aggregate band-free scalars for wandb. Means over features with a
            # defined value (None features are skipped).
            _srccs = [v["srcc"] for v in bf_pf.values() if v["srcc"] is not None]
            _nmaes = [v["nmae"] for v in bf_pf.values() if v["nmae"] is not None]
            _covs = [v["coverage"] for v in bf_pf.values()]
            bf_srcc_mean = (sum(_srccs) / len(_srccs)) if _srccs else 0.0
            bf_nmae_mean = (sum(_nmaes) / len(_nmaes)) if _nmaes else 0.0
            bf_coverage_mean = (sum(_covs) / len(_covs)) if _covs else 0.0

            # Per-epoch averages of B-full's three loss terms — diagnostic curves so you
            # can see whether the prose CE, the nums CE, and the aux-head MSE are each
            # converging as expected.
            avg_val_lm_prose = val_metrics_sum["loss_lm_prose"] / max(1, n_val)
            avg_val_lm_nums = val_metrics_sum["loss_lm_nums"] / max(1, n_val)
            avg_val_mse = val_metrics_sum["loss_mse"] / max(1, n_val)

            log_dict = {
                "val_loss": avg_val_loss,
                "val_loss_lm_prose": avg_val_lm_prose,
                "val_loss_lm_nums": avg_val_lm_nums,
                "val_loss_mse": avg_val_mse,
                "train_loss_epoch": avg_train_loss,
                "epoch": epoch + 1,
                "val_sfs_precision": avg_sfs_p,
                "val_sfs_recall": avg_sfs_r,
                "val_sfs_f1": avg_sfs_f1,
            }
            # val_sfs_f1 is now a DEPRECATED selection axis (saturates; computed on
            # a small subset) — kept logged for back-compat/curves only. The
            # composite below is the selection signal when select_metric=='composite'.
            if table is not None:  # gated by log_val_samples_table (default False)
                log_dict["val_samples"] = table
            # [section_readout] held-out grounding loss + per-feature MAE. Watch
            # val_readout_mae/f0_mean, .../overlap_ratio — falling means the
            # section's z (hence its attention) is learning to find that evidence.
            if val_readout_n > 0:
                log_dict["val_loss_readout"] = val_readout_loss_sum / val_readout_n
                for k, v in val_readout_mae_sum.items():
                    log_dict[f"val_{k}"] = v / val_readout_n
            # [decoupled_grounding] held-out parallel grounding loss + per-feature MAE.
            if val_decoupled_n > 0:
                log_dict["val_loss_decoupled"] = val_decoupled_loss_sum / val_decoupled_n
                for k, v in val_decoupled_mae_sum.items():
                    log_dict[f"val_{k}"] = v / val_decoupled_n
            if gen_metrics["bleu"] is not None:
                log_dict["val_bleu"] = gen_metrics["bleu"]
            if gen_metrics["rouge_l"] is not None:
                log_dict["val_rouge_l"] = gen_metrics["rouge_l"]
            if gen_metrics["bertscore_f1"] is not None:
                log_dict["val_bertscore_f1"] = gen_metrics["bertscore_f1"]
            # ── Band-free selection metrics (research Q1 protocol) ─────────────
            # Aggregate continuous scalars + the composite (raw + EMA) + every
            # per-feature SRCC / nMAE. These stream online (they are plain scalars,
            # not a heavy table artifact), so the selection curve is visible even
            # with log_val_samples_table off.
            log_dict["val/srcc_mean"] = bf_srcc_mean
            log_dict["val/nmae_mean"] = bf_nmae_mean
            log_dict["val/coverage_mean"] = bf_coverage_mean
            log_dict["val/composite"] = avg_composite
            log_dict["val/composite_ema"] = composite_ema
            for _feat, _stats in bf_pf.items():
                if _stats["srcc"] is not None:
                    log_dict[f"val/srcc_{_feat}"] = _stats["srcc"]
                if _stats["nmae"] is not None:
                    log_dict[f"val/nmae_{_feat}"] = _stats["nmae"]
                log_dict[f"val/coverage_{_feat}"] = _stats["coverage"]
            wandb.log(log_dict)

        # ── Print epoch summary ──
        print("\tTrain Loss {:.04f}".format(avg_train_loss))
        if avg_val_loss is not None:
            print("\tVal Loss {:.04f}".format(avg_val_loss))
        # [section_readout] echo grounding metrics to stdout so they're visible
        # in the tee'd log even when wandb is disabled (smoke runs).
        if avg_val_loss is not None and val_readout_n > 0:
            mae_str = "  ".join(
                f"{k.split('/')[-1]}={v / val_readout_n:.3f}"
                for k, v in sorted(val_readout_mae_sum.items())
            )
            print("\tVal Readout Loss {:.04f}".format(val_readout_loss_sum / val_readout_n))
            print("\tVal Readout MAE (normalized): {}".format(mae_str))
        # [decoupled_grounding] echo the parallel grounding metrics too.
        if avg_val_loss is not None and val_decoupled_n > 0:
            d_mae_str = "  ".join(
                f"{k.split('/')[-1]}={v / val_decoupled_n:.3f}"
                for k, v in sorted(val_decoupled_mae_sum.items())
            )
            print("\tVal Decoupled Loss {:.04f}".format(val_decoupled_loss_sum / val_decoupled_n))
            print("\tVal Decoupled MAE (normalized): {}".format(d_mae_str))
        print("\tLearning Rate {:.07f}".format(curr_lr))

        # Save best — selected on val_sfs_f1 (higher = better). avg_sfs_f1 is set
        # when val-time generation runs (val_sfs_n > 0); skip selection on epochs where it didn't.
        def _ckpt_payload(epoch_idx: int, save_optimizer: bool = True) -> dict:
            # SLIM checkpoints (2026-06): the LLM portion is the ADAPTER ONLY — LoRA
            # tensors + any requires_grad rows (tagged-mode embed/lm_head). The frozen
            # 8B base is dropped (it is restored at load time by from_pretrained), so a
            # LoRA-r16 ckpt shrinks ~17 GB → ~0.2 GB. See src/ckpt_io.py.
            #   - `llm_state_dict` is the canonical key; `lora_state_dict` is the legacy
            #     alias — both point at the same slim dict (back-compat with the getter
            #     in inference.py / the resume path).
            #   - `ckpt_format = "peft_slim"` tags the new format so loaders take the
            #     strict=False path; old fat ckpts lack the tag and are auto-detected.
            #   - optimizer/scheduler are written ONLY for last.pt (save_optimizer=True),
            #     since best.pt is inference-only. This drops the full Adam state (which,
            #     even slimmed, only tracks trainable params) from best.pt.
            llm_sd = slim_llm_state_dict(llm)
            payload = {
                "epoch": epoch_idx,
                "ckpt_format": CKPT_FORMAT_SLIM,
                "adapter_state_dict": adapter.state_dict(),
                "llm_state_dict": llm_sd,
                "lora_state_dict": llm_sd,  # legacy alias
                "best_val_sfs_f1": best_val_sfs_f1,
                # Band-free composite selection state (research Q1 protocol). Carried
                # so --resume_from restores the running EMA + best-so-far without a
                # cold start. Absent in pre-band-free ckpts → cold start on resume.
                "best_val_composite": best_val_composite,
                "composite_ema": composite_ema,
                "wandb_run_id": wandb_run_id,
                "config": config,
                "added_special_tokens": list(TAG_SPECIAL_TOKENS) if config.get("tagged_mode") else [],
            }
            if save_optimizer:
                payload["optimizer_state_dict"] = optimizer.state_dict()
                payload["scheduler_state_dict"] = scheduler.state_dict()
            if section_head is not None:
                payload["section_head_state_dict"] = section_head.state_dict()
            if readout_head is not None:  # [section_readout]
                payload["readout_head_state_dict"] = readout_head.state_dict()
            if decoupled_head is not None:  # [decoupled_grounding]
                payload["decoupled_head_state_dict"] = decoupled_head.state_dict()
            if snr_map_head is not None:  # [snr_map]
                payload["snr_map_head_state_dict"] = snr_map_head.state_dict()
            if srmr_map_head is not None:  # [srmr_map]
                payload["srmr_map_head_state_dict"] = srmr_map_head.state_dict()
            # [encoder-unfreeze] persist ONLY the unfrozen (requires_grad) encoder
            # params so the fine-tuned top-N blocks survive a reload without bloating
            # the ckpt with the frozen backbone (restored from from_pretrained / the
            # BEATs checkpoint at load time). Keyed by the full param name.
            if encoder_unfreeze_params:
                payload["encoder_unfreeze_top_n"] = {
                    "wavlm": int(config.get("unfreeze_wavlm_top_n", 0) or 0),
                    "beats": int(config.get("unfreeze_beats_top_n", 0) or 0),
                }
                if wavlm_encoder is not None:
                    payload["wavlm_unfrozen_state_dict"] = {
                        n: p.detach().cpu()
                        for n, p in wavlm_encoder.named_parameters()
                        if p.requires_grad
                    }
                if spec_encoder is not None and int(config.get("unfreeze_beats_top_n", 0) or 0) > 0:
                    payload["beats_unfrozen_state_dict"] = {
                        n: p.detach().cpu()
                        for n, p in spec_encoder.model.named_parameters()
                        if p.requires_grad
                    }
            return payload

        # Atomic save: write to .tmp then os.replace (POSIX atomic). Without
        # this, a SIGKILL / OOM / quota-truncate during torch.save leaves a
        # partially-written file at the target path, corrupting the checkpoint.
        # Lost v7-lora-8b's best.pt to exactly this kind of partial-write event.
        def _atomic_save(payload, path):
            tmp = path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, path)

        # Upload best.pt to wandb as a versioned Artifact — off-site backup so
        # local file loss (quota truncation, Lustre OST failure, accidental
        # deletion, etc.) doesn't kill the training run. Failures here do NOT
        # interrupt training — wandb upload is best-effort.
        def _upload_to_wandb_artifact(path: str, name: str, metadata: dict) -> None:
            try:
                artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
                artifact.add_file(path)
                wandb.log_artifact(artifact)
                print(f"  [wandb-artifact] uploaded {os.path.basename(path)} "
                      f"as {name}  (metadata={metadata})")
            except Exception as e:
                print(f"  [wandb-artifact] WARNING upload of {path} failed: "
                      f"{type(e).__name__}: {e}")

        # ── Best-checkpoint selection ─────────────────────────────────────────
        # select_metric switches the axis:
        #   'composite' (DEFAULT) — EMA-smoothed band-free composite (continuous
        #     SRCC/nMAE, snr excluded) with a HARD BLEU fluency floor baked into
        #     composite_score, plus the same degeneration guard as a backstop.
        #     Lower-variance + non-saturating per the research Q1 protocol.
        #   'sfs_f1' — the legacy degeneration-gated val_sfs_f1 argmax, BYTE-FOR-BYTE
        #     unchanged (the block below is identical to the pre-band-free code).
        # best_val_bleu is updated identically on BOTH paths so the relative BLEU
        # floor tracks the same running max regardless of which axis selects.
        if select_metric == "sfs_f1":
            # ── Legacy path (unchanged) ───────────────────────────────────────
            # Degeneration-aware selection: SFS only parses numbers, so it is blind to
            # fluency/structural collapse (tag-spam, repetition, foreign-token runs) —
            # an SFS argmax can select a degenerate checkpoint. Gate it on a BLEU /
            # rep-n / non-ASCII guard over this epoch's generations (ckpt_selection.py).
            _save_best, _save_reason = should_save_best(
                avg_sfs_f1, best_val_sfs_f1, val_bleu, best_val_bleu, val_gen_texts,
            )
            if val_bleu is not None:
                best_val_bleu = val_bleu if best_val_bleu is None else max(best_val_bleu, val_bleu)
            if (not _save_best) and avg_sfs_f1 is not None and avg_sfs_f1 > best_val_sfs_f1:
                print(f"  [select] withheld best.pt despite val_sfs_f1={avg_sfs_f1:.4f}: {_save_reason}")
            if _save_best:
                best_val_sfs_f1 = avg_sfs_f1
                best_path = os.path.join(config["save_dir"], "best.pt")
                # best.pt is inference-only → omit optimizer/scheduler.
                _atomic_save(_ckpt_payload(epoch, save_optimizer=False), best_path)
                print(f"Saved best val model (val_sfs_f1={best_val_sfs_f1:.4f})")
                if config.get("upload_ckpt_to_wandb", True):
                    run_name = (wandb.run.name if wandb.run is not None else None) or "run"
                    _upload_to_wandb_artifact(
                        best_path,
                        name=f"best-{run_name}",
                        metadata={"epoch": epoch, "val_sfs_f1": best_val_sfs_f1},
                    )
        else:
            # ── Band-free composite path (DEFAULT) ────────────────────────────
            # Select on the EMA-smoothed composite. The hard BLEU floor lives
            # INSIDE composite_score (raw_composite == -inf below the floor → its
            # EMA can't beat best), and the rep-n/non-ASCII degeneration guard is
            # still applied as a backstop for high-BLEU-but-structurally-broken
            # epochs (e.g. tag-spam that keeps BLEU up). Skip on non-eval epochs
            # (avg_composite is None then).
            if val_bleu is not None:
                best_val_bleu = val_bleu if best_val_bleu is None else max(best_val_bleu, val_bleu)
            _save_best = False
            _save_reason = "no eval this epoch"
            if avg_composite is not None:
                improved = composite_ema is not None and composite_ema > best_val_composite
                _deg = degeneration_stats(val_gen_texts)
                guard_ok, guard_reason = passes_degeneration_guard(
                    val_bleu, best_val_bleu,
                    _deg["rep_n_max"], _deg["nonascii_frac"],
                    _deg["frac_clips_nonascii"], _deg["frac_clips_high_rep"],
                )
                if not improved:
                    _save_reason = (f"no composite_ema improvement "
                                    f"(ema={composite_ema:.4f} <= best={best_val_composite:.4f})")
                elif not guard_ok:
                    _save_reason = f"composite improved but degenerate ({guard_reason})"
                else:
                    _save_best = True
                    _save_reason = "composite_ema improved, clean"
            if (not _save_best) and avg_composite is not None and \
                    composite_ema is not None and composite_ema > best_val_composite:
                print(f"  [select] withheld best.pt despite composite_ema={composite_ema:.4f}: {_save_reason}")
            if _save_best:
                best_val_composite = composite_ema
                # keep best_val_sfs_f1 tracking the chosen epoch's SFS for telemetry
                if avg_sfs_f1 is not None:
                    best_val_sfs_f1 = max(best_val_sfs_f1, avg_sfs_f1)
                best_path = os.path.join(config["save_dir"], "best.pt")
                _atomic_save(_ckpt_payload(epoch, save_optimizer=False), best_path)
                print(f"Saved best val model (composite_ema={best_val_composite:.4f}, "
                      f"raw_composite={avg_composite:.4f}, val_sfs_f1={avg_sfs_f1})")
                if config.get("upload_ckpt_to_wandb", True):
                    run_name = (wandb.run.name if wandb.run is not None else None) or "run"
                    _upload_to_wandb_artifact(
                        best_path,
                        name=f"best-{run_name}",
                        metadata={"epoch": epoch, "val_composite": best_val_composite,
                                  "val_sfs_f1": avg_sfs_f1},
                    )

        # Save last (for resuming) — keeps optimizer + scheduler so --resume_from works.
        _atomic_save(
            _ckpt_payload(epoch, save_optimizer=True),
            os.path.join(config["save_dir"], "last.pt"),
        )
        print("Saved epoch model")

    wandb.finish()
    if select_metric == "sfs_f1":
        print("\nTraining complete. Best val_sfs_f1: {:.04f}".format(best_val_sfs_f1))
    else:
        print("\nTraining complete. Best val_composite (EMA): {:.04f} "
              "(best val_sfs_f1 seen: {:.04f})".format(best_val_composite, best_val_sfs_f1))


# ── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    args, unknown = parser.parse_known_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with command line args (expects --key value pairs)
    if len(unknown) % 2 != 0:
        parser.error(f"CLI overrides must be --key value pairs, got odd number of args: {unknown}")
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-")
        val = unknown[i + 1]
        if key in config and config[key] is not None:
            if isinstance(config[key], bool):
                val = val.lower() in ("true", "1", "yes")
            elif isinstance(config[key], int):
                val = int(val)
            elif isinstance(config[key], float):
                val = float(val)
        config[key] = val

    os.makedirs(config["save_dir"], exist_ok=True)
    train(config)
