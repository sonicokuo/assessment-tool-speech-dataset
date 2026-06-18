"""
Training script for Overlap-Aware Speech Quality Description.
Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --adapter_variant concat-only --epochs 3
    python src/train.py --config configs/config.yaml --resume_from ./checkpoints/last.pt
"""

import argparse
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
from feature_set import FEATURE_SCALES
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
# Degeneration-aware, lower-variance best-checkpoint selection.
from ckpt_selection import seeded_val_indices, should_save_best
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
            from token_init import build_semantic_tag_init, semantic_init_new_rows
            # IMPORTANT: the "new token" boundary for semantic init is the
            # tokenizer length BEFORE the add (= the id of the first new token),
            # NOT `old_vocab_size` (the embedding matrix row count). Qwen3-8B pads
            # its embedding matrix to 151936 rows while the tokenizer only has
            # 151669 real tokens, so add_tokens assigns the new <sec_*>/<f_*> ids
            # in [151669, 151687] — all of which are < old_vocab_size (151936).
            # Using old_vocab_size as the threshold rejects every open tag and
            # the warm-start matches 0 (the R10 bug). The pre-add tokenizer length
            # is the correct boundary for both the source-row guard and the
            # new-row guard inside semantic_init_new_rows.
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
) -> torch.Tensor:
    """Run one LM forward (prefix + prompt + target) and return CE loss on the target tokens.

    Tokens of the prefix and prompt are masked out via -100 labels so the loss reflects
    only the autoregressive prediction of the target tokens.

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
    return outputs.loss


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
    out = adapter(audio_features, overlap_info)
    if isinstance(out, tuple):
        prefix_embeds, scalar_pred = out
    else:
        prefix_embeds, scalar_pred = out, None

    metrics: dict[str, float] = {}

    # Prose CE loss (always computed). When use_sections=true, section_ctx is
    # threaded through so the per-section audio summaries get residually
    # injected into the target embeddings at <sec_X> open positions.
    lm_loss_prose = _ce_against_target(
        llm, embed_layer, tokenizer,
        prefix_embeds, prompt_ids, target_text,
        max_length=config["max_target_length"],
        device=device,
        section_ctx=section_ctx,
    )
    metrics["loss_lm_prose"] = float(lm_loss_prose.detach().item())

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

    lambda_prose = float(config.get("lambda_prose", 1.0))
    lambda_nums = float(config.get("lambda_nums", 0.0))
    lambda_mse = float(config.get("lambda_mse", 0.0))

    total = lambda_prose * lm_loss_prose + lambda_nums * lm_loss_nums + lambda_mse * mse_loss
    # [section_readout] add the grounding term (readout_loss is 0 when disabled).
    total = total + lambda_readout * readout_loss
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
    lm_hidden_size = llm.config.hidden_size
    adapter = build_adapter(config["adapter_variant"], lm_dim=lm_hidden_size).to(device).to(torch.bfloat16)

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
        else:
            print(f"[sections] using precomputed BEATs patches from .pt files (beats_cached=true)")

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

    # Trainable-parameter summary — printed once at startup and stashed for wandb.run.summary
    # after wandb.init() below. Helps compare adapter vs LoRA/full-FT footprint across runs.
    lm_total = sum(p.numel() for p in llm.parameters())
    lm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    trainable_total = lm_trainable + adapter_trainable
    param_summary = {
        "params/lm_total": lm_total,
        "params/lm_trainable": lm_trainable,         # full-FT: lm_total; LoRA: small subset
        "params/lora_trainable": lm_trainable if not full_ft else 0,  # legacy compat key
        "params/adapter_trainable": adapter_trainable,
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

    train_set = PreprocessedDataset(train_dir, config["descriptions_path"], features_csv=train_csv)
    val_set = PreprocessedDataset(val_dir, config["descriptions_path"], features_csv=val_csv)
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

    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], weights_only=False)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        # New checkpoints use "llm_state_dict"; pre-2026-05-11 ones used "lora_state_dict".
        llm_sd = checkpoint.get("llm_state_dict") or checkpoint["lora_state_dict"]
        llm.load_state_dict(llm_sd)
        if section_head is not None and "section_head_state_dict" in checkpoint:
            section_head.load_state_dict(checkpoint["section_head_state_dict"])
        if readout_head is not None and "readout_head_state_dict" in checkpoint:  # [section_readout]
            readout_head.load_state_dict(checkpoint["readout_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # Backwards-compat: old checkpoints stored "best_val_loss"; new ones store "best_val_sfs_f1".
        if "best_val_sfs_f1" in checkpoint:
            best_val_sfs_f1 = checkpoint["best_val_sfs_f1"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Resumed from epoch {start_epoch}, best_val_sfs_f1={best_val_sfs_f1:.4f}")

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

    # Readout-grounding warmup: the readout gradient destabilizes generation in
    # dynamic mode (v12: lambda 0.5 -> 31% degenerate). Ramp it in over the first
    # few epochs so the LM stabilizes first. Target stored once; the effective
    # lambda_readout is overwritten per-epoch below.
    _readout_target = float(config.get("lambda_readout", 0.0))
    _readout_warmup = int(config.get("lambda_readout_warmup_epochs", 0))

    for epoch in range(start_epoch, config["epochs"]):
        if _readout_target > 0.0:
            config["lambda_readout"] = warmup_lambda(_readout_target, epoch, _readout_warmup)
            print(f"[section_readout] epoch {epoch+1}: effective lambda_readout="
                  f"{config['lambda_readout']:.4f} (target {_readout_target}, "
                  f"warmup {_readout_warmup} epochs)")
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
                # [section_readout] surface the grounding loss + per-feature MAE
                # (computed in compute_loss but otherwise discarded). These are
                # THE signal for whether attention is becoming grounded.
                if "loss_readout" in loss_metrics:
                    log_payload["train_loss_readout"] = loss_metrics["loss_readout"]
                    log_payload["lambda_readout"] = float(config.get("lambda_readout", 0.0))
                    for k, v in loss_metrics.items():
                        if k.startswith("readout_mae/"):
                            log_payload[f"train_{k}"] = v
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
            # val_sfs_n controls the val-time generation sample count. Default 32 (was 8) —
            # 8 was too noisy to read F1 trends across epochs; 32 cuts noise floor in half
            # for ~2 minutes additional generation time per epoch.
            # Seeded RANDOM subset (stable across epochs), not the biased first-N
            # prefix slice — see ckpt_selection.seeded_val_indices.
            val_idx = seeded_val_indices(
                len(val_set),
                config.get("val_sfs_n", 32),
                seed=config.get("val_sfs_seed", 1234),
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
                    if sample.get("overlap_segments"):
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

            # wandb.Table renders as a browseable table in the UI per epoch.
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
                "val_samples": table,
                "val_sfs_precision": avg_sfs_p,
                "val_sfs_recall": avg_sfs_r,
                "val_sfs_f1": avg_sfs_f1,
            }
            # [section_readout] held-out grounding loss + per-feature MAE. Watch
            # val_readout_mae/f0_mean, .../overlap_ratio — falling means the
            # section's z (hence its attention) is learning to find that evidence.
            if val_readout_n > 0:
                log_dict["val_loss_readout"] = val_readout_loss_sum / val_readout_n
                for k, v in val_readout_mae_sum.items():
                    log_dict[f"val_{k}"] = v / val_readout_n
            if gen_metrics["bleu"] is not None:
                log_dict["val_bleu"] = gen_metrics["bleu"]
            if gen_metrics["rouge_l"] is not None:
                log_dict["val_rouge_l"] = gen_metrics["rouge_l"]
            if gen_metrics["bertscore_f1"] is not None:
                log_dict["val_bertscore_f1"] = gen_metrics["bertscore_f1"]
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
        print("\tLearning Rate {:.07f}".format(curr_lr))

        # Save best — selected on val_sfs_f1 (higher = better). avg_sfs_f1 is set
        # when val-time generation runs (val_sfs_n > 0); skip selection on epochs where it didn't.
        def _ckpt_payload(epoch_idx: int) -> dict:
            # New canonical key is `llm_state_dict` (covers both LoRA and full-FT
            # weights). The pre-existing `lora_state_dict` alias is kept so older
            # tools that look for that key still find the same tensor.
            llm_sd = llm.state_dict()
            payload = {
                "epoch": epoch_idx,
                "adapter_state_dict": adapter.state_dict(),
                "llm_state_dict": llm_sd,
                "lora_state_dict": llm_sd,  # legacy alias
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_sfs_f1": best_val_sfs_f1,
                "wandb_run_id": wandb_run_id,
                "config": config,
                "added_special_tokens": list(TAG_SPECIAL_TOKENS) if config.get("tagged_mode") else [],
            }
            if section_head is not None:
                payload["section_head_state_dict"] = section_head.state_dict()
            if readout_head is not None:  # [section_readout]
                payload["readout_head_state_dict"] = readout_head.state_dict()
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
            _atomic_save(_ckpt_payload(epoch), best_path)
            print(f"Saved best val model (val_sfs_f1={best_val_sfs_f1:.4f})")
            if config.get("upload_ckpt_to_wandb", True):
                run_name = (wandb.run.name if wandb.run is not None else None) or "run"
                _upload_to_wandb_artifact(
                    best_path,
                    name=f"best-{run_name}",
                    metadata={"epoch": epoch, "val_sfs_f1": best_val_sfs_f1},
                )

        # Save last (for resuming)
        _atomic_save(_ckpt_payload(epoch), os.path.join(config["save_dir"], "last.pt"))
        print("Saved epoch model")

    wandb.finish()
    print("\nTraining complete. Best val_sfs_f1: {:.04f}".format(best_val_sfs_f1))


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
