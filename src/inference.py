"""
Inference script: load trained checkpoint, generate descriptions, evaluate with SFS.

Usage:
    # Evaluate on in-domain test set
    python src/inference.py --config configs/config.yaml --checkpoint ./checkpoints/best.pt --test_dir ./data/processed/test

    # Evaluate on cross-domain test set
    python src/inference.py --config configs/config.yaml --checkpoint ./checkpoints/best.pt --test_dir ./data/processed/libricss

    # Single clip
    python src/inference.py --config configs/config.yaml --checkpoint ./checkpoints/best.pt --single ./data/processed/test/clip_001.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model

from model.adapter import build_adapter
from data.ckpt_io import load_llm_state_dict
from data.dataset import PreprocessedDataset
from eval.inference_resume import (
    FINGERPRINT_KEY,
    checkpoint_fingerprint,
    resolve_resume_output_path,
    results_fingerprints,
)
from eval.sfs import HybridClaimParser, SFSScorer
from eval.text_metrics import compute_generation_metrics
from data.section_tags import (
    SPECIAL_TOKENS as TAG_SPECIAL_TOKENS,
    SECTION_TAGS,
    N_SECTIONS,
    RANGE_OPEN_TAG,
    RANGE_CLOSE_TAG,
    section_open_token_ids,
    strip_all_tags,
)
from model.section_query import SectionQueryHead
from model.spec_encoder import SpecEncoder


# ── Generation ──────────────────────────────────────────────
def _apply_repetition_penalty(logits: torch.Tensor, prev_ids, penalty: float) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar 2019): divide the logit of each already-emitted
    token by `penalty` (>1 discourages repeats). Deterministic — compatible with greedy decoding
    (top_k=1) and reproducible paper numbers. penalty==1.0 -> no-op."""
    if penalty == 1.0 or not prev_ids:
        return logits
    for tid in set(int(t) for t in prev_ids):
        v = logits[0, tid]
        logits[0, tid] = v / penalty if v > 0 else v * penalty
    return logits


def _block_repeat_ngrams(logits: torch.Tensor, prev_ids, n: int) -> torch.Tensor:
    """No-repeat-ngram (Paulus 2018): -inf any token that would complete an n-gram already seen
    in prev_ids. Deterministic. n<=0 -> no-op. Useful against the templated-report late-loop."""
    if n <= 0 or len(prev_ids) < n:
        return logits
    prefix = tuple(int(t) for t in prev_ids[-(n - 1):]) if n > 1 else ()
    for i in range(len(prev_ids) - n + 1):
        if tuple(int(t) for t in prev_ids[i:i + n - 1]) == prefix:
            logits[0, int(prev_ids[i + n - 1])] = float("-inf")
    return logits


def sample_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
    prev_ids=None, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Sample a token from logits with temperature, top-k, top-p (nucleus) filtering, and
    optional deterministic repetition control (D4). repetition_penalty=1.0 + no_repeat_ngram_size=0
    (the defaults) are byte-identical to before. When set, they are applied to the raw logits
    BEFORE temperature/top-k/top-p, so they also bias greedy (top_k=1) decoding."""
    if prev_ids:
        logits = _apply_repetition_penalty(logits, prev_ids, repetition_penalty)
        logits = _block_repeat_ngrams(logits, prev_ids, no_repeat_ngram_size)
    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k).values[:, -1, None]
        logits[logits < threshold] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    adapter: torch.nn.Module,
    llm: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    audio_features: torch.Tensor,
    overlap_info: torch.Tensor,
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,      # D4: deterministic; 1.0 = off
    no_repeat_ngram_size: int = 0,        # D4: deterministic; 0 = off
    section_ctx: dict | None = None,
) -> tuple[str, dict]:
    """Generate a quality description from pre-computed features.

    If section_ctx is provided (use_sections=true at training time), this also
    runs the per-section cross-attention hook: when a <sec_X> token is emitted,
    the corresponding query attends over spec patches, the attention map is
    saved into the returned dict, and the audio summary is residually injected
    into the next-input embedding so the section body is conditioned on it.

    Args:
        section_ctx: dict with keys:
            "head":              SectionQueryHead instance.
            "K", "V":            (1, P, d) — precomputed patch projections.
            "section_id_to_idx": {token_id: section_idx}.
            "id_to_section_name": {token_id: "noise"/"reverb"/...}.

    Returns:
        (decoded_text, attention_maps) — attention_maps is {} if section_ctx
        is None, else {section_name: tensor (P,)} for every section the model
        emitted in this clip.
    """
    audio_features = audio_features.unsqueeze(0).to(device).to(torch.bfloat16)
    overlap_info = overlap_info.unsqueeze(0).to(device).to(torch.bfloat16)

    # AdapterWithAuxHead returns (prefix, scalar_pred[, log_var]); legacy adapters return prefix only.
    out = adapter(audio_features, overlap_info)
    if isinstance(out, tuple):
        prefix_embeds = out[0]
        aux_scalar_pred = out[1] if len(out) > 1 else None   # E12: aux-head mean (was discarded here)
        aux_log_var = out[2] if len(out) > 2 else None        # reliability head log-variance (if present)
        # Reliability-head adapters return (prefix, (mean, log_var)) — a NESTED
        # tuple (adapter.py forward). Normalize to flat mean/log_var.
        if isinstance(aux_scalar_pred, tuple):
            aux_scalar_pred, aux_log_var = aux_scalar_pred
    else:
        prefix_embeds, aux_scalar_pred, aux_log_var = out, None, None

    embed_layer = llm.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_ids)

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    # Generate token by token with KV cache
    generated_ids: list[int] = []
    attention_maps: dict[str, torch.Tensor] = {}
    past_key_values = None
    pending_injection: torch.Tensor | None = None   # set after a section-open is emitted

    # F1 (2026-07-16): stop on BOTH Qwen enders, matching the model's shipped
    # generation_config eos_token_id=[<|im_end|>, <|endoftext|>]. Training
    # supervises <|im_end|> (tokenizer.eos_token_id on the post-trained model),
    # but on raw (non-chat-template) inputs the pretrained prior can emit the
    # document ender <|endoftext|> instead; stopping on only one id sails past
    # the other and the generation continues as fresh off-task text (the
    # "boilerplate" degeneration class). skip_special_tokens=True then hides
    # the missed ender from the logs.
    eos_ids = {tokenizer.eos_token_id}
    _eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if _eot is not None and _eot != tokenizer.unk_token_id:
        eos_ids.add(_eot)

    # Whether to ask the LM to return hidden states. Both the dynamic section
    # path and any <r>-marker firing need them; turning the flag off when not
    # needed avoids the extra memory copy.
    needs_hidden = section_ctx is not None and section_ctx.get("mode") == "dynamic"

    # Token ids for range markers (<r>, </r>). Stored on the ctx so generate()
    # doesn't have to do tokenizer lookups per step.
    range_open_id = section_ctx.get("range_open_id") if section_ctx else None
    range_close_id = section_ctx.get("range_close_id") if section_ctx else None

    # State machine for the in-flight <r>...</r> span. While open we collect
    # body token ids; at </r> we decode, parse the value, and key the saved
    # attention map by that value (e.g. "overlap@0.5-1.0s").
    range_pending_alpha: torch.Tensor | None = None
    range_body_ids: list[int] = []

    # First forward: process prefix + prompt
    outputs = llm(inputs_embeds=inputs_embeds, use_cache=True, output_hidden_states=needs_hidden)
    past_key_values = outputs.past_key_values
    next_token_id = sample_token(outputs.logits[:, -1, :], temperature, top_k, top_p,
                                 prev_ids=generated_ids, repetition_penalty=repetition_penalty,
                                 no_repeat_ngram_size=no_repeat_ngram_size)
    token_id = next_token_id.item()
    generated_ids.append(token_id)

    if token_id in eos_ids:
        max_new_tokens = 1   # first token is an ender — skip the loop entirely

    pending_injection, range_pending_alpha, range_body_ids = _maybe_fire_hooks(
        token_id, outputs, needs_hidden, section_ctx,
        range_open_id, range_close_id,
        range_pending_alpha, range_body_ids,
        attention_maps, tokenizer,
    )

    # Subsequent: one token at a time
    for _ in range(max_new_tokens - 1):
        next_embeds = embed_layer(next_token_id)
        if pending_injection is not None:
            # Inject the prior section's audio summary into THIS step's input
            # embedding. The LM's hidden state at this position is now informed
            # by the cross-attention, and the section body it generates from
            # here on will reflect the attended audio.
            next_embeds = next_embeds + pending_injection
            pending_injection = None

        outputs = llm(
            inputs_embeds=next_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=needs_hidden,
        )
        past_key_values = outputs.past_key_values
        next_token_id = sample_token(outputs.logits[:, -1, :], temperature, top_k, top_p,
                                     prev_ids=generated_ids, repetition_penalty=repetition_penalty,
                                     no_repeat_ngram_size=no_repeat_ngram_size)
        token_id = next_token_id.item()
        generated_ids.append(token_id)

        if token_id in eos_ids:
            break

        # Accumulate body tokens while inside an <r>...</r> span.
        if range_pending_alpha is not None and token_id != range_close_id:
            range_body_ids.append(token_id)

        pending, range_pending_alpha, range_body_ids = _maybe_fire_hooks(
            token_id, outputs, needs_hidden, section_ctx,
            range_open_id, range_close_id,
            range_pending_alpha, range_body_ids,
            attention_maps, tokenizer,
        )
        if pending is not None:
            pending_injection = pending

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # E12: stash the aux-head prediction (per-feature mean, and log_var if a reliability head is
    # present) so the caller can log it to inference_results.json for the digit-drift metric
    # (|emitted - aux_mean|). Carried under a reserved key the caller pops; NOT an attention map.
    if aux_scalar_pred is not None:
        aux = {"aux_mean": aux_scalar_pred[0].float().cpu().tolist()}
        if aux_log_var is not None:
            aux["aux_log_var"] = aux_log_var[0].float().cpu().tolist()
        attention_maps["_aux"] = aux
    return text, attention_maps


def _maybe_fire_hooks(
    token_id: int,
    outputs,
    needs_hidden: bool,
    section_ctx: dict | None,
    range_open_id: int | None,
    range_close_id: int | None,
    range_pending_alpha: torch.Tensor | None,
    range_body_ids: list[int],
    attention_maps: dict,
    tokenizer,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int]]:
    """Dispatch section-open / range-open / range-close events.

    Returns:
        pending_injection: e_t (1, 1, d_lm) to add to the LM's next input
                           embedding, or None.
        range_pending_alpha: the in-flight <r>...</r> attention map, or None
                             if no range is currently open.
        range_body_ids: token ids collected so far inside the open range.
    """
    if section_ctx is None:
        return None, range_pending_alpha, range_body_ids

    pending_injection: torch.Tensor | None = None

    # <sec_X> open — fire the section query hook.
    if token_id in section_ctx["section_id_to_idx"]:
        h_t = outputs.hidden_states[-1][:, -1, :] if needs_hidden else None
        pending_injection = _section_hook(token_id, section_ctx, attention_maps, h_t=h_t)

    # <r> open — fire the per-range query hook. Reuses dynamic-mode forward;
    # body tokens accumulate via the generate-loop caller until </r>.
    elif range_open_id is not None and token_id == range_open_id:
        h_t = outputs.hidden_states[-1][:, -1, :] if needs_hidden else None
        e_t, alpha = _range_hook(section_ctx, h_t)
        pending_injection = e_t.unsqueeze(1) if e_t is not None else None
        range_pending_alpha = alpha
        range_body_ids = []                                    # reset buffer

    # </r> close — finalise the in-flight range: parse its body, key the
    # attention map by the parsed value (e.g. "overlap@0.5-1.0s").
    elif range_close_id is not None and token_id == range_close_id and range_pending_alpha is not None:
        body = tokenizer.decode(range_body_ids, skip_special_tokens=True).strip()
        key = _range_attention_key(body)
        # If the key collides with a prior occurrence, suffix with a counter.
        final_key = key
        suffix = 2
        while final_key in attention_maps:
            final_key = f"{key}#{suffix}"
            suffix += 1
        attention_maps[final_key] = range_pending_alpha.detach().squeeze(0).cpu()
        range_pending_alpha = None
        range_body_ids = []

    return pending_injection, range_pending_alpha, range_body_ids


def _range_hook(
    section_ctx: dict,
    h_t: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run the per-range cross-attention.

    Returns (e_t, alpha) where e_t is (1, d_lm) to be injected at the next
    position and alpha is (1, P) for the saved attention map. Static mode
    has no meaningful per-range query, so this returns (None, None) there —
    range markers are a dynamic-mode-only feature.
    """
    if section_ctx.get("mode") != "dynamic" or h_t is None:
        return None, None
    head = section_ctx["head"]
    K, V = section_ctx["K"], section_ctx["V"]
    return head.forward_dynamic(h_t, K, V)


def _range_attention_key(body: str) -> str:
    """Key for an attention map saved at </r>, based on the parsed range body.

    Examples:
        "0.5-1.0s"          → "overlap@0.5-1.0s"
        "0.5 to 1.0 s"      → "overlap@0.5-1.0s"
        (unparseable body)  → "overlap@malformed:<raw text>"
    """
    from data.section_tags import extract_overlap_segments
    ranges = extract_overlap_segments(body)
    if ranges:
        s, e = ranges[0]
        return f"overlap@{s}-{e}s"
    return f"overlap@malformed:{body[:24]}"


def _section_hook(
    section_token_id: int,
    section_ctx: dict,
    attention_maps: dict,
    h_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the section query for a just-emitted <sec_X> token; record alpha; return e_t.

    Static mode (default): query is looked up by section index.
    Dynamic mode: query is `W_q · h_t` where h_t is the LM's last-layer hidden
    state at the position of the just-emitted <sec_X> token. The caller must
    pass `h_t` of shape (1, d_lm).
    """
    head = section_ctx["head"]
    K, V = section_ctx["K"], section_ctx["V"]
    mode = section_ctx.get("mode", "static")

    if mode == "dynamic":
        assert h_t is not None, "dynamic mode needs h_t from the just-finished LM forward"
        e_t, alpha = head.forward_dynamic(h_t, K, V)   # (1, d_lm), (1, P)
    else:
        section_idx = section_ctx["section_id_to_idx"][section_token_id]
        idx_t = torch.tensor([section_idx], device=K.device, dtype=torch.long)
        e_t, alpha = head(idx_t, K, V)                 # (1, d_lm), (1, P)

    section_name = section_ctx["id_to_section_name"][section_token_id]
    attention_maps[section_name] = alpha.detach().squeeze(0).cpu()
    # Reshape to (1, 1, d_lm) so it can be added to next_embeds (1, 1, d_lm).
    return e_t.unsqueeze(1)


# ── Evaluation ──────────────────────────────────────────────
# Keys that must match the training run — read from the checkpoint's embedded
# config rather than the YAML so you can evaluate any run without --key flags.
_STRUCTURAL_KEYS = (
    "lm_name",
    "adapter_variant",
    "lora_rank",
    "lora_alpha",
    "lora_targets",
    "lora_dropout",
    "compression",         # conv stride: a 4 ckpt in an 8 adapter loads the wrong geometry
    "aux_pool",            # mean vs linear_softmax changes how the head reads the prefix
    "use_dora",            # DoRA must match train/inference or the delta won't load
    "init_lora_weights",   # PiSSA/standard init must match (see peft_config.py)
    "tagged_mode",  # tags vs legacy untagged prose — determines tokenizer setup
    # Section-query path. These MUST come from the checkpoint: the YAML default
    # is use_sections=false, so without syncing these a section_head checkpoint
    # would run inference WITHOUT loading section_head — a train/inference
    # mismatch (no e_t injection at <sec_*> positions) that corrupts generation
    # and produces no attention maps. This was the v11 garbage-output bug.
    "use_sections",
    "section_query_mode",
    "beats_cached",
    "spec_encoder_name",
    "spec_checkpoint_name",
    "spec_d_patch",
    "section_d_k",
    "section_d_v",
    # Decoupled-grounding head. grounding_mode MUST come from the checkpoint so any
    # head-loading path (grounding_validate, extraction) builds the SAME forward
    # (softmax vs bottleneck) the head was trained with. The other decoupled_* keys
    # ride along to reconstruct the head with matching dims/β/temperature.
    "grounding_mode",
    "decoupled_grounding",
    "decoupled_d_model",
    "decoupled_n_heads",
    "decoupled_readout_hidden",
    "bits_beta_per_feature",
    "concrete_temp_start",
    "concrete_temp_end",
)


def _sync_config_with_checkpoint(config: dict, checkpoint_path: str) -> dict:
    """Override structural keys in config with whatever the checkpoint was trained with.

    train.py pickles the full config into every checkpoint. For eval that config is
    the source of truth — the YAML might list a different default LM than the run
    being evaluated.
    """
    ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    ck_cfg = ck.get("config", {})
    for k in _STRUCTURAL_KEYS:
        if k in ck_cfg and ck_cfg[k] != config.get(k):
            print(f"[config] {k}: {config.get(k)!r} → {ck_cfg[k]!r} (from checkpoint)")
            config[k] = ck_cfg[k]
    # Carry wandb_run_id so we can re-attach to the same run at test-log time.
    if "wandb_run_id" in ck:
        config.setdefault("_ckpt_wandb_run_id", ck["wandb_run_id"])
    return config


def _pick_device() -> torch.device:
    """Prefer CUDA, then Apple-Silicon MPS, then CPU. MPS lets a Mac run Qwen-class
    models at ~5-10× CPU speed for this kind of single-stream decode."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(config: dict, checkpoint_path: str, test_dir: str) -> None:
    device = _pick_device()
    print(f"[device] {device}")

    # Pull lm_name / adapter_variant / lora_* from the checkpoint itself so we
    # don't need --lm_name / --adapter_variant on the CLI.
    config = _sync_config_with_checkpoint(config, checkpoint_path)

    # Write inference outputs next to the checkpoint, so each ablation's results
    # sit beside its own best.pt instead of all clobbering one YAML-level save_dir.
    config["save_dir"] = os.path.dirname(os.path.abspath(checkpoint_path))
    print(f"[config] save_dir → {config['save_dir']} (inference outputs will land here)")

    # Load tokenizer + LLM
    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # If the checkpoint was trained with tagged-mode, register the feature
    # tokens and resize the embedding matrix BEFORE loading state_dict so
    # the embedding shapes match what the checkpoint expects.
    #
    # `add_tokens(..., special_tokens=False)` is deliberate: with
    # `additional_special_tokens=...` the tags get stripped by the default
    # `decode(skip_special_tokens=True)` call below in generate(), which
    # silently makes every <f_*> tag invisible to the SFS parser.
    if config.get("tagged_mode"):
        added = tokenizer.add_tokens(TAG_SPECIAL_TOKENS, special_tokens=False)
        if added:
            llm.resize_token_embeddings(len(tokenizer))
        print(f"[tagged-mode] vocab size = {len(tokenizer)} ({added} new tokens added)")

    # LoRA wrap only for LoRA-trained checkpoints. Full-FT checkpoints contain
    # the LM weights directly under llm_state_dict / lora_state_dict.
    full_ft = not bool(config.get("lora_rank"))
    if not full_ft:
        from model.peft_config import lora_config_kwargs, uses_pissa
        llm = get_peft_model(llm, LoraConfig(**lora_config_kwargs(config)))
        if uses_pissa(config):
            print("[LoRA] WARNING: PiSSA init requested but inference rebuilds from the "
                  "vanilla HF base; without a saved pissa-residual base the adapter loads "
                  "against the wrong weights. Use DoRA, or implement the base-restore.")
        print(f"[LoRA] rank={config['lora_rank']} (legacy ckpt path)")
    else:
        print(f"[full-FT] lora_rank={config.get('lora_rank')!r} → loading LM weights directly")

    # Load checkpoint (use --checkpoint_device cpu for smaller GPUs)
    # F12: fingerprint the checkpoint file (first 1 MiB + size + mtime, hex[:12])
    # so resume can detect results written by DIFFERENT weights (see resume block).
    ckpt_fingerprint = checkpoint_fingerprint(checkpoint_path)
    print(f"[fingerprint] checkpoint {ckpt_fingerprint} ({checkpoint_path})")
    map_loc = config.get("checkpoint_device", "cuda")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_loc)
    lm_hidden_size = llm.config.hidden_size

    adapter = (
        build_adapter(
            config["adapter_variant"],
            lm_dim=lm_hidden_size,
            # BUGFIX 2026-07-23: the sigma-head flag was NOT forwarded here, so a
            # reliability_head=true checkpoint was loaded into a PLAIN aux head at
            # RANDOM init (strict=False silently dropped the trained head weights).
            # Every aux_mean logged by E12 was garbage, aux_log_var never existed,
            # and slot_decode mode="verified" substituted random values. Forward the
            # flag from the checkpoint's embedded config so the trained head loads.
            reliability_head=bool(config.get("reliability_head", False)),
            # Same class of bug as above: a compression=4 checkpoint loaded into a
            # compression=8 adapter changes the conv stride, so the trained conv2
            # weights would load into the wrong geometry. Read it from the ckpt config.
            compression=int(config.get("compression", 8)),
            # must match training or the head is applied at the wrong granularity
            aux_pool=str(config.get("aux_pool", "mean")),
        )
        .to(device)
        .to(torch.bfloat16)
    )
    missing, unexpected = adapter.load_state_dict(
        checkpoint["adapter_state_dict"], strict=False
    )
    # Fail LOUD if head weights didn't line up: a missing aux/reliability head at
    # inference means aux_mean/aux_log_var (E12, slot verified mode, sigma
    # analyses) would silently be random-init garbage.
    _head_missing = [k for k in missing if "head" in k or "regress" in k]
    if _head_missing:
        raise RuntimeError(
            f"adapter head weights missing from checkpoint load: {_head_missing} — "
            "reliability_head/config mismatch between training and inference?"
        )
    # New checkpoints use `llm_state_dict`; legacy ones used `lora_state_dict`.
    # SLIM ckpts (ckpt_format="peft_slim") carry only LoRA + unfrozen rows and load
    # strict=False over the already-built (from_pretrained + get_peft_model) base;
    # old FAT ckpts carry the full base and load strict. See src/ckpt_io.py.
    llm_sd = checkpoint.get("llm_state_dict") or checkpoint["lora_state_dict"]
    _missing, _unexpected = load_llm_state_dict(
        llm, llm_sd, ckpt_format=checkpoint.get("ckpt_format"),
    )
    if _unexpected:
        raise RuntimeError(f"Unexpected keys loading LLM checkpoint: {_unexpected[:5]} ...")

    # Section-query head (EMNLP rework, Path 3). Same gate as in train.py.
    section_head: SectionQueryHead | None = None
    section_id_to_idx: dict[int, int] = {}
    id_to_section_name: dict[int, str] = {}
    sq_mode = config.get("section_query_mode", "static").lower()
    if config.get("use_sections") and "section_head_state_dict" in checkpoint:
        d_patch = int(config.get("spec_d_patch", 768))
        section_head = SectionQueryHead(
            n_sections=N_SECTIONS, d_patch=d_patch, d_lm=lm_hidden_size,
            d_k=int(config.get("section_d_k", 256)),
            d_v=int(config.get("section_d_v", 256)),
        ).to(device).to(torch.bfloat16)
        section_head.load_state_dict(checkpoint["section_head_state_dict"])
        section_head.eval()
        sec_name_to_id = section_open_token_ids(tokenizer)
        section_id_to_idx = {sec_name_to_id[s.name]: i for i, s in enumerate(SECTION_TAGS)}
        id_to_section_name = {sec_name_to_id[s.name]: s.name for s in SECTION_TAGS}
        print(f"[sections] loaded section_head from checkpoint; mode={sq_mode}; "
              f"section ids = {list(section_id_to_idx.keys())}")

    adapter.eval()
    llm.eval()

    # New checkpoints (post 2026-04-28) save best_val_sfs_f1 (higher is better);
    # legacy ones saved best_val_loss (lower is better). Print whichever is present.
    if "best_val_sfs_f1" in checkpoint:
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_sfs_f1={checkpoint['best_val_sfs_f1']:.4f}")
    elif "best_val_loss" in checkpoint:
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['best_val_loss']:.4f}")
    else:
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']} (no best-metric scalar in ckpt)")

    # Prompt
    # Inference always uses the prose prompt. prompt_prose is the canonical key going
    # forward; fall back to legacy `prompt` if not set (e.g., old YAMLs).
    inference_prompt = config.get("prompt_prose") or config["prompt"]
    prompt_ids = tokenizer(inference_prompt, return_tensors="pt").input_ids.to(device)
    print(f"[prompt-prose] {inference_prompt!r}")

    # Dataset
    # ⚠️ 4th OCCURRENCE OF THE UNFORWARDED-CONFIG BUG (fixed 2026-08-14). Previously this
    # read `PreprocessedDataset(test_dir, config.get("descriptions_path"))` and dropped
    # `zero_overlap_input`, whose default is False (`data/dataset.py:37`). The L7 audio-only
    # arm was TRAINED with the overlap channel zeroed, so its `OverlapEmbedding` Linear(4,32)
    # (`model/adapter.py:37`) received EXACTLY ZERO gradient and sits at random init —
    # feeding it the oracle overlap at eval is structured OOD noise, and it silently
    # corrupted a 6000-clip headline. Same class as the 2026-07-23 `reliability_head` fix
    # and the 2026-08-10 `run_m3b_deletion` fix.
    #
    # The countermeasure is NOT another hand-forwarded kwarg: resolve every data-pipeline
    # flag from the CHECKPOINT'S OWN training config (the model is what it was trained as),
    # fall back to the eval YAML, and SHOUT when the two disagree.
    ck_cfg = (checkpoint.get("config") or {}) if isinstance(checkpoint, dict) else {}

    def _pipeline_flag(name: str, default):
        train_v, yaml_v = ck_cfg.get(name, None), config.get(name, None)
        if train_v is not None and yaml_v is not None and bool(train_v) != bool(yaml_v):
            print(f"[warn] {name}: checkpoint trained with {train_v!r} but eval YAML says "
                  f"{yaml_v!r} — using the CHECKPOINT value ({train_v!r}).")
        return train_v if train_v is not None else (yaml_v if yaml_v is not None else default)

    zero_overlap = bool(_pipeline_flag("zero_overlap_input", False))
    test_set = PreprocessedDataset(
        test_dir,
        config.get("descriptions_path"),
        zero_overlap_input=zero_overlap,
    )
    assert len(test_set) > 0, f"No .pt files in {test_dir}"
    print(f"Test set: {len(test_set)} samples from {test_dir}  "
          f"zero_overlap_input={zero_overlap}")
    if zero_overlap:
        # Positive control: with the channel zeroed the abstain gate's overlap_ratio is 0.0
        # for EVERY clip, so the hard-coded `overlap_ratio >= HEDGE_OVERLAP_TAU` rule in
        # slot_decode.py can never fire. Any abstention observed under this flag is
        # therefore model-driven, not rule-driven — state which one produced a given table.
        _s = test_set[0]
        _oi = _s.get("overlap_info") if isinstance(_s, dict) else None
        if _oi is not None:
            print(f"[check] overlap_info abs-sum on sample 0 = {float(_oi.abs().sum()):.6f} "
                  f"(must be 0.0)")

    # Load SP ground truth features if available (from Person A's Praat measurements)
    features_path = config.get("features_path")
    sp_features = None
    if features_path and os.path.exists(features_path):
        with open(features_path) as f:
            sp_features = json.load(f)
        print(f"Loaded SP ground truth from {features_path}")
    else:
        print("No features_path in config — falling back to parsing target text for SFS ground truth")

    # Decide the index range to process. --start / --end default to the full set
    # but can be narrowed for parallelization or range-resume.
    start_idx = max(0, int(config.get("start", 0)))
    end_idx = config.get("end")
    end_idx = len(test_set) if end_idx is None else min(int(end_idx), len(test_set))
    if end_idx <= start_idx:
        raise ValueError(f"--end ({end_idx}) must be > --start ({start_idx})")
    print(f"Range: clips [{start_idx}, {end_idx}) of {len(test_set)} total")

    # Resume / parallel-safe behaviour: if inference_results.json already exists
    # in save_dir, load it; any clip whose filename is already there is skipped.
    # Fresh completed entries are appended and the file is flushed every 50 clips
    # (atomic tmp-then-rename) so a crash only costs the last <50 clips.
    #
    # F12 guard: every entry is stamped with the checkpoint fingerprint. If the
    # existing file was written by a DIFFERENT checkpoint, appending would silently
    # merge generations from different weights — refuse, and divert this run to
    # inference_results.<fingerprint>.json instead. Legacy files without
    # fingerprints resume as before (unverifiable, not refused).
    os.makedirs(config["save_dir"], exist_ok=True)
    # --out lets several nodes work DISJOINT --start/--end ranges of the same checkpoint into
    # SEPARATE shard files. Without it every process writes the one fixed path, so parallel
    # shards race on the atomic tmp-then-rename and silently clobber each other — which is why
    # a 6000-clip eval previously had to run for ~24 h on a single node. Merge shards afterwards
    # with `scripts/merge_inference_shards.py`.
    default_output_path = (args.out if getattr(args, "out", None)
                           else os.path.join(config["save_dir"], "inference_results.json"))

    def _load_results(path: str) -> list:
        try:
            with open(path) as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                print(f"[resume] Existing {path} is not a list; starting fresh.")
                return []
            return loaded
        except (json.JSONDecodeError, OSError) as e:
            print(f"[resume] Could not parse existing {path} ({e}); starting fresh.")
            return []

    existing = _load_results(default_output_path) if os.path.exists(default_output_path) else []
    output_path, resume_ok = resolve_resume_output_path(
        default_output_path, ckpt_fingerprint, results_fingerprints(existing),
    )
    if not resume_ok:
        print(f"[resume] REFUSING to append: {default_output_path} holds generations from a "
              f"DIFFERENT checkpoint (fingerprints {sorted(results_fingerprints(existing))} "
              f"!= this checkpoint's {ckpt_fingerprint}).")
        print(f"[resume] Writing this run's results to {output_path} instead.")
        # The diverted file (if present from an earlier crash of THIS checkpoint) is
        # still resumable — it was named by this fingerprint.
        existing = _load_results(output_path) if os.path.exists(output_path) else []

    all_outputs: list = existing
    done_filenames: set = {
        e["filename"] for e in all_outputs if isinstance(e, dict) and "filename" in e
    }
    if done_filenames:
        print(f"[resume] Found {len(done_filenames)} already-scored clips in {output_path}")

    FLUSH_EVERY = 50

    def flush_outputs() -> None:
        tmp = output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_outputs, f, indent=2)
        os.replace(tmp, output_path)

    # Generate + evaluate. HybridClaimParser auto-detects tagged spans for the
    # EMNLP rework checkpoints and falls back to the legacy regex parser for
    # older Phase-2 checkpoints.
    claim_parser = HybridClaimParser()
    scorer = SFSScorer()
    n_new = 0

    for i in range(start_idx, end_idx):
        sample = test_set[i]
        if sample["filename"] in done_filenames:
            continue
        stem = os.path.splitext(sample["filename"])[0]

        # Build the per-clip section context if we're using sections AND patches
        # were cached into the .pt. Without cached patches the generate() call
        # falls back to plain autoregressive generation (no attention maps saved).
        clip_section_ctx = None
        if section_head is not None and "beats_patches" in sample:
            patches = sample["beats_patches"].unsqueeze(0).to(device).to(torch.bfloat16)
            K, V = section_head.precompute_kv(patches)
            clip_section_ctx = {
                "mode": sq_mode,
                "head": section_head,
                "K": K, "V": V,
                "section_id_to_idx": section_id_to_idx,
                "id_to_section_name": id_to_section_name,
                # <r>...</r> sub-spans inside <f_overlap_segments> produce
                # per-range attention maps when sq_mode == "dynamic".
                "range_open_id": tokenizer.convert_tokens_to_ids(RANGE_OPEN_TAG),
                "range_close_id": tokenizer.convert_tokens_to_ids(RANGE_CLOSE_TAG),
            }

        slot_mode = config.get("slot_decode", "off")
        slot_report = None
        if slot_mode != "off":
            # F18: constrained slot decoding — frame teacher-forced, LM fills
            # numeric slots; "verified" substitutes the aux head's values.
            from slot_decode import slot_generate
            generated, slot_report = slot_generate(
                adapter, llm, tokenizer,
                sample["audio_features"], sample["overlap_info"],
                prompt_ids, device,
                mode=slot_mode,
                # D2 (phase-2.5): .pt files carry no scalar overlap_ratio; derive the
                # clip ratio from overlap_info col 0 (per-frame is_overlap mean) so
                # the ILL_POSED hedge branch is reachable at eval.
                overlap_ratio=float(sample["overlap_info"][:, 0].float().mean()),
            )
            attention_maps = {}
        else:
            generated, attention_maps = generate(
                adapter, llm, tokenizer,
                sample["audio_features"],
                sample["overlap_info"],
                prompt_ids, device,
                max_new_tokens=config.get("max_target_length", 256),
                temperature=config.get("temperature", 1.0),
                top_k=config.get("top_k", 0),
                top_p=config.get("top_p", 1.0),
                repetition_penalty=config.get("repetition_penalty", 1.0),   # D4: config-wired (was unreachable)
                no_repeat_ngram_size=config.get("no_repeat_ngram_size", 0),  # D4
                section_ctx=clip_section_ctx,
            )
        # E12: pop the aux-head prediction (not an attention map) before serializing maps.
        aux_pred = attention_maps.pop("_aux", None)

        # Measure duration from the WavLM frame count (50 Hz frame rate from
        # the encoder's 320-sample stride at 16 kHz). Stored as a sidecar
        # metadata field, NOT prepended to the generated prose — duration is
        # an audio property, not a quality claim, and SFS no longer scores it.
        # Downstream tools that need duration alongside the quality assessment
        # read this field directly from inference_results.json.
        wavlm_frame_rate_hz = 50.0
        measured_duration_sec = sample["audio_features"].shape[0] / wavlm_frame_rate_hz

        output_entry = {
            "filename": sample["filename"],
            # F12: which weights produced this generation (guards resume-merge).
            FINGERPRINT_KEY: ckpt_fingerprint,
            "generated": generated,
            "generated_clean": strip_all_tags(generated),
            "measured_duration_sec": measured_duration_sec,
        }
        if aux_pred is not None:      # E12: aux-head mean (+ log_var) for the digit-drift metric
            output_entry["aux_mean"] = aux_pred.get("aux_mean")
        if slot_report is not None:
            output_entry["slot_report"] = {
                k: (v if k == "_summary" else {kk: vv for kk, vv in v.items()})
                for k, v in slot_report.items()
            }
        # D1 (phase-2.5): aux_pred is None in slot mode (attention_maps={}); guard it.
        if aux_pred is not None and aux_pred.get("aux_log_var") is not None:
            output_entry["aux_log_var"] = aux_pred["aux_log_var"]
        if attention_maps:
            # Save as plain lists in JSON (per-clip); the plotting script reshapes
            # to (T_p, F_p) using the spec encoder's grid metadata.
            output_entry["attention_maps"] = {
                name: tensor.tolist() for name, tensor in attention_maps.items()
            }

        if "target_text" in sample:
            output_entry["target"] = sample["target_text"]

        # Build ground truth: prefer SP measurements, fall back to parsing target text
        ground_truth = {}
        if sp_features and stem in sp_features:
            ground_truth = sp_features[stem].copy()
        elif "target_text" in sample:
            target_claims = claim_parser.parse(sample["target_text"])
            ground_truth = {c.feature: c.value for c in target_claims}

        # Only score overlap spans when the model is trained to emit them (tagged/
        # section path). Untagged targets are built with --no-overlap-segments and
        # never mention spans, so adding an "overlap_span" entry to the SFS recall
        # denominator would cap recall (8/9) for a feature the model never produces.
        # score_overlap_spans defaults True (read from the ckpt's embedded config via
        # _sync_config_with_checkpoint); untagged configs set it False.
        if (config.get("score_overlap_spans", True)
                and sample["overlap_segments"]
                and "overlap_segments" not in ground_truth):
            ground_truth["overlap_segments"] = sample["overlap_segments"]

        # SFS scoring — save per_feature too so the aggregate can be rebuilt
        # from the JSON on resume (avoids keeping all_results in memory).
        if ground_truth:
            claims = claim_parser.parse(generated)
            result = scorer.score(claims, ground_truth)

            output_entry["sfs_precision"] = result["precision"]
            output_entry["sfs_recall"] = result["recall"]
            output_entry["sfs_f1"] = result["f1"]
            output_entry["claims"] = [(c.feature, c.value) for c in claims]
            output_entry["per_feature"] = result["per_feature"]

        all_outputs.append(output_entry)
        done_filenames.add(sample["filename"])
        n_new += 1

        if n_new % FLUSH_EVERY == 0:
            flush_outputs()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{end_idx} done (range); {len(all_outputs)}/{len(test_set)} total on disk")

    flush_outputs()

    # Rebuild all_results from everything on disk so the aggregate covers previous
    # runs too. Downstream code was originally in terms of all_results-of-dicts.
    all_results = [
        {
            "precision": e.get("sfs_precision", 0.0),
            "recall": e.get("sfs_recall", 0.0),
            "f1": e.get("sfs_f1", 0.0),
            "per_feature": e.get("per_feature", []),
        }
        for e in all_outputs if "sfs_f1" in e
    ]
    if len(all_outputs) < len(test_set):
        print(f"\n[partial] {len(all_outputs)}/{len(test_set)} clips scored so far. "
              f"Run again without --start/--end, or with the remaining range, to finish.")

    # Build a summary dict we'll both print and persist.
    summary: dict = {
        "test_dir": test_dir,
        "n_samples": len(all_outputs),
        FINGERPRINT_KEY: ckpt_fingerprint,
    }

    # Print results
    if all_results:
        avg_p = sum(r["precision"] for r in all_results) / len(all_results)
        avg_r = sum(r["recall"] for r in all_results) / len(all_results)
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)

        summary["sfs_precision"] = avg_p
        summary["sfs_recall"] = avg_r
        summary["sfs_f1"] = avg_f1
        summary["n_scored"] = len(all_results)

        print(f"\n{'='*50}")
        print(f"SFS Results on {test_dir}:")
        print(f"  Precision: {avg_p:.4f}")
        print(f"  Recall:    {avg_r:.4f}")
        print(f"  F1:        {avg_f1:.4f}")
        print(f"  Samples:   {len(all_results)}")
        print(f"{'='*50}")

        # Per-feature breakdown
        feature_correct = {}
        feature_total = {}
        for r in all_results:
            for feat in r["per_feature"]:
                name = feat["feature"]
                feature_total[name] = feature_total.get(name, 0) + 1
                if feat["correct"]:
                    feature_correct[name] = feature_correct.get(name, 0) + 1

        per_feature_acc = {}
        print(f"\nPer-feature accuracy:")
        for name in sorted(feature_total.keys()):
            correct = feature_correct.get(name, 0)
            total = feature_total[name]
            per_feature_acc[name] = {"correct": correct, "total": total, "accuracy": correct / total}
            print(f"  {name:20s}: {correct}/{total} = {correct/total:.2f}")
        summary["per_feature_accuracy"] = per_feature_acc

    # ── Generation-quality metrics: BLEU-4 / ROUGE-L / BERTScore-F1 ──
    # Complement to SFS (numerical faithfulness). Only run on pairs where both
    # hyp and ref are present.
    paired = [(e["generated"], e.get("target", "")) for e in all_outputs if e.get("target")]
    if paired:
        hyps, refs = zip(*paired)
        gen_metrics = compute_generation_metrics(
            list(hyps), list(refs),
            use_bertscore=config.get("use_bertscore", True),
        )
        summary["gen_metrics"] = {**gen_metrics, "n_paired": len(paired)}
        print(f"\nGeneration-quality metrics ({len(paired)} pairs):")
        if gen_metrics["bleu"] is not None:
            print(f"  BLEU-4:        {gen_metrics['bleu']:.2f}")
        if gen_metrics["rouge_l"] is not None:
            print(f"  ROUGE-L (F1):  {gen_metrics['rouge_l']:.4f}")
        if gen_metrics["bertscore_f1"] is not None:
            print(f"  BERTScore-F1:  {gen_metrics['bertscore_f1']:.4f}")

    # Per-clip outputs were flushed incrementally during the loop, so just the summary here.
    # F12: mirror the results filename — a fingerprint-diverted run writes
    # inference_summary.<fingerprint>.json so it can't clobber the other checkpoint's summary.
    summary_name = os.path.basename(output_path).replace("inference_results", "inference_summary", 1)
    summary_path = os.path.join(config["save_dir"], summary_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate summary saved to {summary_path}")

    # Log aggregates to wandb under a "test/*" namespace so the same run page shows
    # both train/val curves and test-set numbers. Default on — set `wandb_log_test: false`
    # in the config to disable. Falls back gracefully if wandb isn't available or not logged in.
    if config.get("wandb_log_test", True):
        try:
            import wandb
            wandb_run_id = None
            if "wandb_run_id" in checkpoint:
                wandb_run_id = checkpoint["wandb_run_id"]
            wandb.init(
                project=config.get("wandb_project", "idl-ablation"),
                entity=config.get("wandb_entity"),
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                name=config.get("wandb_run_name"),
            )
            log = {f"test/{k}": v for k, v in summary.items()
                   if isinstance(v, (int, float)) and v is not None}
            if "gen_metrics" in summary:
                for k, v in summary["gen_metrics"].items():
                    if isinstance(v, (int, float)) and v is not None:
                        log[f"test/{k}"] = v
            wandb.log(log)
            print(f"Logged test metrics to wandb under test/* keys")
            wandb.finish()
        except Exception as e:
            print(f"[wandb] test-time logging skipped: {e}")
    print(f"\nResults saved to {output_path}")

    # Print examples
    print(f"\nExample outputs:")
    for entry in all_outputs[:5]:
        print(f"\n  File: {entry['filename']}")
        if "target" in entry:
            print(f"  Target:    {entry['target'][:100]}...")
        print(f"  Generated: {entry['generated'][:100]}...")
        if "sfs_f1" in entry:
            print(f"  SFS-F1:    {entry['sfs_f1']:.2f}")


# ── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--checkpoint_device", type=str, default="cuda", help="Device to load checkpoint (cpu for OOM on smaller GPUs)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path. Use ONE PER SHARD when splitting a checkpoint's "
                             "eval across nodes with --start/--end; the default single path is "
                             "not safe for concurrent writers.")
    parser.add_argument("--start", type=int, default=0,
                        help="First test-set index to process (inclusive). Default 0.")
    parser.add_argument("--end", type=int, default=None,
                        help="Stop index (exclusive). Default = end of test set. "
                             "Combine with --start for range/parallel runs; reruns auto-skip already-scored clips.")
    parser.add_argument("--slot_decode", type=str, default=None,
                        choices=["off", "free_slots", "verified"],
                        help="F18: constrained slot decoding (frame teacher-forced, "
                             "LM fills numeric slots; verified substitutes aux values)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max_target_length for generation. "
                             "Training default was 256 — long descriptions get truncated mid-sentence. "
                             "Try 512 to see if the model extrapolates coherently past the training cap.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI args override config
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.slot_decode is not None:
        config["slot_decode"] = args.slot_decode
    if args.top_k is not None:
        config["top_k"] = args.top_k
    if args.top_p is not None:
        config["top_p"] = args.top_p
    config["checkpoint_device"] = args.checkpoint_device
    config["start"] = args.start
    config["end"] = args.end
    if args.max_new_tokens is not None:
        config["max_target_length"] = args.max_new_tokens

    evaluate(config, args.checkpoint, args.test_dir)
