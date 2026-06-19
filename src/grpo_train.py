"""grpo_train.py — GRPO (RLVR) fine-tuning scaffold for AQUA-NL.

STATUS: STRUCTURED SCAFFOLD. This module parses cleanly and lays out the full
GRPO pipeline, but the marked `[GPU]` / `[TODO]` blocks need a live model, a GPU,
and `trl` installed to actually run. It is NOT runnable on this CPU box and is not
meant to be — it documents the plan and wires together the pieces that ARE done
(the SFS reward in `sfs_reward.py`, the model/adapter loading mirrored from
`train.py` / `inference.py`).

────────────────────────────────────────────────────────────────────────────────
THE PLAN: SFT cold-start  →  RAFT de-risk  →  GRPO-on-SFS
────────────────────────────────────────────────────────────────────────────────
The headline SFT model is `v9_lora_8b_dur` (LoRA r=16 on Qwen3-8B, test SFS-F1
0.52, BLEU 31.5). SFT gets the model fluent and roughly on-topic but leaves two
residual problems RL is well-suited to fix:
  - numbers that are *plausible but wrong* (precision ceiling), and
  - occasional degeneration (repetition loops / foreign-token injection) that SFT
    cannot un-learn because the cross-entropy target never penalizes it.

Three stages, increasing risk:

  1. SFT COLD-START (done): load `v9_lora_8b_dur` as the RL policy's init. RL from
     a random LoRA never bootstraps a parseable claim, so the verifiable reward
     would be ~0 everywhere and GRPO has no gradient. The SFT model already emits
     parseable claims, so the reward is informative on step 0.

  2. RAFT DE-RISK (best-of-n rejection-sampling fine-tune): before full GRPO,
     run a couple of epochs of RAFT — sample G completions per prompt, keep only
     the top-reward one, and SFT on it. RAFT is GRPO without the KL/PPO machinery;
     it is far more stable and surfaces reward-hacking (e.g. number-spam) cheaply
     and safely. If RAFT already lifts SFS-F1 without degenerating, we may not
     need full GRPO. (This stage reuses the SAME reward function.)

  3. GRPO-ON-SFS (the headline RL run): trl.GRPOTrainer with group size G=8,
     a KL penalty to the SFT reference (beta), and `scale_rewards=False` so the
     advantage keeps the *magnitude* of the reward gap (per the project's research
     finding — std-normalizing advantages throws away how much better the best
     completion is, which matters when most of a group is mediocre).

REWARD (from sfs_reward.py):
    reward = SFS_F1(text vs gt) - rep_penalty*rep_n(text) - nonascii_penalty*nonascii_frac(text)
SFS-F1 (NOT recall) so number-spam is punished by precision; the degeneration
penalty so RL cannot drift into high-SFS-but-unreadable text.

────────────────────────────────────────────────────────────────────────────────
THE HARD PART (audio-prefix integration): clearly marked [TODO-AUDIO] below.
────────────────────────────────────────────────────────────────────────────────
AQUA-NL is NOT a plain text LM. The LM is conditioned on an AUDIO PREFIX produced
by `adapter(audio_features, overlap_info)` and prepended (in EMBEDDING space) to
the prompt embeddings. `trl.GRPOTrainer` assumes a text-in / text-out causal LM
and owns the generation loop, so it does not know about a per-sample audio prefix.
Two integration routes, both stubbed:
  (A) inputs_embeds route — subclass GRPOTrainer and override the generation +
      log-prob path to splice the precomputed audio-prefix embeddings in front of
      the prompt embeddings (mirrors `inference.generate`). Most faithful, most code.
  (B) prefix-token route — freeze the adapter, run it offline over the dataset to
      get each clip's prefix embeddings, and feed them via a custom collator /
      `prepare_inputs`. Simpler but still needs a trainer subclass.
Either way the reward itself is audio-agnostic (it only sees text + GT), so the
SFS reward and its tests are already complete and correct regardless of route.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reward — the ONE piece that is fully implemented + unit-tested (no trl, no GPU).
from sfs_reward import make_sfs_reward_func


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """Load the YAML config (same format as configs/config.psc.emnlp.yaml)."""
    import yaml  # local import: keep module import-parse light

    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Model + adapter loading — mirrors src/train.py and src/inference.py.
# [GPU] Everything in here needs CUDA + transformers + peft + a checkpoint.
# ──────────────────────────────────────────────────────────────────────────────
def load_policy(config: dict, sft_checkpoint: str):
    """Load the Qwen3-8B + LoRA policy and the audio adapter from the SFT
    cold-start checkpoint (`v9_lora_8b_dur`). Mirrors inference.py's loader so the
    RL policy starts byte-identical to the evaluated SFT model.

    [GPU] Requires torch+CUDA, transformers, peft, and the checkpoint file.

    Returns (llm, tokenizer, adapter). The adapter is loaded and, by default,
    FROZEN for stage-1 GRPO (only the LoRA params are the RL policy params — the
    audio adapter is a fixed feature extractor). Unfreezing it is a later ablation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    from adapter import build_adapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    # LoRA wrap — same path as train.py's `_use_lora` branch. The headline recipe
    # is LoRA r=16 (NOT full FT — full FT forgot the LM prior).
    if config.get("lora_rank"):
        from peft_config import lora_config_kwargs

        llm = get_peft_model(llm, LoraConfig(**lora_config_kwargs(config)))

    lm_hidden_size = llm.config.hidden_size
    adapter = (
        build_adapter(config["adapter_variant"], lm_dim=lm_hidden_size)
        .to(device)
        .to(torch.bfloat16)
    )

    # Restore SFT weights (cold-start). Checkpoint key conventions copied verbatim
    # from inference.py: new ckpts use `llm_state_dict`, legacy use `lora_state_dict`.
    ckpt = torch.load(sft_checkpoint, weights_only=False, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    llm_sd = ckpt.get("llm_state_dict") or ckpt["lora_state_dict"]
    llm.load_state_dict(llm_sd)

    # Stage-1 GRPO: freeze the audio adapter (fixed feature extractor).
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()

    return llm, tokenizer, adapter


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def build_rl_dataset(config: dict):
    """Build the prompt dataset for GRPO.

    GRPO needs, per row: the PROMPT (what the policy completes) and enough side
    info to compute the reward — here, the per-clip ground-truth feature dict and
    a clip id. We forward `gt_features` as a dataset column so trl passes it to the
    reward function as an aligned kwargs list (the path `make_sfs_reward_func`
    handles directly).

    [TODO-DATA] Wire this to the real preprocessed split + the SP ground-truth
    feature JSONs (Person A's outputs / features CSV). Each row should be:

        {
          "prompt":      <the prose prompt string, same as train.py's prose_prompt_str>,
          "clip_id":     <clip stem>,
          "gt_features": {"snr": 16.1, "f0_mean": 121.0, ...,
                          "overlap_segments": [(s, e), ...]},   # SFSScorer shape
          # [TODO-AUDIO] plus the audio-prefix handle (path to .pt, or precomputed
          #              prefix embeddings) for the inputs_embeds splice.
        }

    Build GT dicts restricted to SFSScorer.TOLERANCES keys (+ overlap_segments)
    so recall's denominator is honest — see CLAUDE.md's note on GT filtering.

    Returns a datasets.Dataset (or a list[dict] for a smoke test).
    """
    # from datasets import Dataset
    # rows = _load_rows(config)  # TODO: read features CSV + descriptions, build GT dicts
    # return Dataset.from_list(rows)
    raise NotImplementedError(
        "build_rl_dataset is a scaffold stub — wire it to the preprocessed split "
        "and SP ground-truth feature dicts (SFSScorer shape). See [TODO-DATA]."
    )


# ──────────────────────────────────────────────────────────────────────────────
# GRPO config + trainer
# ──────────────────────────────────────────────────────────────────────────────
def build_grpo_config(config: dict):
    """Build trl.GRPOConfig with the project's research-backed RL settings.

    [GPU] Needs `trl` installed.

    Key choices:
      - num_generations (G) = 8        : group size for the GRPO advantage.
      - beta (KL to ref)    = 0.04     : keep the policy near the SFT reference so
                                          it does not forget fluency while chasing SFS.
      - scale_rewards = False          : per research, DON'T std-normalize the
                                          group advantage — keep the reward-gap
                                          magnitude (Dr.GRPO-style).
      - temperature ~ 1.0, top_p ~ 1.0 : need diverse samples within a group or all
                                          G completions are identical and advantage = 0.
    """
    from trl import GRPOConfig  # [GPU] import here so the module parses without trl

    rl = config.get("grpo", {})
    return GRPOConfig(
        output_dir=rl.get("output_dir", "./checkpoints_grpo"),
        # ── GRPO core ──
        num_generations=rl.get("num_generations", 8),     # G = group size
        beta=rl.get("beta", 0.04),                         # KL penalty to ref policy
        scale_rewards=rl.get("scale_rewards", False),      # keep reward-gap magnitude
        # ── sampling (diversity within a group) ──
        temperature=rl.get("temperature", 1.0),
        top_p=rl.get("top_p", 1.0),
        max_completion_length=rl.get("max_completion_length", 512),
        max_prompt_length=rl.get("max_prompt_length", 256),
        # ── optimization ──
        learning_rate=float(rl.get("learning_rate", 1e-6)),  # RL LR << SFT LR
        per_device_train_batch_size=rl.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=rl.get("gradient_accumulation_steps", 8),
        num_train_epochs=rl.get("num_train_epochs", 1),
        bf16=True,
        gradient_checkpointing=rl.get("gradient_checkpointing", True),
        logging_steps=rl.get("logging_steps", 1),
        save_steps=rl.get("save_steps", 50),
        report_to=rl.get("report_to", "wandb"),
    )


def build_reward(config: dict, dataset) -> object:
    """Construct the SFS reward function for GRPO.

    This is the COMPLETED piece. GT is forwarded as the dataset's `gt_features`
    column, so `make_sfs_reward_func()` with no explicit lookup is correct: trl
    passes that column as an aligned kwargs list to the reward function, and the
    wrapper maps each completion → sfs_reward(text, gt).

    Reward weights are read from config so they can be tuned per run.
    """
    rl = config.get("grpo", {})
    return make_sfs_reward_func(
        gt_lookup=None,  # GT arrives via the `gt_features` dataset column (kwargs)
        f1_weight=float(rl.get("f1_weight", 1.0)),
        rep_penalty=float(rl.get("rep_penalty", 0.5)),
        nonascii_penalty=float(rl.get("nonascii_penalty", 1.0)),
        rep_n=int(rl.get("rep_n", 4)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# [TODO-AUDIO] Audio-prefix integration — the non-trivial part.
# ──────────────────────────────────────────────────────────────────────────────
class AudioPrefixGRPOTrainer:  # pragma: no cover — scaffold, needs trl + GPU
    """Subclass-GRPOTrainer placeholder for splicing the audio prefix.

    [TODO-AUDIO] The real implementation should subclass `trl.GRPOTrainer` and
    override the generation + per-token log-prob computation so that, for each
    sample, the precomputed audio-prefix embeddings (from the frozen `adapter`)
    are concatenated in front of the prompt token embeddings — exactly as
    `inference.generate` does:

        out = adapter(audio_features, overlap_info)         # (1, T_pref, d)
        prefix_embeds = out[0] if isinstance(out, tuple) else out
        prompt_embeds = embed_layer(prompt_ids)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    Both the policy AND the reference model must see the SAME prefix so the KL is
    well-defined. The reward function is unaffected (it only reads the decoded
    completion text + GT), so nothing here changes `build_reward`.

    Two routes (see module docstring): (A) override generation to pass
    inputs_embeds; (B) precompute prefixes offline and inject via a collator.
    Left as a class stub on purpose — it is the one part that genuinely needs the
    live model to get right.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AudioPrefixGRPOTrainer is a [TODO-AUDIO] scaffold. Subclass "
            "trl.GRPOTrainer and splice adapter() prefix embeddings into "
            "inputs_embeds for both policy and reference. See the class docstring."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():  # pragma: no cover — needs trl + GPU + checkpoint
    parser = argparse.ArgumentParser(description="GRPO (RLVR) fine-tuning for AQUA-NL.")
    parser.add_argument("--config", required=True, help="YAML config (configs/config.psc.emnlp.yaml).")
    parser.add_argument("--sft_checkpoint", required=True,
                        help="SFT cold-start checkpoint, e.g. v9_lora_8b_dur/best.pt.")
    parser.add_argument("--stage", choices=["raft", "grpo"], default="grpo",
                        help="raft = best-of-n rejection-sampling de-risk; grpo = full GRPO.")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Policy (SFT cold-start) — [GPU]
    llm, tokenizer, adapter = load_policy(config, args.sft_checkpoint)

    # 2. Dataset (prompts + GT feature dicts + audio handles) — [TODO-DATA]
    dataset = build_rl_dataset(config)

    # 3. Reward (DONE — SFS-F1 minus degeneration)
    reward_func = build_reward(config, dataset)

    if args.stage == "raft":
        # [TODO-RAFT] best-of-n rejection sampling on the same reward, then SFT on
        # the kept completions. trl has RLOO/best-of-n building blocks; or roll a
        # small loop: generate G, keep argmax reward, SFT on it. Stable de-risk.
        raise NotImplementedError("RAFT stage is a scaffold stub. See [TODO-RAFT].")

    # 4. GRPO config (DONE except it needs trl) — [GPU]
    grpo_config = build_grpo_config(config)

    # 5. Trainer — [TODO-AUDIO] the audio-prefix splice is the non-trivial part.
    #    Plain text GRPO would be:
    #        from trl import GRPOTrainer
    #        trainer = GRPOTrainer(model=llm, reward_funcs=[reward_func],
    #                              args=grpo_config, train_dataset=dataset,
    #                              processing_class=tokenizer)
    #    but AQUA-NL needs the audio prefix, so use AudioPrefixGRPOTrainer instead.
    trainer = AudioPrefixGRPOTrainer(
        model=llm,
        adapter=adapter,
        reward_funcs=[reward_func],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(grpo_config.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
