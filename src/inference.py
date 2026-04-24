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

from adapter import build_adapter
from dataset import PreprocessedDataset
from sfs import ClaimParser, SFSScorer
from text_metrics import compute_generation_metrics


# ── Generation ──────────────────────────────────────────────
def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Sample a token from logits with temperature, top-k, and top-p (nucleus) filtering."""
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
) -> str:
    """Generate a quality description from pre-computed features."""
    audio_features = audio_features.unsqueeze(0).to(device).to(torch.bfloat16)
    overlap_info = overlap_info.unsqueeze(0).to(device).to(torch.bfloat16)

    prefix_embeds = adapter(audio_features, overlap_info)

    embed_layer = llm.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_ids)

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    # Generate token by token with KV cache
    generated_ids = []
    past_key_values = None

    # First forward: process prefix + prompt
    outputs = llm(inputs_embeds=inputs_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = sample_token(outputs.logits[:, -1, :], temperature, top_k, top_p)
    generated_ids.append(next_token_id.item())

    # Subsequent: one token at a time
    for _ in range(max_new_tokens - 1):
        next_embeds = embed_layer(next_token_id)
        outputs = llm(
            inputs_embeds=next_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_id = sample_token(outputs.logits[:, -1, :], temperature, top_k, top_p)

        token_id = next_token_id.item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


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

    # Load tokenizer + LLM + LoRA
    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    llm = get_peft_model(
        llm,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["lora_targets"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Load checkpoint (use --checkpoint_device cpu for smaller GPUs)
    map_loc = config.get("checkpoint_device", "cuda")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_loc)
    lm_hidden_size = llm.config.hidden_size

    adapter = (
        build_adapter(config["adapter_variant"], lm_dim=lm_hidden_size)
        .to(device)
        .to(torch.bfloat16)
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    llm.load_state_dict(checkpoint["lora_state_dict"])

    adapter.eval()
    llm.eval()

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['best_val_loss']:.4f}")

    # Prompt
    prompt_ids = tokenizer(config["prompt"], return_tensors="pt").input_ids.to(device)

    # Dataset
    test_set = PreprocessedDataset(test_dir, config.get("descriptions_path"))
    assert len(test_set) > 0, f"No .pt files in {test_dir}"
    print(f"Test set: {len(test_set)} samples from {test_dir}")

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
    os.makedirs(config["save_dir"], exist_ok=True)
    output_path = os.path.join(config["save_dir"], "inference_results.json")
    all_outputs: list = []
    done_filenames: set = set()
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                all_outputs = json.load(f)
            done_filenames = {e["filename"] for e in all_outputs if "filename" in e}
            print(f"[resume] Found {len(done_filenames)} already-scored clips in {output_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[resume] Could not parse existing {output_path} ({e}); starting fresh.")
            all_outputs = []
            done_filenames = set()

    FLUSH_EVERY = 50

    def flush_outputs() -> None:
        tmp = output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_outputs, f, indent=2)
        os.replace(tmp, output_path)

    # Generate + evaluate
    claim_parser = ClaimParser()
    scorer = SFSScorer()
    n_new = 0

    for i in range(start_idx, end_idx):
        sample = test_set[i]
        if sample["filename"] in done_filenames:
            continue
        stem = os.path.splitext(sample["filename"])[0]

        generated = generate(
            adapter, llm, tokenizer,
            sample["audio_features"],
            sample["overlap_info"],
            prompt_ids, device,
            max_new_tokens=config.get("max_target_length", 256),
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k", 0),
            top_p=config.get("top_p", 1.0),
        )

        output_entry = {
            "filename": sample["filename"],
            "generated": generated,
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

        if sample["overlap_segments"] and "overlap_segments" not in ground_truth:
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
    summary: dict = {"test_dir": test_dir, "n_samples": len(all_outputs)}

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
    summary_path = os.path.join(config["save_dir"], "inference_summary.json")
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
    parser.add_argument("--start", type=int, default=0,
                        help="First test-set index to process (inclusive). Default 0.")
    parser.add_argument("--end", type=int, default=None,
                        help="Stop index (exclusive). Default = end of test set. "
                             "Combine with --start for range/parallel runs; reruns auto-skip already-scored clips.")
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
