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

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model

from adapter import build_adapter
from sfs import ClaimParser, SFSScorer


# ── Dataset ──────────────────────────────────────────────
class PreprocessedDataset(Dataset):
    """Loads pre-computed WavLM features + overlap info from .pt files."""

    def __init__(self, data_dir: str, descriptions_path: str = None):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

        self.descriptions = None
        if descriptions_path and os.path.exists(descriptions_path):
            with open(descriptions_path) as f:
                self.descriptions = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cached = torch.load(
            os.path.join(self.data_dir, self.files[idx]),
            weights_only=False,
        )
        stem = os.path.splitext(self.files[idx])[0]

        result = {
            "audio_features": cached["audio_features"],
            "overlap_info": cached["overlap_info"],
            "filename": cached.get("filename", self.files[idx]),
            "overlap_segments": cached.get("overlap_segments", []),
        }

        if self.descriptions and stem in self.descriptions:
            result["target_text"] = self.descriptions[stem]

        return result


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
def evaluate(config: dict, checkpoint_path: str, test_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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

    # Generate + evaluate
    claim_parser = ClaimParser()
    scorer = SFSScorer()
    all_results = []
    all_outputs = []

    for i in range(len(test_set)):
        sample = test_set[i]
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
        # NOTE: SP features JSON format depends on Person A. SFSScorer expects
        # numeric features as {"f0_mean": 187.0, "snr": 28.0, ...} and overlap
        # as {"overlap_segments": [(start, end), ...]}. If Person A uses a
        # different key (e.g. "overlap_start"/"overlap_end"), update this section.
        ground_truth = {}
        if sp_features and stem in sp_features:
            ground_truth = sp_features[stem].copy()
        elif "target_text" in sample:
            target_claims = claim_parser.parse(sample["target_text"])
            ground_truth = {c.feature: c.value for c in target_claims}

        # Add overlap segments from Pyannote (always available from .pt files)
        if sample["overlap_segments"] and "overlap_segments" not in ground_truth:
            ground_truth["overlap_segments"] = sample["overlap_segments"]

        # SFS scoring
        if ground_truth:
            claims = claim_parser.parse(generated)
            result = scorer.score(claims, ground_truth)
            all_results.append(result)

            output_entry["sfs_precision"] = result["precision"]
            output_entry["sfs_recall"] = result["recall"]
            output_entry["sfs_f1"] = result["f1"]
            output_entry["claims"] = [(c.feature, c.value) for c in claims]

        all_outputs.append(output_entry)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test_set)} done")

    # Print results
    if all_results:
        avg_p = sum(r["precision"] for r in all_results) / len(all_results)
        avg_r = sum(r["recall"] for r in all_results) / len(all_results)
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)

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

        print(f"\nPer-feature accuracy:")
        for name in sorted(feature_total.keys()):
            correct = feature_correct.get(name, 0)
            total = feature_total[name]
            print(f"  {name:20s}: {correct}/{total} = {correct/total:.2f}")

    # Save outputs
    output_path = os.path.join(config["save_dir"], "inference_results.json")
    os.makedirs(config["save_dir"], exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)
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

    evaluate(config, args.checkpoint, args.test_dir)
