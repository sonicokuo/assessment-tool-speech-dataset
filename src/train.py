"""
Training script for Overlap-Aware Speech Quality Description.
Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --adapter_variant concat-only --epochs 3
    python src/train.py --config configs/config.yaml --resume_from ./checkpoints/last.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import wandb
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from tqdm.auto import tqdm

from adapter import build_adapter


# ── Collate ──────────────────────────────────────────────
def collate_fn(batch):
    """Pad variable-length audio features and overlap info to the longest in the batch."""
    audio_features = [item["audio_features"] for item in batch]
    overlap_info = [item["overlap_info"] for item in batch]
    target_text = [item["target_text"] for item in batch]

    max_len = max(f.shape[0] for f in audio_features)
    B = len(batch)
    audio_dim = audio_features[0].shape[-1]
    overlap_dim = overlap_info[0].shape[-1]

    audio_padded = torch.zeros(B, max_len, audio_dim)
    overlap_padded = torch.zeros(B, max_len, overlap_dim)

    for i, (af, oi) in enumerate(zip(audio_features, overlap_info)):
        audio_padded[i, : af.shape[0]] = af
        overlap_padded[i, : oi.shape[0]] = oi

    return {
        "audio_features": audio_padded,
        "overlap_info": overlap_padded,
        "target_text": target_text,
    }


# ── Dataset ──────────────────────────────────────────────
class PreprocessedDataset(Dataset):
    """Loads pre-computed WavLM features + overlap info from .pt files."""

    def __init__(self, data_dir: str, descriptions_path: str):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])

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
        target_text = self.descriptions[stem]

        return {
            "audio_features": cached["audio_features"],
            "overlap_info": cached["overlap_info"],
            "target_text": target_text,
        }


# ── Loss ──────────────────────────────────────────────
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
) -> torch.Tensor:
    """Forward pass with pre-computed features."""
    audio_features = audio_features.to(device).to(torch.bfloat16)
    overlap_info = overlap_info.to(device).to(torch.bfloat16)

    prefix_embeds = adapter(audio_features, overlap_info)

    target_ids = tokenizer(
        text=target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["max_target_length"],
    ).input_ids.to(device)

    prompt_embeds = embed_layer(prompt_ids.expand(prefix_embeds.shape[0], -1))
    target_embeds = embed_layer(target_ids)

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)

    N = prefix_embeds.shape[1]
    P = prompt_embeds.shape[1]

    # -100 is the default ignore index for cross-entropy in PyTorch
    ignore_labels = torch.full((prefix_embeds.shape[0], N + P), -100, device=device)
    target_labels = target_ids.clone()
    target_labels[target_labels == tokenizer.pad_token_id] = -100
    labels = torch.cat([ignore_labels, target_labels], dim=1)

    outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
    return outputs.loss


# ── Training ──────────────────────────────────────────────
def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(config["seed"])

    # Tokenizer + LLM + LoRA
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

    if config["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable()

    embed_layer = llm.get_input_embeddings()

    # Adapter
    lm_hidden_size = llm.config.hidden_size
    adapter = build_adapter(config["adapter_variant"], lm_dim=lm_hidden_size).to(device).to(torch.bfloat16)

    # Prompt
    prompt_ids = tokenizer(config["prompt"], return_tensors="pt").input_ids.to(device)

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

    train_set = PreprocessedDataset(train_dir, config["descriptions_path"])
    val_set = PreprocessedDataset(val_dir, config["descriptions_path"])
    print(f"Loaded: train={len(train_set)}, val={len(val_set)}")

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
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": config["lr_adapter"]},
            {"params": llm.parameters(), "lr": config["lr_lora"]},
        ],
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
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None

    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], weights_only=False)
        adapter.load_state_dict(checkpoint["adapter_state_dict"])
        llm.load_state_dict(checkpoint["lora_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Wandb
    run_name = config.get("wandb_run_name") or f"{config['adapter_variant']}-seed{config['seed']}"
    if wandb_run_id:
        wandb.init(
            project=config["wandb_project"],
            id=wandb_run_id,
            resume="must",
            config=config,
        )
    else:
        wandb.init(
            project=config["wandb_project"],
            name=run_name,
            config=config,
        )
    wandb_run_id = wandb.run.id

    # Training loop
    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"]):
        print("\nEpoch: {}/{}".format(epoch + 1, config["epochs"]))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        # ── Train ──
        adapter.train()
        llm.train()
        train_loss = 0.0
        n_steps = 0

        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for batch_idx, batch in enumerate(train_loader):
            loss = compute_loss(
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
            )
            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), config["grad_clip"])
                torch.nn.utils.clip_grad_norm_(llm.parameters(), config["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            n_steps += 1

            if n_steps % config["log_every"] == 0:
                wandb.log(
                    {
                        "train_loss": loss.item() * accum_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "step": n_steps + epoch * len(train_loader),
                    }
                )

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(train_loss / n_steps)),
                lr="{:.06f}".format(float(curr_lr)))
            batch_bar.update()

        batch_bar.close()
        avg_train_loss = train_loss / n_steps

        # ── Validate ──
        avg_val_loss = None
        if (epoch + 1) % config["eval_every_epoch"] == 0:
            adapter.eval()
            llm.eval()
            val_loss = 0.0
            n_val = 0

            batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

            with torch.no_grad():
                for batch in val_loader:
                    loss = compute_loss(
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
                    )
                    val_loss += loss.item()
                    n_val += 1

                    batch_bar.set_postfix(
                        loss="{:.04f}".format(float(val_loss / n_val)))
                    batch_bar.update()

            batch_bar.close()
            avg_val_loss = val_loss / n_val
            wandb.log(
                {
                    "val_loss": avg_val_loss,
                    "train_loss_epoch": avg_train_loss,
                    "epoch": epoch + 1,
                }
            )

        # ── Print epoch summary ──
        print("\tTrain Loss {:.04f}".format(avg_train_loss))
        if avg_val_loss is not None:
            print("\tVal Loss {:.04f}".format(avg_val_loss))
        print("\tLearning Rate {:.07f}".format(curr_lr))

        # Save best
        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "adapter_state_dict": adapter.state_dict(),
                    "lora_state_dict": llm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "wandb_run_id": wandb_run_id,
                    "config": config,
                },
                os.path.join(config["save_dir"], "best.pt"),
            )
            print("Saved best val model")

        # Save last (for resuming)
        torch.save(
            {
                "epoch": epoch,
                "adapter_state_dict": adapter.state_dict(),
                "lora_state_dict": llm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "wandb_run_id": wandb_run_id,
                "config": config,
            },
            os.path.join(config["save_dir"], "last.pt"),
        )
        print("Saved epoch model")

    wandb.finish()
    print("\nTraining complete. Best val loss: {:.04f}".format(best_val_loss))


# ── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    args, unknown = parser.parse_known_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-")
        val = unknown[i + 1]
        if key in config:
            if config[key] is None:
                pass  # keep as string (e.g., resume_from path)
            elif type(config[key]) == bool:
                val = val.lower() in ("true", "1", "yes")
            elif type(config[key]) == int:
                val = int(val)
            elif type(config[key]) == float:
                val = float(val)
        config[key] = val

    os.makedirs(config["save_dir"], exist_ok=True)
    train(config)
