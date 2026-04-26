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
from sfs import ClaimParser, SFSScorer
from dataset import PreprocessedDataset, collate_fn
from text_metrics import compute_generation_metrics


# ── Loss ──────────────────────────────────────────────
def _ce_against_target(
    llm: nn.Module,
    embed_layer: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_text: list[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one LM forward (prefix + prompt + target) and return CE loss on the target tokens.

    Tokens of the prefix and prompt are masked out via -100 labels so the loss reflects
    only the autoregressive prediction of the target tokens.
    """
    target_ids = tokenizer(
        text=target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    prompt_embeds = embed_layer(prompt_ids.expand(prefix_embeds.shape[0], -1))
    target_embeds = embed_layer(target_ids)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)

    N = prefix_embeds.shape[1]
    P = prompt_embeds.shape[1]
    ignore_labels = torch.full((prefix_embeds.shape[0], N + P), -100, device=device)
    target_labels = target_ids.clone()
    target_labels[target_labels == tokenizer.pad_token_id] = -100
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

    # Prose CE loss (always computed)
    lm_loss_prose = _ce_against_target(
        llm, embed_layer, tokenizer,
        prefix_embeds, prompt_ids, target_text,
        max_length=config["max_target_length"],
        device=device,
    )
    metrics["loss_lm_prose"] = float(lm_loss_prose.detach().item())

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
            lm_loss_nums = _ce_against_target(
                llm, embed_layer, tokenizer,
                prefix_subset, prompt_ids, nums_subset,
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
        # Per-feature squared error, then mask out missing measurements before averaging.
        per_feat_se = (scalar_pred - gt_scalars_d) ** 2          # (B, n_feat)
        masked = per_feat_se * gt_mask_d                          # (B, n_feat)
        denom = gt_mask_d.sum().clamp(min=1.0)
        mse_loss = masked.sum() / denom
        metrics["loss_mse"] = float(mse_loss.detach().item())
    else:
        metrics["loss_mse"] = 0.0

    lambda_prose = float(config.get("lambda_prose", 1.0))
    lambda_nums = float(config.get("lambda_nums", 0.0))
    lambda_mse = float(config.get("lambda_mse", 0.0))

    total = lambda_prose * lm_loss_prose + lambda_nums * lm_loss_nums + lambda_mse * mse_loss
    metrics["loss_total"] = float(total.detach().item())
    return total, metrics


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

    # Trainable-parameter summary — printed once at startup and stashed for wandb.run.summary
    # after wandb.init() below. Helps compare adapter vs LoRA footprint across runs in one glance.
    lm_total = sum(p.numel() for p in llm.parameters())
    lora_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    adapter_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    trainable_total = lora_trainable + adapter_trainable
    param_summary = {
        "params/lm_total": lm_total,
        "params/lora_trainable": lora_trainable,
        "params/adapter_trainable": adapter_trainable,
        "params/trainable_total": trainable_total,
        "params/trainable_pct_of_lm": 100.0 * trainable_total / lm_total,
    }
    print(
        f"Parameters  —  LM total: {lm_total/1e9:.2f}B  |  LoRA trainable: {lora_trainable/1e6:.1f}M  "
        f"|  adapter trainable: {adapter_trainable/1e6:.1f}M  "
        f"|  grand total trainable: {trainable_total/1e6:.1f}M "
        f"({param_summary['params/trainable_pct_of_lm']:.3f}% of LM)"
    )

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

    # Push param counts into the wandb run summary so the run overview shows them without scrolling logs.
    for k, v in param_summary.items():
        wandb.run.summary[k] = v

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
                        "train_loss_lm_prose": loss_metrics["loss_lm_prose"],
                        "train_loss_lm_nums": loss_metrics["loss_lm_nums"],
                        "train_loss_mse": loss_metrics["loss_mse"],
                        "lr": scheduler.get_last_lr()[0],
                        "step": n_steps + epoch * len(train_loader),
                    }
                )

            batch_bar.set_postfix(
                loss="{:.04f}".format(float(train_loss / n_steps)),
                lr="{:.2e}".format(float(scheduler.get_last_lr()[0])))
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

            val_metrics_sum = {"loss_lm_prose": 0.0, "loss_lm_nums": 0.0, "loss_mse": 0.0}
            with torch.no_grad():
                for batch in val_loader:
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
                    )
                    val_loss += loss.item()
                    for k in val_metrics_sum:
                        val_metrics_sum[k] += loss_metrics[k]
                    n_val += 1

                    batch_bar.set_postfix(
                        loss="{:.04f}".format(float(val_loss / n_val)))
                    batch_bar.update()

            batch_bar.close()
            avg_val_loss = val_loss / n_val

            # ── Qualitative: generate text on a fixed slice of val, SFS-score, log to wandb ──
            # Catches degenerate outputs ("AND THE THE THE...") immediately and tracks SFS F1
            # epoch-by-epoch so you can see the faithfulness curve without waiting for inference.py.
            claim_parser = ClaimParser()
            sfs_scorer = SFSScorer()

            sample_rows = []
            sfs_f1s, sfs_precs, sfs_recs = [], [], []
            # val_sfs_n controls the val-time generation sample count. Default 32 (was 8) —
            # 8 was too noisy to read F1 trends across epochs; 32 cuts noise floor in half
            # for ~2 minutes additional generation time per epoch.
            n_samples = min(config.get("val_sfs_n", 32), len(val_set))
            with torch.no_grad():
                for i in range(n_samples):
                    sample = val_set[i]
                    af = sample["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
                    oi = sample["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
                    # Adapter may return (prefix, scalar_pred) when wrapped with aux head.
                    out = adapter(af, oi)
                    prefix = out[0] if isinstance(out, tuple) else out

                    prompt_emb = embed_layer(prompt_ids)
                    inputs_embeds = torch.cat([prefix, prompt_emb], dim=1)
                    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

                    gen_ids = llm.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=config.get("max_target_length", 256),  # matches training target length
                        do_sample=False,                                       # greedy for reproducibility
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
