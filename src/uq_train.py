"""uq_train.py — the STANDALONE decoupled trainer for the UQ bake-off heads.

WHY THIS EXISTS (the kill-fast gate)
------------------------------------
Before paying for a heavy generative diffusion UQ, the gate (src/uq_bakeoff.py) asks:
does a CHEAP uncertainty head beat the INCUMBENT heteroscedastic-sigma head at predicting
per-frame SNR-map ERROR (risk-coverage AURC)? To answer it we must first TRAIN the four
uncertainty channels under a MATCHED budget (same epochs / lr / data / batch) so the
comparison is fair. This module is that trainer.

It is fully DECOUPLED: it never loads the 8B LM, never builds the encoder, never touches
the adapter. Every head is a tiny nn.Module (in_proj -> temporal conv -> per-frame head)
over the FROZEN WavLM features the pipeline already cached, so all four channels fit in
minutes on one GPU. The heads come from src/uq_heads.py + src/snr_map_head.py; this file
is just the training loop + the .pt orchestration that pairs processed clips with the
oracle SNR-map targets (mirroring src/snr_map_validate.py's manifest pairing).

THE FOUR CHANNELS (matched budget; saved where uq_bakeoff.extract_method_arrays expects)
----------------------------------------------------------------------------------------
  1. heteroscedastic (INCUMBENT): HeteroscedasticSNRMapHead (mean + log-sigma), masked
     Gaussian NLL. Saved as snr_sigma.pt {"sigma_head_state_dict", "config"} (K=1-MDN
     state_dict, exactly the carrier the harness loads the incumbent into).
  2. mdn: MDNSNRMapHead (K-component mixture), masked mixture NLL. Saved as mdn.pt
     {"mdn_head_state_dict", "config"}.
  3. mcdropout: MCDropoutSNRMapHead, masked Huber timeline loss (dropout p from config),
     uncertainty harvested at eval via n MC passes. Saved as mcdropout.pt
     {"mcdropout_head_state_dict", "config"}.
  4. ensemble: ENSEMBLE_SIZE independent SupervisedSNRMapHead point regressors, masked
     Huber, DIFFERENT seeds. Saved as ensemble/seed<seed>.pt {"snr_map_head_state_dict",
     "config"} each; their per-frame disagreement is the ensemble variance.

DECOUPLING / TESTABILITY
------------------------
The per-method training STEP (`train_step`) and the loss builders are pure torch and run
on CPU on a synthetic batch, so tests/test_uq_bakeoff.py can assert the loss decreases
without any data on disk. Only `main` / `build_loaders` touch the filesystem.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Heads (pure torch — no transformers / peft / wandb).
try:  # package-relative when imported as src.uq_train
    from .uq_heads import (
        MDNSNRMapHead, MCDropoutSNRMapHead, HeteroscedasticSNRMapHead,
    )
    from .snr_map_head import SupervisedSNRMapHead
except ImportError:  # flat import when src/ is on sys.path (matches the repo style)
    from uq_heads import (
        MDNSNRMapHead, MCDropoutSNRMapHead, HeteroscedasticSNRMapHead,
    )
    from snr_map_head import SupervisedSNRMapHead


# The four channels and the on-disk filenames + state-dict keys the bake-off extractor
# (uq_bakeoff.extract_method_arrays) expects. Keep these IN SYNC with that extractor.
CKPT_SPEC = {
    "heteroscedastic": {"file": "snr_sigma.pt", "key": "sigma_head_state_dict"},
    "mdn":             {"file": "mdn.pt",       "key": "mdn_head_state_dict"},
    "mcdropout":       {"file": "mcdropout.pt", "key": "mcdropout_head_state_dict"},
    # ensemble members live under ensemble/seed<seed>.pt with key snr_map_head_state_dict
    "ensemble":        {"dir": "ensemble",      "key": "snr_map_head_state_dict"},
}


# ════════════════════════════════════════════════════════════════════════════════
# data: pair processed_aug clips with oracle SNR-map targets (manifest, like validate)
# ════════════════════════════════════════════════════════════════════════════════
class SNRMapFrames(Dataset):
    """One example = (audio_features (T,1024), snr_map_target (T,), snr_map_mask (T,)).

    Pairing mirrors src/snr_map_validate.py and uq_bakeoff.extract_method_arrays: walk the
    processed-clip dir, read each clip's `filename`, look it up in the SNR-map manifest, and
    load the matching oracle target .pt. Clips whose filename is not in the manifest (or
    whose target has no supervised frame) are dropped at index-build time, so __getitem__
    always returns a usable, non-empty-mask example.
    """

    def __init__(self, processed_dir: str, snr_map_dir: str, max_clips: int = 0):
        self.processed_dir = processed_dir
        self.snr_map_dir = snr_map_dir
        manifest = json.load(open(os.path.join(snr_map_dir, "manifest.json")))
        pts = sorted(f for f in os.listdir(processed_dir) if f.endswith(".pt"))
        index: list[tuple[str, str]] = []   # (processed_pt_path, target_pt_path)
        for ptname in pts:
            # we need the clip's `filename` to hit the manifest; the .pt stem is usually it
            # but augmented clips carry an explicit `filename`, so we read it lazily only if
            # the stem-based guess misses (cheap: most stems map directly).
            stem_wav = os.path.splitext(ptname)[0] + ".wav"
            rel = manifest.get(stem_wav)
            if rel is None:
                # fall back to the stored filename (augmented clips)
                try:
                    fn = torch.load(os.path.join(processed_dir, ptname),
                                    map_location="cpu", weights_only=False).get("filename")
                except Exception:
                    fn = None
                rel = manifest.get(fn) if fn else None
            if rel is None:
                continue
            index.append((os.path.join(processed_dir, ptname),
                          os.path.join(snr_map_dir, rel)))
            if max_clips and len(index) >= max_clips:
                break
        if not index:
            raise RuntimeError(
                f"no (clip, SNR-target) pairs found under {processed_dir} / {snr_map_dir}"
            )
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ppath, tpath = self.index[i]
        cached = torch.load(ppath, map_location="cpu", weights_only=False)
        tgt = torch.load(tpath, map_location="cpu", weights_only=False)
        af = cached["audio_features"].float()                       # (T, 1024)
        target = tgt["snr_map_target"].float()                      # (T,)
        mask = tgt.get("snr_map_mask")
        mask = (torch.ones_like(target) if mask is None else mask.float())
        T = min(af.shape[0], target.shape[0], mask.shape[0])
        return af[:T], target[:T], mask[:T]


def collate(batch):
    """Right-pad a variable-length batch to the max T; pad frames carry mask=0.

    Returns (audio_features (B,Tmax,D), target (B,Tmax), mask (B,Tmax)) with the padded
    tail masked out so it contributes nothing to any masked loss.
    """
    Tmax = max(af.shape[0] for af, _, _ in batch)
    D = batch[0][0].shape[1]
    B = len(batch)
    af_b = torch.zeros(B, Tmax, D)
    tg_b = torch.zeros(B, Tmax)
    mk_b = torch.zeros(B, Tmax)
    for i, (af, tg, mk) in enumerate(batch):
        T = af.shape[0]
        af_b[i, :T] = af
        tg_b[i, :T] = tg
        mk_b[i, :T] = mk
    return af_b, tg_b, mk_b


def build_loader(processed_dir, snr_map_dir, batch_size, shuffle, num_workers,
                 seed=0, max_clips=0):
    ds = SNRMapFrames(processed_dir, snr_map_dir, max_clips=max_clips)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, generator=g, drop_last=False, pin_memory=True,
    ), len(ds)


# ════════════════════════════════════════════════════════════════════════════════
# per-method loss (pure torch — CPU-testable)
# ════════════════════════════════════════════════════════════════════════════════
def _masked_huber(pred, target, mask, delta=1.0):
    """Masked Huber over mask=True frames (mean; 0 when the mask is empty)."""
    err = pred - target
    per = F.huber_loss(err, torch.zeros_like(err), reduction="none", delta=delta) * mask
    denom = mask.sum().clamp(min=1.0)
    return per.sum() / denom


def compute_loss(method: str, head: torch.nn.Module, af, target, mask, huber_delta=1.0):
    """One method's masked loss on a (B,T,D)/(B,T)/(B,T) batch.

    method in {heteroscedastic, mdn}: the head's masked NLL on the predictive law.
    method in {mcdropout, ensemble}: masked Huber on the point regression timeline.
    Returns a scalar loss tensor.
    """
    mb = mask.to(torch.bool)
    if method in ("heteroscedastic", "mdn"):
        params = head.forward(af)                    # mixture / single-Gaussian params
        return head.nll(params, target, mb)
    elif method == "mcdropout":
        pred = head.forward(af)                       # (B, T) dropout follows training=True
        return _masked_huber(pred, target, mask, delta=huber_delta)
    elif method == "ensemble":
        pred = head.forward_timeline(af)              # (B, T)
        return _masked_huber(pred, target, mask, delta=huber_delta)
    raise ValueError(f"unknown method {method!r}")


def train_step(method, head, opt, af, target, mask, grad_clip=1.0, huber_delta=1.0):
    """One optimizer step; returns the scalar loss value (float). CPU-testable."""
    head.train()
    opt.zero_grad(set_to_none=True)
    loss = compute_loss(method, head, af, target, mask, huber_delta=huber_delta)
    loss.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
    opt.step()
    return float(loss.detach().cpu())


# ════════════════════════════════════════════════════════════════════════════════
# head construction
# ════════════════════════════════════════════════════════════════════════════════
def build_head(method: str, cfg: dict, seed: int = 0) -> torch.nn.Module:
    """Construct the nn.Module for a channel from the config, seeding init."""
    torch.manual_seed(seed)
    audio_dim = int(cfg.get("snr_map_audio_dim", 1024))
    hidden = int(cfg.get("snr_map_hidden", 256))
    ks = int(cfg.get("snr_map_kernel_size", 5))
    snr_bias = float(cfg.get("snr_map_snr_bias", 0.0))
    if method == "heteroscedastic":
        return HeteroscedasticSNRMapHead(
            audio_dim=audio_dim, hidden=hidden, kernel_size=ks, snr_bias=snr_bias)
    if method == "mdn":
        return MDNSNRMapHead(
            audio_dim=audio_dim, n_components=int(cfg.get("mdn_components", 3)),
            hidden=hidden, kernel_size=ks, snr_bias=snr_bias)
    if method == "mcdropout":
        return MCDropoutSNRMapHead(
            audio_dim=audio_dim, hidden=hidden, kernel_size=ks,
            p=float(cfg.get("mc_dropout_p", 0.1)), snr_bias=snr_bias)
    if method == "ensemble":
        return SupervisedSNRMapHead(
            audio_dim=audio_dim, hidden=hidden, kernel_size=ks,
            huber_delta=float(cfg.get("snr_map_huber_delta", 1.0)), snr_bias=snr_bias)
    raise ValueError(f"unknown method {method!r}")


# ════════════════════════════════════════════════════════════════════════════════
# cosine-with-warmup LR (no external scheduler dep)
# ════════════════════════════════════════════════════════════════════════════════
def _lr_at(step, total_steps, base_lr, warmup_ratio):
    warmup = max(1, int(warmup_ratio * total_steps))
    if step < warmup:
        return base_lr * step / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


# ════════════════════════════════════════════════════════════════════════════════
# train one channel
# ════════════════════════════════════════════════════════════════════════════════
def train_channel(method, cfg, loader, n_train, device, seed, save_path, log_prefix=""):
    """Train one channel under the matched budget; save its checkpoint. Returns history."""
    head = build_head(method, cfg, seed=seed).to(device)
    base_lr = float(cfg.get("lr_adapter", 3e-4))
    wd = float(cfg.get("weight_decay", 0.01))
    epochs = int(cfg.get("epochs", 8))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    huber_delta = float(cfg.get("snr_map_huber_delta", 1.0))
    opt = torch.optim.AdamW(head.parameters(), lr=base_lr, weight_decay=wd)

    steps_per_epoch = max(1, math.ceil(n_train / loader.batch_size))
    total_steps = epochs * steps_per_epoch
    history = []
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        head.train()
        run, nb = 0.0, 0
        for af, target, mask in loader:
            af = af.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            for grp in opt.param_groups:
                grp["lr"] = _lr_at(step, total_steps, base_lr, warmup_ratio)
            loss_v = train_step(method, head, opt, af, target, mask,
                                grad_clip=grad_clip, huber_delta=huber_delta)
            run += loss_v
            nb += 1
            step += 1
        ep_loss = run / max(1, nb)
        history.append(ep_loss)
        print(f"{log_prefix}[{method}] epoch {epoch+1}/{epochs} "
              f"loss={ep_loss:.4f} lr={opt.param_groups[0]['lr']:.2e} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # save in the layout the bake-off extractor expects
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    key = (CKPT_SPEC[method]["key"] if method != "ensemble"
           else CKPT_SPEC["ensemble"]["key"])
    torch.save({key: head.state_dict(), "config": dict(cfg),
                "method": method, "seed": seed, "loss_history": history},
               save_path)
    print(f"{log_prefix}[{method}] saved -> {save_path} "
          f"(final loss {history[-1]:.4f}, first {history[0]:.4f})", flush=True)
    return {"method": method, "seed": seed, "save_path": save_path,
            "loss_history": history, "first_loss": history[0],
            "final_loss": history[-1]}


# ════════════════════════════════════════════════════════════════════════════════
# config loader (tiny YAML subset — no pyyaml dep needed, but use it if present)
# ════════════════════════════════════════════════════════════════════════════════
def load_config(path: str) -> dict:
    try:
        import yaml  # noqa
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        # minimal fallback parser for flat key: value YAML (handles the keys we use)
        cfg: dict = {}
        for line in open(path):
            line = line.split("#", 1)[0].rstrip()
            if not line or ":" not in line or line[0] in " \t-":
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            if not v:
                continue
            try:
                cfg[k.strip()] = json.loads(v)
            except Exception:
                cfg[k.strip()] = v
        return cfg


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--processed_dir", default=None,
                    help="override the TRAIN processed-clip dir (default: data_dir/train)")
    ap.add_argument("--snr_map_dir", default=None,
                    help="override the TRAIN SNR-map target dir (default: cfg snr_map_dir.train)")
    ap.add_argument("--save_dir", default=None, help="override cfg save_dir")
    ap.add_argument("--methods", nargs="+",
                    default=["heteroscedastic", "mdn", "mcdropout", "ensemble"],
                    help="which channels to train")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="cap training clips (0 = all; for a smoke run)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=0, help="override cfg epochs (0 = use cfg)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    save_dir = args.save_dir or cfg.get("save_dir")
    os.makedirs(save_dir, exist_ok=True)

    data_dir = cfg.get("data_dir")
    processed_dir = args.processed_dir or os.path.join(data_dir, "train")
    snr_map_cfg = cfg.get("snr_map_dir", {})
    snr_map_dir = args.snr_map_dir or (
        snr_map_cfg["train"] if isinstance(snr_map_cfg, dict) else snr_map_cfg)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))

    print(f"[uq_train] device={args.device} processed={processed_dir}")
    print(f"[uq_train] snr_map={snr_map_dir} save_dir={save_dir}")
    print(f"[uq_train] epochs={cfg.get('epochs')} lr={cfg.get('lr_adapter')} "
          f"batch={batch_size} methods={args.methods}", flush=True)

    # ONE shared, seeded loader (same data + order budget across point-estimate channels).
    # The ensemble seeds re-init the head but keep the same data; that is the matched-budget
    # protocol (only init/seed differs, per Lakshminarayanan).
    loader, n_train = build_loader(
        processed_dir, snr_map_dir, batch_size, shuffle=True,
        num_workers=num_workers, seed=seed, max_clips=args.max_clips)
    print(f"[uq_train] {n_train} (clip, SNR-target) training pairs", flush=True)

    summary = {"n_train": n_train, "config_path": args.config, "channels": {}}

    for method in args.methods:
        if method == "ensemble":
            ens_dir = os.path.join(save_dir, CKPT_SPEC["ensemble"]["dir"])
            os.makedirs(ens_dir, exist_ok=True)
            seeds = cfg.get("ensemble_seeds") or list(range(int(cfg.get("ensemble_size", 5))))
            members = []
            for sidx, ms in enumerate(seeds):
                sp = os.path.join(ens_dir, f"seed{int(ms)}.pt")
                members.append(train_channel(
                    "ensemble", cfg, loader, n_train, args.device, int(ms), sp,
                    log_prefix=f"  (member {sidx+1}/{len(seeds)}) "))
            summary["channels"]["ensemble"] = members
        else:
            sp = os.path.join(save_dir, CKPT_SPEC[method]["file"])
            summary["channels"][method] = train_channel(
                method, cfg, loader, n_train, args.device, seed, sp)

    sumpath = os.path.join(save_dir, "uq_train_summary.json")
    json.dump(summary, open(sumpath, "w"), indent=2)
    print(f"[uq_train] wrote {sumpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
