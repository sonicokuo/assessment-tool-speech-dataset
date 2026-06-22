#!/usr/bin/env python
"""End-to-end driver for the AQUA-NL risk-coverage THESIS-PROOF.

Headline claim: a model that abstains on uncertain features by its OWN predicted
sigma is more faithful — risk (error rate on the answered subset) drops as it
abstains, faster than abstaining at random.

Pipeline
--------
1. Load a reliability-head checkpoint exactly the way src/inference.py does:
   LM (from config["lm_name"]) + LoRA wrap + AdapterWithAuxHead(reliability_head=True),
   structural config synced from the embedded checkpoint config.
2. For a test subset of preprocessed .pt clips, run ONLY the adapter forward
   (no LM generation) to get per-feature (mean, log_var) -> sigma = exp(0.5*log_var).
   sigma is the model's own per-feature "this number is unreliable" score.
3. Ground truth per (clip, feature): parse the observability TARGET text for that
   clip (src/sfs.ClaimParser). This is the same GT src/inference.py uses when it
   falls back to parsing the target, and it is honest: GT is naturally restricted
   to the features the target actually reports. A feature ABSENT from the target
   means the target ABSTAINED on it (F0 under overlap) -> excluded / reported
   separately, never counted as a recoverable item.
4. "correct" per (clip, feature) = the reliability head's MEAN prediction is within
   the SFS tolerance of GT. We use the head's mean (not an LM generation pass)
   because the mean is the natural uncertainty-paired prediction and avoids a
   2-3 hr generation run. (Documented choice: the headline model emits numbers via
   the LM; the head's mean is the value the sigma is a sigma OF.)
5. risk_coverage_report(sigma, correct) per feature and POOLED over the recoverable
   features: AURC(sigma-ordered) vs AURC(random) vs always-answer risk, plus
   aurc_gain_vs_random. Also a CONSTANT-uncertainty sanity baseline (zero-information
   sigma -> should ~equal random). The thesis holds iff sigma-ordered AURC < random
   AURC (aurc_gain_vs_random > 0).
6. Risk-coverage curve points at coverage {1.0, 0.8, 0.6, 0.4} for plotting.

Usage
-----
    python scripts/run_reliability_eval.py \
        --config configs/config.psc.emnlp.yaml \
        --checkpoint $SHARED/checkpoints/v21_observability/best.pt \
        --test_dir   $SHARED/data/processed_aug/test \
        --target_json $SHARED/data/descriptions_observability_test.json \
        --max_clips 500 \
        --out /tmp/reliability_v21.json

The GT source is the per-clip target text (--target_json). Pass --features_csv to
instead pull GT scalars from a feature CSV via feature_set.extract_scalars (kept for
completeness; the target-text path is the default because it carries the abstention
labels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from adapter import build_adapter                       # noqa: E402
from ckpt_io import load_llm_state_dict                 # noqa: E402
from feature_set import (                               # noqa: E402
    SUPERVISED_FEATURES,
    FEATURE_NAMES,
    RECOVERABLE_FEATURES,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
    extract_scalars,
)
from reliability_eval import risk_coverage_report, risk_coverage_curve  # noqa: E402
from sfs import ClaimParser, SFSScorer                  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model                    # noqa: E402


# Structural keys we must copy from the checkpoint's embedded config so the model is
# rebuilt the way it was trained (mirrors src/inference._STRUCTURAL_KEYS, trimmed to
# what the adapter forward needs).
_STRUCTURAL_KEYS = (
    "lm_name", "adapter_variant", "lora_rank", "lora_alpha", "lora_dropout",
    "lora_targets", "reliability_head", "tagged_mode",
)

# SFS short_name -> tolerance, using the feature_set short names. f0_sd is keyed as
# "f0_sd" in TOLERANCES; everything else maps 1:1.
_SFS = SFSScorer()
_TOL = _SFS.TOLERANCES


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sync_structural(config: dict, ckpt_cfg: dict) -> dict:
    for k in _STRUCTURAL_KEYS:
        if k in ckpt_cfg:
            if config.get(k) != ckpt_cfg[k]:
                log(f"[config] {k}: {config.get(k)!r} -> {ckpt_cfg[k]!r} (from ckpt)")
            config[k] = ckpt_cfg[k]
    return config


def build_model(config: dict, checkpoint: dict, device: torch.device):
    """Rebuild LM + LoRA + reliability-head adapter and load the checkpoint weights."""
    lm_name = config["lm_name"]
    log(f"[load] tokenizer + LM {lm_name}")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        lm_name, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # LoRA wrap (v21 is LoRA r16). Mirror inference.py exactly via the shared helper.
    if config.get("lora_rank"):
        from peft_config import lora_config_kwargs
        llm = get_peft_model(llm, LoraConfig(**lora_config_kwargs(config)))
        log(f"[LoRA] rank={config['lora_rank']} alpha={config.get('lora_alpha')}")

    lm_hidden = llm.config.hidden_size
    log(f"[load] lm_hidden_size={lm_hidden}")

    n_aux = len(SUPERVISED_FEATURES)
    adapter = (
        build_adapter(
            config["adapter_variant"],
            lm_dim=lm_hidden,
            n_aux_features=n_aux,
            reliability_head=bool(config.get("reliability_head", False)),
        )
        .to(device)
        .to(torch.bfloat16)
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    log(f"[load] adapter loaded (reliability_head={config.get('reliability_head')}, "
        f"n_aux={n_aux})")

    llm_sd = checkpoint.get("llm_state_dict") or checkpoint.get("lora_state_dict")
    missing, unexpected = load_llm_state_dict(
        llm, llm_sd, ckpt_format=checkpoint.get("ckpt_format"),
    )
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading LLM checkpoint: {unexpected[:5]}")
    log(f"[load] LLM state loaded ({len(missing)} missing/frozen-base keys, expected)")

    adapter.eval()
    llm.eval()
    return tokenizer, llm, adapter


def gt_from_target(target_text: str, parser: ClaimParser) -> dict[str, float]:
    """Parse the observability target text -> {short_name: gt_value}.

    Features ABSENT from the dict were abstained on by the target (e.g. F0 under
    overlap). Restricted to features in the SFS tolerance table.
    """
    gt = {}
    for c in parser.parse(target_text):
        if c.feature in _TOL:
            gt[c.feature] = c.value
    return gt


def gt_from_csv_row(row: dict) -> dict[str, float]:
    """GT from a feature CSV row via feature_set.extract_scalars.

    Returns {short_name: gt_value} only for features whose mask is True (present).
    No abstention information (CSV always has F0), so the target-text path is
    preferred when available.
    """
    scalars, mask = extract_scalars(row)
    out = {}
    for i, (short, _csv, _fmt) in enumerate(SUPERVISED_FEATURES):
        if bool(mask[i]):
            out[short] = float(scalars[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--target_json", default=None,
                    help="descriptions_observability_test.json — per-clip GT + abstention.")
    ap.add_argument("--features_csv", default=None,
                    help="fallback GT source (no abstention labels).")
    ap.add_argument("--max_clips", type=int, default=500)
    ap.add_argument("--out", default="/tmp/reliability_eval.json")
    args = ap.parse_args()

    if not args.target_json and not args.features_csv:
        raise SystemExit("need --target_json (preferred) or --features_csv for GT")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[device] {device}")

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    ckpt_cfg = checkpoint.get("config", {})
    log(f"[ckpt] epoch={checkpoint.get('epoch')} "
        f"best_val_sfs_f1={checkpoint.get('best_val_sfs_f1')}")
    config = sync_structural(config, ckpt_cfg)

    tokenizer, llm, adapter = build_model(config, checkpoint, device)

    # GT sources
    targets = None
    if args.target_json:
        with open(args.target_json) as f:
            targets = json.load(f)
        log(f"[gt] loaded {len(targets)} target descriptions (target-text GT)")
    csv_map = None
    if args.features_csv:
        import csv as _csv
        csv_map = {}
        with open(args.features_csv) as f:
            for row in _csv.DictReader(f):
                csv_map[os.path.splitext(row["filename"])[0]] = row
        log(f"[gt] loaded {len(csv_map)} CSV rows (fallback GT)")

    parser = ClaimParser()
    feat_idx = {short: i for i, (short, _c, _f) in enumerate(SUPERVISED_FEATURES)}

    files = sorted(f for f in os.listdir(args.test_dir) if f.endswith(".pt"))
    files = files[: args.max_clips]
    log(f"[data] scoring {len(files)} clips from {args.test_dir}")

    # Per-feature accumulators: sigma list + correct list (only where GT present
    # AND not abstained). Abstained features simply have no GT entry -> skipped.
    sigmas = {short: [] for short in FEATURE_NAMES}
    correct = {short: [] for short in FEATURE_NAMES}
    abst_skipped = {short: 0 for short in FEATURE_NAMES}   # GT absent (abstained/missing)
    n_used = 0

    for n, fn in enumerate(files):
        cached = torch.load(os.path.join(args.test_dir, fn), weights_only=False)
        stem = os.path.splitext(fn)[0]
        filename = cached.get("filename", fn)
        wav_stem = os.path.splitext(filename)[0]

        # GT lookup
        gt = None
        if targets is not None and stem in targets:
            gt = gt_from_target(targets[stem], parser)
        elif targets is not None and wav_stem in targets:
            gt = gt_from_target(targets[wav_stem], parser)
        elif csv_map is not None:
            row = csv_map.get(wav_stem) or csv_map.get(stem)
            if row is not None:
                gt = gt_from_csv_row(row)
        if not gt:
            continue

        af = cached["audio_features"].unsqueeze(0).to(device).to(torch.bfloat16)
        oi = cached["overlap_info"].unsqueeze(0).to(device).to(torch.bfloat16)
        with torch.no_grad():
            _prefix, scalar_pred = adapter(af, oi)
        mean_t, log_var_t = scalar_pred
        mean_v = mean_t.float().cpu().numpy().reshape(-1)        # (8,) raw units
        log_var_v = log_var_t.float().cpu().numpy().reshape(-1)  # (8,)
        sigma_v = np.exp(0.5 * log_var_v)                        # (8,) normalized-unit sigma

        n_used += 1
        for short in FEATURE_NAMES:
            i = feat_idx[short]
            if short not in gt:
                abst_skipped[short] += 1
                continue
            err = abs(float(mean_v[i]) - float(gt[short]))
            tol = _TOL[short]
            sigmas[short].append(float(sigma_v[i]))
            correct[short].append(1.0 if err <= tol else 0.0)

        if (n + 1) % 100 == 0:
            log(f"  ... {n + 1}/{len(files)} processed ({n_used} with GT)")

    log(f"[data] {n_used} clips with usable GT")

    # ── sigma sanity / degeneracy check ──
    log("\n=== SIGMA STATS (per feature, normalized units) ===")
    for short in FEATURE_NAMES:
        s = np.asarray(sigmas[short])
        if s.size == 0:
            log(f"  {short:14s}  (no scored items; abstained/missing on {abst_skipped[short]})")
            continue
        log(f"  {short:14s}  n={s.size:4d}  sigma min={s.min():.4f} "
            f"max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}  "
            f"acc={np.mean(correct[short]):.3f}  (skipped/abstained={abst_skipped[short]})")

    # ── per-feature risk-coverage ──
    def constant_baseline(corr):
        """Zero-information uncertainty: all sigma equal -> stable-sort keeps input
        order, so AURC ~ random base error rate (sanity floor)."""
        return risk_coverage_report(np.ones(len(corr)), corr)

    def curve_points(sig, corr, covs=(1.0, 0.8, 0.6, 0.4)):
        cov, risk = risk_coverage_curve(sig, corr)
        out = {}
        for c in covs:
            # nearest coverage grid point at or above c
            idx = int(np.searchsorted(cov, c - 1e-9))
            idx = min(idx, len(cov) - 1)
            out[f"cov_{c}"] = {"coverage": float(cov[idx]), "risk": float(risk[idx])}
        return out

    report = {
        "checkpoint": args.checkpoint,
        "epoch": checkpoint.get("epoch"),
        "best_val_sfs_f1": checkpoint.get("best_val_sfs_f1"),
        "n_clips_with_gt": n_used,
        "recoverable_features": sorted(RECOVERABLE_FEATURES),
        "ill_posed_features": sorted(ILL_POSED_UNDER_OVERLAP_FEATURES),
        "per_feature": {},
        "abstained_or_missing": abst_skipped,
    }

    log("\n=== PER-FEATURE RISK-COVERAGE (sigma-ordered vs random) ===")
    for short in FEATURE_NAMES:
        sig = np.asarray(sigmas[short])
        corr = np.asarray(correct[short])
        cls = "recoverable" if short in RECOVERABLE_FEATURES else "ill-posed"
        if sig.size < 2 or len(np.unique(corr)) < 2:
            log(f"  {short:14s} ({cls}): n={sig.size}, "
                f"acc={corr.mean() if sig.size else float('nan'):.3f} "
                f"-> skipped (need >=2 items and both correct/wrong present)")
            report["per_feature"][short] = {
                "class": cls, "n": int(sig.size),
                "acc": float(corr.mean()) if sig.size else None,
                "note": "insufficient variation for AURC",
            }
            continue
        rep = risk_coverage_report(sig, corr)
        const = constant_baseline(corr)
        pts = curve_points(sig, corr)
        report["per_feature"][short] = {
            "class": cls,
            "n": rep["n"],
            "acc": float(corr.mean()),
            "aurc_model": rep["aurc_model"],
            "aurc_random": rep["aurc_random"],
            "aurc_constant_sigma": const["aurc_model"],
            "aurc_gain_vs_random": rep["aurc_gain_vs_random"],
            "always_answer_risk": rep["always_answer_risk"],
            "curve_points": pts,
        }
        gain = rep["aurc_gain_vs_random"]
        verdict = "THESIS HOLDS" if gain > 0 else "no gain"
        log(f"  {short:14s} ({cls}): n={rep['n']:4d} acc={corr.mean():.3f}  "
            f"AURC_model={rep['aurc_model']:.4f}  AURC_rand={rep['aurc_random']:.4f}  "
            f"AURC_const={const['aurc_model']:.4f}  gain={gain:+.4f}  [{verdict}]")

    # ── POOLED over recoverable features ──
    pool_sig, pool_corr = [], []
    for short in RECOVERABLE_FEATURES:
        pool_sig.extend(sigmas[short])
        pool_corr.extend(correct[short])
    pool_sig = np.asarray(pool_sig)
    pool_corr = np.asarray(pool_corr)

    log("\n=== POOLED RISK-COVERAGE (recoverable features) ===")
    if pool_sig.size >= 2 and len(np.unique(pool_corr)) >= 2:
        rep = risk_coverage_report(pool_sig, pool_corr)
        const = constant_baseline(pool_corr)
        pts = curve_points(pool_sig, pool_corr)
        # spearman sigma vs error-indicator (1-correct): does higher sigma => more wrong?
        err_ind = 1.0 - pool_corr
        try:
            from scipy.stats import spearmanr, pointbiserialr
            rho, p = spearmanr(pool_sig, err_ind)
            pb, ppb = pointbiserialr(pool_corr.astype(int), pool_sig)
        except Exception as e:  # noqa: BLE001
            rho = p = pb = ppb = None
            log(f"  (scipy unavailable for correlation: {e})")
        report["pooled_recoverable"] = {
            "n": rep["n"],
            "acc": float(pool_corr.mean()),
            "aurc_model": rep["aurc_model"],
            "aurc_random": rep["aurc_random"],
            "aurc_constant_sigma": const["aurc_model"],
            "aurc_gain_vs_random": rep["aurc_gain_vs_random"],
            "always_answer_risk": rep["always_answer_risk"],
            "curve_points": pts,
            "spearman_sigma_vs_error": None if rho is None else float(rho),
            "spearman_p": None if p is None else float(p),
            "pointbiserial_correct_vs_sigma": None if pb is None else float(pb),
        }
        gain = rep["aurc_gain_vs_random"]
        verdict = "THESIS HOLDS" if gain > 0 else "THESIS FAILS"
        log(f"  n={rep['n']}  acc={pool_corr.mean():.3f}  base_err={rep['always_answer_risk']:.3f}")
        log(f"  AURC_model(sigma-ordered) = {rep['aurc_model']:.4f}")
        log(f"  AURC_random               = {rep['aurc_random']:.4f}")
        log(f"  AURC_constant-sigma(sanity)= {const['aurc_model']:.4f}")
        log(f"  aurc_gain_vs_random       = {gain:+.4f}   -> {verdict}")
        log(f"  curve points: " + "  ".join(
            f"cov={v['coverage']:.2f}->risk={v['risk']:.3f}" for v in pts.values()))
        if rho is not None:
            log(f"  Spearman(sigma, error)    = {rho:+.3f} (p={p:.2e})  "
                f"[positive => higher sigma predicts more error]")
            log(f"  point-biserial(correct,sigma) = {pb:+.3f} (p={ppb:.2e})  "
                f"[negative => correct items have lower sigma]")
    else:
        log("  POOLED: insufficient variation for AURC")
        report["pooled_recoverable"] = {"note": "insufficient variation"}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
