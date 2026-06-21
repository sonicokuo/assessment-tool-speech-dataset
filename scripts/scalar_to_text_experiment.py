"""Scalar->text routing experiment for AQUA-NL (v17 decoupled head, NO retrain).

WHAT THIS MEASURES
------------------
v17's trustworthy SFS-F1 is ~0.564 and PRECISION-bound (~0.41): the LM STATES
every feature (recall ~0.99) but gets the VALUES wrong, worst on f0_mean (~12.6%
within tol), f0_sd (~13.6%), srmr (~22.6%). The DecoupledGroundingHead regresses
those same scalars from the BEATs patches, but its predictions never reach the
scored text. This script tests the ceiling of "scalar->text routing": if the
generated prose carried the HEAD's number instead of the LM's hallucinated digits,
how much would SFS precision (and thus F1) improve?

It does NOT retrain and does NOT regenerate text. For each test clip it:
  1. runs ONLY the decoupled head (adapter is NOT needed — the head reads BEATs
     patches straight off the .pt) -> pred_scalars in RAW units (8 features,
     SUPERVISED_FEATURES order: snr, srmr, f0_mean, f0_sd, speaking_rate,
     pause_count, pause_rate, overlap_ratio). One cheap forward per clip.
  2. takes v17's EXISTING generated text (inference_results.json) and SPLICES the
     head's value into each feature's claim, keeping the sentence + units identical
     so src/sfs.py's parser re-reads the new number.
  3. re-scores BOTH the original LM text and the head-spliced text with the
     committed src/sfs.py (HybridClaimParser + SFSScorer), and reports
     per-feature accuracy + aggregate precision/recall/F1 with bootstrap 95% CIs.

The delta (spliced - original) is the scalar->text routing ceiling.

GT SOURCE: features.json is absent on PSC, so (matching inference.py and
scripts/eval_trustworthy.py) GT is parsed from each clip's TARGET prose and the
scorer restricts the recall denominator to TOLERANCES keys. This keeps the
original and spliced text on the EXACT same GT footing as the trustworthy eval.

DESIGN NOTES
------------
- Splice is PURE replacement by default. If the LM did not mention a feature, that
  feature is left alone (the head's value is NOT injected) so we first measure the
  pure routing ceiling. `--inject-missing` adds the head's value in the canonical
  template sentence for any scored feature the LM omitted (an upper bound that also
  fixes recall, but recall is already ~0.99 so the gain is small).
- The splice + scoring is PURE-PYTHON and CPU-testable (tests/test_scalar_to_text.py).
  Only step 1 (head forward) needs a GPU, and only because BEATs patches live on the
  GPU-resident model dim; the head itself is tiny.
- Resume-friendly: head predictions are cached to <out_dir>/head_preds.json, flushed
  every --flush clips (atomic tmp->rename). A crash re-runs only the un-cached clips.

USAGE (head forward on GPU, then score):
    python scripts/scalar_to_text_experiment.py \
        --config        configs/config.psc.emnlp.v17decoupled.yaml \
        --checkpoint    /ocean/.../checkpoints/v17_decoupled/best.pt \
        --test_dir      /ocean/.../data/processed_aug/test \
        --inference_results /ocean/.../checkpoints/v17_decoupled/inference_results.json \
        --out           /ocean/.../checkpoints/v17_decoupled/scalar_to_text_eval.json \
        --bootstrap 2000 --seed 0

SCORE-ONLY (re-run scoring from a finished head_preds.json cache, no GPU):
    python scripts/scalar_to_text_experiment.py ... --score_only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# NOTE: we deliberately do NOT import feature_set at module load — it pulls in torch,
# and the splice + scoring path is meant to be torch-free / CPU-testable. The head's
# feature order is hardcoded as HEAD_KEYS below and asserted against
# feature_set.SUPERVISED_FEATURES inside compute_head_predictions() (which needs torch
# anyway), so a drift in the canonical catalog fails loudly before any GPU work.
from sfs import HybridClaimParser, SFSScorer

# Reuse the trustworthy eval's bootstrap + degeneracy + GT-from-target logic so the
# numbers are directly comparable with checkpoints/v17_decoupled/trustworthy_eval.json.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_trustworthy import (  # noqa: E402
    bootstrap_ci,
    per_feature_bootstrap,
    is_degenerate,
)


# ── Per-feature splice specification ─────────────────────────────────────────
# For each of the 8 head-predicted features (SUPERVISED_FEATURES order) we define:
#   sfs_key : the feature name SFSScorer scores it under.
#   fmt     : a callable value -> string, matching the digit format the LM emits and
#             the verbalizer's SUPERVISED_FEATURES format (so the spliced sentence is
#             byte-for-byte what a correct LM would have produced).
#   anchor  : a compiled regex with exactly ONE capturing group around the NUMBER,
#             anchored on the SAME lead-in src/sfs.py's parser uses to bind that
#             feature, so substituting group 1 leaves the sentence/units intact and
#             the parser re-reads the new value as the same feature. Negative classes
#             keep snr/f0_mean from cross-firing on neighbouring sentences.
#
# These anchors are intentionally TIGHTER than the SFS parser (they target the exact
# untagged-prose phrasings v17 emits, see inference_results.json) so we never rewrite
# the wrong number. A feature whose anchor does not match (LM phrased it differently,
# or omitted it) is left untouched unless --inject-missing is set.

def _fmt_int(v: float) -> str:
    return str(int(round(v)))


def _fmt2(v: float) -> str:
    return f"{v:.2f}"


def _fmt3(v: float) -> str:
    return f"{v:.3f}"


def _fmt4(v: float) -> str:
    return f"{v:.4f}"


# number sub-pattern: optional sign, digits, optional decimals
_NUM = r"(-?\d+\.?\d*)"

SPLICE_SPECS: dict[str, dict] = {
    "snr": {
        "fmt": _fmt2,
        "anchor": re.compile(
            r"(?:signal-to-noise\s+ratio\s*(?:\(SNR\))?\s*|SNR\s*)(?:=|≈|~|is|of)\s*"
            + _NUM + r"\s*dB",
            re.IGNORECASE,
        ),
        "template": "The signal-to-noise ratio SNR is {v} dB.",
    },
    "srmr": {
        "fmt": _fmt4,
        "anchor": re.compile(
            r"(?:SRMR|reverberation\s+score\s*(?:\(SRMR\))?)\s*(?:=|≈|~|is|of)\s*" + _NUM,
            re.IGNORECASE,
        ),
        "template": "The SRMR is {v}.",
    },
    "f0_mean": {
        "fmt": _fmt2,
        # "F0 mean is X Hz" but NOT "F0 standard deviation ... Hz" — require the word
        # "mean" (or bare "F0 is") and forbid "deviation"/"SD" between F0 and the number.
        "anchor": re.compile(
            r"F0\s+mean\s*(?:=|≈|~|is|of)\s*" + _NUM + r"\s*Hz",
            re.IGNORECASE,
        ),
        "template": "The F0 mean is {v} Hz.",
    },
    "f0_sd": {
        "fmt": _fmt2,
        "anchor": re.compile(
            r"F0\s+(?:standard\s+)?deviation(?:\s*\(?\s*SD\s*\)?)?\s*(?:=|≈|~|is|of)\s*"
            + _NUM + r"\s*Hz",
            re.IGNORECASE,
        ),
        "template": "The F0 standard deviation SD is {v} Hz.",
    },
    "speaking_rate": {
        "fmt": _fmt3,
        "anchor": re.compile(
            r"speaking\s+rate\s*(?:=|≈|~|is|of|:)\s*" + _NUM
            + r"\s*syl(?:lables?)?\s*(?:/\s*|per\s+)s(?:ec(?:ond)?)?",
            re.IGNORECASE,
        ),
        "template": "The speaking rate is {v} syl/sec.",
    },
    "pause_count": {
        "fmt": _fmt_int,
        # integer; anchor on "pause count is N" (no decimals).
        "anchor": re.compile(
            r"pause\s+count\s*(?:=|≈|~|is|of)\s*(\d+)",
            re.IGNORECASE,
        ),
        "template": "The pause count is {v}.",
    },
    "pause_rate": {
        "fmt": _fmt3,
        "anchor": re.compile(
            r"pause\s+rate\s*(?:=|≈|~|is|of)\s*" + _NUM + r"\s*per\s+min(?:ute)?",
            re.IGNORECASE,
        ),
        "template": "The pause rate is {v} per min.",
    },
    "overlap_ratio": {
        "fmt": _fmt4,
        "anchor": re.compile(
            r"overlap\s+ratio(?:\s+of\s+the\s+sample)?\s*(?:=|≈|~|is|of)\s*" + _NUM,
            re.IGNORECASE,
        ),
        "template": "The overlap ratio is {v}.",
    },
}

# head index -> sfs_key, in feature_set.SUPERVISED_FEATURES order. The head emits one
# scalar per entry in this exact order; HEAD_KEYS[j] names the j-th head output. This
# is hardcoded (not imported) to keep the splice/scoring path torch-free; it is
# asserted == feature_set.SUPERVISED_FEATURES order inside compute_head_predictions()
# so a catalog drift fails loudly before any GPU work. All 8 are SFS-scored.
HEAD_KEYS: list[str] = [
    "snr", "srmr", "f0_mean", "f0_sd",
    "speaking_rate", "pause_count", "pause_rate", "overlap_ratio",
]
assert all(k in SPLICE_SPECS for k in HEAD_KEYS), (
    f"missing splice spec for {[k for k in HEAD_KEYS if k not in SPLICE_SPECS]}"
)


def splice_text(generated: str, head_values: dict[str, float], inject_missing: bool = False) -> str:
    """Replace each feature's numeric value in `generated` with the head's prediction.

    For every (feature -> value) in `head_values`:
      - if the feature's anchor regex matches `generated`, substitute the FIRST match's
        captured number with the head's value (formatted to that feature's digit format),
        leaving the rest of the sentence and the units untouched so the SFS parser reads
        the new number as the same feature.
      - if the anchor does NOT match (the LM phrased it differently or omitted it):
          * inject_missing=False (default): leave the text alone (pure-replacement ceiling).
          * inject_missing=True: append the head's value in the feature's canonical
            template sentence (also lifts recall; upper bound).

    Pure string op, no model deps — CPU-testable.
    """
    out = generated
    for key, value in head_values.items():
        spec = SPLICE_SPECS.get(key)
        if spec is None:
            continue
        new_num = spec["fmt"](value)
        anchor = spec["anchor"]
        m = anchor.search(out)
        if m is not None:
            # Replace ONLY the captured number (group 1), keep everything else.
            s, e = m.span(1)
            out = out[:s] + new_num + out[e:]
        elif inject_missing:
            sentence = " " + spec["template"].format(v=new_num)
            out = out.rstrip()
            if out and not out.endswith((".", "!", "?")):
                out += "."
            out += sentence
    return out


# ── Head forward (GPU) ───────────────────────────────────────────────────────
def compute_head_predictions(config: dict, checkpoint_path: str, test_dir: str,
                             out_dir: str, flush_every: int = 200) -> dict[str, dict]:
    """Run ONLY the DecoupledGroundingHead on every test clip -> {filename: {feat: val}}.

    Reconstructs the head from the v17 config + checkpoint's decoupled_head_state_dict
    (NO adapter, NO LM, NO autoregressive generation). Reads each clip's precomputed
    BEATs patches from the .pt and pools the per-feature queries over them. Caches to
    <out_dir>/head_preds.json, flushed every `flush_every` clips; re-runs skip cached
    filenames so a crash is cheap.
    """
    import torch  # local import so --score_only path stays torch-free
    from decoupled_grounding import DecoupledGroundingHead
    from dataset import PreprocessedDataset
    from feature_set import N_FEATURES, SUPERVISED_FEATURES

    # Guard against catalog drift: HEAD_KEYS (hardcoded for the torch-free splice path)
    # MUST equal the canonical SUPERVISED_FEATURES order the head was trained on.
    catalog = [name for name, _csv, _fmt in SUPERVISED_FEATURES]
    if HEAD_KEYS != catalog:
        raise RuntimeError(
            f"HEAD_KEYS {HEAD_KEYS} != feature_set.SUPERVISED_FEATURES {catalog}; "
            "update HEAD_KEYS + SPLICE_SPECS to match the catalog."
        )

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "head_preds.json")
    preds: dict[str, dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                preds = json.load(f)
            print(f"[head][resume] {len(preds)} clips already cached in {cache_path}")
        except json.JSONDecodeError:
            print(f"[head][resume] could not parse {cache_path}; starting fresh")
            preds = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[head] device = {device}")

    # Rebuild the head EXACTLY as train.py did (configs/config.psc.emnlp.v17decoupled.yaml).
    feature_init_bias = config.get("decoupled_feature_init_bias")
    head = DecoupledGroundingHead(
        d_model=int(config.get("decoupled_d_model", 256)),
        d_patch=int(config.get("spec_d_patch", 768)),
        n_features=N_FEATURES,
        n_heads=int(config.get("decoupled_n_heads", 1)),
        readout_hidden=config.get("decoupled_readout_hidden"),
        huber_delta=float(config.get("decoupled_huber_delta", 1.0)),
        feature_init_bias=feature_init_bias,
    ).to(device)

    # Load ONLY the head's state dict — avoid materializing the 17GB LM into RAM.
    # mmap=True lazy-maps the archive so only the tiny head tensors we actually touch
    # are paged in (the LM/adapter blocks are never read). weights_only is False
    # because the ckpt also holds the full config dict.
    print(f"[head] loading decoupled_head_state_dict from {checkpoint_path} ...")
    try:
        ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu", mmap=True)
    except TypeError:
        # older torch without mmap kwarg — falls back to a full CPU load (then freed).
        ck = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    hsd = ck.get("decoupled_head_state_dict")
    if hsd is None:
        raise RuntimeError(
            "checkpoint has no decoupled_head_state_dict — is this a v17 decoupled run?"
        )
    head.load_state_dict(hsd)
    head.eval()
    del ck  # free the (large) checkpoint dict ASAP

    ds = PreprocessedDataset(test_dir, descriptions_path=None)
    n_total = len(ds)
    print(f"[head] test set: {n_total} clips from {test_dir}")

    def flush():
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(preds, f)
        os.replace(tmp, cache_path)

    n_new = 0
    head_dtype = head.K_proj.weight.dtype
    with torch.no_grad():
        for i in range(n_total):
            sample = ds[i]
            fn = sample["filename"]
            if fn in preds:
                continue
            patches = sample.get("beats_patches")
            if patches is None:
                # No BEATs patches in this .pt — head can't run; record empty.
                preds[fn] = {}
            else:
                p = patches.unsqueeze(0).to(device).to(head_dtype)  # (1, P, d_patch)
                _A, _z, pred_scalars = head(p)                       # (1, n_features)
                vals = pred_scalars.squeeze(0).float().cpu().tolist()
                preds[fn] = {HEAD_KEYS[j]: float(vals[j]) for j in range(N_FEATURES)}
            n_new += 1
            if n_new % flush_every == 0:
                flush()
                print(f"[head] {i + 1}/{n_total} done ({len(preds)} cached)")
    flush()
    print(f"[head] complete: {len(preds)} clips cached in {cache_path}")
    return preds


# ── Scoring (CPU) ────────────────────────────────────────────────────────────
def score_entries(entries: list[dict], head_preds: dict[str, dict],
                  inject_missing: bool, score_overlap_spans: bool) -> tuple[list, list]:
    """Score original LM text and head-spliced text for every entry.

    Returns (orig_results, spliced_results), each a list of per-clip dicts with
    precision/recall/f1/per_feature/degenerate (same shape as eval_trustworthy).
    GT = parse(target) restricted to TOLERANCES by the scorer (matches inference.py).
    """
    parser = HybridClaimParser()
    scorer = SFSScorer()
    orig_results, spliced_results = [], []

    for e in entries:
        gen = e.get("generated", "") or ""
        tgt = e.get("target", "") or ""
        fn = e.get("filename")

        gt_claims = parser.parse(tgt)
        ground_truth = {c.feature: c.value for c in gt_claims}
        if (score_overlap_spans and e.get("overlap_segments")
                and "overlap_segments" not in ground_truth):
            ground_truth["overlap_segments"] = e["overlap_segments"]

        # original
        oc = parser.parse(gen)
        ores = scorer.score(oc, ground_truth)
        orig_results.append({
            "filename": fn,
            "precision": ores["precision"], "recall": ores["recall"], "f1": ores["f1"],
            "per_feature": ores["per_feature"], "degenerate": is_degenerate(gen),
        })

        # spliced
        hv = head_preds.get(fn, {})
        spliced = splice_text(gen, hv, inject_missing=inject_missing) if hv else gen
        sc = parser.parse(spliced)
        sres = scorer.score(sc, ground_truth)
        spliced_results.append({
            "filename": fn,
            "precision": sres["precision"], "recall": sres["recall"], "f1": sres["f1"],
            "per_feature": sres["per_feature"], "degenerate": is_degenerate(spliced),
            "spliced_text": spliced,
        })

    return orig_results, spliced_results


def aggregate_block(results: list, B: int, seed: int) -> dict:
    """Aggregate SFS p/r/f1 + per-feature accuracy with bootstrap CIs."""
    f1 = [r["f1"] for r in results]
    pr = [r["precision"] for r in results]
    rc = [r["recall"] for r in results]
    out = {
        "n": len(results),
        "f1": bootstrap_ci(f1, B, seed),
        "precision": bootstrap_ci(pr, B, seed + 1),
        "recall": bootstrap_ci(rc, B, seed + 2),
    }
    feats = set()
    for r in results:
        for f in r["per_feature"]:
            feats.add(f["feature"])
    out["per_feature"] = {}
    for i, ft in enumerate(sorted(feats)):
        pf = per_feature_bootstrap(results, ft, B, seed + 100 + i)
        if pf is not None:
            out["per_feature"][ft] = pf
    return out


def head_standalone_accuracy(entries: list[dict], head_preds: dict[str, dict],
                             B: int, seed: int) -> dict:
    """Per-feature: is the HEAD's raw scalar within SFS tolerance of GT?

    GT = parse(target). Measured ONLY over clips whose target prose states that
    feature (so the denominator matches what the LM/spliced accuracy is measured on).
    Returns {feature: {accuracy, n, ci}}.
    """
    import numpy as np
    parser = HybridClaimParser()
    scorer = SFSScorer()
    tol = scorer.TOLERANCES

    # Per feature: list of (correct_int) over clips where GT has the feature AND the
    # head predicted it.
    per_feat_clips: dict[str, list[int]] = {k: [] for k in HEAD_KEYS}
    for e in entries:
        fn = e.get("filename")
        tgt = e.get("target", "") or ""
        gt = {c.feature: c.value for c in parser.parse(tgt)}
        hv = head_preds.get(fn, {})
        for key in HEAD_KEYS:
            if key not in gt or key not in tol or key not in hv:
                continue
            correct = abs(hv[key] - gt[key]) <= tol[key]
            per_feat_clips[key].append(1 if correct else 0)

    out: dict[str, dict] = {}
    for key, vals in per_feat_clips.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        point, lo, hi = bootstrap_ci(arr, B, seed + abs(hash(key)) % 1000)
        out[key] = {"accuracy": point, "n": len(vals), "ci": [lo, hi]}
    return out


def fmt_ci(triple) -> str:
    p, lo, hi = triple
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--inference_results", required=True,
                    help="v17's existing inference_results.json (generated + target text).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flush", type=int, default=200, help="flush head_preds cache every N clips")
    ap.add_argument("--inject-missing", dest="inject_missing", action="store_true",
                    help="inject head value for features the LM omitted (default off = pure replacement)")
    ap.add_argument("--score_only", action="store_true",
                    help="skip the GPU head forward; reuse the cached head_preds.json next to --out")
    ap.add_argument("--score_overlap_spans", action="store_true",
                    help="add overlap_segments to GT denominator (default off, matches untagged v17)")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Step 1 — head predictions (GPU), or load the cache (--score_only).
    if args.score_only:
        cache_path = os.path.join(out_dir, "head_preds.json")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"--score_only but no cache at {cache_path}")
        with open(cache_path) as f:
            head_preds = json.load(f)
        print(f"[score_only] loaded {len(head_preds)} head predictions from {cache_path}")
    else:
        head_preds = compute_head_predictions(
            config, args.checkpoint, args.test_dir, out_dir, flush_every=args.flush,
        )

    # Step 2 — load v17's generated/target text.
    with open(args.inference_results) as f:
        entries = json.load(f)
    print(f"[score] {len(entries)} inference entries from {args.inference_results}")

    # Step 3 — score original vs spliced.
    B, seed = args.bootstrap, args.seed
    orig_results, spliced_results = score_entries(
        entries, head_preds, args.inject_missing, args.score_overlap_spans,
    )
    print(f"[bootstrap] B={B}")
    orig_agg = aggregate_block(orig_results, B, seed)
    spliced_agg = aggregate_block(spliced_results, B, seed + 500)
    head_acc = head_standalone_accuracy(entries, head_preds, B, seed + 1500)

    import numpy as np
    # paired per-clip F1 delta (spliced - original) on identical clips
    o_by = {r["filename"]: r for r in orig_results}
    s_by = {r["filename"]: r for r in spliced_results}
    common = sorted(set(o_by) & set(s_by))
    diffs = np.array([s_by[fn]["f1"] - o_by[fn]["f1"] for fn in common])
    paired = {
        "n_common": len(common),
        "mean_diff": bootstrap_ci(diffs, B, seed + 2500),
        "frac_spliced_better": float((diffs > 0).mean()) if len(diffs) else 0.0,
        "frac_orig_better": float((diffs < 0).mean()) if len(diffs) else 0.0,
        "frac_tie": float((diffs == 0).mean()) if len(diffs) else 0.0,
    }

    summary = {
        "bootstrap_B": B,
        "seed": seed,
        "inject_missing": args.inject_missing,
        "score_overlap_spans": args.score_overlap_spans,
        "n_clips": len(entries),
        "n_head_preds": len(head_preds),
        "original_lm": orig_agg,
        "head_spliced": spliced_agg,
        "head_standalone_accuracy": head_acc,
        "paired_spliced_minus_original": paired,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print tables ──
    print("\n" + "=" * 80)
    print(f"SCALAR->TEXT ROUTING EXPERIMENT  (v17 decoupled head, B={B}, inject_missing={args.inject_missing})")
    print("=" * 80)
    print(f"\n{'variant':16} {'n':>5}  {'SFS-F1 [95% CI]':30} {'precision':28} {'recall':28}")
    for name, agg in [("original LM", orig_agg), ("head-spliced", spliced_agg)]:
        print(f"{name:16} {agg['n']:>5}  {fmt_ci(agg['f1']):30} {fmt_ci(agg['precision']):28} "
              f"{fmt_ci(agg['recall']):28}")

    p, lo, hi = paired["mean_diff"]
    excludes_zero = (lo > 0) or (hi < 0)
    print(f"\nPAIRED  spliced - original  per-clip F1 delta (n={paired['n_common']})")
    print(f"  mean diff: {p:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"CI excludes 0: {excludes_zero}")
    print(f"  frac spliced>orig: {paired['frac_spliced_better']:.3f}  "
          f"orig>spliced: {paired['frac_orig_better']:.3f}  tie: {paired['frac_tie']:.3f}")

    print(f"\nPER-FEATURE SFS accuracy  [fraction of MADE claims within tolerance]")
    print(f"  {'feature':16} {'original LM':26} {'head-spliced':26} {'HEAD standalone':26}")
    feats = sorted(set(orig_agg["per_feature"]) | set(spliced_agg["per_feature"]) | set(head_acc))
    for ft in feats:
        o = orig_agg["per_feature"].get(ft)
        s = spliced_agg["per_feature"].get(ft)
        h = head_acc.get(ft)
        os_ = f"{o['accuracy']:.3f}({o['n_claims']})" if o else "—"
        ss_ = f"{s['accuracy']:.3f}({s['n_claims']})" if s else "—"
        hs_ = f"{h['accuracy']:.3f}({h['n']})" if h else "—"
        print(f"  {ft:16} {os_:26} {ss_:26} {hs_:26}")

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
