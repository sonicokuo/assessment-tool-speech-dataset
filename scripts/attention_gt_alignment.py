#!/usr/bin/env python3
"""Validate section_head overlap attention against ground-truth overlap windows.

Turns the qualitative "the maps look discriminative" claim into a quantitative
"the attention is grounded in real evidence" result.

For each clip, the sec_overlap attention map (T_p × F_p over BEATs patches) is
marginalized over frequency → a 1D time-attention. We then measure how much of
that attention MASS falls inside the ground-truth VAD overlap segments vs
outside, and compare to the chance baseline (the fraction of clip time that is
overlap — what a uniform attention would score).

    concentration_ratio = (attention mass inside GT overlap) / (fraction of time that is overlap)

  ratio > 1  → overlap query attends to actual overlap regions MORE than chance (grounded)
  ratio ≈ 1  → no better than uniform
  ratio < 1  → attends AWAY from overlap (anti-grounded)

Only clips with nonzero GT overlap are scored (clean clips have no overlap to
attend to). sec_pauses can be validated the same way against pause locations if
you pass a pause-segment CSV column, but overlap is the cleanest GT we have
(oracle VAD on s1/s2 stems).

Usage
-----
    python scripts/attention_gt_alignment.py \
      --inference_results $SHARED/checkpoints/v11_section_head_lora/inference_results.json \
      --features_csv      $SHARED/data/features_pyannote/test.csv \
      --output_md         $SHARED/checkpoints/v11_section_head_lora/attention_gt_alignment.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

F_P = 8          # BEATs frequency patches (128 mel / 16 patch)
SR = 16000       # sample rate for the *_vad sample-index segments


def parse_segments(raw: str) -> list[tuple[float, float]]:
    """'a-b;c-d' (sample indices) → [(start_s, end_s), ...]."""
    out = []
    for seg in (raw or "").split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            a, b = seg.split("-", 1)
            s0, s1 = int(a) / SR, int(b) / SR
            if s1 > s0:
                out.append((s0, s1))
        except (ValueError, IndexError):
            continue
    return out


def load_overlap_gt(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    """filename → GT overlap segments (prefers VAD-on-stems oracle)."""
    gt = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            raw = row.get("overlap_segments_vad") or row.get("overlap_segments") or ""
            gt[fn] = parse_segments(raw)
    return gt


def clip_duration(entry: dict) -> float | None:
    """Clip duration in seconds. Prefer the measured_duration_sec sidecar."""
    d = entry.get("measured_duration_sec")
    if d:
        return float(d)
    return None


def alignment_for_clip(entry: dict, segs: list[tuple[float, float]]) -> dict | None:
    """Compute the overlap attention-vs-GT concentration ratio for one clip."""
    import numpy as np
    amap = entry.get("attention_maps", {})
    if "overlap" not in amap or not segs:
        return None
    dur = clip_duration(entry)
    if not dur:
        return None

    alpha = np.asarray(amap["overlap"], dtype=float)
    T_p = alpha.size // F_P
    if T_p == 0:
        return None
    time_attn = alpha[: T_p * F_P].reshape(T_p, F_P).mean(axis=1)   # (T_p,)
    total = time_attn.sum()
    if total <= 0:
        return None
    time_attn = time_attn / total                                   # mass per bin
    bin_dur = dur / T_p

    mass_in = 0.0
    for i in range(T_p):
        t0, t1 = i * bin_dur, (i + 1) * bin_dur
        if any(t0 < s1 and t1 > s0 for s0, s1 in segs):            # bin intersects a GT seg
            mass_in += time_attn[i]

    overlap_dur = sum(s1 - s0 for s0, s1 in segs)
    frac_time = min(overlap_dur / dur, 1.0)
    if frac_time <= 0:
        return None
    return {
        "filename": entry["filename"],
        "mass_in_overlap": mass_in,
        "frac_time_overlap": frac_time,
        "concentration_ratio": mass_in / frac_time,
        "n_segments": len(segs),
        "duration": dur,
        "sfs_f1": entry.get("sfs_f1"),
    }


def main() -> int:
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inference_results", type=Path, required=True)
    p.add_argument("--features_csv", type=Path, required=True,
                   help="Per-split feature CSV with overlap_segments_vad")
    p.add_argument("--output_md", type=Path, default=None)
    p.add_argument("--min_sfs", type=float, default=None,
                   help="Only score clips with sfs_f1 >= this (drop degenerate generations)")
    args = p.parse_args()

    results = json.loads(args.inference_results.read_text())
    gt = load_overlap_gt(args.features_csv)
    print(f"Loaded {len(results)} clips; {len(gt)} GT overlap rows")

    rows = []
    for e in results:
        if args.min_sfs is not None and (e.get("sfs_f1") or 0) < args.min_sfs:
            continue
        r = alignment_for_clip(e, gt.get(e["filename"], []))
        if r is not None:
            rows.append(r)

    if not rows:
        print("No scorable clips (need attention_maps['overlap'] + GT overlap + duration).")
        return 1

    ratios = np.array([r["concentration_ratio"] for r in rows])
    masses = np.array([r["mass_in_overlap"] for r in rows])
    fracs = np.array([r["frac_time_overlap"] for r in rows])

    # Sign test: how many clips have ratio > 1 (attends to overlap more than chance)?
    n_grounded = int((ratios > 1.0).sum())
    # One-sided: is mean mass_in > mean frac_time? (paired)
    try:
        import scipy.stats as st
        wil = st.wilcoxon(masses, fracs, alternative="greater")
        pval = wil.pvalue
    except Exception:
        pval = None

    md = [
        "# Overlap attention vs. ground-truth alignment",
        "",
        f"**N clips scored** (have overlap GT + attention + duration): {len(rows)}",
        f"{'(filtered to sfs_f1 >= ' + str(args.min_sfs) + ')' if args.min_sfs is not None else ''}",
        "",
        "## Concentration ratio = (attention mass in GT overlap) / (fraction of time that is overlap)",
        "",
        f"- **mean ratio**: {ratios.mean():.3f}  (1.0 = chance / uniform)",
        f"- **median ratio**: {np.median(ratios):.3f}",
        f"- **clips with ratio > 1** (grounded): {n_grounded}/{len(rows)} = {100*n_grounded/len(rows):.1f}%",
        f"- mean attention mass inside GT overlap: {masses.mean():.3f}",
        f"- mean fraction of time that is overlap: {fracs.mean():.3f}",
    ]
    if pval is not None:
        md.append(f"- **Wilcoxon** (mass_in > frac_time, one-sided): p = {pval:.2e}")
    md += [
        "",
        "## Interpretation",
        "",
        "- mean ratio meaningfully > 1 → the overlap query attends to actual overlap",
        "  regions more than a uniform baseline would. This is quantitative evidence",
        "  that the section attention is grounded in real acoustic evidence, not decorative.",
        "- ratio ≈ 1 → attention is no better than chance at locating overlap.",
        "- The Wilcoxon p-value tests whether attention mass inside overlap windows",
        "  is significantly greater than the time fraction those windows occupy.",
    ]
    out = "\n".join(md)
    if args.output_md:
        args.output_md.write_text(out)
        print(f"Wrote {args.output_md}\n")
    print(out)
    # Also dump per-clip JSON for plotting / inspection
    if args.output_md:
        args.output_md.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
