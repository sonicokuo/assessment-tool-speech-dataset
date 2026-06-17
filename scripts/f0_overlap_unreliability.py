#!/usr/bin/env python3
"""Model-independent justification for overlap-aware F0 hedging.

Shows that the F0 MEASUREMENT itself becomes unreliable as speaker overlap rises:
the disagreement between the mixture F0 (Praat on the 2-speaker mix, the current
GT) and the well-posed clean-frame F0 (Praat restricted to non-overlapped voiced
frames) grows with overlap, and the fraction of clips whose F0 is entirely
undefined (no clean voiced frames) grows too. No model is involved, so this is a
clean signal-level result: the model's overlap-driven hedge fires exactly where
the underlying measurement is genuinely unreliable.

Usage:
  python scripts/f0_overlap_unreliability.py \
    --features_csv $SHARED/data/features_pyannote/test.csv \
    --clean_f0     $SHARED/data/clean_f0_test.json
"""
import argparse
import csv
import json
import math


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--clean_f0", required=True)
    ap.add_argument("--bins", default="0,0.25,0.5,0.75,1.01")
    ap.add_argument("--output_md", default=None)
    args = ap.parse_args()

    clean = json.loads(open(args.clean_f0).read())
    rows = list(csv.DictReader(open(args.features_csv)))
    edges = [float(x) for x in args.bins.split(",")]
    bins = [{"lo": a, "hi": b, "dis": [], "undef": 0, "n": 0}
            for a, b in zip(edges[:-1], edges[1:])]

    for r in rows:
        cf = clean.get(r.get("filename"))
        if cf is None:
            continue
        ov = _f(r.get("overlap_ratio_vad") or r.get("overlap_ratio"))
        if ov is None:
            continue
        b = next((x for x in bins if x["lo"] <= ov < x["hi"]), None)
        if b is None:
            continue
        b["n"] += 1
        cm = cf.get("f0_mean_hz")
        cm = cm if not (cm is None or (isinstance(cm, float) and math.isnan(cm))) else None
        if cm is None:
            b["undef"] += 1
            continue
        mix = _f(r.get("f0_mean_hz"))
        if mix is not None:
            b["dis"].append(abs(mix - cm))

    lines = [
        "# F0 measurement unreliability vs speaker overlap (model-independent)",
        "",
        "| overlap | n | % F0 undefined | mean |mix−clean| Hz | median Hz |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in bins:
        d = sorted(b["dis"])
        n = b["n"]
        mean = sum(d) / len(d) if d else float("nan")
        med = d[len(d) // 2] if d else float("nan")
        undef = 100 * b["undef"] / n if n else 0.0
        lines.append(f"| [{b['lo']:.2f},{b['hi']:.2f}) | {n} | {undef:.1f}% | "
                     f"{mean:.1f} | {med:.1f} |")
    lines += [
        "",
        "Rising disagreement and rising %-undefined with overlap mean the F0 "
        "measurement degrades with overlap, so overlap is a valid reliability "
        "signal and hedging on it is justified at the signal level.",
    ]
    out = "\n".join(lines)
    print(out)
    if args.output_md:
        open(args.output_md, "w").write(out)
        print(f"\nWrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
