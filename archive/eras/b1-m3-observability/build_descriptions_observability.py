#!/usr/bin/env python3
"""Build OBSERVABILITY-AWARE descriptions deterministically from the per-clip
feature CSVs (no LLM in the loop).

Motivation (paper pivot: "Observability-Aware Speech-Feature Description")
-------------------------------------------------------------------------
The old template builder (scripts/build_descriptions_deterministic.py) emits a
fixed-slot, 22-mold robotic paragraph:

    "The signal-to-noise ratio SNR is 15.63 dB. The SRMR is 5.1569. The F0 mean
     is 202.89 Hz and the F0 standard deviation SD is 10.74 Hz. ..."

86.8% of that text is boilerplate, so SFS recall is a STRUCTURAL ARTIFACT of the
fixed slots rather than a measurement of coverage. Worse, it always asserts an
F0 number even on heavily overlapped clips where single-speaker pitch is
physically unrecoverable from a 2-speaker mixture (an ill-posed estimate
presented as fact).

This builder instead emits a SELECTIVE, CONDITIONAL, QUALITATIVE-BAND target:

  1. RECOVERABLE features (snr, srmr, speaking_rate, pause_count, pause_rate,
     overlap_ratio) are reported as MEASURED NUMBERS (the grounded moat that SFS
     scores), but with VARIED phrasing and light qualitative bands so the prose
     reads naturally instead of as 22 identical molds.

  2. ILL-POSED-under-overlap features (f0_mean, f0_sd) are reported as a NUMBER
     ONLY WHEN the signal can physically support the estimate (low overlap AND a
     well-defined clean-frame F0). When overlap is high or the clean-frame F0 is
     undefined / too sparse, the target ABSTAINS: it emits a calibrated
     conditional hedge ("Because the two speakers overlap heavily, the pitch
     cannot be reliably estimated from the mixture") INSTEAD of a number.

  3. Causal connectors and conditional phrasing tie the claims together so the
     hedge is grounded in the measured overlap, not a constant string.

Because F0 is mentioned iff it is estimable, SFS recall becomes a REAL coverage
measurement and the hedge becomes a calibrated conditional rather than a fixed
trailing sentence.

F0 abstention decision (`--f0_overlap_threshold`, default 0.30)
---------------------------------------------------------------
F0 is ABSTAINED (hedged, no number) when ANY of:
  - the clip's overlap ratio >= threshold (heavy speaker overlap -> mixture F0
    is ill-posed), OR
  - the clean-frame F0 (from --clean_f0) is UNDEFINED (no non-overlap voiced
    frames), OR
  - the clean-frame F0 is defined but rests on too few clean voiced frames
    (`--min_clean_voiced_frac`, default 0.05) to be trustworthy.
Otherwise F0 is ASSERTED as a number.

The overlap ratio used for the decision is the VAD-on-stems ratio
(overlap_ratio_vad) when present (the true oracle overlap), else the
overlap_ratio column, else 0.0 (a clip with no overlap column is a clean
single-speaker recording -> F0 is well-posed).

Recoverable-vs-ill-posed (paper decision, flagged here for the verifier)
------------------------------------------------------------------------
  RECOVERABLE (kept numeric, SFS-scored):
      snr, srmr, speaking_rate, pause_count, pause_rate, overlap_ratio
  ILL-POSED under overlap (band / abstain):
      f0_mean, f0_sd
  NOTE: SRMR is treated as RECOVERABLE on purpose. It is a reverberation metric;
  it does not "recover" on clean speech the way pitch does, so abstaining on it
  under overlap would be wrong. Pitch is the one confirmed ill-posed case.

Inputs
------
  --features-dir   dir containing {train-100,dev,test}.csv (default
                   $SHARED/data/features_pyannote). Reuses the VAD-vs-pyannote
                   overlap source-of-truth logic from
                   build_descriptions_deterministic.py (prefers *_vad columns).
  --clean_f0       optional clean-frame F0 JSON(s) {filename: {f0_mean_hz,
                   f0_sd_hz, clean_voiced_frac}} from scripts/compute_clean_f0.py.
                   When present, the asserted F0 number is the WELL-POSED
                   clean-frame F0, not the ill-posed mixture F0.

Outputs
-------
  One JSON per split: data/descriptions_observability_{train,dev,test}.json
  (keyed by clip stem, value = the observability-aware description string).

Usage
-----
    python scripts/build_descriptions_observability.py \
        --features-dir $SHARED/data/features_pyannote \
        --clean_f0 $SHARED/data/clean_f0_train.json \
        --clean_f0 $SHARED/data/clean_f0_dev.json \
        --clean_f0 $SHARED/data/clean_f0_test.json \
        --out-dir data

The target format and SFS scoring of it are intentionally co-designed: the
recoverable numbers use phrasings the SFS regex accepts ("of"/"is"/"="), and the
hedge sentences are recognized by the SFS selective-scoring path
(SFSScorer.score_selective) as ABSTENTION for f0_mean / f0_sd.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# Reuse the canonical overlap source-of-truth logic (VAD vs pyannote columns).
from build_descriptions_deterministic import (  # noqa: E402
    _clean_overlap_segments,  # noqa: F401  (imported for completeness / reuse)
    _prepare_row_for_build,
)


# ── Recoverable / ill-posed feature partition (paper decision) ──────────────
RECOVERABLE_FEATURES = (
    "snr", "srmr", "speaking_rate", "pause_count", "pause_rate", "overlap_ratio",
)
ILL_POSED_FEATURES = ("f0_mean", "f0_sd")

DEFAULT_F0_OVERLAP_THRESHOLD = 0.30
DEFAULT_MIN_CLEAN_VOICED_FRAC = 0.05


def _stable_choice(stem: str, salt: str, n: int) -> int:
    """Deterministic 0..n-1 index from (stem, salt). Gives per-clip phrasing
    variety that is reproducible run-to-run (no RNG, no global state)."""
    h = hashlib.sha1(f"{stem}|{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def _to_float(val):
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in ("nan", "n/a", "na", "none"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Qualitative bands (light, never replace the number) ─────────────────────
def _snr_band(v: float) -> str:
    if v < 5:
        return "very low"
    if v < 12:
        return "low"
    if v < 20:
        return "moderate"
    if v < 30:
        return "high"
    return "very high"


def _srmr_band(v: float) -> str:
    # Higher SRMR ~ less reverberation / clearer modulation.
    if v < 3:
        return "heavily reverberant"
    if v < 5:
        return "moderately reverberant"
    if v < 8:
        return "lightly reverberant"
    return "clean"


def _rate_band(v: float) -> str:
    if v < 3.5:
        return "slow"
    if v < 5.5:
        return "measured"
    if v < 7.0:
        return "brisk"
    return "fast"


def _overlap_band(v: float) -> str:
    if v <= 0.001:
        return "no"
    if v < 0.2:
        return "light"
    if v < 0.5:
        return "moderate"
    if v < 0.75:
        return "heavy"
    return "near-constant"


# ── Per-feature numeric sentence templates (varied phrasing) ────────────────
# Each function returns one sentence asserting the MEASURED number in a form the
# SFS regex accepts. `stem` drives a deterministic template choice so the corpus
# has variety while each clip is reproducible.
def _snr_sentence(stem: str, v: float) -> str:
    band = _snr_band(v)
    opts = [
        f"The recording carries a {band} signal-to-noise ratio SNR of {v:.2f} dB.",
        f"Background noise is {band}, with an SNR of {v:.2f} dB.",
        f"At {v:.2f} dB, the signal-to-noise ratio SNR is {band}.",
        f"The SNR is {v:.2f} dB, a {band} noise level.",
    ]
    return opts[_stable_choice(stem, "snr", len(opts))]


def _srmr_sentence(stem: str, v: float) -> str:
    band = _srmr_band(v)
    opts = [
        f"The signal is {band}, reflected in an SRMR of {v:.4f}.",
        f"Reverberation is {band} here, with an SRMR of {v:.4f}.",
        f"The SRMR of {v:.4f} indicates {band} conditions.",
        f"The SRMR is {v:.4f}, so the recording is {band}.",
    ]
    return opts[_stable_choice(stem, "srmr", len(opts))]


def _speaking_rate_sentence(stem: str, v: float) -> str:
    band = _rate_band(v)
    opts = [
        f"The talker speaks at a {band} pace, a speaking rate of {v:.3f} syl/sec.",
        f"Delivery is {band}, with a speaking rate of {v:.3f} syl/sec.",
        f"The speaking rate is {v:.3f} syl/sec, a {band} pace.",
        f"The speaking rate runs to a {band} {v:.3f} syl/sec.",
    ]
    return opts[_stable_choice(stem, "srate", len(opts))]


def _pause_sentences(stem: str, count: float, rate: float | None) -> str:
    """Pause count + (optional) pause rate, woven into one sentence."""
    c = int(round(count))
    if c == 0:
        opts = [
            "The talker runs through without pausing, so the pause count is 0 and the pause rate is 0.000 per min.",
            "There are no breaks in delivery; the pause count is 0 and the pause rate is 0.000 per min.",
            "The pause count is 0 and the pause rate is 0.000 per min, an unbroken stretch of speech.",
        ]
        return opts[_stable_choice(stem, "pause0", len(opts))]
    word = "pause" if c == 1 else "pauses"
    if rate is None:
        opts = [
            f"Delivery is broken by {c} {word}, so the pause count is {c}.",
            f"The talker takes {c} {word}; the pause count is {c}.",
            f"The pause count is {c}, marking {c} {word} in the recording.",
        ]
        return opts[_stable_choice(stem, "pauseN_norate", len(opts))]
    opts = [
        f"Delivery is broken by {c} {word}, giving a pause count of {c} and a pause rate of {rate:.3f} per min.",
        f"The talker takes {c} {word}, so the pause count is {c} and the pause rate is {rate:.3f} per min.",
        f"With {c} {word} in all, the pause count is {c} and the pause rate is {rate:.3f} per min.",
    ]
    return opts[_stable_choice(stem, "pauseN", len(opts))]


def _overlap_sentence(stem: str, v: float) -> str:
    band = _overlap_band(v)
    if v <= 0.001:
        opts = [
            "Only one speaker is active, so the overlap ratio is 0.0000.",
            "There is no concurrent speech; the overlap ratio is 0.0000.",
            "The overlap ratio is 0.0000, a single-speaker recording.",
        ]
        return opts[_stable_choice(stem, "ov0", len(opts))]
    opts = [
        f"The two speakers overlap to a {band} degree, an overlap ratio of {v:.4f}.",
        f"Concurrent speech is {band}, with an overlap ratio of {v:.4f}.",
        f"The overlap ratio is {v:.4f}, indicating {band} co-channel speech.",
    ]
    return opts[_stable_choice(stem, "ovN", len(opts))]


# ── F0: assert (number) vs abstain (hedge) ──────────────────────────────────
def _f0_assert_sentence(stem: str, f0_mean: float, f0_sd: float | None) -> str:
    if f0_sd is None:
        opts = [
            f"Pitch is recoverable here: the F0 mean is {f0_mean:.2f} Hz.",
            f"With little overlap, the F0 mean can be estimated at {f0_mean:.2f} Hz.",
            f"The F0 mean is {f0_mean:.2f} Hz.",
        ]
        return opts[_stable_choice(stem, "f0a_nosd", len(opts))]
    opts = [
        f"Pitch is recoverable here: the F0 mean is {f0_mean:.2f} Hz and the F0 standard deviation SD is {f0_sd:.2f} Hz.",
        f"With little overlap, pitch can be measured: the F0 mean is {f0_mean:.2f} Hz with an F0 standard deviation SD of {f0_sd:.2f} Hz.",
        f"The F0 mean is {f0_mean:.2f} Hz and the F0 standard deviation SD is {f0_sd:.2f} Hz.",
    ]
    return opts[_stable_choice(stem, "f0a", len(opts))]


def _f0_hedge_sentence(stem: str, reason: str, overlap: float | None) -> str:
    """Calibrated conditional hedge. `reason` in {'overlap', 'undefined'}."""
    if reason == "overlap":
        ovtxt = f" ({overlap:.2f} overlap ratio)" if overlap is not None else ""
        opts = [
            f"Because the two speakers overlap heavily{ovtxt}, the pitch cannot be reliably estimated from the mixture, so no F0 value is reported.",
            f"The speakers overlap too much{ovtxt} for single-speaker pitch to be recovered, so the F0 is left unstated.",
            f"Given the heavy speaker overlap{ovtxt}, F0 is ill-posed on this mixture and is not asserted.",
        ]
        return opts[_stable_choice(stem, "f0h_ov", len(opts))]
    # undefined / too few clean frames
    opts = [
        "Too few clean, non-overlapped voiced frames are available to estimate pitch, so no F0 value is reported.",
        "There is not enough clean voiced speech to recover a trustworthy pitch, so the F0 is left unstated.",
        "Pitch cannot be estimated from the available clean frames, so no F0 is asserted.",
    ]
    return opts[_stable_choice(stem, "f0h_undef", len(opts))]


def f0_decision(row: dict, clean_f0: dict | None,
                overlap_threshold: float,
                min_clean_voiced_frac: float) -> tuple[str, dict]:
    """Decide whether to ASSERT (number) or ABSTAIN (hedge) on F0 for this row.

    Returns (action, info) where action in {"assert", "abstain"} and info holds
    the values / reason needed to render the sentence:
        assert  -> {"f0_mean": float, "f0_sd": float|None}
        abstain -> {"reason": "overlap"|"undefined", "overlap": float|None}

    Decision (any one triggers abstention):
      - overlap ratio >= overlap_threshold, OR
      - clean-frame F0 undefined (no non-overlap voiced frames), OR
      - clean-frame F0 rests on < min_clean_voiced_frac clean voiced frames.
    Otherwise assert the (clean-frame, when available) F0 number.
    """
    # Overlap ratio: prefer the VAD oracle, else pyannote, else 0 (clean clip).
    ov = _to_float(row.get("overlap_ratio_vad"))
    if ov is None:
        ov = _to_float(row.get("overlap_ratio"))
    if ov is None:
        ov = 0.0

    # Clean-frame F0 (well-posed) substitution when provided.
    f0_mean = None
    f0_sd = None
    clean_undefined = False
    too_sparse = False
    if clean_f0 is not None:
        fname = (row.get("filename") or "").strip()
        cf = clean_f0.get(fname)
        if cf is not None:
            cm = cf.get("f0_mean_hz")
            if cm is None or (isinstance(cm, float) and math.isnan(cm)):
                clean_undefined = True
            else:
                f0_mean = float(cm)
                cs = cf.get("f0_sd_hz")
                if cs is not None and not (isinstance(cs, float) and math.isnan(cs)):
                    f0_sd = float(cs)
                cvf = cf.get("clean_voiced_frac")
                if cvf is not None and float(cvf) < min_clean_voiced_frac:
                    too_sparse = True
        else:
            # No clean-F0 entry for this clip -> fall back to the mixture F0 in
            # the row (only trustworthy when overlap is low; the overlap gate
            # below still governs).
            f0_mean = _to_float(row.get("f0_mean_hz"))
            f0_sd = _to_float(row.get("f0_sd_hz"))
    else:
        f0_mean = _to_float(row.get("f0_mean_hz"))
        f0_sd = _to_float(row.get("f0_sd_hz"))

    # Abstention gates.
    if ov >= overlap_threshold:
        return "abstain", {"reason": "overlap", "overlap": ov}
    if clean_undefined or too_sparse:
        return "abstain", {"reason": "undefined", "overlap": ov}
    if f0_mean is None:
        # No usable F0 number even though overlap is low -> abstain (undefined).
        return "abstain", {"reason": "undefined", "overlap": ov}
    return "assert", {"f0_mean": f0_mean, "f0_sd": f0_sd}


# ── Description assembly ─────────────────────────────────────────────────────
def build_observability_description(
    row: dict,
    stem: str,
    clean_f0: dict | None,
    overlap_threshold: float,
    min_clean_voiced_frac: float,
    fallback_warned: dict | None = None,
) -> str:
    """Compose the observability-aware description for one CSV row."""
    fallback_warned = fallback_warned if fallback_warned is not None else {"warned": False}
    # Normalize overlap columns to the canonical VAD-or-clamped-pyannote values.
    row = _prepare_row_for_build(row, fallback_warned)

    sentences: list[str] = []

    # 1) Noise (SNR) — recoverable.
    snr = _to_float(row.get("snr_db"))
    if snr is not None:
        sentences.append(_snr_sentence(stem, snr))

    # 2) Reverb (SRMR) — recoverable.
    srmr = _to_float(row.get("srmr"))
    if srmr is not None:
        sentences.append(_srmr_sentence(stem, srmr))

    # 3) Pitch (F0 mean / F0 SD) — ill-posed under overlap: assert or abstain.
    action, info = f0_decision(row, clean_f0, overlap_threshold, min_clean_voiced_frac)
    if action == "assert":
        sentences.append(_f0_assert_sentence(stem, info["f0_mean"], info.get("f0_sd")))
    else:
        sentences.append(_f0_hedge_sentence(stem, info["reason"], info.get("overlap")))

    # 4) Tempo (speaking rate) — recoverable.
    srate = _to_float(row.get("praat_speaking_rate_syl_sec"))
    if srate is not None:
        sentences.append(_speaking_rate_sentence(stem, srate))

    # 5) Pauses (count + rate) — recoverable.
    pcount = _to_float(row.get("praat_pause_count"))
    prate = _to_float(row.get("praat_pause_rate_per_min"))
    if pcount is not None:
        sentences.append(_pause_sentences(stem, pcount, prate))

    # 6) Overlap (ratio) — recoverable.
    ovr = _to_float(row.get("overlap_ratio"))
    if ovr is not None:
        sentences.append(_overlap_sentence(stem, ovr))

    return " ".join(s for s in sentences if s).strip()


def _iter_rows(csv_path: Path):
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            yield row


SPLIT_FILES = {
    "train": "train-100.csv",
    "dev": "dev.csv",
    "test": "test.csv",
}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features-dir", type=Path,
                   default=Path(os.environ.get("SHARED",
                                "/ocean/projects/cis260125p/shared"))
                   / "data" / "features_pyannote")
    p.add_argument("--out-dir", type=Path, default=Path("data"),
                   help="output dir for descriptions_observability_{split}.json")
    p.add_argument("--splits", nargs="*", default=["train", "dev", "test"],
                   choices=list(SPLIT_FILES.keys()),
                   help="standard splits to build (default all three). Pass "
                        "'--splits' with no value to build ONLY the "
                        "--split-file entries (e.g. a clean-control-only run).")
    p.add_argument("--split-file", action="append", default=None,
                   metavar="NAME=PATH",
                   help="override/extend a split's CSV path, e.g. "
                        "clean=features_clean_control.csv (repeatable). The NAME "
                        "is the output suffix; PATH is resolved against "
                        "--features-dir if not absolute.")
    p.add_argument("--clean_f0", type=Path, action="append", default=None,
                   help="clean-frame F0 JSON(s) {filename:{f0_mean_hz,f0_sd_hz,"
                        "clean_voiced_frac}} (repeatable, merged). When present "
                        "the asserted F0 is the well-posed clean-frame value.")
    p.add_argument("--f0_overlap_threshold", type=float,
                   default=DEFAULT_F0_OVERLAP_THRESHOLD,
                   help="abstain on F0 when the clip's overlap ratio "
                        ">= this value (default %(default)s). The decision "
                        "ratio is overlap_ratio_vad when present, else "
                        "overlap_ratio, else 0.")
    p.add_argument("--min_clean_voiced_frac", type=float,
                   default=DEFAULT_MIN_CLEAN_VOICED_FRAC,
                   help="abstain on F0 when the clean-frame estimate rests on "
                        "fewer than this fraction of clean voiced frames "
                        "(default %(default)s).")
    p.add_argument("--combined-output", type=Path, default=None,
                   help="if set, also write a single merged JSON over all splits")
    args = p.parse_args()

    clean_f0: dict | None = None
    if args.clean_f0:
        clean_f0 = {}
        for cf_path in args.clean_f0:
            clean_f0.update(json.loads(Path(cf_path).read_text()))
        print(f"[clean_f0] loaded {len(clean_f0)} clips of clean-frame F0")

    if not args.features_dir.is_dir():
        print(f"ERROR: features dir {args.features_dir} not found", file=sys.stderr)
        return 2

    # Resolve the split -> csv mapping (defaults + overrides).
    split_csv: dict[str, Path] = {
        s: args.features_dir / SPLIT_FILES[s] for s in args.splits
    }
    if args.split_file:
        for spec in args.split_file:
            if "=" not in spec:
                print(f"ERROR: --split-file expects NAME=PATH, got {spec!r}",
                      file=sys.stderr)
                return 2
            name, path = spec.split("=", 1)
            path_p = Path(path)
            if not path_p.is_absolute():
                path_p = args.features_dir / path_p
            split_csv[name] = path_p

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fallback_warned = {"warned": False}
    combined: dict[str, str] = {}

    n_assert_total = n_abstain_total = 0
    for split, csv_path in split_csv.items():
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found", file=sys.stderr)
            return 2
        out: dict[str, str] = {}
        n_assert = n_abstain = 0
        for row in _iter_rows(csv_path):
            fname = (row.get("filename") or "").strip()
            stem = os.path.splitext(fname)[0]
            if not stem:
                continue
            text = build_observability_description(
                row, stem, clean_f0,
                args.f0_overlap_threshold, args.min_clean_voiced_frac,
                fallback_warned,
            )
            out[stem] = text
            combined[stem] = text
            action, _ = f0_decision(row, clean_f0,
                                    args.f0_overlap_threshold,
                                    args.min_clean_voiced_frac)
            if action == "assert":
                n_assert += 1
            else:
                n_abstain += 1
        out_path = args.out_dir / f"descriptions_observability_{split}.json"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        tmp.replace(out_path)
        n_assert_total += n_assert
        n_abstain_total += n_abstain
        print(f"  {split:8s}: {len(out):6d} clips  "
              f"(F0 asserted {n_assert}, abstained {n_abstain})  -> {out_path}")

    if args.combined_output is not None:
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.combined_output.with_suffix(args.combined_output.suffix + ".tmp")
        tmp.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
        tmp.replace(args.combined_output)
        print(f"  combined: {len(combined)} clips -> {args.combined_output}")

    print()
    print(f"F0 asserted (number): {n_assert_total}  "
          f"abstained (hedge): {n_abstain_total}  "
          f"abstain frac: {n_abstain_total / max(1, n_assert_total + n_abstain_total):.3f}")
    print(f"threshold: overlap_ratio >= {args.f0_overlap_threshold} -> abstain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
