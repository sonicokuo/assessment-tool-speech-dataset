#!/usr/bin/env python3
"""Build CANONICAL deterministic-template descriptions for the AQUA-NL-successor
project (codename undecided), with NO LLM in the loop.

Why this replaces gemma re-verbalization
-----------------------------------------
Running the feature rows back through gemma4:e2b produced MULTI-STYLE DRIFT: the
same fact ("SNR 15.63 dB") was paraphrased a dozen ways across the corpus, some
of which the SFS regex parser silently missed. The research mandate for the
successor project is the opposite of stylistic diversity:

    ONE canonical style
      = fixed-slot CONTENT (every reported feature -> exactly one clause)
      + fixed ORDERING (so parser recall is a guaranteed property, not luck)
      + mild paraphrase of CONNECTIVES ONLY (intro / between-clause glue),
        chosen deterministically by a hash of the filename (reproducible, not
        gemma drift) — never varying WHICH facts appear or their order
      + OBSERVABILITY ABSTENTION (do not state a pitch / voice-quality number
        when the speakers overlap too much for it to be recoverable).

Every reported clause uses a phrasing the SFS `HybridClaimParser`
(src/sfs.py) provably round-trips (see tests/test_canonical_descriptions.py,
which asserts extraction of all 12 values). The connective variants are pure
glue text with no digits, so they cannot perturb claim extraction.

The 12 canonical features (name -> CSV column -> unit phrasing)
--------------------------------------------------------------
    snr               -> snr_db                          (dB)
    srmr              -> srmr                             (unitless)
    hnr               -> hnr_db                           (dB)
    f0_mean           -> f0_mean_hz                       (Hz)
    f0_sd             -> f0_sd_hz                         (Hz)
    jitter            -> jitter_local_pct                 (percent)
    shimmer           -> shimmer_pct                      (percent)
    speaking_rate     -> praat_speaking_rate_syl_sec      (syl/sec)
    articulation_rate -> praat_articulation_rate_syl_sec  (syl/sec)
    pause_count       -> praat_pause_count                (count, int)
    pause_rate        -> praat_pause_rate_per_min         (per min)
    overlap_ratio     -> overlap_ratio                    (ratio 0-1)

NO duration, NO sample_rate (those are deterministic metadata, not learned
targets, and are re-added at inference if needed).

Fixed clause order (always emitted in THIS order; abstained features drop out)
------------------------------------------------------------------------------
    snr, srmr, hnr, f0_mean, f0_sd, jitter, shimmer,
    speaking_rate, articulation_rate, pause_count, pause_rate, overlap_ratio

Observability abstention
------------------------
When the clip's overlap ratio is high (>= --overlap_threshold, default 0.5),
the voice-quality / pitch features that REQUIRE clean voiced speech become
unreliable on a 2-speaker mixture and are ABSTAINED: no number is stated.

    ABSTAIN-UNDER-OVERLAP = {f0_mean, f0_sd, jitter, shimmer, hnr}
    ALWAYS-REPORT         = {snr, srmr, speaking_rate, articulation_rate,
                             pause_count, pause_rate, overlap_ratio}

All abstained features are grouped into ONE hedge sentence, emitted in place of
the f0_mean clause (the first abstained slot in fixed order), e.g.

    "Because the speakers overlap heavily (0.8162 overlap ratio), the pitch and
     voice-quality measures (F0, jitter, shimmer, HNR) cannot be reliably
     estimated from the mixture and are not reported."

The SFS selective-scoring path (src/sfs.py AbstentionDetector) recognizes this
as a calibrated F0 abstention.

GT source policy (clean GT preferred; observability-aware)
----------------------------------------------------------
  --clean_features  clean_features_{split}.json  {filename.wav: {snr_db,
                    praat_speaking_rate_syl_sec, praat_articulation_rate_syl_sec,
                    praat_pause_count, praat_pause_rate_per_min, srmr, ...}}
                    -> the CLEAN GT for the recoverable scalars. Preferred over
                    the (mixture) CSV columns when present.
  --clean_f0        clean_f0_{split}.json  {filename.wav: {f0_mean_hz, f0_sd_hz,
                    clean_voiced_frac}} -> the CLEAN GT for pitch (only reported
                    when observable, i.e. low overlap).
  --features_csv    per-split feature CSV. Source for overlap_ratio (oracle
                    overlap_ratio_vad preferred), hnr/jitter/shimmer (no clean
                    JSON exists for those), and a fallback for any recoverable
                    scalar missing from --clean_features.

If a GT value is missing / NaN for a (reported) feature, its clause is OMITTED
rather than fabricated. The denominator of SFS recall is the set of features
SFS can score against, so an honestly-omitted clause does not inflate recall.

KNOWN PARSER INTERACTION (negative SNR / HNR)
---------------------------------------------
On overlapped Libri2Mix mixtures the clean-GT SNR is frequently NEGATIVE
(~45% of dev/test clips, ~15% of train). The builder emits the true value with
its sign ("The SNR is -3.50 dB.") — fabricating a sign would be worse than an
omission. HOWEVER the current SFS regex (src/sfs.py, snr / hnr patterns) uses
`(\d+\.?\d*)` which does NOT match a leading minus, so a negative SNR / HNR
clause is emitted but extracted by the parser as zero claims. The one-line fix
is to widen those two patterns to `(-?\d+\.?\d*)`; that lives in src/sfs.py,
which this builder does not own, so it is flagged here rather than patched.
HNR is never negative in this corpus (0/41700 train rows), so the practical
impact is SNR recall on negative-SNR clips.

Usage
-----
    python scripts/build_canonical_descriptions.py \
        --features_csv   $SHARED/data/features_pyannote/test.csv \
        --clean_features $SHARED/data/clean_features_test.json \
        --clean_f0       $SHARED/data/clean_f0_test.json \
        --overlap_threshold 0.5 \
        --output         $SHARED/data/descriptions_canonical_test.json
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


# ── The 12 canonical features ────────────────────────────────────────────────
# Order is the FIXED CLAUSE ORDER. Each entry:
#   (short_name, csv_column, clean_json_key_or_None, format_string)
# clean_json_key: the key to look up in --clean_features (recoverable scalars).
#   f0_mean / f0_sd come from --clean_f0 (handled specially). hnr / jitter /
#   shimmer have no clean JSON, so their clean key is None and they read the CSV.
FEATURE_ORDER: list[tuple[str, str, str | None, str]] = [
    ("snr",               "snr_db",                          "snr_db",                          "{:.2f}"),
    ("srmr",              "srmr",                            "srmr",                            "{:.4f}"),
    ("hnr",               "hnr_db",                          None,                              "{:.2f}"),
    ("f0_mean",           "f0_mean_hz",                      None,                              "{:.2f}"),  # clean_f0
    ("f0_sd",             "f0_sd_hz",                        None,                              "{:.2f}"),  # clean_f0
    ("jitter",            "jitter_local_pct",                None,                              "{:.2f}"),
    ("shimmer",           "shimmer_pct",                     None,                              "{:.2f}"),
    ("speaking_rate",     "praat_speaking_rate_syl_sec",     "praat_speaking_rate_syl_sec",     "{:.3f}"),
    ("articulation_rate", "praat_articulation_rate_syl_sec", "praat_articulation_rate_syl_sec", "{:.3f}"),
    ("pause_count",       "praat_pause_count",               "praat_pause_count",               "{:d}"),
    ("pause_rate",        "praat_pause_rate_per_min",        "praat_pause_rate_per_min",        "{:.3f}"),
    ("overlap_ratio",     "overlap_ratio",                   None,                              "{:.4f}"),
]

# Features whose number is unreliable under heavy overlap -> abstained (grouped
# into one hedge) when overlap_ratio >= threshold.
ABSTAIN_UNDER_OVERLAP: frozenset[str] = frozenset(
    {"f0_mean", "f0_sd", "jitter", "shimmer", "hnr"}
)
# Always reported (recoverable from the mixture).
ALWAYS_REPORT: frozenset[str] = frozenset(
    {"snr", "srmr", "speaking_rate", "articulation_rate",
     "pause_count", "pause_rate", "overlap_ratio"}
)

DEFAULT_OVERLAP_THRESHOLD = 0.5

# pause_count is an integer feature; its value is rounded before formatting.
_INT_FEATURES = frozenset({"pause_count"})


# ── Value coercion ───────────────────────────────────────────────────────────
def _to_float(val):
    """CSV / JSON cell -> float, or None if missing / NaN / unparseable."""
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


def _fmt(short_name: str, fmt: str, value: float) -> str:
    """Format a numeric value per the canonical NUMBER FORMATTING spec."""
    if short_name in _INT_FEATURES:
        return f"{int(round(value)):d}"
    return fmt.format(value)


# ── Deterministic connective choice ─────────────────────────────────────────
def _stable_choice(stem: str, salt: str, n: int) -> int:
    """Reproducible 0..n-1 index from (stem, salt). No RNG, no global state, so
    the SAME clip always gets the SAME connective across runs / machines."""
    h = hashlib.sha1(f"{stem}|{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


# Intro connectives — pure glue, no digits, never change which facts appear.
_INTRO_VARIANTS = [
    "This recording has the following measured characteristics.",
    "The acoustic measurements for this clip are as follows.",
    "An analysis of this recording yields the following measurements.",
    "The measured signal characteristics of this clip are listed below.",
]

# Per-clause lead-in connectives. Index 0 is always the empty (no-glue) form so
# the first reported clause reads plainly; later clauses pick from light glue.
# These are inserted BEFORE the canonical clause; they carry no numbers, so the
# SFS regex (which anchors on "The <feature> is <num> <unit>") is untouched.
_CLAUSE_CONNECTIVES = [
    "",
    "In addition, ",
    "Furthermore, ",
    "Also, ",
]


def _canonical_clause(short_name: str, value_str: str) -> str:
    """Return the FIXED canonical clause for a feature, in an SFS-parseable form.

    The phrasings are intentionally the exact ones the SFS HybridClaimParser
    round-trips; see tests/test_canonical_descriptions.py.
    """
    if short_name == "snr":
        return f"The SNR is {value_str} dB."
    if short_name == "srmr":
        return f"The SRMR is {value_str}."
    if short_name == "hnr":
        return f"The HNR is {value_str} dB."
    if short_name == "f0_mean":
        return f"The F0 mean is {value_str} Hz."
    if short_name == "f0_sd":
        return f"The F0 standard deviation SD is {value_str} Hz."
    if short_name == "jitter":
        return f"The jitter is {value_str} percent."
    if short_name == "shimmer":
        return f"The shimmer is {value_str} percent."
    if short_name == "speaking_rate":
        return f"The speaking rate is {value_str} syl/sec."
    if short_name == "articulation_rate":
        return f"The articulation rate is {value_str} syl/sec."
    if short_name == "pause_count":
        return f"The pause count is {value_str}."
    if short_name == "pause_rate":
        return f"The pause rate is {value_str} per min."
    if short_name == "overlap_ratio":
        # NOTE: the SFS parser MISSES value-first overlap phrasing
        # ("at 0.81, the overlap ..."); the canonical form is the
        # "The overlap ratio is X" pattern, which it parses.
        return f"The overlap ratio is {value_str}."
    raise KeyError(short_name)  # pragma: no cover (guarded by FEATURE_ORDER)


def _hedge_sentence(overlap_ratio: float | None) -> str:
    """The single grouped hedge for the abstained pitch / voice-quality features.

    Mentions F0 / jitter / shimmer / HNR by NAME but states NO number for them,
    so the SFS parser extracts no claim for any abstained feature while the
    AbstentionDetector recognizes the pitch hedge.
    """
    if overlap_ratio is not None:
        ov = f" ({overlap_ratio:.4f} overlap ratio)"
    else:
        ov = ""
    return (
        f"Because the speakers overlap heavily{ov}, the pitch and voice-quality "
        f"measures (F0, jitter, shimmer, HNR) cannot be reliably estimated from "
        f"the mixture and are not reported."
    )


# ── GT resolution ────────────────────────────────────────────────────────────
def resolve_value(
    short_name: str,
    csv_col: str,
    clean_key: str | None,
    row: dict,
    clean_features: dict | None,
    clean_f0: dict | None,
    filename: str,
) -> float | None:
    """Resolve the GT value for one feature, clean-GT-preferred.

    f0_mean / f0_sd  -> --clean_f0 (clean-frame pitch). No CSV/mixture fallback:
                        if the clean pitch is undefined the clause is omitted
                        (these are abstained under overlap anyway).
    snr/srmr/speaking_rate/articulation_rate/pause_count/pause_rate
                     -> --clean_features (clean key) preferred, else CSV column.
    hnr/jitter/shimmer/overlap_ratio
                     -> CSV column (no clean JSON for these). overlap_ratio
                        prefers the oracle overlap_ratio_vad column.
    Returns None when the value is missing / NaN (caller omits the clause).
    """
    if short_name in ("f0_mean", "f0_sd"):
        if clean_f0 is None:
            return None
        cf = clean_f0.get(filename)
        if not cf:
            return None
        key = "f0_mean_hz" if short_name == "f0_mean" else "f0_sd_hz"
        return _to_float(cf.get(key))

    if short_name == "overlap_ratio":
        # Oracle VAD overlap preferred; else the (pyannote) overlap_ratio column.
        v = _to_float(row.get("overlap_ratio_vad"))
        if v is None:
            v = _to_float(row.get("overlap_ratio"))
        return v

    # Recoverable scalars with a clean-features key: clean GT preferred.
    if clean_key is not None and clean_features is not None:
        cf = clean_features.get(filename)
        if cf is not None and clean_key in cf:
            v = _to_float(cf.get(clean_key))
            if v is not None:
                return v
    # CSV fallback (also the primary path for hnr / jitter / shimmer).
    return _to_float(row.get(csv_col))


def overlap_for_decision(
    row: dict, clean_f0: dict | None, filename: str
) -> float:
    """The overlap ratio used for the abstention decision.

    Same source as the reported overlap_ratio: oracle overlap_ratio_vad when
    present, else overlap_ratio, else 0.0 (a clip with no overlap column is a
    clean single-speaker recording -> nothing abstained).
    """
    v = _to_float(row.get("overlap_ratio_vad"))
    if v is None:
        v = _to_float(row.get("overlap_ratio"))
    if v is None:
        return 0.0
    return v


# ── Description assembly ─────────────────────────────────────────────────────
def build_canonical_description(
    row: dict,
    stem: str,
    filename: str,
    clean_features: dict | None,
    clean_f0: dict | None,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> str:
    """Compose the canonical description for one clip.

    Fixed order; abstained features (under heavy overlap) replaced by ONE grouped
    hedge sentence emitted at the first abstained slot; connectives varied
    deterministically by `stem`; missing GT values omit their clause (never
    fabricated).
    """
    overlap = overlap_for_decision(row, clean_f0, filename)
    abstaining = overlap >= overlap_threshold

    clauses: list[str] = []
    hedge_emitted = False
    n_reported = 0

    for short_name, csv_col, clean_key, fmt in FEATURE_ORDER:
        is_abstained = abstaining and short_name in ABSTAIN_UNDER_OVERLAP

        if is_abstained:
            # Emit the single grouped hedge once, at the first abstained slot in
            # fixed order. Subsequent abstained features are folded into it.
            if not hedge_emitted:
                clauses.append(_hedge_sentence(overlap))
                hedge_emitted = True
            continue

        value = resolve_value(
            short_name, csv_col, clean_key, row,
            clean_features, clean_f0, filename,
        )
        if value is None:
            # Missing / NaN GT -> omit (do not fabricate).
            continue
        value_str = _fmt(short_name, fmt, value)
        clause = _canonical_clause(short_name, value_str)
        # Light deterministic connective glue (never on the very first clause).
        if n_reported == 0:
            connective = ""
        else:
            connective = _CLAUSE_CONNECTIVES[
                _stable_choice(stem, f"clause{n_reported}", len(_CLAUSE_CONNECTIVES))
            ]
        if connective and clause.startswith("The "):
            # Lowercase the clause-leading "The" so the connective reads as glue
            # ("In addition, the SNR is ...") rather than a sentence restart
            # ("In addition, The SNR ..."). The SFS regex is case-insensitive and
            # anchors on the feature word, so this does not affect parsing.
            clause = "the " + clause[4:]
        clauses.append(connective + clause)
        n_reported += 1

    if not clauses:
        return ""

    intro = _INTRO_VARIANTS[_stable_choice(stem, "intro", len(_INTRO_VARIANTS))]
    return (intro + " " + " ".join(clauses)).strip()


# ── CLI / driver ─────────────────────────────────────────────────────────────
def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _iter_rows(csv_path: Path):
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            yield row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features_csv", type=Path, required=True,
                   help="per-split feature CSV (source for overlap_ratio, "
                        "hnr/jitter/shimmer, and recoverable-scalar fallback)")
    p.add_argument("--clean_features", type=Path, default=None,
                   help="clean_features_{split}.json (clean GT for the "
                        "recoverable scalars; preferred over CSV columns)")
    p.add_argument("--clean_f0", type=Path, default=None,
                   help="clean_f0_{split}.json (clean GT for f0_mean / f0_sd)")
    p.add_argument("--overlap_threshold", type=float,
                   default=DEFAULT_OVERLAP_THRESHOLD,
                   help="abstain on the pitch / voice-quality features when the "
                        "clip overlap ratio >= this value (default %(default)s)")
    p.add_argument("--output", type=Path, required=True,
                   help="output JSON {clip_stem: description}")
    args = p.parse_args(argv)

    if not args.features_csv.exists():
        print(f"ERROR: features_csv {args.features_csv} not found", file=sys.stderr)
        return 2

    clean_features = _load_json(args.clean_features)
    clean_f0 = _load_json(args.clean_f0)
    if clean_features is not None:
        print(f"[clean_features] loaded {len(clean_features)} clips")
    if clean_f0 is not None:
        print(f"[clean_f0] loaded {len(clean_f0)} clips")

    out: dict[str, str] = {}
    n_rows = n_abstain = n_empty = 0
    for row in _iter_rows(args.features_csv):
        filename = (row.get("filename") or "").strip()
        stem = os.path.splitext(filename)[0]
        if not stem:
            continue
        n_rows += 1
        overlap = overlap_for_decision(row, clean_f0, filename)
        if overlap >= args.overlap_threshold:
            n_abstain += 1
        text = build_canonical_description(
            row, stem, filename, clean_features, clean_f0,
            args.overlap_threshold,
        )
        if not text:
            n_empty += 1
        out[stem] = text

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    tmp.replace(args.output)

    print()
    print(f"wrote {args.output}")
    print(f"  clips                 : {len(out)}")
    print(f"  abstained (overlap>= {args.overlap_threshold}): {n_abstain} "
          f"({n_abstain / max(1, n_rows):.1%})")
    print(f"  full (all observable) : {n_rows - n_abstain}")
    print(f"  empty (no usable GT)  : {n_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
