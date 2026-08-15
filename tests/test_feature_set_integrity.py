"""Data-integrity tests for src/data/feature_set.py (F11 + F15, 2026-07-16).

F15 — constant-width numeric surface form: every format string in
SUPERVISED_FEATURES must be "{:.2f}" so digit position aligns with magnitude
across features (Singh & Strouse, arXiv:2402.14903; R4 in the 2026-07-13
training-method memo).

F11 — clean-f0 fail-loud (2026-07-15 audit risk 10): assert_clean_f0_csv must
pass on the existing clean conventions (cleanf0/cleangt filenames, sentinel
column, clean-named columns) and raise RuntimeError on a plain noisy CSV.

Runnable under pytest or directly: python3 tests/test_feature_set_integrity.py
"""

import csv
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import feature_set
from data.feature_set import (
    SUPERVISED_FEATURES,
    assert_clean_f0_csv,
    build_nums_target,
    validate_f0_source,
)


# ── Fixture helpers (tempfile-based so the file also runs without pytest) ────
NOISY_HEADER = ["filename", "snr_db", "f0_mean_hz", "f0_sd_hz", "overlap_ratio"]
NOISY_ROW = {
    "filename": "clip_001.wav",
    "snr_db": "15.66",
    "f0_mean_hz": "152.46",
    "f0_sd_hz": "53.18",
    "overlap_ratio": "0.79",
}


def _write_csv(path: str, header: list, rows: list) -> str:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return path


# ── F15: every format is the constant two-decimal form ───────────────────────
def test_every_format_is_two_decimal():
    for short_name, csv_col, fmt in SUPERVISED_FEATURES:
        assert fmt == "{:.2f}", (
            f"{short_name} ({csv_col}) format is {fmt!r}; F15 mandates the "
            f"constant-width two-decimal '{{:.2f}}' surface form"
        )


def test_build_nums_target_emits_two_decimals():
    row = {
        "snr_db": 15.663,
        "srmr": 3.2667,           # the old 4-decimal offender
        "f0_mean_hz": 152.456,
        "f0_sd_hz": 53.184,
        "praat_speaking_rate_syl_sec": 5.6104,
        "praat_pause_count": 1,
        "praat_pause_rate_per_min": 5.3170,
        "overlap_ratio": 0.7928,
        "jitter_local_pct": 1.7458,   # the old 4-decimal offender
        "shimmer": 14.1259,
        "hnr": 8.341,
    }
    out = build_nums_target(row)
    assert "srmr=3.27" in out
    assert "jitter=1.75" in out
    assert "shimmer=14.13" in out
    assert "speaking_rate=5.61" in out
    assert "pause_rate=5.32" in out
    assert "overlap_ratio=0.79" in out
    assert "pause_count=1" in out     # integer feature stays integer
    # No slot other than pause_count/na may deviate from exactly 2 decimals.
    for part in out.split():
        name, val = part.split("=")
        if val == "na" or name == "pause_count":
            continue
        assert "." in val and len(val.rsplit(".", 1)[1]) == 2, (
            f"{part} is not two-decimal"
        )


# ── F11: assert_clean_f0_csv ─────────────────────────────────────────────────
def test_passes_on_cleanf0_named_path():
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev_cleanf0.csv"), NOISY_HEADER, [NOISY_ROW])
        assert_clean_f0_csv(path)  # must not raise


def test_passes_on_cleangt_named_path():
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(
            os.path.join(d, "features_aug_train_cleangt.csv"), NOISY_HEADER, [NOISY_ROW]
        )
        assert_clean_f0_csv(path)  # must not raise


def test_passes_on_sentinel_column():
    header = NOISY_HEADER + ["f0_clean_substituted"]
    row = dict(NOISY_ROW, f0_clean_substituted="1")
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev.csv"), header, [row])
        assert_clean_f0_csv(path)  # must not raise


def test_passes_on_clean_named_columns():
    header = NOISY_HEADER + ["f0_mean_hz_clean", "f0_sd_hz_clean"]
    row = dict(NOISY_ROW, f0_mean_hz_clean="150.10", f0_sd_hz_clean="50.00")
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev.csv"), header, [row])
        assert_clean_f0_csv(path)  # must not raise


def test_passes_when_no_f0_columns_at_all():
    header = ["filename", "snr_db", "overlap_ratio"]
    row = {"filename": "clip_001.wav", "snr_db": "15.66", "overlap_ratio": "0.79"}
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev.csv"), header, [row])
        assert_clean_f0_csv(path)  # nothing to poison -> must not raise


def test_raises_on_plain_noisy_csv():
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev.csv"), NOISY_HEADER, [NOISY_ROW])
        try:
            assert_clean_f0_csv(path)
        except RuntimeError as e:
            msg = str(e)
            assert "dev.csv" in msg          # names the file
            assert "f0_mean_hz" in msg       # names the risky columns
            assert "risk 10" in msg          # cites the audit risk
        else:
            raise AssertionError("plain noisy CSV must raise RuntimeError")


def test_raises_on_sentinel_column_not_1():
    header = NOISY_HEADER + ["f0_clean_substituted"]
    row = dict(NOISY_ROW, f0_clean_substituted="0")
    with tempfile.TemporaryDirectory() as d:
        path = _write_csv(os.path.join(d, "dev.csv"), header, [row])
        try:
            assert_clean_f0_csv(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("sentinel=0 must raise RuntimeError")


def test_rows_mode_with_filename_hint():
    # In-memory rows: name convention applies only via the explicit filename hint.
    assert_clean_f0_csv([dict(NOISY_ROW)], filename="dev_cleanf0.csv")  # passes
    try:
        assert_clean_f0_csv([dict(NOISY_ROW)], filename="dev.csv")
    except RuntimeError:
        pass
    else:
        raise AssertionError("noisy rows with a plain filename must raise")


def test_strict_flag_gates_build_nums_target():
    """STRICT_CLEAN_F0=True blocks target building until validate_f0_source() ran."""
    old_flag = feature_set.STRICT_CLEAN_F0
    old_validated = feature_set._F0_SOURCE_VALIDATED
    try:
        feature_set.STRICT_CLEAN_F0 = True
        feature_set._F0_SOURCE_VALIDATED = False
        try:
            build_nums_target(dict(NOISY_ROW))
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict mode without validation must raise")
        # After a successful validation the gate opens.
        with tempfile.TemporaryDirectory() as d:
            path = _write_csv(os.path.join(d, "dev_cleanf0.csv"), NOISY_HEADER, [NOISY_ROW])
            validate_f0_source(path)
        build_nums_target(dict(NOISY_ROW))  # must not raise now
    finally:
        feature_set.STRICT_CLEAN_F0 = old_flag
        feature_set._F0_SOURCE_VALIDATED = old_validated


def test_default_flag_is_off():
    """Default behavior identical: strict mode must ship disabled."""
    assert feature_set.STRICT_CLEAN_F0 is False


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001 — report and continue
                failures += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failures else 0)


# ── 2026-07-28: schema-drift guard (the hnr/shimmer class of bug) ─────────────
def test_supervised_columns_present_in_real_header():
    """SUPERVISED_FEATURES must name columns that exist in features_corrected_merged/*.csv.

    Regression guard for the bug that erased hnr and shimmer from all 39,800 targets: the
    builder asked for "hnr_db"/"shimmer_pct" after the CSVs were regenerated as
    "hnr"/"shimmer", and row.get() cannot distinguish a rename from a missing measurement.
    """
    from data.feature_set import assert_supervised_columns_exist

    real_header = [
        "filename", "snr_db", "srmr", "f0_mean_hz", "praat_speaking_rate_syl_sec",
        "praat_articulation_rate_syl_sec", "praat_pause_count",
        "praat_pause_rate_per_min", "f0_sd_hz", "jitter_local_pct", "jitter_rap_pct",
        "shimmer", "hnr", "overlap_ratio", "overlap_segments", "overlap_segments_vad",
        "overlap_ratio_vad",
    ]
    assert assert_supervised_columns_exist(real_header, strict=False) == []
    # the historical typos must not creep back into SUPERVISED_FEATURES
    cols = {csv_col for _n, csv_col, _f in SUPERVISED_FEATURES}
    assert "hnr_db" not in cols and "shimmer_pct" not in cols


def test_supervised_columns_missing_raises():
    """A renamed/absent column must stop the run, not train on empty supervision."""
    from data.feature_set import assert_supervised_columns_exist

    starved = [c for _n, c, _f in SUPERVISED_FEATURES if c not in ("hnr", "shimmer")]
    with pytest.raises(RuntimeError) as ei:
        assert_supervised_columns_exist(starved)
    assert "hnr" in str(ei.value) and "shimmer" in str(ei.value)
    # non-strict must downgrade to a warning and report what goes unsupervised
    assert sorted(assert_supervised_columns_exist(starved, strict=False)) == ["hnr", "shimmer"]
