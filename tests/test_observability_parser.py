"""Tests for observability-builder phrasing coverage in the SFS parser, plus the
calibrated/correlation metrics (src/metrics_calibrated.py).

The observability target builder
(scripts/build_descriptions_observability.py) intentionally emits 3-4 PARAPHRASE
VARIANTS per feature to de-template the prose. The SFS regex ClaimParser must
match EVERY one of those variants (assert forms -> correct feature+value) and
the AbstentionDetector must fire on EVERY F0 hedge variant, or measured
precision/recall is silently under-counted.

The authoritative variant strings below are lifted verbatim from the builder's
sentence templates (the `_*_sentence` / `_pause_sentences` / `_f0_*` functions),
with representative numbers substituted, so this file is a regression lock on the
co-design between the builder and the parser.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sfs import ClaimParser, AbstentionDetector  # noqa: E402
import metrics_calibrated as mc  # noqa: E402


PARSER = ClaimParser()
ABST = AbstentionDetector()


def _feat_vals(text):
    return {(c.feature, round(c.value, 4)) for c in PARSER.parse(text)}


# ── ASSERT variants: every builder phrasing -> correct (feature, value) ─────
# (text, expected (feature, value) that MUST appear)
ASSERT_CASES = [
    # SNR — all four builder templates incl. leading-number form.
    ("The recording carries a moderate signal-to-noise ratio SNR of 13.68 dB.", ("snr", 13.68)),
    ("Background noise is moderate, with an SNR of 16.68 dB.", ("snr", 16.68)),
    ("At 21.29 dB, the signal-to-noise ratio SNR is high.", ("snr", 21.29)),
    ("The SNR is 13.19 dB, a moderate noise level.", ("snr", 13.19)),
    # SRMR — all four templates (bare-number, no unit).
    ("The signal is moderately reverberant, reflected in an SRMR of 4.8224.", ("srmr", 4.8224)),
    ("Reverberation is lightly reverberant here, with an SRMR of 6.6493.", ("srmr", 6.6493)),
    ("The SRMR of 4.8224 indicates moderately reverberant conditions.", ("srmr", 4.8224)),
    ("The SRMR is 6.6493, so the recording is lightly reverberant.", ("srmr", 6.6493)),
    # speaking rate — all four, incl. "runs to a brisk X".
    ("The talker speaks at a brisk pace, a speaking rate of 5.531 syl/sec.", ("speaking_rate", 5.531)),
    ("Delivery is brisk, with a speaking rate of 6.250 syl/sec.", ("speaking_rate", 6.25)),
    ("The speaking rate is 6.053 syl/sec, a brisk pace.", ("speaking_rate", 6.053)),
    ("The speaking rate runs to a brisk 6.053 syl/sec.", ("speaking_rate", 6.053)),
    # F0 assert — incl. the previously-MISSED "can be estimated at X Hz".
    ("Pitch is recoverable here: the F0 mean is 202.89 Hz.", ("f0_mean", 202.89)),
    ("With little overlap, the F0 mean can be estimated at 202.89 Hz.", ("f0_mean", 202.89)),
    ("The F0 mean is 202.89 Hz.", ("f0_mean", 202.89)),
    # overlap ratio — all six (0 and non-0 templates).
    ("Only one speaker is active, so the overlap ratio is 0.0000.", ("overlap_ratio", 0.0)),
    ("There is no concurrent speech; the overlap ratio is 0.0000.", ("overlap_ratio", 0.0)),
    ("The overlap ratio is 0.0000, a single-speaker recording.", ("overlap_ratio", 0.0)),
    ("The two speakers overlap to a moderate degree, an overlap ratio of 0.4231.", ("overlap_ratio", 0.4231)),
    ("Concurrent speech is moderate, with an overlap ratio of 0.4231.", ("overlap_ratio", 0.4231)),
    ("The overlap ratio is 0.4231, indicating moderate co-channel speech.", ("overlap_ratio", 0.4231)),
]


@pytest.mark.parametrize("text,expected", ASSERT_CASES)
def test_assert_variant_parses(text, expected):
    feats = _feat_vals(text)
    assert expected in feats, f"{expected} not in {feats} for {text!r}"


# ── F0 assert WITH SD: both mean and sd parse from one sentence ─────────────
F0_WITH_SD_CASES = [
    "Pitch is recoverable here: the F0 mean is 202.89 Hz and the F0 standard deviation SD is 10.74 Hz.",
    "With little overlap, pitch can be measured: the F0 mean is 202.89 Hz with an F0 standard deviation SD of 10.74 Hz.",
    "The F0 mean is 202.89 Hz and the F0 standard deviation SD is 10.74 Hz.",
]


@pytest.mark.parametrize("text", F0_WITH_SD_CASES)
def test_f0_with_sd(text):
    feats = _feat_vals(text)
    assert ("f0_mean", 202.89) in feats
    assert ("f0_sd", 10.74) in feats


# ── Pause variants: count + rate woven into one sentence ────────────────────
PAUSE_CASES = [
    ("The talker runs through without pausing, so the pause count is 0 and the pause rate is 0.000 per min.",
     {("pause_count", 0.0), ("pause_rate", 0.0)}),
    ("There are no breaks in delivery; the pause count is 0 and the pause rate is 0.000 per min.",
     {("pause_count", 0.0), ("pause_rate", 0.0)}),
    ("Delivery is broken by 1 pause, so the pause count is 1.",
     {("pause_count", 1.0)}),
    ("The talker takes 1 pause; the pause count is 1.",
     {("pause_count", 1.0)}),
    ("Delivery is broken by 1 pause, giving a pause count of 1 and a pause rate of 7.500 per min.",
     {("pause_count", 1.0), ("pause_rate", 7.5)}),
    ("The talker takes 1 pause, so the pause count is 1 and the pause rate is 7.500 per min.",
     {("pause_count", 1.0), ("pause_rate", 7.5)}),
    ("With 1 pause in all, the pause count is 1 and the pause rate is 7.500 per min.",
     {("pause_count", 1.0), ("pause_rate", 7.5)}),
]


@pytest.mark.parametrize("text,expected", PAUSE_CASES)
def test_pause_variant(text, expected):
    feats = _feat_vals(text)
    assert expected <= feats, f"{expected} not subset of {feats}"


# ── F0 ABSTAIN: every hedge variant -> AbstentionDetector fires on pitch ────
ABSTAIN_CASES = [
    "Because the two speakers overlap heavily (0.62 overlap ratio), the pitch cannot be reliably estimated from the mixture, so no F0 value is reported.",
    "The speakers overlap too much (0.62 overlap ratio) for single-speaker pitch to be recovered, so the F0 is left unstated.",
    "Given the heavy speaker overlap (0.62 overlap ratio), F0 is ill-posed on this mixture and is not asserted.",
    "Too few clean, non-overlapped voiced frames are available to estimate pitch, so no F0 value is reported.",
    "There is not enough clean voiced speech to recover a trustworthy pitch, so the F0 is left unstated.",
    "Pitch cannot be estimated from the available clean frames, so no F0 is asserted.",
]


@pytest.mark.parametrize("text", ABSTAIN_CASES)
def test_f0_abstain_detected(text):
    abst = ABST.detect(text)
    assert "f0_mean" in abst and "f0_sd" in abst, f"abstain not detected: {text!r}"


@pytest.mark.parametrize("text", ABSTAIN_CASES)
def test_f0_abstain_emits_no_f0_number(text):
    """A hedge must NOT parse as an F0 number, and the embedded overlap-ratio
    number in the hedge must NOT mis-bind to f0/snr."""
    feats = _feat_vals(text)
    assert not any(f in ("f0_mean", "f0_sd") for f, _ in feats), feats
    assert not any(f == "snr" for f, _ in feats), feats


# ── Cross-feature mis-binding guards ────────────────────────────────────────
def test_snr_leading_number_does_not_cross_sentence_to_f0():
    txt = "At 21.29 dB, the signal-to-noise ratio SNR is high. The F0 mean is 202.89 Hz."
    feats = _feat_vals(txt)
    assert ("snr", 21.29) in feats
    assert ("f0_mean", 202.89) in feats
    # SNR value must not also appear as an f0 number.
    assert ("f0_mean", 21.29) not in feats


def test_full_clip_all_features_parse():
    txt = (
        "The recording carries a high signal-to-noise ratio SNR of 28.50 dB. "
        "The SRMR is 6.6493, so the recording is lightly reverberant. "
        "With little overlap, the F0 mean can be estimated at 150.00 Hz. "
        "The talker speaks at a brisk pace, a speaking rate of 6.053 syl/sec. "
        "Delivery is broken by 1 pause, giving a pause count of 1 and a pause rate of 7.500 per min. "
        "Only one speaker is active, so the overlap ratio is 0.0000."
    )
    feats = _feat_vals(txt)
    for exp in [("snr", 28.5), ("srmr", 6.6493), ("f0_mean", 150.0),
                ("speaking_rate", 6.053), ("pause_count", 1.0),
                ("pause_rate", 7.5), ("overlap_ratio", 0.0)]:
        assert exp in feats, f"{exp} missing from full clip parse: {feats}"


# ── Calibrated metrics: correlation + relative tolerance ────────────────────
def test_pearson_spearman_mae_toy():
    pred = [1.0, 2.0, 3.0, 4.0]
    gt = [1.1, 1.9, 3.2, 3.8]
    assert mc.pearson(pred, gt) > 0.99
    assert mc.spearman(pred, gt) == 1.0  # monotone -> perfect rank corr
    assert abs(mc.mae(pred, gt) - 0.15) < 1e-9


def test_spearman_handles_ties():
    # Two identical predicted values; average-rank handling must not crash.
    assert mc.spearman([1.0, 1.0, 2.0], [5.0, 6.0, 7.0]) is not None


def test_correlation_undefined_on_constant():
    assert mc.pearson([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) is None


def test_relative_tolerance_floor_and_scaling():
    # snr: abs_tol 2.0, rel_frac 0.10 -> max(2, 0.1*|gt|)
    assert mc.relative_tolerance("snr", 5.0) == 2.0     # floor wins
    assert mc.relative_tolerance("snr", 30.0) == 3.0    # rel wins
    # pause_count rel_frac 0 -> always the absolute floor (1.0)
    assert mc.relative_tolerance("pause_count", 50.0) == 1.0


def test_correlation_report_on_toy_resultset():
    results = [
        {"filename": "a", "generated": "The SNR is 20.0 dB.", "target": "The SNR is 21.0 dB."},
        {"filename": "b", "generated": "The SNR is 10.0 dB.", "target": "The SNR is 12.0 dB."},
        {"filename": "c", "generated": "The SNR is 30.0 dB.", "target": "The SNR is 29.0 dB."},
    ]
    rep = mc.correlation_report(results)
    assert rep["per_feature"]["snr"]["n"] == 3
    assert rep["per_feature"]["snr"]["srcc"] == 1.0
    assert rep["aggregate"]["n_pairs"] == 3


def test_calibrated_sfs_relative_vs_absolute():
    # A 2.5 dB miss on a 30 dB GT: absolute (±2) = WRONG, relative (±3) = CORRECT.
    results = [
        {"filename": "a", "generated": "The SNR is 27.5 dB.", "target": "The SNR is 30.0 dB."},
    ]
    rel = mc.calibrated_sfs_report(results)
    assert rel["per_feature"]["snr"]["precision"] == 1.0  # within relative band
    # rel_frac=0 reproduces the absolute path: now the same miss is WRONG.
    absol = mc.calibrated_sfs_report(results, rel_frac={"snr": 0.0})
    assert absol["per_feature"]["snr"]["precision"] == 0.0


# ── Automated coverage on REAL builder output (skips if not present) ────────
def _find_builder_json():
    """Locate a descriptions_observability_*.json the builder produced.

    On PSC the worktree ships the real split JSONs; locally they may be absent
    (skip). Checks env override OBS_JSON first, then a few standard locations.
    """
    candidates = []
    env = os.environ.get("OBS_JSON")
    if env:
        candidates.append(env)
    here = os.path.dirname(__file__)
    for base in (os.path.join(here, "..", "data"),
                 "/ocean/projects/cis260125p/shared/assessment-tool-redirect/data",
                 "/ocean/projects/cis260125p/shared/data"):
        for name in ("descriptions_observability_test.json",
                     "descriptions_observability_dev.json",
                     "descriptions_observability_all.json",
                     "descriptions_observability_train.json"):
            candidates.append(os.path.join(base, name))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def test_real_builder_output_parse_coverage():
    """For real builder targets: every clip's asserted feature number must parse,
    and every abstained pitch must be detected as abstention. Targets ~100%.
    """
    path = _find_builder_json()
    if path is None:
        pytest.skip("no descriptions_observability_*.json found (run on PSC or set OBS_JSON)")
    data = json.loads(open(path).read())
    items = list(data.items())[:500]  # sample for speed

    # The builder always asserts snr, srmr, speaking_rate, overlap_ratio, and
    # pause_count (recoverable, unconditional). F0 is assert-or-abstain. We
    # verify the unconditional recoverables parse on every clip, and that F0 is
    # EITHER parsed as a number OR detected as a hedge (never silently lost).
    miss = {"snr": 0, "srmr": 0, "speaking_rate": 0, "overlap_ratio": 0,
            "pause_count": 0, "f0_resolved": 0}
    for _stem, text in items:
        feats = {c.feature for c in PARSER.parse(text)}
        for f in ("snr", "srmr", "speaking_rate", "overlap_ratio", "pause_count"):
            if f not in feats:
                miss[f] += 1
        f0_num = "f0_mean" in feats
        f0_hedge = bool(ABST.detect(text))
        if not (f0_num or f0_hedge):
            miss["f0_resolved"] += 1

    n = len(items)
    # Allow a tiny slack only for clips with a genuinely missing CSV column;
    # the recoverables should be ~100%.
    for f in ("snr", "srmr", "speaking_rate", "overlap_ratio", "pause_count"):
        cov = 1.0 - miss[f] / n
        assert cov >= 0.98, f"{f} parse coverage {cov:.3f} on {n} real clips ({miss[f]} misses)"
    f0_cov = 1.0 - miss["f0_resolved"] / n
    assert f0_cov >= 0.99, f"F0 resolved (number or hedge) coverage {f0_cov:.3f} ({miss['f0_resolved']} misses)"
