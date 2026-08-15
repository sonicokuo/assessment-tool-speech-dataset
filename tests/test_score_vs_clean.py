"""Unit tests for scripts/score_inference_vs_clean.py — the trustworthy vs-clean-GT scorer.

Updated 2026-07-15 for the metric-consistency fixes (F8/F10): the headline
`mean_reliable` is the mean SRCC over exactly the frozen ROBUST5
(eval.selection_metric.HEADLINE_FEATURES), coverage is reported beside every SRCC,
and undefined SRCC is None (not NaN).

The script imports eval.selection_metric, which transitively imports
data.feature_set (torch at module level, owned elsewhere) — so this module skips
cleanly when torch is unavailable; it runs in full on PSC.
"""
import importlib.util
import math
import os
import sys

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip(
        "eval.selection_metric -> data.feature_set imports torch at module level",
        allow_module_level=True,
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import score_inference_vs_clean as S  # noqa: E402
from eval.selection_metric import HEADLINE_FEATURES  # noqa: E402


def _mk(n):
    """n clips; srmr claims perfectly rank-correlated with clean GT, snr anti-correlated."""
    results, feats, f0 = [], {}, {}
    for i in range(n):
        fn = f"clip{i}.wav"
        results.append({
            "filename": fn,
            "per_feature": [
                {"feature": "srmr", "claimed": float(i)},          # increasing
                {"feature": "snr",  "claimed": float(n - i)},      # decreasing
            ],
        })
        feats[fn] = {"srmr": float(i) + 0.01, "snr_db": float(i)}  # srmr ↑ (corr +1), snr ↑
        f0[fn] = {}
    return results, feats, f0


def test_perfect_and_anti_correlation():
    results, feats, f0 = _mk(40)
    out = S.score_vs_clean(results, feats, f0)
    assert abs(out["per_feature"]["srmr"]["srcc"] - 1.0) < 1e-9   # claimed↑ vs clean↑
    assert abs(out["per_feature"]["snr"]["srcc"] - (-1.0)) < 1e-9  # claimed↓ vs clean↑
    # F8: snr is IN the frozen headline -> mean over {srmr:+1, snr:-1} = 0
    assert abs(out["mean_reliable"] - 0.0) < 1e-9
    assert out["n_reliable_features"] == 2
    # secondary no-snr mean (SNR-circularity check) = srmr only
    assert abs(out["mean_reliable_no_snr"] - 1.0) < 1e-9


def test_claims_schema_and_coverage():
    # 'claims' [feature,value] form; all 30 clips emit srmr but only 20 have clean
    # GT -> n (pairs) and coverage (emissions/clips) are tracked SEPARATELY.
    results, feats, f0 = [], {}, {}
    for i in range(30):
        fn = f"c{i}.wav"
        results.append({"filename": fn, "claims": [["srmr", float(i)]]})
        if i < 20:
            feats[fn] = {"srmr": float(i)}
        f0[fn] = {}
    out = S.score_vs_clean(results, feats, f0)
    assert out["per_feature"]["srmr"]["n"] == 20
    assert out["per_feature"]["srmr"]["n_emitted"] == 30
    assert out["per_feature"]["srmr"]["coverage"] == pytest.approx(1.0)
    assert abs(out["per_feature"]["srmr"]["srcc"] - 1.0) < 1e-9


def test_min_pairs_guard():
    results, feats, f0 = _mk(5)  # below default min_pairs=10
    out = S.score_vs_clean(results, feats, f0)
    assert out["per_feature"]["srmr"]["srcc"] is None  # undefined, NOT NaN
    assert out["n_reliable_features"] == 0
    assert math.isnan(out["mean_reliable"])
    # coverage stays visible even when the SRCC is undefined (F10)
    assert out["per_feature"]["srmr"]["coverage"] == pytest.approx(1.0)


def test_never_emitted_feature_present_with_zero_coverage():
    # A headline feature emitted on 0 clips must still appear: coverage 0.0,
    # srcc None, no crash (2026-07-15 audit risk 3 — asymmetric degeneration).
    results, feats, f0 = _mk(40)
    out = S.score_vs_clean(results, feats, f0)
    for feat in ("speaking_rate", "pause_count", "pause_rate"):
        pf = out["per_feature"][feat]
        assert pf["srcc"] is None
        assert pf["coverage"] == 0.0
        assert pf["n"] == 0 and pf["n_emitted"] == 0
    # headline mean coverage averages over ALL 5 headline features
    assert out["mean_coverage_reliable"] == pytest.approx(2.0 / 5.0)


def test_headline_set_is_imported_not_duplicated():
    # F8: the script must consume the canonical frozen set, not its own copy.
    assert tuple(S.HEADLINE_FEATURES) == tuple(HEADLINE_FEATURES)
    assert set(HEADLINE_FEATURES) <= set(S.FEATURE_MAP)


def test_spearman_ties():
    # constant -> zero variance -> NaN, not a crash
    assert S._spearman([1, 1, 1, 1], [1, 2, 3, 4]) != S._spearman([1, 1, 1, 1], [1, 2, 3, 4])
