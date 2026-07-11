"""Unit tests for scripts/score_inference_vs_clean.py — the trustworthy vs-clean-GT scorer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import score_inference_vs_clean as S  # noqa: E402


def _mk(n):
    """n clips; per_feature srmr perfectly rank-correlated with clean, snr anti-correlated."""
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
    # snr is degenerate-excluded -> headline mean is srmr only here
    assert abs(out["mean_excl_snr"] - 1.0) < 1e-9
    assert out["n_reliable_features"] == 1


def test_claims_schema_and_coverage():
    # 'claims' [feature,value] form; only 20 clips have a clean srmr -> coverage tracked
    results, feats, f0 = [], {}, {}
    for i in range(30):
        fn = f"c{i}.wav"
        results.append({"filename": fn, "claims": [["srmr", float(i)]]})
        if i < 20:
            feats[fn] = {"srmr": float(i)}
        f0[fn] = {}
    out = S.score_vs_clean(results, feats, f0)
    assert out["per_feature"]["srmr"]["n"] == 20
    assert abs(out["per_feature"]["srmr"]["srcc"] - 1.0) < 1e-9


def test_min_pairs_guard():
    results, feats, f0 = _mk(5)  # below default min_pairs=10
    out = S.score_vs_clean(results, feats, f0)
    assert out["per_feature"]["srmr"]["srcc"] != out["per_feature"]["srmr"]["srcc"]  # NaN
    assert out["n_reliable_features"] == 0


def test_spearman_ties():
    # constant -> zero variance -> NaN, not a crash
    assert S._spearman([1, 1, 1, 1], [1, 2, 3, 4]) != S._spearman([1, 1, 1, 1], [1, 2, 3, 4])
