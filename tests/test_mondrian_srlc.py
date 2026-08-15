"""Unit tests for src/eval/mondrian_srlc.py (Mondrian selective-risk-control).

Pure numpy — no torch. Verifies: the risk guarantee holds on the emitted set, an
all-wrong cell abstains by construction, an all-correct cell emits everything,
overlap binning, and that recoverable/counting features are never hedged.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval.mondrian_srlc import (  # noqa: E402
    crc_cell,
    mondrian_calibrate,
    overlap_bin,
    should_emit,
)


def test_overlap_bin():
    assert overlap_bin(0.0) == "none"
    assert overlap_bin(0.3) == "partial"
    assert overlap_bin(0.5) == "heavy"
    assert overlap_bin(0.99) == "heavy"
    assert overlap_bin(float("nan")) == "none"


def test_crc_cell_finds_threshold_and_controls_risk():
    # low-sigma clips are correct (err<k), high-sigma clips are wrong (err>k)
    rng = np.random.default_rng(1)
    n = 400
    sig = np.concatenate([rng.uniform(0.0, 1.0, n // 2), rng.uniform(3.0, 5.0, n // 2)])
    err = np.concatenate([rng.uniform(0.0, 0.5, n // 2), rng.uniform(2.0, 4.0, n // 2)])
    out = crc_cell(sig, err, alpha=0.1, delta=0.05, k=1.0)
    assert out["feasible"]
    assert 0.0 < out["coverage"] < 1.0            # abstains on the high-sigma half
    assert out["ucb"] <= 0.1 + 1e-9               # guarantee holds
    # emitted set (sigma <= threshold) has empirical miscoverage <= alpha
    emit = sig <= out["threshold"]
    emp = np.mean((err[emit] > 1.0))
    assert emp <= 0.1 + 1e-9


def test_crc_cell_abstains_by_construction_when_all_wrong():
    sig = np.linspace(0.1, 1.0, 50)
    err = np.full(50, 5.0)                          # everything exceeds k=1
    out = crc_cell(sig, err, alpha=0.1, delta=0.05, k=1.0)
    assert out["feasible"] is False
    assert out["threshold"] is None
    assert out["coverage"] == 0.0


def test_crc_cell_emits_all_when_all_correct():
    sig = np.linspace(0.1, 1.0, 200)
    err = np.full(200, 0.1)                         # all well within k=1
    out = crc_cell(sig, err, alpha=0.1, delta=0.05, k=1.0)
    assert out["feasible"]
    assert out["coverage"] == pytest.approx(1.0)
    assert math.isinf(out["threshold"])


def test_mondrian_recoverable_never_hedged_illposed_gets_cells():
    feats = ["snr", "f0_mean", "speaking_rate", "jitter"]
    scales = [15.0, 44.0, 0.75, 0.6]
    ill = frozenset({"f0_mean", "jitter"})
    rng = np.random.default_rng(2)
    records = []
    for _ in range(300):
        ov = rng.choice([0.0, 0.3, 0.7])
        rec = {"overlap_ratio": ov}
        # ill-posed features: sigma tracks error under heavy overlap
        for f, sc in (("f0_mean", 44.0), ("jitter", 0.6)):
            hard = ov >= 0.5
            rec[f"sigma_{f}"] = rng.uniform(2.0, 4.0) if hard else rng.uniform(0.0, 0.8)
            rec[f"err_{f}"] = (rng.uniform(2.0, 4.0) if hard else rng.uniform(0.0, 0.3)) * sc
        records.append(rec)

    cal = mondrian_calibrate(records, feature_names=feats, scales=scales,
                             ill_posed_features=ill, alpha=0.1, delta=0.05, k=1.0)
    # recoverable feats are "always emit", not in cells
    assert set(cal["always_emit"]) == {"snr", "speaking_rate"}
    assert all(f in {"f0_mean", "jitter"} for (f, _g) in cal["cells"])
    # a recoverable feature always emits regardless of sigma/overlap
    assert should_emit(cal, "snr", sigma_norm=99.0, overlap_ratio=0.9) is True
    # ill-posed feature under heavy overlap with huge sigma should abstain
    assert should_emit(cal, "f0_mean", sigma_norm=99.0, overlap_ratio=0.9) is False
    # ill-posed feature under no overlap with tiny sigma should emit
    assert should_emit(cal, "f0_mean", sigma_norm=0.01, overlap_ratio=0.0) is True


def test_bonferroni_delta_split_recorded():
    feats = ["f0_mean"]
    records = [{"overlap_ratio": 0.0, "sigma_f0_mean": 0.1, "err_f0_mean": 1.0},
               {"overlap_ratio": 0.7, "sigma_f0_mean": 3.0, "err_f0_mean": 100.0}]
    cal = mondrian_calibrate(records, feature_names=feats, scales=[44.0],
                             ill_posed_features=frozenset({"f0_mean"}),
                             alpha=0.1, delta=0.06)
    # 2 active cells (none, heavy) -> delta_cell = 0.03
    assert cal["params"]["n_cells"] == 2
    assert cal["params"]["delta_cell"] == pytest.approx(0.03)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
