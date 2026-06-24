"""CPU-isolated unit tests for src/selection_metric.py.

No GPU and no train-loop imports — only the pure selection primitives plus the
HybridClaimParser / feature_set they depend on. Run with:

    pytest tests/test_selection_metric.py -v
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selection_metric import (  # noqa: E402
    band_free_val_scores,
    composite_score,
    ema,
    avg_state_dicts,
    SELECTION_FEATURES,
    DEGENERATE_SELECTION_FEATURES,
)
from feature_set import RECOVERABLE_FEATURES  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────
def _desc(snr, srmr, sr, f0):
    """A canonical-style description asserting four scorable features."""
    return (
        f"The SNR is {snr:.2f} dB. The SRMR is {srmr:.2f}. "
        f"The speaking rate is {sr:.2f} syl/sec. The F0 mean is {f0:.2f} Hz."
    )


# ═══════════════════════════════════════════════════════════════════════════
# band_free_val_scores — round-trip on a tiny synthetic with KNOWN SRCC signs
# ═══════════════════════════════════════════════════════════════════════════
class TestBandFreeRoundTrip:
    def _build(self):
        # 5 clips. Construct so the parsed prediction tracks the GT in a KNOWN
        # rank relationship per feature:
        #   srmr           : pred increases with gt  -> SRCC = +1
        #   speaking_rate  : pred decreases with gt  -> SRCC = -1
        #   snr            : pred == gt              -> SRCC = +1
        #   f0_mean        : only 1 clip emits it    -> SRCC = None (n<2)
        gens, fnames, gt = [], [], {}
        srmr_gt = [3.0, 3.5, 4.0, 4.5, 5.0]
        sr_gt = [4.0, 4.5, 5.0, 5.5, 6.0]
        snr_gt = [10.0, 12.0, 14.0, 16.0, 18.0]
        for i in range(5):
            key = f"clip_{i:04d}"
            # pred srmr tracks gt (monotone up); pred speaking_rate is the
            # REVERSE order so its rank corr vs gt is -1.
            pred_srmr = srmr_gt[i] + 0.1
            pred_sr = sr_gt[4 - i]
            pred_snr = snr_gt[i]
            # only clip 0 emits an f0 number; the rest omit it
            f0_txt = " The F0 mean is 150.00 Hz." if i == 0 else ""
            gens.append(
                f"The SNR is {pred_snr:.2f} dB. The SRMR is {pred_srmr:.2f}. "
                f"The speaking rate is {pred_sr:.2f} syl/sec.{f0_txt}"
            )
            fnames.append(key + ".wav")  # extension tolerated
            gt[key] = {
                "snr": snr_gt[i],
                "srmr": srmr_gt[i],
                "speaking_rate": sr_gt[i],
                "f0_mean": 150.0,
            }
        return gens, fnames, gt

    def test_srcc_signs_and_coverage(self):
        gens, fnames, gt = self._build()
        out = band_free_val_scores(gens, fnames, gt)
        pf = out["per_feature"]

        assert out["n_clips"] == 5
        # srmr pred is a strictly increasing shift of gt -> perfect +1 rank corr
        assert pf["srmr"]["srcc"] == pytest.approx(1.0, abs=1e-9)
        assert pf["srmr"]["n"] == 5
        assert pf["srmr"]["coverage"] == pytest.approx(1.0)
        # speaking_rate pred is the reversed gt order -> -1 rank corr
        assert pf["speaking_rate"]["srcc"] == pytest.approx(-1.0, abs=1e-9)
        # snr pred == gt -> +1
        assert pf["snr"]["srcc"] == pytest.approx(1.0, abs=1e-9)
        # f0_mean emitted by only 1 of 5 clips -> no SRCC, coverage 1/5
        assert pf["f0_mean"]["srcc"] is None
        assert pf["f0_mean"]["n"] == 1
        assert pf["f0_mean"]["coverage"] == pytest.approx(0.2)

    def test_nmae_is_band_free_normalized(self):
        gens, fnames, gt = self._build()
        pf = band_free_val_scores(gens, fnames, gt)["per_feature"]
        # srmr pred = gt + 0.1 for every clip -> MAE = 0.1.
        # gt = [3.0,3.5,4.0,4.5,5.0] -> sample std (ddof=1) ~ 0.7906
        # nmae = 0.1 / 0.7906 ~ 0.1265
        gt_vals = [3.0, 3.5, 4.0, 4.5, 5.0]
        m = sum(gt_vals) / len(gt_vals)
        std = math.sqrt(sum((x - m) ** 2 for x in gt_vals) / (len(gt_vals) - 1))
        assert pf["srmr"]["nmae"] == pytest.approx(0.1 / std, rel=1e-6)

    def test_all_12_features_present_in_output(self):
        gens, fnames, gt = self._build()
        pf = band_free_val_scores(gens, fnames, gt)["per_feature"]
        assert set(pf.keys()) == set(SELECTION_FEATURES)
        # a feature with no GT/no emission has n=0, coverage 0, srcc/nmae None
        assert pf["jitter"]["n"] == 0
        assert pf["jitter"]["srcc"] is None
        assert pf["jitter"]["coverage"] == 0.0

    def test_missing_filename_skipped(self):
        gens, fnames, gt = self._build()
        gens.append("The SNR is 99.00 dB.")
        fnames.append("not_in_gt.wav")
        out = band_free_val_scores(gens, fnames, gt)
        # the extra clip has no GT, so snr still only has the original 5 pairs
        assert out["per_feature"]["snr"]["n"] == 5
        assert out["n_clips"] == 6

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            band_free_val_scores(["a", "b"], ["x"], {})


# ═══════════════════════════════════════════════════════════════════════════
# composite_score
# ═══════════════════════════════════════════════════════════════════════════
class TestCompositeScore:
    def _pf(self):
        # snr is degenerate (excluded); srmr + speaking_rate are reliable +
        # non-degenerate; f0_mean is ill-posed (NOT reliable) so excluded even
        # though it has a high srcc.
        return {
            "snr": {"srcc": -0.9, "nmae": 2.0, "n": 50},          # degenerate-excluded
            "srmr": {"srcc": 0.8, "nmae": 0.2, "n": 50},          # reliable
            "speaking_rate": {"srcc": 0.6, "nmae": 0.4, "n": 50}, # reliable
            "f0_mean": {"srcc": 0.95, "nmae": 0.1, "n": 50},      # ill-posed (excluded)
            "pause_count": {"srcc": 0.7, "nmae": 0.3, "n": 2},    # reliable but n<min_pairs
        }

    def test_excludes_snr_and_illposed_and_low_n(self):
        pf = self._pf()
        s = composite_score(pf, RECOVERABLE_FEATURES, lam_nmae=0.5, min_pairs=5)
        # usable = srmr, speaking_rate (snr degenerate, f0_mean ill-posed,
        # pause_count below min_pairs).
        mean_srcc = (0.8 + 0.6) / 2
        mean_nmae = (0.2 + 0.4) / 2
        assert s == pytest.approx(mean_srcc - 0.5 * mean_nmae)

    def test_snr_truly_excluded_by_name(self):
        # Even if snr were reliable and high-n, it must be dropped.
        pf = {"snr": {"srcc": 1.0, "nmae": 0.0, "n": 100},
              "srmr": {"srcc": 0.5, "nmae": 0.0, "n": 100}}
        reliable = RECOVERABLE_FEATURES | {"snr"}
        s = composite_score(pf, reliable, lam_nmae=0.5, min_pairs=5)
        # snr dropped -> mean over {srmr} only = 0.5
        assert s == pytest.approx(0.5)
        assert "snr" in DEGENERATE_SELECTION_FEATURES

    def test_bleu_floor_returns_neg_inf(self):
        pf = self._pf()
        s = composite_score(pf, RECOVERABLE_FEATURES, bleu=3.0, bleu_floor=5.0)
        assert s == float("-inf")

    def test_bleu_above_floor_scores_normally(self):
        pf = self._pf()
        s_floor = composite_score(pf, RECOVERABLE_FEATURES, bleu=9.0, bleu_floor=5.0)
        s_nofloor = composite_score(pf, RECOVERABLE_FEATURES)
        assert s_floor == pytest.approx(s_nofloor)
        assert math.isfinite(s_floor)

    def test_none_bleu_with_floor_is_rejected(self):
        pf = self._pf()
        s = composite_score(pf, RECOVERABLE_FEATURES, bleu=None, bleu_floor=5.0)
        assert s == float("-inf")

    def test_none_srcc_features_ignored(self):
        pf = {"srmr": {"srcc": None, "nmae": None, "n": 1},
              "speaking_rate": {"srcc": 0.4, "nmae": 0.2, "n": 50}}
        s = composite_score(pf, RECOVERABLE_FEATURES, lam_nmae=0.5, min_pairs=5)
        # only speaking_rate usable
        assert s == pytest.approx(0.4 - 0.5 * 0.2)

    def test_empty_usable_set_is_zero(self):
        pf = {"snr": {"srcc": 0.9, "nmae": 0.1, "n": 50}}  # only the excluded feature
        s = composite_score(pf, RECOVERABLE_FEATURES, lam_nmae=0.5)
        assert s == pytest.approx(0.0)

    def test_lam_nmae_weights_penalty(self):
        pf = {"srmr": {"srcc": 0.8, "nmae": 1.0, "n": 50}}
        s0 = composite_score(pf, RECOVERABLE_FEATURES, lam_nmae=0.0)
        s1 = composite_score(pf, RECOVERABLE_FEATURES, lam_nmae=1.0)
        assert s0 == pytest.approx(0.8)
        assert s1 == pytest.approx(0.8 - 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# ema
# ═══════════════════════════════════════════════════════════════════════════
class TestEMA:
    def test_first_call_is_identity(self):
        assert ema(None, 0.5) == pytest.approx(0.5)

    def test_smoothing_formula(self):
        # ema_t = beta*prev + (1-beta)*new
        assert ema(1.0, 0.0, beta=0.7) == pytest.approx(0.7)
        assert ema(0.0, 1.0, beta=0.7) == pytest.approx(0.3)

    def test_monotonic_ish_tracks_toward_new(self):
        # A constant stream of a higher value pulls the EMA monotonically up
        # toward it without overshooting.
        prev = 0.0
        target = 1.0
        vals = []
        for _ in range(20):
            prev = ema(prev, target, beta=0.7)
            vals.append(prev)
        # strictly increasing, bounded above by target, converging to it
        assert all(b > a for a, b in zip(vals, vals[1:]))
        assert all(v < target for v in vals)
        assert vals[-1] == pytest.approx(target, abs=1e-2)

    def test_nan_new_passes_through_prev(self):
        assert ema(0.42, float("nan"), beta=0.7) == pytest.approx(0.42)

    def test_bad_beta_raises(self):
        with pytest.raises(ValueError):
            ema(0.0, 1.0, beta=1.0)
        with pytest.raises(ValueError):
            ema(0.0, 1.0, beta=-0.1)


# ═══════════════════════════════════════════════════════════════════════════
# avg_state_dicts  (needs torch)
# ═══════════════════════════════════════════════════════════════════════════
torch = pytest.importorskip("torch")


class TestAvgStateDicts:
    def test_averages_hand_example(self):
        a = {"w": torch.tensor([0.0, 2.0, 4.0]), "b": torch.tensor([10.0])}
        b = {"w": torch.tensor([2.0, 4.0, 6.0]), "b": torch.tensor([20.0])}
        out = avg_state_dicts([a, b])
        assert torch.allclose(out["w"], torch.tensor([1.0, 3.0, 5.0]))
        assert torch.allclose(out["b"], torch.tensor([15.0]))

    def test_three_way_mean(self):
        sds = [
            {"x": torch.tensor([3.0])},
            {"x": torch.tensor([6.0])},
            {"x": torch.tensor([9.0])},
        ]
        out = avg_state_dicts(sds)
        assert out["x"].item() == pytest.approx(6.0)

    def test_single_dict_is_copy(self):
        a = {"w": torch.tensor([1.0, 2.0])}
        out = avg_state_dicts([a])
        assert torch.allclose(out["w"], a["w"])
        # must be a new top-level dict
        assert out is not a

    def test_preserves_dtype(self):
        a = {"w": torch.tensor([0, 4], dtype=torch.int64)}
        b = {"w": torch.tensor([2, 8], dtype=torch.int64)}
        out = avg_state_dicts([a, b])
        assert out["w"].dtype == torch.int64
        assert torch.equal(out["w"], torch.tensor([1, 6], dtype=torch.int64))

    def test_mismatched_keys_raise(self):
        a = {"w": torch.tensor([1.0])}
        b = {"v": torch.tensor([1.0])}
        with pytest.raises(ValueError):
            avg_state_dicts([a, b])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            avg_state_dicts([])

    def test_non_tensor_entries_kept_from_first(self):
        a = {"w": torch.tensor([2.0]), "meta": "epoch_5"}
        b = {"w": torch.tensor([4.0]), "meta": "epoch_6"}
        out = avg_state_dicts([a, b])
        assert out["w"].item() == pytest.approx(3.0)
        assert out["meta"] == "epoch_5"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
