"""Unit tests for the scalar->text routing splice (scripts/scalar_to_text_experiment.py).

CPU-only: no torch / no model. Verifies that splice_text() correctly rewrites the
numeric value of each feature's claim with the head's prediction (keeping the
sentence + units so src/sfs.py's parser reads the new number), and that a feature
the LM omitted is left alone when --inject-missing is off.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scalar_to_text_experiment import splice_text, SPLICE_SPECS, HEAD_KEYS
from sfs import ClaimParser


# The exact untagged-prose phrasing v17 emits (from inference_results.json).
V17_GENERATED = (
    "The signal-to-noise ratio SNR is 18.05 dB. The SRMR is 4.5857. "
    "The F0 mean is 137.17 Hz and the F0 standard deviation SD is 50.65 Hz. "
    "The speaking rate is 5.853 syl/sec. The pause count is 2 and the pause rate "
    "is 11.329 per min. The overlap ratio is 0.6975."
)


def _claims_dict(text: str) -> dict:
    return {c.feature: c.value for c in ClaimParser().parse(text)}


class TestSpliceReplaces:
    def test_f0_mean_replaced_and_parsed(self):
        """'The F0 mean is 110.0 Hz' -> head value, and the parser reads the new number."""
        text = "The F0 mean is 110.0 Hz."
        out = splice_text(text, {"f0_mean": 142.93})
        # number changed, sentence + unit intact
        assert "142.93" in out
        assert "110.0" not in out
        assert "Hz" in out
        # parser re-reads the spliced number as f0_mean
        assert _claims_dict(out)["f0_mean"] == 142.93

    def test_all_eight_features_replaced(self):
        head = {
            "snr": 25.50, "srmr": 9.1234, "f0_mean": 200.11, "f0_sd": 33.33,
            "speaking_rate": 4.500, "pause_count": 7, "pause_rate": 3.210,
            "overlap_ratio": 0.1234,
        }
        out = splice_text(V17_GENERATED, head)
        got = _claims_dict(out)
        assert got["snr"] == 25.50
        assert got["srmr"] == 9.1234
        assert got["f0_mean"] == 200.11
        assert got["f0_sd"] == 33.33
        assert got["speaking_rate"] == 4.500
        assert got["pause_count"] == 7.0
        assert got["pause_rate"] == 3.210
        assert got["overlap_ratio"] == 0.1234

    def test_f0_mean_does_not_clobber_f0_sd(self):
        """Splicing f0_mean must not touch the f0 standard deviation value."""
        out = splice_text(V17_GENERATED, {"f0_mean": 999.99})
        got = _claims_dict(out)
        assert got["f0_mean"] == 999.99
        assert got["f0_sd"] == 50.65  # unchanged

    def test_pause_count_formatted_as_integer(self):
        out = splice_text("The pause count is 2.", {"pause_count": 5.0})
        assert "The pause count is 5." in out
        assert "5.0" not in out

    def test_units_preserved_for_snr(self):
        out = splice_text("The signal-to-noise ratio SNR is 18.05 dB.", {"snr": 30.00})
        assert "30.00 dB" in out
        assert _claims_dict(out)["snr"] == 30.00

    def test_overlap_ratio_four_decimals(self):
        out = splice_text("The overlap ratio is 0.6975.", {"overlap_ratio": 0.5})
        assert "0.5000" in out
        assert _claims_dict(out)["overlap_ratio"] == 0.5000


class TestSpliceLeavesOmittedFeatures:
    def test_omitted_feature_left_alone_when_inject_off(self):
        """LM omitted speaking_rate -> with inject off, text is untouched for it."""
        text = "The signal-to-noise ratio SNR is 18.05 dB. The overlap ratio is 0.6975."
        # head predicts speaking_rate but the LM never mentioned it
        out = splice_text(text, {"speaking_rate": 4.2}, inject_missing=False)
        # no speaking-rate sentence was added, no speaking_rate claim appears
        assert "speaking rate" not in out.lower()
        assert "speaking_rate" not in _claims_dict(out)
        # the features that WERE present are still there, unchanged
        assert out == text

    def test_only_present_feature_is_changed(self):
        text = "The signal-to-noise ratio SNR is 18.05 dB. The overlap ratio is 0.6975."
        out = splice_text(text, {"snr": 9.99, "speaking_rate": 4.2}, inject_missing=False)
        got = _claims_dict(out)
        assert got["snr"] == 9.99
        assert "speaking_rate" not in got
        assert got["overlap_ratio"] == 0.6975  # untouched


class TestInjectMissing:
    def test_inject_missing_adds_canonical_sentence(self):
        """With inject on, an omitted feature gets its canonical template sentence,
        and the parser reads the injected value."""
        text = "The signal-to-noise ratio SNR is 18.05 dB."
        out = splice_text(text, {"speaking_rate": 4.250}, inject_missing=True)
        got = _claims_dict(out)
        assert got["speaking_rate"] == 4.250
        # original snr claim survives
        assert got["snr"] == 18.05


class TestSpecsCoverAllHeadFeatures:
    def test_every_head_feature_has_a_spec(self):
        for k in HEAD_KEYS:
            assert k in SPLICE_SPECS, f"no splice spec for head feature {k}"

    def test_anchor_has_single_capture_group(self):
        # splice_text relies on group(1) being the number for every spec.
        for k, spec in SPLICE_SPECS.items():
            assert spec["anchor"].groups == 1, f"{k} anchor must have exactly 1 group"
