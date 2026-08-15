"""Adversarial tests for the REBUILT RL reward (F19) — observability-gated,
per-slot, bounded in [0, 1].

The old tolerance-band SFS-F1 reward was RETIRED as the RL objective: bands
saturate under constant-mode collapse (emitting the population-median value for
every feature on ~all clips scores near-ceiling), so under RL pressure it is a
dead objective. Each test class here encodes an ATTACK; the invariant under
test is that gaming the reward strictly LOWERS it relative to honest behavior:

  1. perfect clip           -> R near 1
  2. CONSTANT-MODE          -> median-spam averages strictly below truth (no plateau)
  3. EASY-FEATURE-ONLY      -> 2 easy slots < a full honest attempt (ABSENT = 0)
  4. HEDGE-SPAM             -> hedging a clean clip caps at the 0.2 hedge floor
  5. attributed hedge       -> 0.8 on an unobservable slot > wrong value (0) > generic (0.4)
  6. FORMAT-GAMING          -> first claim only; unparseable mention = 0 < any hedge
  7. degeneration           -> G(y) collapses R toward 0 (loops, non-ASCII)

Torch-free: the reward is pure python over parsed text. data/feature_set.py
imports torch at module level for tensor helpers the reward never calls, so a
bare module stub satisfies that import in a torch-free environment.
"""

import math
import sys
import types

import pytest

try:  # real torch present — nothing to stub
    import torch  # noqa: F401
except ModuleNotFoundError:  # torch-free env (the intended dev/CI path)
    _stub = types.ModuleType("torch")
    _stub.Tensor = type("Tensor", (), {})

    def _needs_real_torch(*_a, **_k):
        raise ModuleNotFoundError(
            "real torch required — stubbed out for torch-free reward tests")

    _stub.tensor = _needs_real_torch
    _stub.zeros = _needs_real_torch
    _stub.bool = "bool"
    _stub.float32 = "float32"
    sys.modules["torch"] = _stub

from data.feature_set import (  # noqa: E402
    FEATURE_NAMES,
    HEDGE_OVERLAP_TAU,
    ILL_POSED_UNDER_OVERLAP_FEATURES,
)
from training.sfs_reward import (  # noqa: E402
    HEDGE_ATTRIBUTED_UNOBSERVABLE,
    HEDGE_GENERIC_UNOBSERVABLE,
    HEDGE_OBSERVABLE,
    classify_slots,
    deprecated_band_f1_reward,
    extract_completion_text,
    make_reward_func,
    make_sfs_reward_func,
    observability_from_row,
    observability_reward,
    reward_components,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
# A full features-CSV-shaped GT row (keys = feature_set CSV column names).
# overlap_ratio 0.10 < HEDGE_OVERLAP_TAU=0.5 -> every slot observable.
ROW = {
    "snr_db": 16.10,
    "srmr": 5.00,
    "f0_mean_hz": 121.00,
    "f0_sd_hz": 25.00,
    "praat_speaking_rate_syl_sec": 4.30,
    "praat_pause_count": 3,
    "praat_pause_rate_per_min": 12.00,
    "overlap_ratio": 0.10,
    "jitter_local_pct": 1.20,
    "shimmer": 5.00,
    "hnr": 11.00,
}

# High-overlap variant: overlap_ratio 0.80 >= tau -> ill-posed slots unobservable.
HIGH_OVERLAP_ROW = dict(ROW, overlap_ratio=0.80)

_COL = {
    "snr": "snr_db",
    "srmr": "srmr",
    "f0_mean": "f0_mean_hz",
    "f0_sd": "f0_sd_hz",
    "speaking_rate": "praat_speaking_rate_syl_sec",
    "pause_count": "praat_pause_count",
    "pause_rate": "praat_pause_rate_per_min",
    "overlap_ratio": "overlap_ratio",
    "jitter": "jitter_local_pct",
    "shimmer": "shimmer",
    "hnr": "hnr",
}


def honest_text(row):
    """A fully parseable 11-slot description stating the row's values."""
    return (
        f"The SNR is {row['snr_db']:.2f} dB. "
        f"The SRMR is {row['srmr']:.4f}. "
        f"The F0 mean is {row['f0_mean_hz']:.2f} Hz. "
        f"The F0 standard deviation is {row['f0_sd_hz']:.2f} Hz. "
        f"The speaking rate is {row['praat_speaking_rate_syl_sec']:.3f} syl/sec. "
        f"The pause count is {row['praat_pause_count']:d}. "
        f"The pause rate is {row['praat_pause_rate_per_min']:.3f} per minute. "
        f"The overlap ratio is {row['overlap_ratio']:.4f}. "
        f"The jitter is {row['jitter_local_pct']:.4f} percent. "
        f"The shimmer is {row['shimmer']:.4f} percent. "
        f"The HNR is {row['hnr']:.2f} dB."
    )


# ── (1) Perfect clip ──────────────────────────────────────────────────────────
class TestPerfectClip:
    def test_perfect_reward_near_one(self):
        assert observability_reward(honest_text(ROW), ROW) > 0.95

    def test_every_slot_is_a_value_slot(self):
        comps = reward_components(honest_text(ROW), ROW)
        for name in FEATURE_NAMES:
            assert comps["slots"][name]["class"] == "value", name
            assert comps["slots"][name]["reward"] >= 0.999, name
        assert comps["template_validity"] == pytest.approx(1.0)

    def test_reward_bounded_zero_one(self):
        texts = [
            honest_text(ROW),
            "",
            "The SNR is 99.00 dB. " * 20,
            "乱码" * 50,
            "The F0 is unreliable during overlap.",
        ]
        for t in texts:
            r = observability_reward(t, ROW)
            assert 0.0 <= r <= 1.0, t[:40]

    def test_empty_text_zero(self):
        assert observability_reward("", ROW) == 0.0


# ── (2) Constant-mode attack ──────────────────────────────────────────────────
def _spread_row(i):
    """Synthetic clip i in {-2..2}; i=0 is the population median row."""
    return {
        "snr_db": 16.0 + 6.0 * i,
        "srmr": 5.0 + 1.2 * i,
        "f0_mean_hz": 130.0 + 30.0 * i,
        "f0_sd_hz": 25.0 + 10.0 * i,
        "praat_speaking_rate_syl_sec": 4.0 + 0.5 * i,
        "praat_pause_count": 3 + i,
        "praat_pause_rate_per_min": 12.0 + 5.0 * i,
        "overlap_ratio": 0.20 + 0.05 * i,
        "jitter_local_pct": 1.2 + 0.4 * i,
        "shimmer": 5.0 + 1.5 * i,
        "hnr": 11.0 + 2.0 * i,
    }


class TestConstantModeAttack:
    def test_median_spam_strictly_below_truth(self):
        """Emitting the population-median value for every feature on a spread
        of clips must AVERAGE strictly below emitting the true values. This is
        the exact failure mode that saturated the retired band metric
        (f0_min=75 on 96% of clips)."""
        rows = [_spread_row(i) for i in (-2, -1, 0, 1, 2)]
        median_text = honest_text(_spread_row(0))

        true_rewards = [observability_reward(honest_text(r), r) for r in rows]
        const_rewards = [observability_reward(median_text, r) for r in rows]

        mean_true = sum(true_rewards) / len(true_rewards)
        mean_const = sum(const_rewards) / len(const_rewards)
        assert mean_const < mean_true
        # No plateau: the gap must be material, not epsilon.
        assert mean_true - mean_const > 0.2

    def test_constant_penalized_on_every_non_median_clip(self):
        median_text = honest_text(_spread_row(0))
        for i in (-2, -1, 1, 2):
            row = _spread_row(i)
            assert (observability_reward(median_text, row)
                    < observability_reward(honest_text(row), row)), i

    def test_error_scales_continuously(self):
        """Reward degrades monotonically with |v - g| — the continuous ramp
        that replaces the saturating in/out-of-band step."""
        rewards = []
        # Errors 0.0, 3.9, 9.9, 13.9, 29.9 vs the snr scale of 15: the first
        # four sit on the ramp (strictly decreasing), the last clamps to 0.
        for claimed in (16.10, 20.0, 26.0, 30.0, 46.0):
            text = f"The SNR is {claimed:.2f} dB."
            comps = reward_components(text, ROW)
            rewards.append(comps["slots"]["snr"]["reward"])
        assert all(a > b for a, b in zip(rewards, rewards[1:]))
        assert rewards[-1] == 0.0  # error >= scale saturates at 0, not below


# ── (3) Easy-feature-only attack ──────────────────────────────────────────────
class TestEasyFeatureOnly:
    def test_two_easy_slots_below_full_honest_attempt(self):
        easy = (f"The SNR is {ROW['snr_db']:.2f} dB. "
                f"The overlap ratio is {ROW['overlap_ratio']:.4f}.")
        # A full honest ATTEMPT with mild errors everywhere still wins,
        # because ABSENT slots score 0 and drag the mean down.
        from data.feature_set import FEATURE_SCALES
        scale = dict(zip(FEATURE_NAMES, FEATURE_SCALES))
        off = {col: (val + 0.1 * scale[name] if name != "pause_count" else val)
               for name, col in _COL.items() for val in [ROW[col]]}
        full_attempt = honest_text(off)

        r_easy = observability_reward(easy, ROW)
        r_full = observability_reward(full_attempt, ROW)
        assert r_easy < r_full
        assert r_easy < 0.1  # (2/11 validity) * (2/11 mean) — omission is costly

    def test_absent_slots_score_zero(self):
        comps = reward_components("The SNR is 16.10 dB.", ROW)
        for name in FEATURE_NAMES:
            if name == "snr":
                continue
            assert comps["slots"][name]["class"] == "absent"
            assert comps["slots"][name]["reward"] == 0.0


# ── (4) Hedge-spam attack ─────────────────────────────────────────────────────
HEDGE_ALL_TEXT = (
    "The F0 is unreliable. The jitter is unreliable. The shimmer is "
    "unreliable. The HNR is unreliable. The SNR is unreliable. The SRMR is "
    "unreliable. The speaking rate is unreliable. The pause count is "
    "unreliable. The pause rate is unreliable. The overlap ratio is unreliable."
)


class TestHedgeSpam:
    def test_hedge_everything_on_clean_clip_below_honest(self):
        r_spam = observability_reward(HEDGE_ALL_TEXT, ROW)
        r_honest = observability_reward(honest_text(ROW), ROW)
        assert r_spam < r_honest
        # Hedging observable slots earns at most the 0.2 hedge floor per slot,
        # so the sequence reward is capped well below it.
        assert r_spam <= HEDGE_OBSERVABLE

    def test_hedged_observable_slots_get_hedge_floor(self):
        comps = reward_components(HEDGE_ALL_TEXT, ROW)
        for name in ILL_POSED_UNDER_OVERLAP_FEATURES:
            assert comps["slots"][name]["class"] == "hedge", name
            assert comps["slots"][name]["reward"] == pytest.approx(
                HEDGE_OBSERVABLE), name

    def test_recoverable_features_cannot_be_hedged(self):
        # "The SNR is unreliable" is not a trackable hedge — it classifies as
        # an unparseable mention and scores 0.
        comps = reward_components(HEDGE_ALL_TEXT, ROW)
        assert comps["slots"]["snr"]["class"] == "mentioned_unparseable"
        assert comps["slots"]["snr"]["reward"] == 0.0


# ── (5) Hedge attribution on a high-overlap clip ──────────────────────────────
class TestHedgeAttributionReward:
    ATTRIBUTED = "Due to overlapping speech, the F0 cannot be reliably measured."
    GENERIC = "The F0 cannot be reliably measured."
    WRONG_VALUE = "The F0 mean is 300.00 Hz."
    TRUE_VALUE = "The F0 mean is 121.00 Hz."

    def test_attributed_hedge_scores_08_on_unobservable_slot(self):
        comps = reward_components(self.ATTRIBUTED, HIGH_OVERLAP_ROW)
        slot = comps["slots"]["f0_mean"]
        assert slot["class"] == "hedge"
        assert slot["attributed"] is True
        assert slot["observable"] == 0
        assert slot["reward"] == pytest.approx(HEDGE_ATTRIBUTED_UNOBSERVABLE)

    def test_generic_hedge_scores_04_on_unobservable_slot(self):
        comps = reward_components(self.GENERIC, HIGH_OVERLAP_ROW)
        assert comps["slots"]["f0_mean"]["reward"] == pytest.approx(
            HEDGE_GENERIC_UNOBSERVABLE)

    def test_value_on_unobservable_slot_scores_zero(self):
        # ANY number on an unobservable slot is an over-claim — even the
        # mix-measured "true" value earns nothing.
        for text in (self.WRONG_VALUE, self.TRUE_VALUE):
            comps = reward_components(text, HIGH_OVERLAP_ROW)
            assert comps["slots"]["f0_mean"]["class"] == "value"
            assert comps["slots"]["f0_mean"]["reward"] == 0.0

    def test_ordering_attributed_gt_generic_gt_value(self):
        r_attr = observability_reward(self.ATTRIBUTED, HIGH_OVERLAP_ROW)
        r_gen = observability_reward(self.GENERIC, HIGH_OVERLAP_ROW)
        r_val = observability_reward(self.WRONG_VALUE, HIGH_OVERLAP_ROW)
        assert r_attr > r_gen > r_val

    def test_recoverable_value_still_scored_under_overlap(self):
        comps = reward_components("The SNR is 16.10 dB.", HIGH_OVERLAP_ROW)
        assert comps["slots"]["snr"]["observable"] == 1
        assert comps["slots"]["snr"]["reward"] == pytest.approx(1.0)

    def test_value_takes_precedence_over_hedge(self):
        text = ("The F0 mean is 121.00 Hz, though the F0 is unreliable "
                "during overlap.")
        comps = reward_components(text, HIGH_OVERLAP_ROW)
        assert comps["slots"]["f0_mean"]["class"] == "value"
        assert comps["slots"]["f0_mean"]["reward"] == 0.0

    def test_calibrated_report_beats_overclaiming(self):
        """THE target behavior: values for recoverable slots + attributed
        hedges for unobservable slots beats asserting numbers everywhere."""
        calibrated = (
            f"The SNR is {HIGH_OVERLAP_ROW['snr_db']:.2f} dB. "
            f"The SRMR is {HIGH_OVERLAP_ROW['srmr']:.4f}. "
            f"The speaking rate is "
            f"{HIGH_OVERLAP_ROW['praat_speaking_rate_syl_sec']:.3f} syl/sec. "
            f"The pause count is {HIGH_OVERLAP_ROW['praat_pause_count']:d}. "
            f"The pause rate is "
            f"{HIGH_OVERLAP_ROW['praat_pause_rate_per_min']:.3f} per minute. "
            f"The overlap ratio is {HIGH_OVERLAP_ROW['overlap_ratio']:.4f}. "
            "Due to the overlapping speech, the F0, jitter, shimmer and HNR "
            "cannot be reliably measured."
        )
        overclaim = honest_text(HIGH_OVERLAP_ROW)
        r_cal = observability_reward(calibrated, HIGH_OVERLAP_ROW)
        r_over = observability_reward(overclaim, HIGH_OVERLAP_ROW)
        assert r_cal > r_over


# ── (6) Format-gaming attacks ─────────────────────────────────────────────────
class TestFormatGaming:
    def test_value_spread_earns_first_claim_only(self):
        row = dict(ROW, snr_db=10.0)
        comps = reward_components(
            "The SNR is 10.00 dB. The SNR is 20.00 dB.", row)
        assert comps["slots"]["snr"]["value"] == 10.0
        assert comps["slots"]["snr"]["reward"] == pytest.approx(1.0)

    def test_value_spread_first_claim_wrong_stays_wrong(self):
        row = dict(ROW, snr_db=10.0)
        comps = reward_components(
            "The SNR is 20.00 dB. The SNR is 10.00 dB.", row)
        assert comps["slots"]["snr"]["value"] == 20.0
        # 1 - 10/15 with the snr scale of 15
        assert comps["slots"]["snr"]["reward"] == pytest.approx(1.0 - 10.0 / 15.0)

    def test_spread_no_better_than_single_claim(self):
        row = dict(ROW, snr_db=10.0)
        r_spread = observability_reward(
            "The SNR is 10.00 dB. The SNR is 20.00 dB.", row)
        r_single = observability_reward("The SNR is 10.00 dB.", row)
        assert r_spread <= r_single + 1e-9

    def test_unparseable_mention_scores_zero_below_any_hedge(self):
        comps = reward_components("The SNR was analyzed carefully.", ROW)
        slot = comps["slots"]["snr"]
        assert slot["class"] == "mentioned_unparseable"
        assert slot["reward"] == 0.0
        assert slot["reward"] < HEDGE_OBSERVABLE < HEDGE_GENERIC_UNOBSERVABLE \
            < HEDGE_ATTRIBUTED_UNOBSERVABLE

    def test_mention_does_not_count_toward_validity(self):
        comps = reward_components("The SNR was analyzed carefully.", ROW)
        assert comps["template_validity"] == 0.0
        assert comps["reward"] == 0.0


# ── (7) Degeneration gate ─────────────────────────────────────────────────────
class TestDegenerationGate:
    def test_repetition_loop_collapses_reward(self):
        loop = " ".join(["broken broken broken broken"] * 40)
        clean = honest_text(ROW)
        r_clean = observability_reward(clean, ROW)
        r_loop = observability_reward(clean + " " + loop, ROW)
        assert r_loop < 0.5 * r_clean

    def test_pure_loop_scores_zero(self):
        loop = " ".join(["broken broken broken broken"] * 40)
        assert observability_reward(loop, ROW) == 0.0  # validity 0

    def test_nonascii_injection_lowers_reward(self):
        clean = honest_text(ROW)
        injected = clean + " " + "这是乱码注入文本严重降低生成质量。" * 5
        r_clean = observability_reward(clean, ROW)
        r_inj = observability_reward(injected, ROW)
        assert r_inj < r_clean
        comps = reward_components(injected, ROW)
        assert comps["nonascii_fraction"] > 0.0

    def test_gate_components_reported(self):
        comps = reward_components(honest_text(ROW), ROW)
        assert comps["gate"] == pytest.approx(
            comps["template_validity"]
            * (1.0 - comps["rep_fraction"])
            * (1.0 - comps["nonascii_fraction"])
        )
        assert comps["reward"] == pytest.approx(
            comps["gate"] * comps["mean_slot_reward"])


# ── Observability rule (single source of truth: feature_set) ──────────────────
class TestObservabilityRule:
    def test_clean_clip_all_observable(self):
        obs = observability_from_row(ROW)
        assert all(obs[name] == 1 for name in FEATURE_NAMES)

    def test_high_overlap_masks_ill_posed_only(self):
        obs = observability_from_row(HIGH_OVERLAP_ROW)
        for name in FEATURE_NAMES:
            expected = 0 if name in ILL_POSED_UNDER_OVERLAP_FEATURES else 1
            assert obs[name] == expected, name

    def test_tau_boundary_is_inclusive(self):
        row = dict(ROW, overlap_ratio=HEDGE_OVERLAP_TAU)
        obs = observability_from_row(row)
        assert obs["f0_mean"] == 0

    def test_missing_overlap_ratio_means_observable(self):
        row = {k: v for k, v in ROW.items() if k != "overlap_ratio"}
        row["overlap_ratio"] = float("nan")
        assert all(v == 1 for v in observability_from_row(row).values())

    def test_explicit_observability_override(self):
        obs = {name: 1 for name in FEATURE_NAMES}
        obs["snr"] = 0
        comps = reward_components("The SNR is 16.10 dB.", ROW, observability=obs)
        assert comps["slots"]["snr"]["reward"] == 0.0


# ── Slot classification ───────────────────────────────────────────────────────
class TestClassifySlots:
    def test_four_way_classification(self):
        text = ("The SNR is 16.10 dB. The SRMR was hard to assess. "
                "The F0 is unreliable during overlap.")
        slots = classify_slots(text)
        assert slots["snr"]["class"] == "value"
        assert slots["snr"]["value"] == 16.10
        assert slots["srmr"]["class"] == "mentioned_unparseable"
        assert slots["f0_mean"]["class"] == "hedge"
        assert slots["f0_mean"]["attributed"] is True
        assert slots["shimmer"]["class"] == "absent"


# ── TRL batch wrapper ─────────────────────────────────────────────────────────
class TestExtractCompletionText:
    def test_plain_string(self):
        assert extract_completion_text("hello") == "hello"

    def test_chat_message_list(self):
        comp = [{"role": "assistant", "content": "The SNR is 16.10 dB."}]
        assert extract_completion_text(comp) == "The SNR is 16.10 dB."

    def test_single_message_dict(self):
        assert extract_completion_text({"role": "assistant", "content": "x"}) == "x"

    def test_none_and_empty(self):
        assert extract_completion_text(None) == ""
        assert extract_completion_text([]) == ""


class TestBatchWrapper:
    def test_aligned_rows_via_kwargs(self):
        reward_func = make_reward_func()
        completions = [honest_text(ROW), "The recording is clean.", ""]
        rewards = reward_func(
            prompts=["p"] * 3, completions=completions,
            gt_features=[ROW, ROW, ROW],
        )
        assert isinstance(rewards, list) and len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert rewards[0] > rewards[1] == rewards[2] == 0.0

    def test_aligned_rows_via_constructor_list(self):
        reward_func = make_reward_func([ROW, ROW])
        rewards = reward_func(
            prompts=["p", "p"],
            completions=[honest_text(ROW), "nothing here"])
        assert rewards[0] > rewards[1]

    def test_mapping_lookup_keyed_by_clip_id(self):
        reward_func = make_reward_func({"clip_A": ROW, "clip_B": ROW})
        rewards = reward_func(
            prompts=["pA", "pB"],
            completions=[honest_text(ROW), "no numbers"],
            clip_ids=["clip_A", "clip_B"],
        )
        assert rewards[0] > rewards[1]

    def test_chat_completions(self):
        reward_func = make_reward_func()
        completions = [
            [{"role": "assistant", "content": honest_text(ROW)}],
            [{"role": "assistant", "content": "clean prose, no claims"}],
        ]
        rewards = reward_func(completions=completions, gt_features=[ROW, ROW])
        assert rewards[0] > rewards[1]

    def test_missing_row_yields_zero(self):
        reward_func = make_reward_func()
        rewards = reward_func(completions=["The SNR is 16.10 dB."])
        assert rewards == [0.0]

    def test_per_sample_observability_column(self):
        reward_func = make_reward_func()
        obs_all_off = {name: 0 for name in FEATURE_NAMES}
        rewards = reward_func(
            completions=[honest_text(ROW), honest_text(ROW)],
            gt_features=[ROW, ROW],
            observability=[None, obs_all_off],
        )
        assert rewards[0] > 0.9
        assert rewards[1] == 0.0  # every VALUE slot gated off


# ── Back-compat surface ───────────────────────────────────────────────────────
class TestBackCompat:
    def test_deprecated_factory_name_returns_new_reward(self):
        """grpo_train.py imports make_sfs_reward_func with the legacy kwargs —
        it must keep working AND produce the NEW observability reward."""
        with pytest.warns(DeprecationWarning):
            reward_func = make_sfs_reward_func(
                gt_lookup=None, f1_weight=1.0, rep_penalty=0.5,
                nonascii_penalty=1.0, rep_n=4,
            )
        rewards = reward_func(
            prompts=["p"], completions=[honest_text(ROW)], gt_features=[ROW])
        assert rewards[0] == pytest.approx(
            observability_reward(honest_text(ROW), ROW))

    def test_deprecated_band_f1_reward_still_callable(self):
        gt = {"snr": 16.10, "f0_mean": 121.00}
        text = "The SNR is 16.10 dB. The F0 mean is 121.00 Hz."
        r = deprecated_band_f1_reward(text, gt)
        assert r == pytest.approx(1.0)  # exact claims, clean text: band-F1 = 1

    def test_new_reward_is_not_the_band_metric(self):
        """The constant-mode text that saturates the band metric must NOT
        saturate the new reward."""
        row = _spread_row(2)
        median_text = honest_text(_spread_row(0))
        r_new = observability_reward(median_text, row)
        assert r_new < 0.5
