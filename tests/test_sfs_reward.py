"""Tests for sfs_reward.py — the RLVR reward built on the SFS verifier.

These prove SOUNDNESS of the reward (no trl, no GPU, no cluster):
  - a faithful description (correct GT numbers) scores HIGH
  - a number-spam description (many WRONG numbers) scores LOWER than the faithful
    one — proving F1's precision term punishes wrong/extra claims (F1 > recall)
  - degenerate completions (heavy repetition; foreign / non-ASCII tokens) are
    penalized below an equally-faithful clean completion
  - empty / no-claims text scores ~0 or negative
  - the TRL batch wrapper returns a list[float] of the right length

GT feature dicts use the exact shape SFSScorer.score expects (see test_sfs.py):
scalar features keyed by name, plus "overlap_segments": [(start, end), ...].
"""

import pytest

from sfs_reward import (
    extract_completion_text,
    make_sfs_reward_func,
    sfs_f1,
    sfs_reward,
)


# A realistic per-clip ground-truth dict. Values chosen to sit comfortably
# inside SFSScorer.TOLERANCES so a correctly-stated number scores as correct.
GT = {
    "snr": 16.10,            # ±2 dB
    "f0_mean": 121.00,       # ±5 Hz
    "hnr": 11.00,            # ±2 dB
    "srmr": 5.00,            # ±0.5
    "speaking_rate": 4.30,   # ±0.5 syl/s
}


# A faithful description that states the correct GT numbers in SFS-parseable
# phrasing ("The SNR is 16.10 dB" — the parser accepts "is"/"of"/"=", rejects
# the colon form per the project's verbalization contract).
FAITHFUL = (
    "The SNR is 16.10 dB. The F0 mean is 121.00 Hz. "
    "The HNR is 11.00 dB. The SRMR is 5.00. "
    "The speaking rate is 4.30 syl/s."
)


class TestSfsF1:
    def test_faithful_text_high_f1(self):
        assert sfs_f1(FAITHFUL, GT) >= 0.99  # all correct, all GT mentioned

    def test_empty_text_zero_f1(self):
        assert sfs_f1("", GT) == 0.0

    def test_no_claims_zero_f1(self):
        assert sfs_f1("The audio sounds clean and pleasant.", GT) == 0.0


class TestSfsReward:
    def test_faithful_gets_high_reward(self):
        r = sfs_reward(FAITHFUL, GT)
        # F1 ~1.0, no repetition, no non-ASCII → reward ~1.0.
        assert r > 0.9

    def test_number_spam_lower_than_faithful(self):
        """A spammer emits a number for every feature but most are WRONG.

        Recall would be ~1.0 (every feature 'mentioned'), but precision tanks
        because the wrong claims are counted incorrect, so F1 < faithful's F1.
        This is the whole reason we reward F1 and not recall.
        """
        spam = (
            "The SNR is 99.00 dB. The F0 mean is 9.00 Hz. "
            "The HNR is 99.00 dB. The SRMR is 99.00. "
            "The speaking rate is 99.00 syl/s. "
            # extra wrong claims for features not even in GT — hurts precision more
            "The jitter is 50.00%. The shimmer is 80.00%."
        )
        faithful_r = sfs_reward(FAITHFUL, GT)
        spam_r = sfs_reward(spam, GT)
        assert spam_r < faithful_r
        # And the spam reward should be clearly worse, not a hair below.
        assert faithful_r - spam_r > 0.3

    def test_partial_spam_recall_high_but_f1_punished(self):
        """Directly contrast F1 vs recall on the spam text: recall stays high
        (everything mentioned) yet F1 is dragged down by precision."""
        from sfs import HybridClaimParser, SFSScorer

        spam = (
            "The SNR is 99.00 dB. The F0 mean is 9.00 Hz. "
            "The HNR is 99.00 dB. The SRMR is 99.00. "
            "The speaking rate is 99.00 syl/s."
        )
        res = SFSScorer().score(HybridClaimParser().parse(spam), dict(GT))
        # Every GT feature is mentioned → recall is high...
        assert res["recall"] >= 0.99
        # ...but precision collapses (all wrong) → F1 << recall.
        assert res["precision"] == 0.0
        assert res["f1"] < res["recall"]

    def test_repetition_penalized_below_clean(self):
        """An equally-faithful completion with a heavy repetition loop appended
        scores below the clean faithful one."""
        loop = " ".join(["broken broken broken broken"] * 40)
        repetitive = FAITHFUL + " " + loop
        clean_r = sfs_reward(FAITHFUL, GT)
        rep_r = sfs_reward(repetitive, GT)
        assert rep_r < clean_r

    def test_nonascii_penalized_below_clean(self):
        """Foreign / non-ASCII token injection is penalized below clean text of
        the same faithfulness."""
        injected = FAITHFUL + " 这是一段乱码注入文本严重降低质量。"
        clean_r = sfs_reward(FAITHFUL, GT)
        inj_r = sfs_reward(injected, GT)
        assert inj_r < clean_r

    def test_empty_reward_non_positive(self):
        assert sfs_reward("", GT) <= 0.0

    def test_no_claims_reward_near_zero(self):
        r = sfs_reward("The recording is pleasant and clear.", GT)
        # No correct claims → F1 0; clean prose → no penalty → reward ~0.
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_degenerate_no_claims_negative(self):
        """No correct claims AND degenerate → strictly negative."""
        garbage = " ".join(["乱码乱码乱码乱码"] * 30)
        assert sfs_reward(garbage, GT) < 0.0

    def test_weights_respected(self):
        """Zeroing the penalties makes a repetitive-but-faithful completion tie
        the clean one — confirms the penalty terms are what separated them."""
        loop = " ".join(["broken broken broken broken"] * 40)
        repetitive = FAITHFUL + " " + loop
        clean = sfs_reward(FAITHFUL, GT, rep_penalty=0.0, nonascii_penalty=0.0)
        rep = sfs_reward(repetitive, GT, rep_penalty=0.0, nonascii_penalty=0.0)
        assert rep == pytest.approx(clean, abs=1e-6)


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
    def test_aligned_gt_via_kwargs(self):
        """GT forwarded as a TRL dataset column (kwargs list aligned with
        completions) — the recommended path."""
        reward_func = make_sfs_reward_func()
        completions = [FAITHFUL, "The recording is clean.", ""]
        gts = [GT, GT, GT]
        rewards = reward_func(prompts=["p", "p", "p"], completions=completions, gt_features=gts)
        assert isinstance(rewards, list)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        # faithful > no-claims >= empty
        assert rewards[0] > rewards[1]
        assert rewards[1] >= rewards[2]

    def test_aligned_gt_via_constructor_list(self):
        reward_func = make_sfs_reward_func([GT, GT])
        rewards = reward_func(prompts=["p", "p"], completions=[FAITHFUL, "nothing here"])
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]

    def test_prompt_keyed_mapping(self):
        """Mapping lookup keyed by clip id forwarded via a clip_ids column."""
        lookup = {"clip_A": GT, "clip_B": GT}
        reward_func = make_sfs_reward_func(lookup)
        rewards = reward_func(
            prompts=["pA", "pB"],
            completions=[FAITHFUL, "no numbers"],
            clip_ids=["clip_A", "clip_B"],
        )
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]

    def test_chat_completions_in_batch(self):
        reward_func = make_sfs_reward_func()
        completions = [
            [{"role": "assistant", "content": FAITHFUL}],
            [{"role": "assistant", "content": "clean prose, no claims"}],
        ]
        rewards = reward_func(completions=completions, gt_features=[GT, GT])
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]

    def test_missing_gt_yields_finite_reward(self):
        """No GT resolvable → empty-GT fallback, reward is finite (penalties only)."""
        reward_func = make_sfs_reward_func()
        rewards = reward_func(completions=["The SNR is 16.10 dB."])
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.0, abs=1e-6)
