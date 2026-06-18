"""Tests for degeneration-aware, lower-variance checkpoint selection.

These are pure-function tests (no torch / model), so they run anywhere.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ckpt_selection import (  # noqa: E402
    rep_n,
    nonascii_frac,
    degeneration_stats,
    passes_degeneration_guard,
    should_save_best,
    seeded_val_indices,
)

CLEAN = (
    "The signal-to-noise ratio SNR is 18.14 dB. The SRMR is 4.11. "
    "The F0 mean is 121.79 Hz and the F0 standard deviation is 49.7 Hz."
)
# A longer, realistic templated quality description (several distinct
# "The <feature> is <value>." sentences). Templated prose is naturally a little
# repetitive, so this is the honest stress case for the per-clip rep-n threshold;
# test_clean_real_below_per_clip_thresh confirms rep_n(n=4) stays well under 0.5.
CLEAN_REAL = (
    "The duration of the speech sample is 4.275 s. The signal-to-noise ratio "
    "SNR is 30.55 dB. The harmonics-to-noise ratio HNR is 8.34 dB. The SRMR is "
    "4.11. The F0 mean is 121.79 Hz and the F0 standard deviation is 49.7 Hz. "
    "The jitter local is 1.82 percent. The shimmer local is 9.41 percent. The "
    "speaking rate is 3.2 syl per sec. The articulation rate is 4.1 syl per sec. "
    "The clip contains 3 pauses. The overlap ratio is 0.42."
)
TAGSPAM = "<sec_noise>. </sec>. </sec>. </sec>. </sec>. </sec>. </sec>. </sec>. </sec>."
REPLOOP = "the the the the the the the the the the the the the the"
# The exact v14 failure: a single clip stuck in a sentence-level loop.
V14_LOOP = ("The recording is about 12 minutes long. The speaker is a man. " * 6).strip()
CHINESE = "<sec_noise>编辑播报编辑播报编辑播报 The SNR is 1.04 dB 编辑播报编辑播报"


# ── rep_n ────────────────────────────────────────────────────────────────────
def test_rep_n_clean_is_low():
    assert rep_n(CLEAN, n=4) < 0.1


def test_rep_n_loop_is_high():
    assert rep_n(REPLOOP, n=4) > 0.5
    assert rep_n(TAGSPAM, n=2) > 0.5


def test_rep_n_short_text_is_zero():
    assert rep_n("two words", n=4) == 0.0   # fewer tokens than n → no n-grams


def test_clean_real_below_per_clip_thresh():
    # A realistic, longer templated description is naturally a little repetitive
    # but must stay safely below the 0.5 per-clip repetition threshold, so it is
    # never miscounted as a degenerate loop. (Measured ~0.013 at n=4.)
    assert rep_n(CLEAN_REAL, n=4) < 0.5
    assert rep_n(CLEAN_REAL, n=4) < 0.1   # in practice well under, not just under


def test_v14_loop_above_per_clip_thresh():
    # The looping clip from the v14 run must clear the per-clip threshold so it
    # is counted toward frac_clips_high_rep. (Measured ~0.83 at n=4.)
    assert rep_n(V14_LOOP, n=4) > 0.5


# ── nonascii_frac ────────────────────────────────────────────────────────────
def test_nonascii_clean_is_zero():
    assert nonascii_frac(CLEAN) == 0.0


def test_nonascii_chinese_is_high():
    assert nonascii_frac(CHINESE) > 0.1


# ── degeneration_stats ───────────────────────────────────────────────────────
def test_stats_flags_one_bad_clip_in_batch():
    stats = degeneration_stats([CLEAN, CLEAN, REPLOOP], n=4)
    assert stats["rep_n_max"] > 0.5          # the bad clip shows in the max (telemetry)
    assert stats["rep_n_mean"] < stats["rep_n_max"]


def test_stats_reports_frac_clips_high_rep():
    # 1 looping clip out of 4 → frac_clips_high_rep = 1/4 = 0.25. The clean clips
    # (CLEAN, CLEAN_REAL) are below the 0.5 per-clip threshold and do not count.
    stats = degeneration_stats([CLEAN, CLEAN_REAL, CLEAN, V14_LOOP], n=4)
    assert abs(stats["frac_clips_high_rep"] - 0.25) < 1e-9


def test_stats_frac_clips_high_rep_all_clean_is_zero():
    stats = degeneration_stats([CLEAN, CLEAN_REAL, CLEAN], n=4)
    assert stats["frac_clips_high_rep"] == 0.0


def test_stats_empty():
    s = degeneration_stats([], n=4)
    assert s["rep_n_mean"] == 0.0 and s["nonascii_frac"] == 0.0
    assert s["frac_clips_high_rep"] == 0.0


# ── passes_degeneration_guard ────────────────────────────────────────────────
def test_guard_passes_clean():
    ok, _ = passes_degeneration_guard(bleu=31.5, best_bleu=31.5, rep_n_max=0.02, nonascii_frac_val=0.0)
    assert ok


def test_guard_rejects_bleu_collapse():
    # v11-style: SFS looked fine but BLEU collapsed vs the clean running max.
    ok, reason = passes_degeneration_guard(bleu=8.0, best_bleu=31.5, rep_n_max=0.1, nonascii_frac_val=0.0)
    assert not ok and "bleu" in reason


def test_guard_rejects_repetition_when_many_clips_loop():
    # New semantics: the gate is the FRACTION of clips that loop, not the single
    # worst clip. A high frac_clips_high_rep (most of the batch looping) → reject.
    ok, reason = passes_degeneration_guard(
        bleu=20.0, best_bleu=20.0, rep_n_max=0.8, nonascii_frac_val=0.0,
        frac_clips_high_rep=0.6,
    )
    assert not ok and "frac_clips_high_rep" in reason


def test_guard_tolerates_a_few_looping_clips():
    # The exact v14 case: 3/32 looping clips → frac_clips_high_rep ~ 0.094, below
    # the 0.15 clip-fraction threshold. rep_n_max=0.62 is below the 0.95
    # catastrophic backstop. The checkpoint is otherwise clean → NOT withheld.
    ok, reason = passes_degeneration_guard(
        bleu=31.0, best_bleu=31.5, rep_n_max=0.62, nonascii_frac_val=0.0,
        frac_clips_high_rep=3 / 32,
    )
    assert ok, reason


def test_guard_rejects_catastrophic_single_clip_backstop():
    # A single clip that is essentially one token repeated (rep_n ~ 0.99) is so
    # broken it trips the high catastrophic backstop even alone.
    ok, reason = passes_degeneration_guard(
        bleu=20.0, best_bleu=20.0, rep_n_max=0.99, nonascii_frac_val=0.0,
        frac_clips_high_rep=1 / 32,   # only one clip, below the fraction gate
    )
    assert not ok and "rep_n_max" in reason


def test_guard_rejects_nonascii():
    ok, reason = passes_degeneration_guard(bleu=20.0, best_bleu=20.0, rep_n_max=0.0, nonascii_frac_val=0.3)
    assert not ok and "nonascii" in reason


def test_stats_report_frac_clips_nonascii():
    # 1 of 3 clips has a foreign char
    s = degeneration_stats([CLEAN, CLEAN, "The SNR is 8 dB 网络"], n=4)
    assert abs(s["frac_clips_nonascii"] - 1 / 3) < 1e-9


def test_guard_rejects_many_clips_with_few_foreign_chars():
    # The v12 epoch-2 case: foreign tokens in 34% of clips but <1% of all chars,
    # not repetitive. The char-fraction guard MISSES it; the clip-fraction guard
    # must catch it. (Simulate: low nonascii_frac, high frac_clips_nonascii.)
    ok, reason = passes_degeneration_guard(
        bleu=20.0, best_bleu=20.0, rep_n_max=0.1,
        nonascii_frac_val=0.01,          # under the 0.05 char threshold
        frac_clips_nonascii=0.34,        # over the 0.15 clip threshold
    )
    assert not ok and "frac_clips_nonascii" in reason


def test_save_rejects_epoch2_style_foreign_injection():
    # Build a 32-clip batch: 11 with a foreign token, 21 clean. Aggregate char
    # fraction is tiny; per-clip fraction is 11/32 = 0.34 -> must be withheld.
    batch = [CLEAN] * 21 + [CLEAN + " 网络"] * 11
    save, reason = should_save_best(0.30, 0.22, bleu=8.0, best_bleu=31.5, gen_texts=batch)
    assert not save and "degenerate" in reason


def test_guard_relative_floor_not_absolute():
    # A genuinely lower-BLEU-but-clean config must NOT be rejected just for low BLEU.
    ok, _ = passes_degeneration_guard(bleu=12.0, best_bleu=15.0, rep_n_max=0.0, nonascii_frac_val=0.0)
    assert ok   # 12 >= 0.6*15 = 9 → clean


def test_guard_first_epoch_no_bleu_ref_still_checks_repetition():
    # best_bleu None (epoch 1): BLEU check skipped, but repetition still caught
    # via the clip-fraction gate (most of the batch looping).
    ok, reason = passes_degeneration_guard(
        bleu=5.0, best_bleu=None, rep_n_max=0.9, nonascii_frac_val=0.0,
        frac_clips_high_rep=0.5,
    )
    assert not ok and "frac_clips_high_rep" in reason


# ── should_save_best (top-level decision) ────────────────────────────────────
def test_save_when_improved_and_clean():
    save, _ = should_save_best(0.55, 0.50, bleu=31.0, best_bleu=31.0, gen_texts=[CLEAN, CLEAN])
    assert save


def test_no_save_when_not_improved():
    save, reason = should_save_best(0.49, 0.50, bleu=31.0, best_bleu=31.0, gen_texts=[CLEAN])
    assert not save and "improvement" in reason


def test_no_save_when_improved_but_degenerate():
    # Higher SFS but tag-spam generations → must NOT be selected (the v11 trap).
    save, reason = should_save_best(0.55, 0.50, bleu=8.0, best_bleu=31.5,
                                    gen_texts=[TAGSPAM, TAGSPAM])
    assert not save and "degenerate" in reason


def test_save_v14_one_looping_clip_among_clean():
    # The exact v14 regression: a 32-clip val batch, mostly clean templated
    # descriptions with a single looping clip. Under the old rep_n_max gate this
    # was withheld EVERY epoch (max 0.62 > 0.5) and no best.pt was ever saved.
    # Under the fraction gate (1/32 ~ 0.031 < 0.15) it must be allowed.
    batch = [CLEAN_REAL] * 31 + [V14_LOOP]
    save, reason = should_save_best(0.30, 0.22, bleu=31.0, best_bleu=31.5,
                                    gen_texts=batch)
    assert save, reason


def test_save_v14_a_few_looping_clips_among_clean():
    # The literal v14 description: 3 looping clips out of 32 (~9%) → still below
    # the 0.15 clip-fraction threshold → NOT withheld.
    batch = [CLEAN_REAL] * 29 + [V14_LOOP] * 3
    save, reason = should_save_best(0.30, 0.22, bleu=31.0, best_bleu=31.5,
                                    gen_texts=batch)
    assert save, reason


def test_no_save_when_majority_of_clips_loop():
    # A batch where MOST clips are degenerate-looping must still be withheld.
    # 20/32 looping → frac_clips_high_rep = 0.625 > 0.15.
    batch = [CLEAN_REAL] * 12 + [V14_LOOP] * 20
    save, reason = should_save_best(0.55, 0.50, bleu=20.0, best_bleu=20.0,
                                    gen_texts=batch)
    assert not save and "degenerate" in reason


# ── seeded_val_indices ───────────────────────────────────────────────────────
def test_seeded_indices_stable_and_valid():
    a = seeded_val_indices(3000, 128, seed=1234)
    b = seeded_val_indices(3000, 128, seed=1234)
    assert a == b                                  # stable across calls (epochs)
    assert len(a) == 128 and len(set(a)) == 128    # no dupes
    assert all(0 <= i < 3000 for i in a)
    assert a == sorted(a)


def test_seeded_indices_not_prefix():
    # The whole point: not the biased first-N convenience slice.
    a = seeded_val_indices(3000, 32, seed=1234)
    assert a != list(range(32))


def test_seeded_indices_clamps_to_total():
    a = seeded_val_indices(10, 128, seed=1)
    assert len(a) == 10 and sorted(a) == list(range(10))


def test_seeded_indices_stratified_covers_all_strata():
    # 100 items, 4 overlap bins; a stratified sample must hit every bin.
    strata = [i % 4 for i in range(100)]
    idx = seeded_val_indices(100, 20, seed=7, strata=strata)
    assert len(idx) == 20
    hit_bins = {strata[i] for i in idx}
    assert hit_bins == {0, 1, 2, 3}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
