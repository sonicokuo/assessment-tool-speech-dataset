# Key Findings

The load-bearing experimental findings for AQUA-NL. Each is something a returning
reader (or a reviewer) needs to know. Most are documented negative results.

---

## (a) R10 semantic token-init is the CULPRIT for section-path degeneration

The section / tagged path adds 19 structural vocab tokens (`<sec_*>` / `<f_*>` open,
close, and range markers). "R10" was a proposed warm-start that initialized those new
token embeddings from their catalog display names (e.g. the `<sec_pitch>` embedding
seeded from the word "pitch").

**Result: this made degeneration WORSE.** Seeding structural tokens from real words
makes them "word-like", so the LM treats them as ordinary vocabulary and generates
PROSE around them instead of using them as content-free structural markers. Disabling
R10 (mean-init, `semantic_tag_init=false`, config flag added on commit 8ae49f8)
dropped epoch-1 degeneration from **44% -> 16%** (v15 -> v15v2).

Consequence for the recipe: the section path must use `semantic_tag_init=FALSE`.
R10 is a documented NEGATIVE RESULT for the ablation. This corroborates the
literature finding that structural tokens must be content-free and word-DISTINCT
(ChatML, ViT-registers, Llama-3 reserved tokens) -- see research synthesis doc.

(Separate note: R10 also had a real plumbing bug initially, fixed on commit 8626e0c:
`_register_feature_tags` passed the PADDED embedding row count as `old_vocab_size`, so
the new tag IDs fell below it and zero tags were initialized. After the fix 15 of 19
tags warm-start. But warm-starting them is itself the wrong move per the result above.)

---

## (b) lambda_readout=0.05 is too aggressive on AUGMENTED data

On non-augmented data, lambda_readout=0.05 with warmup is fine (v13 settles at 9%
degen). On the augmented corpus the same lambda spikes degeneration at full strength
(v15v2: 16% / 16% / 41% spike). Dropping to ~0.02 is gentler (v15v3) but the path is
**still unstable** (41% / 31%, high variance). Neither lambda value tames the
augmented section path. lambda=0.5 (v12) is catastrophic everywhere (31% degen).

Root cause of why grounding fights generation: in DYNAMIC query mode the attention
query IS the LM hidden state at the `<sec_>` positions, so the readout grounding loss
flows gradient back into the LM hidden state (to make a grounding-friendly query),
COMPETING with prose generation. At lambda=0.5 the grounding pull corrupts generation
-> foreign-token / derailment degeneration. The decoupled head (redirect) removes this
by making the queries free PARAMETERS, not LM hidden states.

---

## (c) The degeneration is ARCHITECTURAL: special-token EMISSION is fragile

The cleanest single fact across all runs: **every untagged run is 0% degeneration**
(v7/v8/v9, all adapter variants, 0 tag-spam, ~0.7% non-ASCII). **Every tagged /
section run degenerates** (3-44%). The variable that flips degeneration on is whether
the LM must EMIT under-trained new vocab tokens.

Under-trained new tokens collapse on hard / out-of-distribution inputs: the model
emits the foreign token inside otherwise-clean prose, and once emitted the generation
derails. This is not a tuning miss; it is the same failure documented in the
literature ("Magikarp" glitch / under-trained tokens; the audio precedent
arXiv:2602.10230 where timestamp-token models "collapse completely" OOD). The robust
fix is to NOT route grounding through emitted tokens at all -> the decoupled,
token-free grounding head (see decisions + literature docs).

---

## (d) F0 ground-truth was ILL-POSED (computed on the 2-speaker mixture)

The F0 features (f0_mean, f0_sd) were computed by running pitch tracking on the
2-SPEAKER MIXTURE waveform. Under Libri2Mix's 78% base overlap this is ill-posed: two
overlapping speakers produce a pitch estimate that matches neither. Verified on the
full 3000-clip test set, the mixture-F0 vs clean-frame-F0 disagreement RISES
monotonically with overlap:

| overlap bin | mean Hz error | median Hz error | % F0 undefined |
|-------------|---------------|-----------------|-----------------|
| [0.25, 0.5) | 11.8 | 7.5 | 0% |
| [0.5, 0.75) | 21.1 | 15.0 | 0.1% |
| [0.75, 1.0] | 27.6 | 19.5 | 5.4% |

All bins are 2-5x the +-5 Hz SFS tolerance. The [0, 0.25) bin is EMPTY (Libri2Mix
lacks low-overlap clips, motivating clean-LibriSpeech ingest). This proves two things:
(1) the F0 metric reference was broken (low f0_mean SFS was a measurement artifact,
not a model failure), and (2) overlap is a valid reliability signal at the signal
level, which JUSTIFIES the hedging contribution.

**Fix** (do NOT drop F0 -- it is the hedging contribution): compute F0 on voiced,
non-overlap frames only (`src/f0_clean.py`, `src/feature_extractor_mix.py` additive
`f0_*_clean` columns), retrain on clean-F0 targets, AND score against clean-F0 CSVs
(`features_pyannote/{dev,test}_cleanf0.csv` + `clean_f0_{dev,test}.json`).
**f0_mean recovered 11% -> 23%** after this fix.

> METHOD GOTCHA: re-scoring a mixture-F0-TRAINED model against clean-F0 GT is
> CONFOUNDED (~43 Hz flat error = train/test GT mismatch, not overlap reliability).
> Model-level hedging calibration requires RE-TRAINING on clean-F0 targets, not just
> re-scoring. Always rescore EVERY trained model against clean-F0 GT before reading
> SFS, because the default dev/test feature CSVs still carry mixture F0.

---

## (e) The degeneration-aware checkpoint guard

SFS is degeneration-BLIND: it only regex-parses NUMBERS, so a tag-spamming or
repetition-looping checkpoint can still contain parseable numbers and score non-zero
(even high) SFS while being unreadable garbage. A naive argmax-SFS selector saved
exactly such a checkpoint (the v11 trap).

The fix (`src/ckpt_selection.py`, `should_save_best`): gate the SFS argmax on a
degeneration guard. Two design points that matter:

- Gate on a **RELATIVE BLEU floor** (~30-40% below the running BLEU max) plus
  repetition-n and non-ASCII fraction guards -- not on absolute thresholds.
- Gate on the **FRACTION of clips** that loop / inject foreign tokens, NOT on the
  single worst clip (commit b279ba4). A single bad clip on a 32-clip val slice is
  noise; a high fraction is real collapse.
- Also replaced the biased fixed-first-32 val slice with a seeded random subset
  (`seeded_val_indices`).

Proven in production: the guard withheld v12's degenerate best.pt
("[select] withheld best.pt despite val_sfs_f1=0.2591: degenerate (rep_n_max 0.949)")
and withheld v13's 22% epoch-2 despite higher SFS, saving a clean epoch instead. The
SAME two degeneration signals (rep_n, nonascii_frac) are reused as the RL penalty in
`src/sfs_reward.py`.

---

## (f) Validated cross-cutting metrics

- **SFS vs human: Spearman rho = 0.69** (Pearson 0.70, p < 1e-4, n=50, single rater;
  terciles 1.94 / 2.56 / 3.22). This is the key card -- SFS catches numerical errors
  that BLEU / BERTScore miss. SFS is non-redundant with BERTScore/FENSE.
- **Overlap attention grounded on 95% of clips** (Wilcoxon p < 1e-6, mean
  concentration ratio 1.10, capped ~1.27 by Libri2Mix's 78% base overlap). Modest but
  significant; the cap motivates adding clean non-overlap training data.
