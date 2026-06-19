# Decisions and Trajectory

The decision trajectory for AQUA-NL, in the order it actually happened. The headline
story changed twice; this records why.

---

## Step 0 — established headline (untagged LoRA Qwen3-8B)

- LoRA r=16 on Qwen3-8B is the headline, NOT full FT. Full FT on a 1.7B model
  catastrophically forgot the LM prior (off-topic ramble, mode collapse). LoRA keeps
  the frozen prior as an anchor. Checkpoint `v9_lora_8b_dur`.
- Target format: `build_descriptions_deterministic.py --untagged --no-overlap-segments
  --no-duration`. Duration is measured from the wav at inference, not predicted (it was
  mode-collapsing). Overlap segments removed (pooled WavLM can't emit timestamps, so
  the model hallucinated a tile-the-clip pattern the IoU SFS rejects). SFS no longer
  scores duration_sec / sample_rate.

## Step 1 — tried to FUSE 2D-map grounding + augmentation (3 runs, all unstable)

The user's stated goal was one integrated headline: the novel 2D-map (tagged section
path) trained ON the augmented corpus = metric + hedging + 2D-map + augmentation in a
single model. Three fusion runs (see `../experiments/training-runs.md`):

- v15: R10 semantic-init ON + augmented + lambda=0.05 -> 44% degen.
- v15v2: R10 OFF + augmented + lambda=0.05 -> 16% / 16% / 41% spike.
- v15v3: R10 OFF + augmented + lambda=0.02 -> 41% / 31%, high variance.

All swing 16-44% degeneration vs v13 (2D-map, no augmentation) stable at 9%. Both
isolated knobs (R10 off, lambda down) help but neither tames the augmented tagged
section path. **Conclusion: the augmented tagged section path is unstable.** Stop
burning GPU on fusing.

## Step 2 — interim TWO-MODEL fallback (superseded)

Initial recommendation after the failed fusion was a two-model paper, both novel,
UNFUSED:

- `v13_section_warmup` — the 2D-map grounding figure (stable 9% degen, non-augmented).
- `v14_aug` — the augmentation / metric leg (untagged, SFS peaked 0.515, f0_mean
  11% -> 23% on clean-F0).

This is an honest but MODEST story. It was the plan for ~a day, then superseded by the
redirect once the research established the instability was architectural.

## Step 3 — the REDIRECT (current plan, branch `redirect-decoupled-rl`)

Three independent literature-review agents confirmed the section-path degeneration is
ARCHITECTURAL: special-token EMISSION is intrinsically fragile, and the strongest audio
precedent (arXiv:2602.10230, Frame-Level Internal Tool Use) fixed our exact failure by
DECOUPLING grounding from token emission. So instead of patching the fragile section
path, replace the mechanism. Two new robust mechanisms, both built + unit-tested on the
redirect branch (commit 5173672):

### (i) DECOUPLED token-free 2D grounding (`src/decoupled_grounding.py`)

- `DecoupledGroundingHead`: a fixed set of LEARNED per-feature query embeddings
  (`nn.Parameter`, NOT vocab tokens) cross-attend the audio's BEATs time-frequency
  patches. Each query yields a per-feature 2D attention map `A`, a pooled vector
  `z = A @ V.detach()`, and a shallow readout `y` regressed to that feature's scalar.
- The LM stays UNTAGGED and never emits a special token (so it inherits the untagged
  path's 0% degeneration). Grounding lives entirely in the head; generation lives
  entirely in the LM; they share only the audio features, never the decoding channel.
- The `V.detach()` pool is the grounding property (copied from `section_readout.py`):
  the regression gradient cannot rewrite V to smuggle the feature in, so the only
  descent direction is to reshape the attention map onto real evidence. An optional
  `query_orthogonality_penalty` guards the map-collapse failure mode.
- 19 tests including a grounding-gradient proof.

### (ii) RLVR-on-SFS (`src/sfs_reward.py` + `src/grpo_train.py`)

- The SFS metric is a deterministic, human-validated (rho=0.69) verifier -> an ideal
  RL-with-verifiable-rewards (RLVR / GRPO) target.
- Reward = SFS-**F1** (NOT recall, which number-spamming hacks) minus a
  repetition / non-ASCII degeneration penalty, with a KL leash to the SFT reference.
- Plan: SFT cold-start (from v9) -> RAFT best-of-n de-risk -> GRPO-on-SFS.
- `grpo_train.py` is a structured scaffold; the reward (21 tests) is fully done. The
  hard remaining piece is splicing the audio prefix into TRL's generation loop
  (marked `[TODO-AUDIO]`). TRL 1.6.0 installed.

### Why the redirect makes the paper STRONGER

It is not the modest two-model story. The paper becomes: SFS metric + overlap-aware
hedging + DECOUPLED grounded per-attribute 2D maps + RLVR-on-SFS + augmentation. This
is a strong main-conference target (research reinforces ICLR 2027 over AAAI-27 -- the
mandatory faithfulness experiments and the F0-GT retrain don't fit AAAI's ~6-week,
no-rebuttal window).

---

## Supporting decisions (carried forward)

- **F0: re-ground, do NOT drop.** F0 is the hedging contribution. Fix = clean-frame F0
  GT + clean-F0 scoring (see findings doc). This invalidates v9's original SFS / rho /
  grounding numbers -> they must be recomputed. Biggest schedule risk.
- **Checkpoint selection: gated composite, not argmax-SFS.** SFS is degeneration-blind;
  gate on a relative-BLEU floor + rep-n + non-ASCII guard, on the FRACTION of clips.
- **Framing: claim "intrinsically grounded generation", NOT "self-explainable".** The
  LM also sees the full audio prefix (leaky bottleneck), so it fails the SENN/CBM bar.
- **Descriptions stay deterministic with EXACT numbers** (the moat). Never
  LLM-verbalize targets. Optional deterministic rule-based qualitative glosses are OK.
- **No semantic metric added** (BERTScore already reported; FENSE / CLAIR-A / LLM-judge
  would collapse toward the QualiSpeech perceptual lane we position away from).
- **Drop the OLD "intrinsically grounded generation via section readout" as a live
  contribution** -- `lambda_readout=0` in all shipped configs, so v9 got zero grounding
  gradient and the only section run (v11) was degenerate. Revived in the NEW form via
  the decoupled head, which is grounded by construction and token-free.
- **No Claude watermark** in any commit (no Co-Authored-By, no "Generated with").
