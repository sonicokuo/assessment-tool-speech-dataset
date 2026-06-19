# AQUA-NL Research Knowledge Base

A durable record of AQUA-NL's results, findings, and decisions so the project state is
never lost. Distilled from the project's auto-memory, `.claude/CLAUDE.md`, and the
source on the `redirect-decoupled-rl` branch. Last consolidated: 2026-06-19.

## Project in one paragraph

**AQUA-NL** (working codename only; final paper title undecided) is a speech
recording-quality description system: it takes a speech waveform and generates a
natural-language description that states MEASURED signal-processing numbers (SNR, F0,
speaking rate, overlap ratio, ...), hedging when a feature is unreliable under
overlapping speech. Two core contributions: (1) the **Signal Faithfulness Score (SFS)**,
a deterministic regex-based metric that scores numerical claims against SP ground truth
(validated rho=0.69 vs human), and (2) **overlap-aware grounded description**, where
per-attribute 2D time-frequency attention maps tie each claim to the audio evidence and
the model hedges where the signal estimate is unreliable. The headline model is LoRA
r=16 on Qwen3-8B (full fine-tuning forgot the LM prior). This is a CMU 11-785 project
targeting a main NLP/ML conference (research currently favors ICLR 2027 over AAAI-27).

## Documents

| Doc | What's in it |
|-----|--------------|
| [experiments/training-runs.md](experiments/training-runs.md) | Full run history v9 -> v15v3: each checkpoint, its recipe delta, degeneration %, and SFS. The fusion-instability and v13-warmup tables. |
| [findings/key-findings.md](findings/key-findings.md) | The load-bearing findings: R10 token-init is the degeneration culprit; lambda sensitivity; degeneration is architectural; ill-posed F0 GT and its fix; the checkpoint guard; rho=0.69 + 95% grounding. |
| [decisions/decisions.md](decisions/decisions.md) | The decision trajectory: fuse (failed) -> two-model fallback -> the research-backed redirect to decoupled grounding + RLVR-on-SFS. |
| [literature/research-synthesis.md](literature/research-synthesis.md) | The 3 lit reviews behind the redirect (token-free 2D grounding / special-token weakness / RLVR), with arXiv ids. |
| [pipeline/data-and-tooling.md](pipeline/data-and-tooling.md) | Feature extraction, clean-frame F0, SFS, deterministic descriptions, noise/reverb augmentation, the corpora on disk, and the redirect modules. |

## Current status (2026-06-19)

- **Headline (untagged, clean):** `v9_lora_8b_dur` -- LoRA Qwen3-8B, 0% degeneration,
  test SFS ~0.52 (pre clean-F0 rescore), BLEU 31.5.
- **Grounding figure:** `v13_section_warmup` -- grounded 2D maps at stable 9%
  degeneration (the old section path).
- **Augmentation leg:** `v14_aug` -- untagged + noise augmentation, 0% degeneration,
  SFS peaked 0.515, f0_mean recovered 11% -> 23% on clean-F0.
- **Decided:** the fused 2D-map + augmentation path is unstable (16-44% degen); the
  degeneration is architectural (special-token emission is fragile). REDIRECT to a
  token-free decoupled grounding head + RLVR-on-SFS. Both mechanisms built + tested on
  branch `redirect-decoupled-rl`.

## Next steps

1. **Integrate `DecoupledGroundingHead` into `train.py`** as a new mode -- adapter ->
   BEATs patches feed the head as a PARALLEL branch; LM trains on the UNTAGGED
   `descriptions_aug.json`; loss = `untagged_lm_ce + lambda_ground * grounding_loss`.
   Add a config flag + config. Run on `processed_aug` (41700 clips, has BEATs).
2. **Rescore** every clean-F0-trained model against the clean-F0 CSVs before reading SFS.
3. **GRPO-on-SFS** from a v9 / untagged SFT cold-start with the SFS-F1 reward (finish the
   `[TODO-AUDIO]` audio-prefix splice in `grpo_train.py`; do SFT -> RAFT -> GRPO).
4. **Pending experiments:** equal clean non-overlap training data (fix the 80% hedge
   false-positive, raise the grounding ceiling) and AMI cross-domain test.

## Conventions

- Do NOT touch cluster / SSH / GPU from this knowledge base; these are docs only.
- Paper prose: no em-dashes, no colon-as-pause, no e.g./i.e./etc.
- No Claude watermark in commits.
- Paper draft lives separately at `/Users/sheng-tselin/papers/aquaNl/` (not this repo).
