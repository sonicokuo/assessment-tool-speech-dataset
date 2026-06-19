# Data and Tooling

The data + tooling state for AQUA-NL. The end-to-end pipeline (audio -> features ->
descriptions -> processed .pt -> train -> inference -> SFS) is in the repo's
`.claude/CLAUDE.md`; this doc records the CURRENT data state and the redirect tooling.

---

## Feature extraction

- **`src/feature_extractor.py`** -- Praat-based SP feature extraction with Pyannote
  overlap (27-column CSV). Kept for cross-domain datasets without clean stems; NOT the
  default Libri2Mix path.
- **`src/feature_extractor_mix.py`** -- Libri2Mix-specific extractor, default
  `--overlap min_max_vad` runs Silero VAD on the s1/s2 clean stems for oracle overlap
  labels. The default pipeline extractor. Now also emits additive `f0_*_clean` columns.
- Audio encoders: **WavLM-Large** (1024-d frame features, the prefix backbone) and
  **BEATs** (768-d time-frequency patch features, consumed by the grounding heads).
  BEATs patches are cached in `processed_pyannote/*.pt` (the section / grounding path
  needs them; the augmented clips need a one-time BEATs precompute -- see corpora).
- **Clean-frame F0** (`src/f0_clean.py`): restricts pitch estimation to voiced,
  non-overlap frames. Fixes the ill-posed mixture-F0 GT (see
  `../findings/key-findings.md` (d)). The mixture-vs-clean F0 disagreement rises with
  overlap (11.8 / 21.1 / 27.6 Hz across overlap bins), all 2-5x the +-5 Hz tolerance.

## The SFS metric (`src/sfs.py`)

- `ClaimParser` / `HybridClaimParser` (regex + tagged-span first) parse numerical
  claims out of generated text; `SFSScorer` scores them against SP ground truth with
  per-feature tolerances -> precision / recall / F1.
- Phrasing-sensitive by design: accepts "SNR of 30.55 dB", "= 30.55 dB", "is 30.55 dB"
  but REJECTS colon form "SNR: 30.55 dB" -- the deterministic builder is aligned to
  this.
- **13 scorable features** at 98.6-100% parse coverage: snr, hnr, f0_mean, f0_sd,
  jitter, shimmer, srmr, overlap_ratio, speaking_rate, articulation_rate, pause_count,
  pause_rate (+ overlap spans via IoU >= 0.8). duration_sec and sample_rate were
  REMOVED from TOLERANCES (duration is measured at inference, not predicted).
- The 8-feature canonical scalar list for the regression heads is `src/feature_set.py`
  (`SUPERVISED_FEATURES`): snr, srmr, f0_mean, f0_sd, speaking_rate, pause_count,
  pause_rate, overlap_ratio (with `FEATURE_SCALES` for loss normalization). This is the
  set the decoupled grounding head and the aux/readout heads regress.
- Recall caveat: `recall = mentioned / |gt_features|`. Filter GT to `TOLERANCES` keys
  before scoring or a non-scorable GT key inflates the denominator and caps recall.

## Descriptions (deterministic, NO LLM)

- **`scripts/build_descriptions_deterministic.py`** is the current target builder.
  Numbers are stamped EXACTLY from the feature CSV -- no LLM verbalization (the old
  gemma4 / Ollama path is legacy). Flags: `--untagged` (strip section tags),
  `--no-overlap-segments`, `--no-duration`, `--clean_f0 {train,dev,test}`, `--tagged`
  (keep `<sec_*>` structural tags for the section path).
- **`scripts/build_clean_train_descriptions.py`** builds no-hedge descriptions for the
  clean single-speaker clips; `--tagged` flag for the section path.
- The deterministic builder replaces the LLM verbalizer for the moat reason: exact
  measured numbers are the contribution; never LLM-verbalize targets.

## Augmentation (controlled synthesis -> exact GT)

- **Noise: `scripts/augment_noise.py`** -- mixes a clean s1 stem with `wham_noise` at a
  KNOWN target SNR. SNR GT is exact by construction
  (`alpha = sqrt(Ps / (Pn * 10^(T/10)))`, verified to 1e-6). Additive noise leaves
  F0 / rate / pauses / SRMR / overlap from the clean source unchanged, so the augmented
  feature row = clean row with snr_db overwritten = exact GT on every feature. 19/19
  tests. RUN: 13900 clean s1 clips -> random SNR in {0,5,10,15,20,30} dB ->
  `processed_aug_noise` + `features_aug_noise.csv` + `descriptions_aug_noise.json`.
  Fixes the previously narrow SNR distribution.
- **Reverb: `scripts/augment_reverb.py`** -- adds reverberation at a known RT60 via
  `pyroomacoustics` (with a numpy exp-decay RIR synth fallback). 11/11 tests, exact
  `rt60_s` column. BUILT but NOT yet run as data-gen. REQUIRED FIX before reverb
  data-gen: stamp the MEASURED rt60 (measure on the actual generated RIR) not the
  nominal target, because `pra.inverse_sabine` is approximate. SRMR is NOT a simple
  function of RT60, so reverb clips must be RE-EXTRACTED through the feature extractor
  to get real SRMR. SRMR is the next-worst feature after f0_mean -> reverb is its lever.

## Corpora on disk

- **Libri2Mix** train-clean-100 / dev-clean / test-clean -- speaker-disjoint by
  filesystem layout (so `scripts/split_data.py` is NOT used). 78% base overlap rate.
- **clean s1 stems** -- single-speaker, overlap=0. NOTE the R3 collision bug: clean s1
  filenames are IDENTICAL to the Libri2Mix mixture names, so they must be suffixed
  (`_s1clean`) consistently across (a) CSV filename, (b) description key, (c) .pt
  on-disk name, AND (d) the .pt internal `filename` field, or they overwrite the
  mixture targets. Fixed by `fix_r3_collision.py`.
- **`processed_aug`** -- the assembled UNTAGGED augmented corpus = **41700 train .pt**:
  13900 Libri2Mix mixtures (clean-F0) + 13900 clean-s1 (`_s1clean`) + 13900 noise-aug
  (`_augN`). val/test linked from `processed_pyannote`. `descriptions_aug.json` = 47700;
  `features_aug_train.csv` = 41700; 0 collisions. Ready to train (untagged headline).
- **`descriptions_aug_tagged.json`** = 47700 TAGGED entries for the section path (built,
  verified). The section path on augmented data additionally needs BEATs patches for
  the augmented clips (one-time GPU precompute or online encoding).
- **clean-F0 scoring CSVs**: `features_pyannote/{dev,test}_cleanf0.csv` +
  `clean_f0_{dev,test}.json` -- the CORRECT answer key. RESCORE every clean-F0-trained
  model against these before reading SFS (default dev/test CSVs still carry mixture F0).
- Also on disk: `wham_noise` (noise synth), AMI (`ami_ihm` / `ami_sdm`, test-only
  cross-domain, never trained on), LibriSpeech (clean source; `processed_clean_control`
  + `_v2` with overlap=0 already built). NOT on disk: MUSAN / DNS / DEMAND / RIR-corpora
  / LibriCSS / CHiME.

## Redirect branch + new modules

Branch **`redirect-decoupled-rl`** (commit 5173672); main is untouched. TRL 1.6.0
installed; 40 redirect tests pass in the PSC env.

- **`src/decoupled_grounding.py`** -- `DecoupledGroundingHead`: learned per-feature
  query embeddings (nn.Parameter, NOT vocab tokens) cross-attend BEATs T-F patches ->
  per-attribute 2D map + `z = A @ V.detach()` + shallow readout regressed to the
  scalar. Token-free, training-only, pure torch (CPU-testable).
  `query_orthogonality_penalty` guards map-collapse. 19 tests
  (`tests/test_decoupled_grounding.py`) incl. a grounding-gradient proof.
- **`src/sfs_reward.py`** -- the RLVR reward: `sfs_reward = f1_weight * SFS_F1
  - rep_penalty * rep_n - nonascii_penalty * nonascii_frac`. F1 not recall (recall is
  number-spam-hackable). `make_sfs_reward_func` is the TRL GRPO batch wrapper. Imports
  only `sfs.py` + `ckpt_selection.py` + stdlib (no trl) -> CPU-testable. 21 tests
  (`tests/test_sfs_reward.py`).
- **`src/grpo_train.py`** -- GRPO scaffold: SFT cold-start (from v9) -> RAFT de-risk ->
  GRPO-on-SFS. Reward is done; model/adapter loading mirrors train.py/inference.py;
  the audio-prefix splice into TRL's generation loop is the remaining `[TODO-AUDIO]`
  (subclass `GRPOTrainer`, concat adapter prefix embeddings into `inputs_embeds` for
  both policy and reference).

## Other supporting modules (for reference)

- `src/section_readout.py` -- the OLD grounding head (regress per-section scalar from
  attention z, detach V). Wired into train.py but `lambda_readout=0` in shipped configs.
  Superseded as a contribution by the decoupled head.
- `src/ckpt_selection.py` -- degeneration-gated checkpoint selection (`should_save_best`,
  `rep_n`, `nonascii_frac`, `seeded_val_indices`). Same signals reused in the RL reward.
- `src/grounding_metrics.py` -- time / freq concentration ratios for the maps.
- `src/hedging_calibration.py` -- risk-coverage + reliability-by-overlap-bin tools.
- `src/token_init.py`, `src/peft_config.py` -- R10 semantic init (negative result) and
  DoRA/PiSSA plumbing.
