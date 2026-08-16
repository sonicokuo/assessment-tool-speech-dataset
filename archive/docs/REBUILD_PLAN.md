<!-- AUTO-BANNER SUPERSEDED 2026-08-03 -->
> # ⚠️ SUPERSEDED — historical record, not a plan of action
>
> Selector bug marked FIXED via clip_rep_thresh -- INCOMPLETE; best.pt still froze via the rep_n_max>0.95 backstop. Real fix was F4. The M3b-vs-B1 comparison this plan is built around was RETIRED by the 2026-07-16 salvage.
>
> Authoritative current state: `.claude/research/STATE_2026-08-03.md`.
> Original text preserved unchanged below.

---

# AQUA-NL Rebuild + Paper Plan (2026-07-08)

LOCAL-ONLY (under .claude/research). Self-contained plan for later execution. Target venue ICLR 2027 (~Sep 2026).

---

## TL;DR
**Partial rebuild via DATA AUGMENTATION (not new datasets).** RIR-reverberate + independent-SNR-noise the
existing Libri2Mix clean stems to (1) fix the feature confounds, (2) shift training toward *general* far-field
realism so the AMI OOD gap narrows, (3) enable voice-quality features. Retrain B1+M3b with the fixed selector +
unlikelihood loss. **AMI + one 2nd real corpus stay HELD-OUT OOD test only** (never train on them; never tune
augmentation to them). Then a disclose-and-defend paper led by grounding maps + abstention.

Key expectation to set: "closing the OOD gap" happens mostly by the **inflated in-domain number falling to its
honest value and meeting AMI in the middle** (convergence), NOT by AMI leaping up. Convergence is the stronger,
more trustworthy result.

---

## WHY (verified findings this session — all measured, not assumed)
1. **Headline +0.077 is fragile.** in-domain M3b 0.5466 vs B1 0.4697 (seed 73, best.pt, n3000) BUT seed-42/last.pt = -0.002.
   Inflated by a checkpoint-SELECTION bug (degeneration guard `clip_rep_thresh=0.15` froze B1's best.pt at ep0).
   FIXED: `src/ckpt_selection.py` clip_rep_thresh 0.15->0.50 (backup train.py.bak / ckpt_selection.py.bak_20260708).
2. **Metric confound (the fundamental issue).** The 5 "robust" features collapse to ~2 independent axes:
   - srmr<->f0 = **0.80** in the GT because Libri2Mix mix_clean is ANECHOIC (SRMR, a reverberation metric [Falk/Zheng/Chan
     2010], has no reverb to measure -> defaults to a pitch proxy). partial-corr(emit_srmr, real_srmr | f0) = +0.17 in-dom,
     NEGATIVE OOD; on AMI emit_srmr corr +0.43 with f0 but +0.07 with real srmr.
   - snr<->pause_count/pause_rate/speaking_rate = +0.57/+0.66/-0.58 (estimate_snr is a frame-ENERGY ratio coupled to
     speech/pause structure). speaking_rate corr +0.49 with clip DURATION (duration proxy; input is 6.25 tok/s, too coarse).
   - Only SNR and f0-when-asserted are clean, independent features.
3. **Fixes PROVEN on subsets** (this is the license to rebuild):
   - RIR reverb: srmr<->f0 0.80 -> **0.30** on 160 clips (RT60 0.53s, matches real AMI 0.56).
   - Independent uniform SNR: snr<->temporal 0.57-0.66 -> **~0** (+0.004/+0.008/-0.011).
   - Combined pipeline on 800 clips: srmr<->f0 0.80->0.303, snr<->pauses 0.57->-0.03, snr<->rate 0.66->-0.06,
     snr<->speaking 0.58->+0.05. ALL confounds broken. -> features_corrected_subset.csv on PSC.
   - Unlikelihood loss: implemented in `src/train.py` (`unlikelihood_token_loss` + `lambda_unlikelihood` flag), unit-
     and end-to-end tested (4 real training steps ran).
4. **AMI cross-domain (already the honest OOD result).** With clean-headset (IHM) GT the model TRANSFERS on intrinsic
   feats: M3b intrinsic-only mean SRCC +0.415 / B1 +0.395; f0-when-asserted +0.572/+0.667 on real audio. (Old SDM-Praat
   GT gave spurious ~0 because Praat measured reverb not speech.) Scripts: scripts/prepare_ami_ihm.py, score_ami_ihm.py.
   AMI IS the decorrelated experiment the rebuild manufactures -> it is already strong evidence the features are real.
5. **EOS is CORRECT** (appended + loss-supervised, tested) -> NOT the degeneration cause. Degeneration = MLE objective
   on templated targets; baseline B1 repeats (frac_high_rep 0.16->0.28), grounding aux loss prevents it (2 seeds).
6. **Grounding maps (novelty, partial but real):** overlap-attention deletion soft-IoU p=0.0008 (flagship); f0 localized
   p~0.03 (borderline, one-sided); SNR grounding NULL at the LM-emitted-number level = CORRECT (SNR is global) and is
   presented as evidence the maps are honest. The "SNR win-rate 1.0" was RETRACTED (map-internal, wrong model).
7. **Abstention:** f0 assert 0.995 clean vs hedge 0.66->0.96 as overlap rises (positive control). Largely confound-free.
8. **Metric name:** SFS (band precision/recall/F1) is RETIRED. Current metric = band-free per-feature Spearman SRCC
   (+ nMAE/coverage). The rho=0.69 LLM-judge validation was for the RETIRED SFS-F1, so the SRCC metric is NOT yet
   validated against a judge. NOTE: it is an OBJECTIVE metric (claim vs DSP truth) -> a HUMAN study is NICE-TO-HAVE, not
   necessary (contrast ALLD's perceptual NISQA MOS which needs humans). Do not over-invest in a human pilot.
9. **External baseline:** Qwen2-Audio-7B zero-shot = constant output (SRCC undefined) -> off-the-shelf can't do this;
   the adapter is necessary. scripts/run_external_baseline.py.
10. **Nearest prior = Sci-Phi (arXiv 2510.05542, Columbia+Microsoft).** SPATIAL-audio descriptor; emits numeric acoustic
    params incl. reverberation; trained on 4000h synthetic Ambisonics WITH room characteristics; generalizes to real RIRs.
    Partial scoop on "numbers in audio-LLM text" -> NARROW our claim to speech-QUALITY prose + claim-level faithfulness +
    causal grounding + abstention (none of which Sci-Phi does, per abstract). Sci-Phi VALIDATES that reverb estimation
    works WHEN trained on reverberant data = confirms our diagnosis. Cite it prominently in related work.

---

## THE APPROACH (decision)
- **Data augmentation, NOT new training datasets.** Reverberate + noise existing Libri2Mix clean stems.
- Augment to GENERAL far-field realism: diverse REAL RIRs (OpenSLR SLR28 / RIRS_NOISES) mixed with simulated
  (pyroomacoustics, RT60 0.3-0.8s via inverse_sabine); noise (MUSAN/WHAM) at INDEPENDENT uniform SNR 0-40 dB (DNS recipe).
- NEVER tune augmentation to AMI. AMI + 1 second real corpus (CHiME / ICSI / VoxConverse) = HELD-OUT OOD test only.
- Real RIRs (not just simulated) specifically to blunt the "domain-matched data augmentation" reviewer criticism.

---

## FEATURE SET (post-audit design + story)
Currently emitted+scored (~8): snr, srmr, f0_mean, f0_sd, speaking_rate, pause_count, pause_rate, overlap_ratio.
jitter/shimmer/hnr/articulation_rate = 0% emitted.

**Pruning/restoration:**
- DROP as proxies (pre-rebuild): srmr (f0-proxy), speaking_rate (duration-proxy), f0_sd (weak 0.09 + srmr-entangled).
- pause_count ~= pause_rate (corr 0.84) -> one "pause" axis (report one as headline, other in appendix).
- RESTORE after rebuild: srmr becomes REAL (reverb) -> KEEP. speaking_rate MAY return if conv 8x->4x lets the model
  count syllables (TEST it; drop honestly if not).
- ADD (gated): jitter, shimmer, hnr = voice-quality, ill-posed-under-overlap class. Strengthens abstention from an f0
  anecdote into a per-feature policy. GATE: only add if the clean positive control shows the model can predict them on
  CLEAN audio (they are hard to estimate; otherwise they are dead features).
- Target clean-independent set post-rebuild (~7-9): snr, srmr(real), f0-when-asserted, pause(count/rate), overlap_ratio,
  + jitter/shimmer/hnr if gated in.

**Story (for the paper):** the feature set covers the dimensions that determine downstream USABILITY (noise, reverb,
voice quality, tempo/fluency, multi-speaker overlap), and it spans TWO axes that ARE the two contributions:
  - OBSERVABILITY (abstention): channel feats (SNR, SRMR) observable in any condition vs speaker-intrinsic feats
    (f0, rate, pauses, jitter/shimmer/hnr) recoverable only when the talker is isolated -> hedge the latter under overlap.
  - LOCALIZATION (grounding maps): temporally localized feats (f0 in voiced frames, overlap in overlap regions ->
    frame-groundable) vs global feats (SNR -> a whole-clip stat, correctly NOT groundable; the SNR-null is a FEATURE).
We further AUDIT the set for spurious correlations (the anechoic-SRMR-to-pitch finding) and select a decorrelated set.
Present the structure as REVEALED by analysis, not designed a priori (honest). The audit itself is a contribution.

---

## RECON-VERIFIED ASSETS (2026-07-09) — everything present except RIRs
- Train mix (input to reverberate): $SH/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/mix_clean = 13,900 wavs (+ s1 stems same tree).
- Noise: $SH/data/wham_noise/tr = 20,000 wavs (NO download needed).
- Clean prosody: clean_features_train.json = 13,900 keys (snr_db, praat speaking/articulation rate, pause_count/rate/dur, srmr).
  Clean f0: clean_f0_train.json. NOT PRESENT anywhere: clean jitter/shimmer/hnr -> extract on s1 stem IF positive control passes.
- Aux head already REGRESSES the canonical 12 incl jitter/shimmer/hnr (src/feature_set.py SUPERVISED_FEATURES) -> positive control
  = a forward pass reading aux-head outputs vs clean GT, NO retrain.
- RIRs: DOWNLOADING now (SLR28 rirs_noises.zip -> $SH/data/rirs/, ~4GB) -- the ONE external prereq.
- Disk: 674G free on /ocean. Regen script seed: $SH/tmp_regen_corrected.py (validated, feature-only demo on 800 clips).

## REBUILD PIPELINE (stages, with assets)
Validated subset script on PSC: `$SH/tmp_regen_corrected.py` (uses feature_extractor_mix.compute_srmr/compute_f0_variation/
compute_praat_speaking_rate/compute_praat_pause_patterns; pyroomacoustics inverse_sabine RIRs; SRMR on reverberant, prosody
on clean stem, independent SNR). Scale -> `scripts/regenerate_corrected.py` with 3 CHANGES the demo lacks: (1) WRITE the
reverberant+noisy WAVs (model trains on them via WavLM), (2) use REAL SLR28 RIRs mixed w/ simulated + ACTUALLY add wham noise
at the independent SNR (demo only labels SNR, does not add noise), (3) REUSE clean_features_train.json/clean_f0_train.json
instead of recomputing prosody/f0. Input = the 13,900 train-100 mix_clean wavs.

STAGE A - Audio regen (CPU, both nodes' cores). For each of ~41,700 train (+ dev) clean stems:
  - reverberate (real SLR28 RIR w/ prob p, else simulated pyroomacoustics RT60 0.3-0.8) -> convolve.
  - add noise (MUSAN/WHAM) at independent uniform SNR 0-40 dB.
  - write new wav. PREREQ: verify full Libri2Mix train s1 stems ($SH/data/Libri2Mix/Libri2Mix/wav16k/min/train-100/s1);
    download SLR28 RIRS_NOISES + a noise corpus.
STAGE B - Feature extraction on new audio (CPU, parallel across 24 cores): SRMR on reverberant, snr = imposed value,
  intrinsic (f0, rate, pauses, jitter, shimmer, hnr) on CLEAN stem. -> features_corrected_train.csv.
  VALIDATE (two checks):
    (B1) DECORRELATION: re-run the confound matrix -> confirm srmr<->f0 and snr<->temporal dropped (target <~0.3 / ~0).
    (B2) DISTRIBUTION-OVERLAP DIAGNOSTIC (do NOT tune to AMI - circularity): compare SRMR/SNR/RT60 percentiles of
      {anechoic, augmented, AMI}. Confirm augmented MOVED into far-field AND AMI falls WITHIN the augmented support
      (interpolation, frac-inside >~0.7) AND augmented stays BROADER than AMI (general, not matched). Params come from
      GENERAL priors (RT60 0.3-0.8, SNR uniform 0-40); verify overlap post-hoc; NEVER optimize the gap to AMI.
      SUBSET RESULT (800 clips, 2026-07-08): SRMR anechoic 8.6 -> augmented 4.4 -> AMI 3.9 (78% of AMI SRMR inside
      augmented [2.1,7.6] = interpolation, GOOD). SNR augmented uniform (mean 20, 0-40) fully CONTAINS AMI (mean 13,
      7.5-20) and is broader = correct (AMI stays genuine OOD). Honest caveat: augmentation is slightly cleaner/dryer
      than AMI -> expect a small residual OOD gap; report it, do NOT tune it away.
STAGE C - Deterministic target regen (FAST, minutes): `scripts/build_descriptions_deterministic.py` with the new feature
  values; add jitter/shimmer/hnr templates IF gated in. NOT the LLM verbalizer (targets are deterministic templates).
STAGE D - Preprocess to .pt (GPU, hours): src/preprocess.py -> WavLM features on the reverberant+noise audio.
STAGE E - Retrain the ABLATION SET on the decorrelated data (GPU; regen A-D is shared, done ONCE). All: data_dir = new set,
  select_metric srcc_robust (fixed selector auto-applies), lambda_unlikelihood 0.1, UNINTERRUPTED to ep12 (do NOT let walls
  cut them off at ep5 like the fair-sel run - resume across walls to CONVERGENCE, since the whole point is a converged number).
  CORE method ablation (the paper's main table):
    - B1  : baseline adapter (lambda_token_grounding=0, lambda_liu=0)
    - M3c : token-grounding head, NO Liu (lambda_token_grounding=0.1, lambda_liu=0)  -> isolates the Liu term
    - M3b : token-grounding + Liu (full method)
  Secondary ablations (prioritize by GPU budget):
    - resolution: conv 8x vs 4x (does 4x rescue speaking_rate counting? affects the temporal features)
    - adapter conditioning: FiLM vs concat vs gate (the reviewer wanted this; no published verdict on FiLM in audio connectors)
    - +/- unlikelihood loss (isolate the degeneration fix's effect)
  SINGLE SEED (73) for ALL variants -- NO multi-seed sweep. Rationale: the +0.077 headline is DEMOTED, so seed-robustness
  of a small delta is no longer needed; the durable results (maps, abstention, AMI) are not seed-fragile. Report seed
  variance as a one-line limitation. NOTE: each variant ~1 GPU-day to ep12; 3-way core ablation (B1/M3c/M3b) is the minimum;
  resolution (8x/4x) + FiLM-vs-concat are the high-value adds.
STAGE F - Re-measure: confound matrix (confirm clean); in-domain M3b-vs-B1 (honest number); AMI-IHM transfer; 2nd OOD
  corpus; positive control for jitter/shimmer/hnr on clean; hedge-rate-vs-overlap-decile.

Rough cost with deterministic targets: ~3-5 days wall-clock (NOT 1-2 weeks — verbalization is a script, not the LLM).

---

## PAPER STRUCTURE (disclose-and-defend; strategy-panel synthesis, adjusted)
LEADS (durable, unscooped): (i) causal token/overlap-attention grounding maps (deletion soft-IoU p=0.0008 flagship, f0
borderline flagged, SNR correctly-null); (ii) observability-driven abstention (f0 0.995 clean -> hedge 0.66->0.96 on real
AMI overlap); (iii) claim-level faithfulness metric with real far-field transfer (AMI-IHM intrinsic mean 0.415 vs 0.395)
AND, post-rebuild, in-domain/OOD CONVERGENCE on decorrelated features.
DISCLOSED as contributions: the confound audit (partial-corr-decorrelated SRCC, anechoic-SRMR diagnosis, ~2-axis result);
the RIR+independent-SNR fix (now DEPLOYED, not just subset-demo); rho=0.69 relabeled LLM-judge; head currently SNR-only.
DROPPED: +0.077 as headline (appendix robustness table w/ seed + selection-bug caveat); confounded feats from the clean
set; retracted SNR win-rate 1.0; human-validated language; broad self-explainable framing (narrow vs Sci-Phi).
SECTIONS: 1 Intro (grounded self-hedging reports; Qwen2-Audio-constant = task non-triviality). 2 Related (ALLD template;
Sci-Phi differ on claim-faithfulness + causal maps). 3 Method (adapter; token-grounding head; metric; abstention). 4
Grounding maps + deletion (LEAD). 5 Abstention on real overlap. 6 Faithfulness eval: confound audit + decorrelated SRCC +
the rebuild convergence result + AMI transfer + 2nd OOD. 7 Limitations. Appendix: +0.077 table, RIR/SNR details, full
confound matrix.

---

## GATES / OPEN QUESTIONS (resolve before/within the rebuild)
- [ ] Clean positive control: can the model predict jitter/shimmer/hnr on CLEAN audio? (decides whether to add them)
- [ ] Does the fair-selection retrain (RUNNING now, old data) give +0.077 or ~0? (settles the "before" number)
- [ ] Does conv 8x->4x rescue speaking_rate (counting)? (decides keep/drop)
- [ ] After full regen, is the confound matrix actually clean at scale (not just 800 clips)?
- [ ] Second OOD corpus choice + does it have clean references for intrinsic-feature GT (or hedge f0/rate there too)?

---

## TASK CHECKLIST (ordered)
1. [ ] Let fair-selection retrain (b1_fairsel_s73 / m3b_fairsel_s73) finish -> TEST eval -> the "before" +0.077 number.
2. [ ] Verify full Libri2Mix train s1 stems; download SLR28 RIRS_NOISES + MUSAN/WHAM to PSC.
3. [ ] Clean positive control for jitter/shimmer/hnr (gate the voice features).
4. [ ] Promote tmp_regen_corrected.py -> scripts/regenerate_corrected.py (real+sim RIRs, noise, parallel over cores).
5. [ ] STAGE A+B regen full train+dev; validate confound matrix at scale.
6. [ ] STAGE C deterministic targets (corrected values + gated voice feats).
7. [ ] STAGE D WavLM preprocess.
8. [ ] STAGE E retrain B1+M3b (fixed selector + lambda_unlikelihood on both) + seeds.
9. [ ] STAGE F re-measure (confound matrix, honest in-domain, AMI, 2nd OOD, voice positive control, hedge-decile).
10. [ ] Paper: apply rho=0.69 relabel; write confound-audit section; rebuild convergence; grounding-maps figure; drop +0.077 lead.

---

## OPS NOTES (PSC)
- SH=/ocean/projects/cis260125p/shared ; ENVPY=$SH/envs/project/bin/python ; CT=$SH/cur_train
- ssh -i ~/.ssh/psc_key slin32@bridges2.psc.edu ; scp via data.bridges2.psc.edu with -O.
- Projects are GPU-ONLY (no RM/CPU allocation) -> use GPU-shared nodes' CPU cores for CPU work.
- salloc --no-shell -A cis260125p -p GPU-shared --gres=gpu:h100-80:1 -t <h> creates a detached alloc (jobid via squeue).
- srun --jobid=J --overlap for compute; HOLD ssh 55-90s after launch so the step registers; verify GPU>5GB.
  GOTCHA: the 16GB blob prewarm-cat STRANGLES a cold node -> launch training WITHOUT prewarm (torch loads directly).
- LOGIN NODE CANNOT RUN TORCH (illegal instruction) -> torch/versa/SRMR jobs must srun onto a compute node. numpy/scipy/
  pandas/parselmouth DO run on the login node.
- Ocean NFS cold reads stall (util 0%, mem held) -> WAIT, do not kill.
- ls *.pt on the 41,700-file train dir hits "argument list too long" -> use find.
- Current retrains: b1_fairsel_s73 (42001172/w010), m3b_fairsel_s73 (42001174/w008), seed 73, 12 ep, fixed selector,
  WANDB_MODE=offline. Monitor armed (ScheduleWakeup) to drive to ep12 + eval.

## KEY NUMBERS TO CITE
- srmr<->f0: 0.80 (anechoic GT) -> 0.30 (RIR subset) -> 0.56 (real AMI). snr<->temporal: 0.57-0.66 -> ~0 (independent SNR).
- AMI-IHM: M3b intrinsic 0.415 / B1 0.395; f0-when-asserted 0.572/0.667. Overlap map p=0.0008; f0 p~0.03; abstention 0.66->0.96.
- +0.077 (seed73 best.pt, BUGGY selector) vs -0.002 (seed42 last.pt). Qwen2-Audio = constant (SRCC undefined).
