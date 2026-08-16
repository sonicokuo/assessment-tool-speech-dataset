<!-- AUTO-BANNER SUPERSEDED 2026-08-03 -->
> # ⚠️ SUPERSEDED — historical record, not a plan of action
>
> Superseded twice: by MASTER_PLAN_2026-07-16, then by the bsigma/architecture campaign. The grounded pair is walled and NOT resumed.
>
> Authoritative current state: `.claude/research/STATE_2026-08-03.md`.
> Original text preserved unchanged below.

---

# AQUA-NL Rebuild — COMPREHENSIVE SPEC (2026-07-09)

Companion to REBUILD_PLAN.md (which holds the verified findings + ops notes). This file is the DETAILED,
answer-every-question spec: exact feature set, exact train set, exact test sets, how OOD is verified, how the
augmented/AMI similarity is measured, and where the deterministic (non-LLM) target builder sits. LOCAL-ONLY.

---

## 0. ONE-PARAGRAPH SUMMARY
We take Libri2Mix 2-speaker mixtures and DATA-AUGMENT them into far-field realism (real+simulated RIR reverberation
+ independent-SNR wham noise) so the two GT confounds break (srmr stops being a pitch proxy; snr decouples from speech
content). GT for channel features comes from the degraded audio; GT for speaker-intrinsic features comes from the CLEAN
source stem. Targets are built by a DETERMINISTIC TEMPLATE SCRIPT (never the gemma/Ollama verbalizer). We retrain a
3-way ablation to convergence, then evaluate on a decorrelated in-domain test plus TWO held-out real-far-field OOD
corpora (AMI + one more), where "OOD" is a MEASURED property (acoustic-distribution distance from the training support),
not an assumption.

---

## 1. FEATURE SET (exact list + role of each)
Organized on the two axes that ARE the paper's contributions (observability -> abstention; localization -> grounding maps).

### 1a. Channel / recording-condition features  (GLOBAL, observable in ANY condition, grounding-NULL, NOT hedged)
| feature | unit | GT source in the rebuilt data |
|---|---|---|
| snr   | dB   | the INDEPENDENT uniform value imposed during augmentation (0-40 dB) |
| srmr  | -    | measured on the DEGRADED (reverb+noise) mixture -> real reverberation post-rebuild |

### 1b. Speaker-intrinsic features  (LOCALIZED, recoverable only when the talker is isolated, HEDGED under overlap)
| feature | unit | GT source | notes |
|---|---|---|---|
| f0_mean         | Hz     | CLEAN s1 stem (clean_f0_train.json)        | strongest clean feature; hedged under overlap |
| f0_sd           | Hz     | CLEAN s1 stem                              | pitch variation; historically weak self-SRCC (flag/appendix) |
| speaking_rate   | syl/s  | CLEAN s1 stem (clean_features_train.json)  | duration-proxy risk; KEEP only if 8x->4x resolution rescues counting |
| articulation_rate | syl/s| CLEAN s1 stem                              | secondary rate feature |
| pause_count     | int    | CLEAN s1 stem                              | one of the two pause features |
| pause_rate      | /min   | CLEAN s1 stem                              | ~0.84 corr with pause_count -> one "pause axis"; headline uses one, appendix the other |
| jitter  (GATED) | -      | CLEAN s1 stem (extract if gate passes)     | voice quality; ill-posed under overlap |
| shimmer (GATED) | -      | CLEAN s1 stem                              | voice quality |
| hnr     (GATED) | dB     | CLEAN s1 stem                              | voice quality |

GATE for jitter/shimmer/hnr: only add to the targets if the positive control shows the model can predict them on CLEAN
audio (aux head already regresses them -> forward-pass correlation vs clean GT). If SRCC ~0 on clean, they are dead
features -> DROP, do not template them.

### 1c. Scene feature
| overlap_ratio | - | oracle-VAD / Pyannote-on-mix | the abstention TRIGGER; also grounding-localizable (overlap regions) |

### 1d. Headline metric set (band-free SRCC, mean over the clean-independent features)
snr, srmr(real), f0-when-asserted, pause_count (ONE pause axis), overlap_ratio, + jitter/shimmer/hnr IF gated in.
Everything else reported per-feature in the appendix. DECISION rule: a feature is "headline" only if (i) decorrelated at
scale (confound matrix), and (ii) predictable on clean audio (positive control / self-SRCC > threshold).

---

## 2. TRAIN SET (exact design)
- BASE AUDIO: Libri2Mix train-100 2-speaker mixtures (mix_clean, 13,900). KEEP the single-speaker s1clean variant
  (13,900) for the clean/non-overlap abstention contrast -> reverberate those too (same pipeline, overlap_ratio=0).
- AUGMENTATION per clip (regenerate_corrected.py):
  1. Reverberate: REAL SLR28 RIR with prob p_real=0.5 (real_rirs_isotropic_noises, ~326 AIR/RWCP recordings),
     else simulated pyroomacoustics ShoeBox RT60 0.3-0.8s. (Real RIRs specifically to blunt "domain-matched aug".)
  2. Noise: add wham_noise/tr at an INDEPENDENT uniform SNR in [0,40] dB (scale noise to target SNR vs signal power).
  3. Write the degraded WAV (model trains on THIS via WavLM).
- GT SOURCE (the decorrelation rule):
  * snr  = the imposed sampled value (independent of content).
  * srmr = measured on the degraded mixture (real reverb).
  * intrinsic (f0, prosody, voice quality) = measured on the CLEAN s1 stem (REUSE clean_features_train.json +
    clean_f0_train.json; extract jitter/shimmer/hnr on the s1 stem only if gated).
  * overlap_info / overlap_ratio = Pyannote-on-mix, recomputed on the reverberant mixture (train==test input dist).
- TARGETS (Stage C): use ONLY `scripts/build_descriptions_deterministic.py` -- VERIFIED deterministic (2026-07-09):
  build_description(row) assembles text from f-string templates, loads clean-frame F0, NO server call. Its lone "gemma"
  string is a docstring comment. Point it at the corrected feature CSV (per-split <split>.csv in --features_dir);
  regenerating from new numbers = re-run it (minutes).
  ** DO NOT USE scripts/regen_descriptions.py ** -- VERIFIED it IS a gemma/Ollama verbalizer (calls localhost:11434,
  gemma4:e2b, n_ollama_err counters). Banned from the rebuild. NEVER gemma/Ollama for targets.
- PREPROCESS (Stage D): WavLM-Large features on the degraded audio -> new processed_corrected/{train,val,test}.

---

## 3. TEST SETS (three, with roles)
1. IN-DOMAIN test (3000): the test mixtures augmented with the SAME pipeline (reverb+noise). This is the HONEST
   in-domain number on decorrelated features -> "did the model learn real features, fairly selected." Train==test dist.
2. AMI-IHM (HELD-OUT OOD #1, real far-field meetings): already built (ami_ihm_matched, 3654 clips). Intrinsic GT from
   the clean HEADSET (IHM) twin; channel GT (srmr) from the far-field SDM. Primary real-data generalization result.
3. SECOND OOD corpus (NEW, HELD-OUT OOD #2): **CHiME-6** (DECIDED 2026-07-09). Dinner-party far-field (real reverb+noise),
   most DIFFERENT from AMI meetings -> probes generalization breadth. Has WORN/close-mic references -> intrinsic-feature
   GT (f0/pauses/voice) available via the same clean-twin pipeline as AMI-IHM. Must be FETCHED (not on PSC; only AMI is).
   Expect it to sit FURTHER from the training support than AMI (Sec 4) = a stronger OOD test. Fetch + segment at Stage F.
NB: AMI and the 2nd corpus are TEST ONLY. Never train on them; never tune augmentation params to them.

---

## 4. OOD VERIFICATION — make "is it OOD" a MEASURED claim (not assumed)
For EACH test set, characterize its acoustic distribution and locate it relative to the AUGMENTED TRAIN support:
- Compute per-clip SRMR, SNR, and an RT60 estimate; build the marginal distributions.
- METRIC 1 (interpolation fraction): fraction of test clips whose SRMR (and SNR) fall inside train [p5,p95].
  High = interpolation (near-domain); low = extrapolation (genuine OOD).
- METRIC 2 (distribution distance): 1-Wasserstein (or symmetric KL) on SRMR and SNR between test and augmented train.
- REPORT a 4-way figure: {original anechoic, augmented train, AMI, 2nd-OOD} SRMR/SNR distributions + the two metrics.
- KNOWN (800-clip subset): augmented SRMR mean 4.4 vs AMI 3.9 (78% of AMI SRMR interpolates); augmented SNR uniform
  mean 20 CONTAINS AMI's narrower mean-13 -> AMI = "far-domain but mostly interpolating." The 2nd corpus SHOULD sit
  FURTHER out (larger distance) to probe breadth: if it still transfers -> strong; if it degrades -> the honest OOD limit.
- This directly answers "did you check augmented~AMI similarity" (yes, metrics above) AND "is the new test set OOD"
  (yes, quantified by distance/interpolation, not asserted).

---

## 5. ABLATIONS (Stage E) — single seed 73, to CONVERGENCE (ep12, resume across walls)
CORE (paper main table):
  - B1  : baseline adapter (lambda_token_grounding=0, lambda_liu=0)
  - M3c : token-grounding head, NO Liu
  - M3b : token-grounding + Liu (full method)
SECONDARY (budget permitting):
  - resolution: conv 8x vs 4x (does 4x rescue speaking_rate/counting? decides if speaking_rate is a headline feature)
  - adapter conditioning: FiLM vs concat vs gate (reviewer asked; no published verdict for audio connectors)
  - +/- unlikelihood loss (isolate the degeneration fix)
All with lambda_unlikelihood=0.1 + the FIXED selector (ckpt_selection.py clip_rep_thresh 0.50). NO multi-seed sweep.

---

## 6. METRICS
- FAITHFULNESS: band-free per-feature Spearman SRCC (headline = mean over the clean-robust set), nMAE, coverage.
- ABSTENTION: hedge rate vs overlap decile; assert-accuracy clean (~0.995) vs hedge; per-intrinsic-feature hedge policy.
- GROUNDING MAPS: attention-deletion soft-IoU (overlap flagship p=0.0008; f0 localized; SNR correctly-NULL).
- FAITHFULNESS-RISK: AURC (risk made continuous) for the abstention/faithfulness trade-off.
- (NOT a headline) in-domain M3b-vs-B1 delta: reported with the seed/selection caveat; the +0.077 is DEMOTED.

---

## 7. NODE PARALLELIZATION (two nodes, both up)
- Node A + Node B both allocated (salloc --no-shell; srun --overlap for steps). Split work, no idle:
  * Stage A/B regen: split 13,900 train (+ dev/test augmentation) across BOTH nodes (each ~half) -> ~2x faster.
  * Positive control: on whichever node frees first (forward pass, ~1h).
  * Stage D preprocess: both nodes (GPU).
  * Stage E retrain: 2 variants in parallel (B1 on A, M3c on B; then M3b) -> 3 variants in ~2 rounds.

---

## 8. STAGE SEQUENCE + GATES
G0. Pre-gates: RIRs downloaded+extracted (DONE, 417 real RIR wavs); stems/noise/clean-features verified (DONE).
G1. SMOKE TEST (50 clips) -> confound matrix must drop (srmr<->f0 0.80->~0.3, snr<->temporal->~0). GO/NO-GO. <-- current
G2. Positive control -> gate jitter/shimmer/hnr into the feature set.
Stage A/B (both nodes): full regen train+dev+test -> validate confound matrix + distribution overlap at FULL scale (Sec 4).
Stage C: deterministic targets (Sec 2) with the final feature set.
Stage D: WavLM preprocess on degraded audio.
Stage E: retrain the ablation (Sec 5) to convergence.
Stage F: measure (Sec 6) on the 3 test sets (Sec 3) + OOD verification (Sec 4).
Rough cost: ~3-5 days wall-clock (deterministic targets, 2-node parallel regen + retrain).

---

## 9. OPEN DECISIONS (need a call before/within the run)
- [ ] 2nd OOD corpus choice (availability + clean reference) -> Sec 3 item 3.
- [ ] Does 8x->4x rescue speaking_rate? -> decides headline membership.
- [ ] jitter/shimmer/hnr positive-control result -> in or out of the feature set.
- [ ] Keep s1clean single-speaker variant in train (recommended: yes, for the abstention contrast).

## VERBALIZER RULE TABLE (2026-07-10) — grounded thresholds for qualitative NL bands
Research verdict (deep-research wf_0190f846): physical anchors exist for SNR + f0; for SRMR/rate/voice/pauses NO physical
threshold exists and the literature PROVES it -> data-relative quartile bins are the honest+citable choice (cite the "no
threshold" finding). Cache of rules + citations for the "Description vocabulary" appendix:
- SNR (dB): <10 noisy | 10-20 fair | 20-30 good | >30 clean.  CITE BS5839-8/BS6259 (10 dB = intelligibility floor); note
  condition-dependence (3-6 dB anechoic vs 15-20 dB reverberant per iscve.org.uk).
- f0_mean (Hz): <150 low-pitched | 150-200 mid | >200 high-pitched.  CITE NCVS + Holmberg/Hillman/Perkell 1988 (male ~116-125, female ~200-210).
- SRMR: <2.5 heavily reverberant | 2.5-4.7 moderately | >4.7 lightly (DATA QUARTILES).  CITE Falk/Zheng/Chan 2010 (metric) +
  arXiv:1510.04707 (SRMR->RT60 is a fitted regression, r~0.6, NO fixed cut points) to justify data-relative.
- speaking_rate (syl/s): <4.5 slow | 4.5-6 measured | >6 brisk (soft).  CITE Pellegrino 2011 (normal 5.2-7.8) + Trouvain 2004 (tempo speaker-relative, no objective cutpoint).
- pause_count: 0 none | 1-3 few | 4-6 several | >6 frequent (descriptive).  Goldman-Eisler 1968 250ms = pause DETECTION only, not band.
- overlap_ratio: 0 single-speaker | <0.3 light | 0.3-0.6 moderate | >0.6 heavy (definitional).
- jitter/shimmer/hnr: low/moderate/high by DATA QUARTILE.  CITE Teixeira 2013 (norms are SUSTAINED-VOWEL) + AJSLP 2020 / ASHA 2018
  (running speech uses CPP instead; jitter/shimmer unreliable on connected speech) -> no running-speech bands exist -> data-relative honest.
- f0_sd: monotone/moderate/expressive by data quartile (prosodic variation, no standard threshold).
Number ALWAYS printed next to the adjective (adjective = flavor, never replaces the value the SFS metric parses).
