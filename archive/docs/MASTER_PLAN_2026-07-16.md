<!-- AUTO-BANNER SUPERSEDED 2026-08-03 -->
> # ⚠️ SUPERSEDED — historical record, not a plan of action
>
> Presents RT1/RT2/RT3 as FUTURE runs -- all executed 2026-07-22..29. Declares a 'V1-V5 all green before any retrain' gate that was NEVER satisfied (V1, V2, V5 never ran) yet the campaign proceeded and succeeded. Lists zero-shot baselines 'as budget allows' -- Qwen2-Audio ran 2026-07-02 on all 3000 clips.
>
> Authoritative current state: `.claude/research/STATE_2026-08-03.md`.
> Original text preserved unchanged below.

---

# MASTER PLAN — fix everything, verify everything, then retrain + RL (2026-07-16)

Sources consolidated (nothing skipped): trajectory audit 13 risks (`audit-trajectory-2026-07-15.md`),
degeneration memo (`degeneration-rootcause-2026-07-15.md`), training-method memo R1–R10
(`research-training-method-2026-07-13.md`), RL design memo (`rl-design-2026-07-15.md`),
plus standing registry items. Doctrine: **no retrain until every fix is empirically verified**
(the 2026-07-13 full-verification pass caught 3 gaps unit tests missed; same standard here).

Traceability: every memo item maps to a plan ID below. Audit risks = A1–A13. Degeneration = D-*.
Training-method = R1–R10. RL memo = RL-*.

---

## PHASE 0 — Offline salvage + diagnostics (zero node-hours; establishes ground truth first)

| ID | Task | Source | Verification |
|----|------|--------|--------------|
| S1 | Replay the selection guard offline from `val_samples/epoch_*.json` + wandb `val_bleu` for BOTH arms → identify exactly why `best.pt` froze at ep0 (BLEU relative floor w/ unpersisted `best_val_bleu`? `frac_clips_high_rep` gate? resume restore?) | A2 | Reproduce the withhold decision per epoch; written finding |
| S2 | Paired matched-coverage re-score of both arms per epoch: SRCC computed only on the intersection of clips where BOTH arms emit each feature; report `coverage_{feat}` beside every SRCC; recompute on non-degenerate clips separately | A3 | Table: paired per-epoch SRCC deltas; separates "b1 catches up" from "degeneration eats m3b" |
| S3 | Per-feature decomposition: srcc_robust WITH and WITHOUT snr, both arms (SNR-circularity check on the oracle-SNR-supervised treatment) | A5 | Does the grounding delta survive snr exclusion? |
| S4 | Paired per-clip bootstrap of the grounding delta at each epoch; formally RETIRE the +0.077 claim for the corrected pipeline (appendix note at most) | A4 | CI table; registry + memory updated |
| S5 | Epoch-wise loop-fraction curve + loop-fraction vs val-CE per checkpoint (tests the overfitting-sharpening account); bootstrap the m3b-40% vs b1-30% loop-rate difference | D-R2, D-diag | Plot + CI |
| S6 | Verify `checkpoint['epoch']` parity in both ep8 `last.pt` files (budget-matched endpoint) | A8 | Logged values |
| S7 | Head-vs-emission disagreement audit from existing `inference_results.json` (aux_mean already logged, E12) — quantifies the aux-head bypass hypothesis | R5c, §1.4 | Per-feature |pred_LM − aux_mean| distribution |
| S8 | Formal decision recorded: A/B stopped at ep8, NOT resumed to ep12; babysit /loop terminated | A8, audit §4 | Registry entry |

## PHASE 1 — Code fixes (local; every fix ships with a test)

### Group A — decode / stopping
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F1 | EOS stop-set: stop on `{<\|im_end\|>(151645), <\|endoftext\|>(151643)}` in `inference.py:215` AND `train.py` val `generate(eos_token_id=[...])` (matches Qwen's shipped generation_config) | D-EOS (Task-1 correction) | Unit test; on-node: EOS audit (V1) then re-measure loop fraction on the SAME ep8 checkpoint |
| F2 | Wire decode backstop into eval configs: `repetition_penalty` (sweep 1.0/1.1/1.15/1.2) + `no_repeat_ngram_size=4`; adopt best SRCC-safe setting | D-fix, D4 | `test_decode_penalty` (exists) + config-wiring test; val sweep with SRCC guard (V2) |

### Group B — checkpoint / selection integrity
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F3 | Numbered per-epoch SLIM checkpoints (adapter+LoRA+heads, ~0.2 GB) in addition to last/best — peak weights can never be lost again | A2 | Save/load unit test; disk-budget check |
| F4 | Persist the FULL selection-tracker state (`best_val_bleu`, best metric, guard state) in the checkpoint payload; restore on resume; fix guard per S1 findings (incl. not updating `best_val_bleu` on withheld epochs if S1 confirms that bug) | A2 | Extend `test_ckpt_selection` resume cases |
| F5 | wandb-log `_save_reason` + `degeneration_stats` (rep_n_mean/max, frac_clips_high_rep, frac_nonascii) EVERY epoch (currently console-only) | A2, A13 | Smoke shows keys nonzero (V3) |
| F6 | Append-mode / timestamped training logs (kill the `>` overwrite; run banner per launch) | A13 | Two launches → both visible |
| F7 | Upload last.pt / slim ckpt as wandb artifact every N epochs during campaigns | A13 | Smoke shows artifact |

### Group C — metric consistency (ONE headline everywhere)
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F8 | Freeze THE headline metric: srcc_robust over {snr, srmr, speaking_rate, pause_count, pause_rate} (snr IN, f0 OUT, overlap OUT) in BOTH `selection_metric.py` and `scripts/score_inference_vs_clean.py`; move f0_mean/f0_sd to the abstention/coverage panel; DELETE the stale `articulation_rate` row (FEATURE_MAP) | A6, B2 | Fixture run through both scorers asserts identical feature set |
| F9 | Fix stale "snr EXCLUDED" comments (`train.py:2273,2337,2482`, `selection_metric.py:270`) + repo-wide grep sweep | A12 | `grep -rni "snr excluded"` clean (labeled historical notes exempt) |
| F10 | Report `coverage_{feat}` next to EVERY SRCC in val + test scorers (coverage-blindness fix, permanent) | A3 | Scorer unit tests updated |

### Group D — data integrity
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F11 | f0 clean-GT fail-loud: read explicitly named `f0_*_hz_clean` (or assert clean-substitution marker); startup assert `descriptions_path*` point at the clean build | A10, f0 diagnosis | Unit test: noisy fixture → raises |
| F12 | Inference resume keyed on checkpoint hash (or per-checkpoint output file) so `inference_results.json` can never mix checkpoints | A11 | Unit test: two ckpts → separate caches |

### Group E — training losses (needed BY the retrain; built + tested now)
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F13 | NTL on the NUMS channel: verify implemented form (prefer NTL-WAS); currently NTL runs on the PROSE logits — extend the nums forward to return ntl tensors and apply NTL there (digit-position masking over `snr=15.66 …`, vocab digit→value map) | R1 | Extend `test_ntl`: digit-mask correctness; closer-digit ⇒ lower loss; nums-channel wiring |
| F14 | DITTO-style sentence-level repetition penalty — IDENTICAL-sentence only (token/n-gram UL would punish the supervised template frame; raising lambda_unlikelihood is a PROVEN dead lever, DITTO Fig 10) — implemented as a config-gated loss term, default off | D-fix, R7 | Unit test on synthetic loops; SRCC-guarded ablation flag |
| F15 | Fixed constant-width numeric surface form: normalize ALL target numbers to 2 decimals + consistent units in the target builders (srmr "3.2667"→"3.27" etc.) for both prose + nums targets | R4 | Build fixture; SFS parser still parses; digit-position alignment check |
| F16 | Expose `lambda_nums` rebalance / separate-LoRA-branch option (config only) to enable the R3 nums-dominance ablation later | D-R3 | Config parse test |
| F17 | Headline σ-head config: `config.bsigma.yaml` = b1 recipe + `reliability_head:true, lambda_nll:0.1, beta_nll:0.5, stirn_stop_grad:true, lambda_mse>0` + NTL on + all new instrumentation. (Code already node-verified: h2b smoke, NLL 6.66→0.86, σ 1.00→1.25) | A1, H2 | `test_stirn_hedge` (green); config-completeness check; smoke V3 |

### Group F — verified-slot decode (top-ranked zero-retrain lever)
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F18 | Constrained slot decode harness: template frame from an FSM/grammar, LM fills numeric slots; optional verified-slot substitution from aux_mean; per-slot digit-distribution spread + head-vs-slot disagreement exposed as the abstention signal; hedge clause as the abstain branch (fuses H6) | R2/G1, RL gate 1 | FSM unit tests; slot-parse tests; on-node: loop rate + SRCC free-decode vs slot-filled on the ep8 checkpoint (V4) |

### Group G — RL prep (code now, run in Phase 5)
| ID | Fix | Source | Test |
|----|-----|--------|------|
| F19 | REBUILD `sfs_reward.py`: retire the dead band-F1 core; new observability-gated per-slot reward — VALUE: `1−min(|v−g|/s,1)` if recoverable else 0; HEDGE: 0.8 (unrecoverable+attributed) / 0.4 (unrecoverable generic) / 0.2 (recoverable); ABSENT/unparseable: 0; sequence reward × `G(y)=template_validity×(1−rep_n)×(1−nonascii)`; keep rep_n/nonascii guards + first-claim-only policy | RL §2, A-sfs | ADVERSARIAL test cases FIRST in `test_sfs_reward`: constant-mode, easy-feature-only, hedge-spam, format-gaming, value-spread — each must strictly lower reward |
| F20 | ClaimParser hedge-attribution flag `A_f` (hedge names overlap as cause AND clip overlap supports it) | RL §2.1 | Parser unit cases |
| F21 | Verify the GRPO rollout path: generation from `inputs_embeds` (audio prefix) under TRL GRPOTrainer / `grpo_train.py`; note vLLM prompt-embeds support status | RL §3 | Local dry-run + on-node micro-smoke (Phase 5 gate) |

## PHASE 2 — Full verification battery (gate: ALL green before any retrain)

Local: entire pytest suite green (including every new test above).

On-node (one shared-node window, ~3–4 h total, all on EXISTING checkpoints):
| ID | Check | Evidence required |
|----|-------|-------------------|
| V1 | EOS audit (~1 h): instrument decode to log rank/logprob of BOTH enders + stop reason, `skip_special_tokens=False`, ~50 looping + ~50 clean clips → then apply F1 and re-measure loop fraction on the same checkpoint | (a) % boilerplate clips emitting 151643 mid-stream (any >0 confirms the missed-stop class) (b) scaffold-loop steps where either ender top-5 ≈ 0 (c) clean clips stop on 151645 |
| V2 | Decode-backstop sweep on val: penalty {1.0,1.1,1.15,1.2} × ngram {0,4} → loop fraction + SRCC dashboard | Chosen setting: max loop reduction with SRCC unharmed |
| V3 | 200-step smoke of `config.bsigma.yaml`: every new wandb key fires nonzero (`train_loss_nll`, `reliability_sigma_mean`, `loss_ntl`, `_save_reason`, degeneration stats), per-epoch slim ckpt written, tracker persists across a forced resume | Key-by-key checklist, like the 2026-07-13 pass |
| V4 | Slot-decode measurement on the ep8 checkpoint: loop rate + SRCC, free-decode vs slot-filled (the G3 two-row) | Decides whether F14 (DITTO) needs to be ON in the retrain |
| V5 | Reduced-LoRA-scaling probe (SALMONN-style) on the ep8 checkpoint: loops vs LoRA α | Adapter-induced vs base-prior diagnosis (R5a) |

## PHASE 2.5 — Independent Fable verification of ALL fixes (user requirement, 2026-07-16)

After Phase 1 code is written and Phase 2 local tests are green, an independent **Fable-5
adversarial verification agent** audits EVERY fix (S1–S8, F1–F21) against its spec in this plan:
reads the diffs, re-derives the expected behavior, tries to refute each fix, checks tests
actually cover the failure mode, and confirms nothing on the list was skipped. Output = a
per-item verdict table (CONFIRMED / INCOMPLETE / WRONG + evidence). Only after this verdict is
clean does the V1–V5 node battery run, and only after V1–V5 does any retrain launch.

**Progress log:**
- 2026-07-16: Phase 0 COMPLETE (S1–S6 ran on PSC; report `$SH/salvage_report_2026-07-16.json`).
  S1: best.pt froze because the `rep_n_max>0.95` single-clip backstop fires at every epoch ≥2
  (statistically guaranteed at n=200; b1 all 6 epochs, m3b ep2-4+7) plus the BLEU floor (m3b ep5-6);
  only ep1 ever saved in both arms — replay reproduces observed behavior exactly.
  S2–S4: paired matched-coverage delta significant ONLY at ep2 (+0.108 CI[+0.027,+0.194], survives
  snr-exclusion → not circular); ep3–7 all straddle 0 → grounding = early-convergence effect only.
  S5: loop-rate difference real but flips (b1 worse at ep2, m3b worse ep4+); m3b coverage collapses
  to ~0.55-0.67 vs b1 ~0.76-0.89 after ep3. S6: both last.pt at epoch=6 (budget parity ✓);
  `best_val_bleu` NOT in ckpt (confirms F4). S7 deferred to the V-window (needs aux_mean at scale).
  S8: A/B formally stopped at ep8, loop ended.
- 2026-07-16: F1 (EOS stop-set) implemented in `inference.py` (incl. first-token edge case) +
  `train.py` `_val_eos_ids()` helper; `tests/test_eos_stopset.py` (4 cases); py_compile clean.
  Torch-dependent test execution happens on PSC (no local torch venv) with the rest of the suite.

## PHASE 3 — Retrain campaign (~10 weeks to ICLR abstract; peak ~ep3-4 ⇒ short schedules)

Standing rules: schedule sized to finish INSIDE the wall (≤6 epochs, cosine completes); per-epoch slim ckpts;
selection on the frozen headline metric with the fixed guard; append logs; seed declared per run;
give the headline run the longest node.

| Run | Config | Purpose | Budget |
|-----|--------|---------|--------|
| **RT1 (headline)** | `bsigma` seed 73: b1 recipe + σ-head (β-NLL 0.5 + Stirn) + NTL-nums + fixed-width targets + hedge two-mask + decode backstop at eval + clean non-overlap data if landed (else document) | THE paper model: numeric faithfulness + trained abstention substrate | ~6 ep ≈ 8 h |
| **RT2 (seed)** | RT1 with seed 42 | Seed error bars on the headline (A7) | ~6 ep |
| **RT3 (adapter ablation cluster)** | concat-only / film / film-attn / film-mamba / qformer under the RT1 recipe, post-init-fix, same seed, SHORT (~4 ep) | Settles the never-actually-chosen adapter (user's flag); table ablation rows | 5 × ~4 ep (can run 2/node) |
| **RT4 (optional, gated on S2–S4)** | Grounding ablation under the final recipe ONLY if the paired re-score shows a real effect: token_grounding WITHOUT liu (single knob), + separate liu arm if needed | Un-confound A5; single-seed convergence observation at most | ~6 ep, only if budget |
| **RT5 (optional)** | λ-sensitivity mini-sweep (R6) | Reviewer armor | last priority |

Explicitly NOT resumed: the walled ep8 grounded pair (compromised selection; audit §4).

## PHASE 4 — Post-retrain evaluation (no retrain)

1. Selective-prediction stack on RT1's σ checkpoint: Mondrian/CRC calibration → risk-coverage / AURC / E-AURC / AUGRC; ENCE + σ coefficient-of-variation. **Degeneracy gate (A9): the abstention claim ships ONLY if σ varies within-feature across overlap bins and Mondrian thresholds differ across bins** (constant-high σ on f0 is vacuous).
2. Hedge verbalization (H6) through the slot-decode abstain branch → description-level guarantee.
3. Test-set protocol: frozen headline metric + coverage, greedy inside the harness, matched test_dir/GT; paired cluster-robust DM test + Holm.
4. Two-row main table: free-decode vs slot-filled (G3) + digit-drift (E12).
5. Baseline rows: regress-then-verbalize upper bound (R10); zero-shot audio-LLM baselines (X3) as budget allows.
6. AMI cross-domain + Mondrian recalibration-without-retrain (X2) if data ready; s1/s2 stem re-sync remains load-bearing for clean f0 GT (B4).
7. Report f0 via abstention/coverage panel, not SRCC (f0 diagnosis).

## PHASE 5 — RL (GRPO), time-boxed, all gates before any launch

Gates (in order): (G1) slot-decode measured — it rebaselines RL headroom; (G2) reward rebuilt + adversarial tests green (F19/F20); (G3) clean non-overlap data landed (batch mixing is the anti-hedge-collapse mitigation); (G4) **best-of-16 probe**: sample 16 @ temp 1.0 on ~500 val clips, rerank by the new reward — the best-of-16 vs greedy reward gap IS the go/no-go headroom number; (G5) `inputs_embeds` rollout path verified (F21).

| Stage | What | Guard |
|-------|------|-------|
| RL-0 | Rejection-sampling SFT (best-of-N distillation) on 5–10k stratified subset — no RL machinery, same reward; doubles as the probe | SRCC + loop dashboards |
| RL-1 | GRPO: group 8, temp 1.0, token-KL to the SFT policy 0.05→0.01, LoRA-only trainable (WavLM + adapter + heads FROZEN), 1–2 epochs over 5–10k stratified subset; batches stratified by overlap bucket with a guaranteed high-overlap floor; consider mean-only advantage if reward variance skews across buckets | Per-feature emission-rate dashboard; hedge rate by overlap bin; length curve; loop fraction |
| RL-2 | **Mandatory random-reward control** (identical pipeline, Bernoulli reward) — Qwen-family GRPO can gain from spurious prior amplification; without this the RL result is not trustworthy | RL gains must exceed the random-reward run |
| RL-3 | Post-RL: RECALIBRATE Mondrian/CRC on held-out (pre-RL guarantees are void after the policy shift; RL inflates expressed confidence); re-run the full selective stack | ENCE/CoV gates again |
| Time-box | If G4 shows negligible headroom or 2 calendar weeks elapse → cut RL; the paper ships SFT + slot-decode | — |

Expected RL wins: hedge policy, AURC/selective risk, bias, format. NOT raw SRCC where information is missing (encoder frozen).

## PHASE 6 — Paper mapping

- Headline: RT1 (+RT2 error bars) → faithfulness table + the abstention/selective-prediction figure set (risk-coverage, Mondrian map, ENCE).
- Ablations: RT3 adapter cluster; slot-decode two-row; NTL on/off (RT1 vs the F13-off smoke if needed); RT4 grounding (only if it survives).
- Retired/reframed: +0.077 → appendix historical note; grounding → single-seed convergence-speed observation unless RT4 + seeds say otherwise.
- Claims all cite the verified memos in `.claude/research/findings/`.

## Traceability check (nothing skipped)
- Audit A1→F17/RT1 · A2→S1,F3,F4,F5 · A3→S2,F10 · A4→S4(+RT2/RT4) · A5→S3,RT4 · A6→F8 · A7→RT2 · A8→S6,S8 · A9→Phase4-gate(+clean data in RT1) · A10→F11 · A11→F12 · A12→F9 · A13→F5,F6,F7
- Degeneration: stop-set→F1/V1 · backstop→F2/V2 · NTL→F13 · DITTO-identical-only→F14/V4-gated · nums-dominance→F16 (ablation later) · UL-is-dead-lever→recorded (no tuning) · epoch-climb test→S5 · loop-vs-CE→S5 · reduced-LoRA probe→V5 · 40-vs-30 bootstrap→S5
- Training-method: R1→F13 · R2→F18/V4 · R3→(FEATURE_SCALES verified; covered by F8 fixture) · R4→F15 · R5a→V5, R5b→S5, R5c→S7 · R6→RT5 · R7→F14 · R8→decision recorded (no diversification) · R9→Phase 5 · R10→Phase 4.5
- RL memo: reward rebuild→F19 · attribution→F20 · adversarial tests→F19 · gates→Phase 5 G1–G5 · GRPO config→RL-1 · random control→RL-2 · recalibration→RL-3 · rollout path→F21 · Hallucination-Tax mixing→RL-1 batch floor
