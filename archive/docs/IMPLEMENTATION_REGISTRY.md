<!-- AUTO-BANNER SUPERSEDED 2026-08-03 -->
> # ⚠️ SUPERSEDED — historical record, not a plan of action
>
> H2 marked 'RETRAIN to realize' -- realized. G3 'TODO' -- executed. H4 Mondrian 'runs post-H2' -- code exists and was run 2026-07-27/29: 0 of 15 cells certify. 'PSC cur_train is FLAT, reconcile before deploy' -- resolved; live tree is repo_verify.
>
> Authoritative current state: `.claude/research/STATE_2026-08-03.md`.
> Original text preserved unchanged below.

---

# AQUA-NL Implementation Registry (2026-07-12)

Single source of truth for every open item, from the investigation tasks and the (corrected + verified)
Fable research. Each card: **What / Now / Do / Expect / Why**. This is my consolidated call per item, not
a report comparison. Track against this; check items off as you go.

## SESSION LOG — 2026-07-12 (Fable audit + node-free build)

### ✅ DONE + TESTED this session (25 local unit tests pass, numpy/scipy venv)
- **`src/eval/faithfulness_metrics.py`** (+ `tests/test_faithfulness_metrics.py`, 19 tests) → **E3** bias, **E4** CCC (1/n moments), **E5** risk-coverage+AURC, **E6** E-AURC (oracle-ordering), **E7** AUGRC, **E9** Gaussian CRPS (1/√π constant, σ→0 = |y−μ|), **E10** Bland-Altman (Krouwer variant), **E11** Diebold-Mariano (**cluster-robust by speaker + Holm**), **E12** digit-drift (metric), σ-units helpers. Pure numpy, no torch.
- **`src/eval/mondrian_srlc.py`** (+ `tests/test_mondrian_srlc.py`, 6 tests) → **H4** Mondrian selective-risk-control. Clopper-Pearson binomial UCB (not Hoeffding), LTT grid-Bonferroni + cross-cell Bonferroni, **first-principles frozen ε_f = k·scale** (normalized units, no back-door tolerance), counting features **excluded from the taxonomy** (never hedged). Guarantee verified in-test (empirical miscoverage ≤ α on emitted set).
- **B2 propagation fixed** → `scripts/score_inference_vs_clean.py` (DEGENERATE=∅, snr kept, key `mean_reliable`), `scripts/bandfree_val_eval.py` (RELIABLE={snr,srmr,speaking_rate,pause_count,pause_rate}; dropped stale articulation_rate/overlap_ratio). B2 now fully propagated.
- **Stale 12→11 comments fixed** → `feature_set.py:1,5,54`, `adapter.py:29`, `reliability_head.py:10`.

### 🔧 AUDIT CORRECTIONS to the plan (wf_c4880704, 5 Fable agents — do NOT implement the old card text)
- **B3** needs **TWO masks**: hedge-aware for MSE + nums-CE ("na"); **presence-only (hedge-UNAWARE) for the NLL** — else masking hedged f0/jitter/etc. deletes exactly the overlap examples the σ-head must learn high-σ on (breaks the headline). Mask domain = ILL_POSED_UNDER_OVERLAP only (never counting feats), recomputed from CSV (overlap≥0.5), **not parsed from prose**. Needs a separate `nll_mask` arg in `compute_loss`.
- **H2** Stirn = `err=(target - mean.detach())/scales` in `heteroscedastic_nll`; **MUST keep lambda_mse>0** (MSE becomes the mean's only gradient; if 0 → garbage means → σ calibrates on garbage).
- **C2** there are NO per-feature blocks in the pooled vector (single `Linear(4096,2N)`); use contribution vectors `h_f=w_f⊙pooled`; **NEVER penalize head OUTPUTS** (would force apart genuinely-0.80-correlated predictions → worse accuracy).
- **D1/B5 (NEW BUG)** unlikelihood **silently DISABLES token-grounding**: `want_ntl=(λ_ntl>0 OR λ_ul>0)` and `if want_ntl…elif want_tg` starve `prose_hidden`. With λ_ul=0.1 in the corrected configs, the **parked M3b/M3c retrains train grounding OFF** (logged as a constant). Fix: return both ntl tensors + hidden from one forward. HIGH priority before any M3 retrain.
- **B4** `scripts/compute_clean_f0.py` ALREADY EXISTS (mix non-overlap frames) → new stem script must be a **new name** `compute_stem_f0.py` (whole-utterance s1 + octave protection); regen prose/nums/scalars together; re-score banked runs.
- **E5/E6/E7** existing `reliability_eval.aurc`/`hedging_calibration` use the **RETIRED tolerance** correctness → rewired to continuous scaled error in `faithfulness_metrics`. **σ-ordered versions BLOCKED on H2**; only the overlap-ratio proxy curve runs now.
- **E11** cluster-robust by SPEAKER (Libri2Mix shares speakers) + Holm across 11 feats — implemented.
- **E12** aux_mean NOT logged (`inference.py:106` discards `scalar_pred`) → code-now, **compute-when-nodes**.
- **E13** a linear/MLP probe is a **lower bound**, not a ceiling; needs PSC-resident WavLM caches (none local). Reword P10 accordingly.
- **F1/F2** the `normal_(0,0.01)` fix is **already APPLIED** (`adapter.py:95`) → **F2 = DONE**; F1 = post-fix training-log inspection only.
- **T1** 8×→4× is NOT one config line (strides hardcoded in `ConvCompressor.__init__`); needs a `compression_factor` kwarg through `build_adapter` + `get_output_length`.
- **H4 ε_f** = k·FEATURE_SCALES, frozen before calibration (in normalized units this is just k) — reconciles the guarantee with the band-free decision; add one paper sentence.

### ➕ NEW cards (were MISSING from the registry — audit completeness pass)
- **B6 voice-quality triad** (hnr/jitter/shimmer): re-added to supervised set 2026-06-24, classified ILL_POSED; clean-stem GT extraction gated behind `regenerate_corrected.py --voice` (positive-control gated). Status: 🟠 GT gated on X4.
- **RL/GRPO branch = DEFER**: current branch `redirect-decoupled-rl`; `src/training/sfs_reward.py` reward is SFS-F1 **with retired tolerances** → would optimize a dead objective. Revive only with (a) band-free continuous reward, (b) reward-hacking audit. Until then: legacy.
- **X3 external baselines** (`src/experiments/run_external_baseline.py`) — ⏸ BLOCKED(nodes), ahead of retrains per external review.
- **X4 positive control** — ⏸ BLOCKED(nodes); also gates B6's --voice GT.
- **σ-units convention**: ReliabilityHead log_var is NORMALIZED (per FEATURE_SCALES), mean is RAW. All σ-metrics operate in normalized units; raw σ = exp(0.5·log_var)·S_f. (Encoded in `faithfulness_metrics`/`mondrian_srlc`.)
- **P11** Cherian/Gibbs/Candès NeurIPS 2024 (nearest published conformal-in-text neighbor; novelty-delta = physical vs self-eval signal). **P12** Kuleshov 2018 σ-recalibration (repair path if E8 ENCE shows miscalibration). **P13** run the Qwen3 tokenizer 1-liner before the G1 design freeze.

### ✅ DONE + NODE-TESTED (2026-07-13, on compute node w003, 187 passed)
- **H2 Stirn** — `heteroscedastic_nll(stop_grad_mean=)` (mean.detach() in the σ-error term) + `train.py`
  wires `stirn_stop_grad` + guards `lambda_mse>0`. Tests: `test_stirn_hedge.py` (grad to mean is zero
  under Stirn, non-zero without; forward value unchanged). GREEN on node.
- **D1/B5 grounding-vs-UL fix** — `_ce_against_target` now composes BOTH ntl + hidden from one forward;
  caller has the `want_ntl and want_tg` branch; stale "not used together" comment fixed. `test_compute_loss_b_full`
  (the integration test) GREEN on node.
- **B3 two-mask** — `feature_set.hedge_mask()` + `build_nums_target(hedge=)` (ILL_POSED-only, overlap≥0.5,
  from CSV not prose) → `dataset` (__getitem__ + collate) → `compute_loss` MSE uses `gt_mask & ~hedge`,
  NLL keeps presence. Tests: `test_stirn_hedge.py` (never hedges recoverable; presence mask hedge-unaware).
  GREEN on node.
- **D4** — `sample_token` deterministic repetition-penalty + no-repeat-ngram (defaults = no-op), wired into
  `generate()`. `test_decode_penalty.py`. 
- **O3** — `test_ntl` (11, both asserts), `test_feature_tags` (12 tags + expected set + dropped set +
  the `qual` **source bug** in `build_cover_lines` fixed with format_map), `test_token_init` (23 special /
  19 open tags). `test_ckpt_selection` REMAINS (tied to your in-flight ckpt_selection edits — left alone).
- Earlier this cycle: `faithfulness_metrics.py`, `mondrian_srlc.py`, B2 propagation, stale comments — all green.

- **E12 aux_mean logging** — DONE (code): `generate()` captures the previously-discarded `scalar_pred`
  (+ `log_var` if a reliability head is present), stashes it under a reserved key, and the caller pops it
  into `output_entry["aux_mean"]`/`["aux_log_var"]`. py_compile-clean. The digit-drift NUMBER still needs
  an inference run on a σ-trained checkpoint (blocked on H2 retrain) — code-now, compute-when-nodes.

### ⏭ STILL TODO
- **B4 `compute_stem_f0.py`** — BLOCKED: the Libri2Mix s1/s2 stems are not on PSC (only the banked
  mix-non-overlap `clean_f0_*.json`), so it's write-only until the stems are located/re-synced. Not
  written yet (would risk inconsistency with the existing `f0_clean.py` pipeline without the data to test).
- **`test_ckpt_selection::test_guard_first_epoch...`** — the one remaining red test; tied to your in-flight
  ckpt_selection edits (`assert not ok` — the guard returns ok=True). Left untouched deliberately.

### CORRECTED critical-path line (supersedes the old "Everything in E … is node-free NOW")
Truly node-free NOW (code **and** result): E3, E4, E10, E11, G3, F1, O3, B2, stale-comments, H4(code), the whole `faithfulness_metrics`+`mondrian_srlc`. Code-now / result-when-nodes: E5/E6/E7 (σ version), E9, E12, B3, H2, B4, D4. Blocked on nodes to run: E13, X2, X3, X4, all retrains.

---

## SESSION LOG — 2026-07-13 (full empirical verification + grounded A/B retrain + f0 diagnosis)

### ✅ FULL EMPIRICAL VERIFICATION (user: "verify every single detail that works as expected") — all node-tested items confirmed FIRING end-to-end on PSC, not just unit-green
- **H2 σ-head/Stirn** — h2b smoke (wandb `aqua-nl-verify/ovkt02nj`): `train_loss_nll` 6.66→0.86, `reliability_sigma_mean` 1.0023→1.254, `loss_mse` 11.4→1.4 (all 113/113 nonzero, all LEARNING). NLL fires + Stirn split (mean via MSE, σ via stop-grad NLL) confirmed.
- **D1/B5 grounding** — m3b smoke: `train_loss_token_grounding` 64/64 nonzero 2.98→0.63. The earlier "grounding-off bug" was a **FALSE ALARM** — `snr_map_dir` IS a dict, grounding was firing; only the wandb LOG LINE was missing.
- **B3 two-mask** — verified on real corrected batches (hedge_mask stacks + splits MSE vs NLL paths).
- **E12 aux_mean** — `inference_results.json`: 3/3 entries carry 11-value `aux_mean`; `aux_log_var` correctly absent (no σ-head in m3b).
- **D4 rep-penalty** — applied end-to-end (inference ran w/ `repetition_penalty=2.0`); at 2.0 output degrades to foreign/full-width chars = over-penalization → use **1.1–1.2 + val sweep**. Inference-only, NOT in training.
- **srcc_robust** arithmetic = 0.6122 exact; **faithfulness_metrics + mondrian_srlc** = 19+6 unit tests + hand-checked arithmetic.

### 🐛 3 GAPS THE FULL VERIFICATION CAUGHT (unit-green but dead end-to-end; all FIXED)
- **grounding-logging** — `loss_token_grounding` computed but never logged → added `train_loss_token_grounding` (+`loss_tg_pooled/liu`) to wandb payload.
- **NLL-logging** — `loss_nll`+`reliability_sigma_mean` computed but never logged → added `train_loss_nll`+`train_reliability_sigma_mean`.
- **D4 config-wiring** — `generate()` supported `repetition_penalty` but the caller never passed it from config → wired `repetition_penalty`/`no_repeat_ngram_size` into the `generate()` call.

### 🚀 GROUNDED A/B RETRAIN — launched 2026-07-13 into NEW wandb project `aqua-nl-grounded`
- **m3b_grounded** (token_grounding=0.1, run `hpdpxprt`) vs **b1_grounded** (0.0 control, run `nqwed752`); identical otherwise (film-mamba, aux-MSE 0.3, UL 0.1, max_target 512, processed_corrected + descriptions_corrected, ep12).
- srcc_robust TREND (headline, m3b>b1 as hypothesized): m3b ep1 0.248 / ep2 0.525 / ep3 **0.594** (prior corrected peaked 0.612 ep4); b1 ep1 0.226 / ep2 0.430. **Grounding delta +0.096 at ep2 — exceeds the prior +0.077.**
- OPS: m3b CUDA-hung mid-epoch (GPU 100%, iter frozen) → `pkill -f config.m3b.grounded` + resume from last.pt cleared it; b1 walled → resumed on fresh node w006. Both resume-safe from last.pt; best-tracker OK (no reset on resume). Babysit `/loop` running.

### 🔬 f0 DEGRADATION DIAGNOSIS (user flagged "f0 got worse, esp f0 st"; workflow `wf_cd3d9ac1`, adversarially CONFIRMED)
- **VERIFIED**: f0_sd val SRCC collapses (m3b 0.29→0.15→0.09, worse than b1 0.29→0.20); f0_mean declines in m3b (0.34→0.28) while b1 RISES (0.39→0.52). f0_sd has ALWAYS been near-0 (all prior runs −0.14 to 0.19).
- **NOT a GT mismatch** (the bug I suspected, RULED OUT adversarially): val f0 GT is parsed from the SAME prose target text as the prediction (`train.py:2126-2186`) — no CSV in the val path → a clean/noisy column name **cannot** corrupt the metric. Self-consistent by construction.
- **Actual cause = f0 mode-collapse + grounding capacity reallocation** (expected trade-off, not a scorer bug): f0 is ILL_POSED, hedged out on overlap≥0.5 (~78% clips) → little gradient; prose-CE dominance → emit-the-mode → variance collapse → SRCC→0. Confirmed empirically: `coverage_f0_sd` stays 0.62–0.86 (model EMITS values, so NOT abstention/parse-miss) while `nmae_f0_sd` RISES 0.86→0.90 → near-constant predictions. `token_grounding` backprops into the SHARED adapter prefix (`train.py:955-988`, SNR@idx0) → reshapes it toward SNR structure, starving the fragile pitch signal → m3b f0 < b1 f0. Robust-UP/pitch-DOWN = capacity-reallocation signature.
- **H2 CONNECTION**: H2 is **OFF** in these runs (`reliability_head:false`, `lambda_nll:0` → aux path byte-identical to plain MSE), so β-NLL/Stirn are NOT the cause. But H2 is the **designed fix**: naive NLL would WORSEN f0 (attenuate the high-σ mean = the same "lazy on hard points" pathology, `reliability_head.py:118-123`); β-NLL (β=0.5, "recommended for F0 under overlap" per the docstring) + Stirn `stop_grad_mean` (mean stays MSE-faithful) counteract it, and the σ-head enables report-f0-via-abstention (the headline framing). Guard already in place: `stirn_stop_grad` requires `lambda_mse>0`.
- **LATENT BUG to harden** (doesn't affect the val metric, but fragile): `feature_set.py:44` maps `f0_sd→f0_sd_hz` (noisy-NAMED column) relying on it being overwritten in-place with clean values by `make_clean_f0_csv.py`. If a config points at a raw-mixture CSV, aux-MSE/nums silently train noisy f0. FIX: read `f0_sd_hz_clean` explicitly + fail-loud if absent; assert `descriptions_path_val` is the clean build.
- **PAPER TAKEAWAY**: f0 is genuinely ill-posed on 78%-overlap Libri2Mix → report via abstention/coverage NOT SRCC (already excluded from the selector). The grounding robust-up/pitch-down trade-off is an honest reportable finding. Natural next run: **m3b_grounded + H2** (`reliability_head:true`, `lambda_nll>0`, `beta 0.5`, `stirn_stop_grad`) to combine grounding (robust) with σ-abstention (f0).

---

## Status legend
- ✅ DONE — implemented + verified
- 🟡 PARTIAL — some sub-parts done, some open
- ⬜ TODO-FREE — code/experiment that needs NO GPU (do while nodes are down)
- 🟠 TODO-RETRAIN — needs a training run to realize/validate
- ⏸ BLOCKED — waiting on nodes to RUN (code may be ready)
- 📄 PAPER — writing/framing, no code
- ⛔ SKIP — evaluated and rejected (kept so we don't re-litigate)

## Summary (counts)
| Workstream | Items | Done | Node-free TODO | Retrain | Paper | Skip |
|---|---|---|---|---|---|---|
| B Bugs/fixes | 5 | 2 | 3 | – | – | – |
| H Headline abstention | 7 | 2 | 3 | 2 | – | – |
| E Evaluation scaffolding | 13 | 3 | 9 | 1 | – | – |
| G Numeric generation | 6 | – | 1 | 2 | – | 3 |
| D Degeneration | 5 | 1 | 1 | 2 | – | 1 |
| C Confounds/decorrelation | 6 | 1 | – | 3 | 1 | 1 (2 skips) |
| F Conditioning/FiLM | 3 | – | 1 | – | 1 | 1 |
| T Token rate | 2 | – | – | 1 | 1 | – |
| X Cross-domain/AMI | 2 | – | 1 | – | – | – (1 blocked) |
| P Paper framing | 10 | – | – | – | 10 | – |
| O Ops/cleanup | 4 | 2 | 1 | – | – | – (1 blocked) |

**Critical path (headline):** H2 β-NLL+Stirn (retrain) → unlocks H4 Mondrian calibration + E8 ENCE + H6 hedge verbalization.
Everything node-free (A-list) can be done NOW without waiting.

---

## B — Bugs & fixes

### B1 FEATURE_SCALES miscalibration — ✅ DONE
- **What**: aux-MSE per-feature normalizers were miscalibrated to the corrected data (effective weights 0.33–4×), over-weighting snr/f0/hnr, under-weighting pause_count.
- **Now**: fixed to MAD-based scales (1.4826·MAD) in `src/data/feature_set.py`; pushed to PSC; tests green.
- **Do**: nothing.
- **Expect**: aux-MSE gradient balanced across features; already in effect for the next retrain.
- **Why**: measurement-noise-floor normalization is first-principles, not tuned.

### B2 DEGENERATE selection feature stale — ✅ DONE
- **What**: checkpoint selector excluded `snr` from the headline metric; snr is actually learnable (~0.9), and the real leak is `overlap_ratio` (a FiLM conditioning input).
- **Now**: flipped to `DEGENERATE_SELECTION_FEATURES = {overlap_ratio}` in `src/eval/selection_metric.py`; tests updated + green.
- **Do**: nothing.
- **Expect**: selector headline includes snr, excludes the leaked overlap_ratio → honest selection.
- **Why**: snr contributes real signal; overlap_ratio is copyable from the input.

### B3 Hedge-only-on-prose bug — ✅ DONE + NODE-TESTED (two-mask)
- **What**: the overlap hedge is applied ONLY to the prose target; the aux-MSE, nums-CE, and token-grounding targets read the UN-hedged CSV, so they train the noisy F0 on hedged clips (contradicts the abstention story).
- **Now**: FIXED (2026-07-13). `hedge_mask()` + `build_nums_target(hedge=)` (ILL_POSED-only, overlap≥0.5, from CSV) → `dataset` → `compute_loss` MSE uses `gt_mask & ~hedge`, nums emits "na"; NLL keeps presence (hedge-unaware, so the σ-head still sees hard clips). `test_stirn_hedge.py` green on node. CAVEAT: aux-MSE reads the noisy-NAMED `f0_sd_hz` column (clean-substituted in-place) — see the 2026-07-13 f0 diagnosis for the fail-loud hardening TODO.
- **Do**: thread the per-clip/per-feature hedge mask into `extract_scalars` (mask=False on hedged features) and `build_nums_target` (emit "na" on hedged features), so ALL three numeric channels agree with the prose.
- **Expect**: aux head + nums channel stop learning noisy F0 on overlap clips; σ-head (H1/H2) gets a cleaner "unmeasurable here" signal; F0 SRCC on non-overlap should hold or improve.
- **Why**: consistency across the 3 numeric channels is a prerequisite for the abstention head to mean anything.

### B4 f0 clean-stem GT regeneration — ⬜ TODO-FREE (write) / ⏸ (run)
- **What**: F0 GT was computed by Praat on the REVERBERANT MIX non-overlap frames → sparse (~17% frames), 26% of f0_sd values implausible (>60 Hz) = octave errors.
- **Now**: not done. GT still mix-derived. Script pending.
- **Do**: write `compute_clean_f0.py` — Praat pitch on the clean s1 stem over the WHOLE utterance + octave-jump protection (pitch-range clamp / median-continuity), then merge into `descriptions_corrected.json`. Run on a node.
- **Expect**: dense, octave-clean F0 GT; f0_sd implausible-rate → near 0; F0 SRCC ceiling rises; the hedge becomes about observability, not GT noise.
- **Why**: you cannot separate "model can't recover F0" from "GT is garbage" until the GT is clean-stem.

### B5 Known train.py bugs (best-tracker / M3c repetition) — ✅ DONE (verify provisional)
- **What**: best-tracker reset on resume corrupted best.pt; M3c late-epoch repetition.
- **Now**: best-tracker FIXED (per external review); M3c repetition provisionally addressed.
- **Do**: nothing new; keep an eye on rep_n_max trend on the next M3c run.
- **Expect**: best.pt tracks the true best across resumes.
- **Why**: already banked; listed for completeness.

---

## H — Headline: observability-aware abstention (this IS original item 4 / "#2")

### H1 Heteroscedastic σ-head (μ, log σ²) — ✅ DONE
- **What**: aux head emits per-feature mean AND log-variance; σ = the learned "this feature is unreliable here" signal.
- **Now**: `ReliabilityHead(d→2·N)` exists in `src/model/reliability_head.py`; default-off.
- **Do**: nothing (H2 turns it on).
- **Expect**: per-feature σ available at inference once trained.
- **Why**: σ is the substrate the whole abstention guarantee consumes.

### H2 β-NLL + Stirn stop-gradient — ✅ CODE DONE + NODE-VERIFIED (β-NLL + Stirn firing); 🟠 RETRAIN to realize
- **What**: train the σ-head with β-NLL (Seitzer 2022, β=0.5) so the gradient isn't down-weighted 1/σ² on high-σ overlap frames; Stirn (2023) stop-gradient so the σ-head can't degrade the mean that SRCC/nMAE score.
- **Now**: `beta` + `stop_grad_mean` params in `heteroscedastic_nll`, wired in `train.py` (`stirn_stop_grad`, guard `lambda_mse>0`). VERIFIED FIRING on the h2b smoke (2026-07-13, `aqua-nl-verify/ovkt02nj`): loss_nll 6.66→0.86, sigma_mean 1.0023→1.254, loss_mse 11.4→1.4. Not yet on in a HEADLINE retrain (the grounded A/B has `reliability_head:false`). **Now also the candidate fix for the f0_sd collapse** (see 2026-07-13 log).
- **Do**: (a) add the Stirn stop-gradient (detach the mean path when computing the σ loss term); (b) set `reliability_head:true, lambda_nll>0, beta_nll:0.5` in a config arm; (c) retrain the σ-head.
- **Expect**: σ becomes a *ranking-faithful* reliability signal (high on overlap-F0, low on snr); fixes the recorded "worse-than-constant-mean on high-σ features"; mean accuracy protected.
- **Why**: THE single best scarce retrain — it upgrades the signal every downstream abstention piece depends on. Plain NLL would give a poor σ.

### H3 Zaoui optimality justification — ✅ DONE (framing)
- **What**: thresholding conditional variance is the Bayes-optimal reject rule for regression (Zaoui 2020).
- **Now**: cited; turns our σ-threshold from heuristic into "the optimal rule, uncalibrated."
- **Do**: use in the paper (see P).
- **Expect**: reviewers can't call the σ-threshold ad-hoc.
- **Why**: retroactively justifies the shape of the hand rule we're replacing.

### H4 Mondrian selective-risk-control calibration — ⬜ TODO-FREE (runs post-H2)
- **What**: turn the σ-threshold into a per-(feature × overlap-bin) finite-sample guarantee. Cells = feature f × overlap-bin g (none/partial/heavy). Emit claim f iff σ_f ≤ λ_{f,g}; pick λ per cell so selective risk P(nMAE_f>ε_f | emit) ≤ α, via Learn-then-Test (FWER over ~11×4 cells) + Pareto Testing, CRC fallback for starved cells.
- **Now**: not written. Current abstain = hand rule (recoverability<0.20).
- **Do**: write `src/eval/mondrian_srlc.py` — bin overlap; per cell run LTT/Pareto to choose λ; output the abstention map. Post-hoc on dev-split σ outputs; NO retrain (but needs a σ-trained checkpoint from H2 to be non-trivial).
- **Expect**: the theorem — "w.p. ≥1−δ, for every feature f and overlap bin g, P(claim-error/scale>ε_f | emitted, bin g) ≤ α." Counting features stay outside the taxonomy (mix-measured GT), un-hedged.
- **Why**: converts the headline from a magic number into a distribution-free guarantee; node-free once σ exists.

### H5 Honesty ledger for the guarantee — 📄 PAPER
- **What**: state the guarantee's limits: marginal within a bin (per-clip conditional coverage is impossible distribution-free, Barber 2021); (1−δ) over calibration not a.s.; AMI needs recalibration (no retrain); F0×heavy-overlap = "abstains by construction."
- **Now**: not written.
- **Do**: one paragraph in the eval/limitations section.
- **Expect**: pre-empts the obvious reviewer attacks on the guarantee.
- **Why**: honesty is cheaper than a rebuttal.

### H6 Verbalize the guarantee (Mohri–Hashimoto back-off) — 🟠 RETRAIN (ties to G1)
- **What**: a non-emitted claim becomes a hedge sentence; back-off lifts the per-claim guarantee to the whole description.
- **Now**: hedging is prose-only + heuristic.
- **Do**: at decode, when the Mondrian map says abstain on f, emit the hedge clause instead of the number (implemented cleanly as the "abstain" branch of the slot decode, G1).
- **Expect**: description-level guarantee; hedging and numeric-fill become one discrete decision per slot.
- **Why**: makes the abstention *appear in the generated text*, which is the novelty.

### H7 Aleatoric = observability framing — 📄 PAPER
- **What**: overlap-masked F0 is textbook input-dependent aleatoric uncertainty (Kendall & Gal 2017); Nix & Weigend 1994 for the head lineage.
- **Now**: not written.
- **Do**: framing sentence.
- **Expect**: legitimizes "observability = heteroscedastic noise."
- **Why**: the setup is unusually favorable (F0 GT from clean stems, input is the mix).

---

## E — Evaluation scaffolding (band-free, mostly node-free on existing inference_results.json)

### E1 SRCC per feature — 🟡 PARTIAL
- **What**: Spearman rank corr of emitted vs measured GT, per feature (no tolerance).
- **Now**: `srcc_robust` tracked; band-free scorer exists (`scripts/score_matched_test.py`).
- **Do**: confirm it emits per-feature SRCC for all 11 (not just the 5 robust) + the headline `srcc_robust`.
- **Expect**: the primary faithfulness number.
- **Why**: rank correlation needs no tolerance — the whole reason SFS-precision was retired.

### E2 nMAE per feature — 🟡 PARTIAL
- **What**: scale-normalized mean abs error.
- **Now**: believed present in band-free scorer; verify.
- **Do**: confirm/emit per-feature nMAE.
- **Expect**: magnitude companion to SRCC (SRCC is invariant to monotone distortion).
- **Why**: SRCC alone misses "2× true SNR every clip."

### E3 bias (signed mean error) column — ⬜ TODO-FREE
- **What**: signed mean error per feature — completes the ACE triple (bias/error/ρ).
- **Now**: not reported.
- **Do**: add a bias column to the scorer.
- **Expect**: exposes systematic offset (e.g., always-high SNR).
- **Why**: ACE precedent; one line, you already have the data.

### E4 CCC (Lin) column — ⬜ TODO-FREE
- **What**: concordance corr = correlation with the 45° line; penalizes scale/location shift + scatter in one bounded [−1,1] number.
- **Now**: not reported.
- **Do**: add CCC per feature (few lines numpy).
- **Expect**: fixes SRCC's blind spot; AVEC-style headline-grade agreement stat.
- **Why**: the citable "measurement instrument vs reference" statistic.

### E5 Risk-coverage curve + AURC — ⬜ TODO-FREE (needs σ for the sweep var)
- **What**: sweep the abstention threshold; plot risk (nMAE among emitted) vs coverage; summarize as AURC.
- **Now**: not built.
- **Do**: per-feature RC curve + AURC; the operating point the model actually chooses as a dot.
- **Expect**: THE abstention figure (your planned AURC plot).
- **Why**: the standard selective-prediction summary; reviewers in 2026 expect it.

### E6 E-AURC — ⬜ TODO-FREE
- **What**: AURC minus the oracle-ordering AURC → unitless, cross-model comparable.
- **Now**: not built.
- **Do**: add alongside AURC.
- **Expect**: lets you compare M3b vs B1 abstention quality fairly.
- **Why**: raw AURC isn't comparable across models.

### E7 AUGRC — ⬜ TODO-FREE
- **What**: area under the *generalized* RC curve (Traub 2024) — fixes AURC's monotonicity flaw.
- **Now**: not built.
- **Do**: add as the robust variant (same curve).
- **Expect**: a 2026-reviewer-aware metric that can't rank a worse system above a better one.
- **Why**: cheap metric hygiene.

### E8 ENCE + σ coefficient-of-variation — ⬜ TODO-FREE (needs σ from H2)
- **What**: Expected Normalized Calibration Error — bin by predicted σ, compare per-bin RMSE to mean σ; + CoV that σ actually varies.
- **Now**: not built (and σ isn't trained yet — needs H2).
- **Do**: histogram diagnostic once a σ-checkpoint exists.
- **Expect**: proves the hedge signal is trustworthy (hedges where error would be large), not just present.
- **Why**: the field-standard heteroscedastic-σ calibration check.

### E9 CRPS (Gaussian, closed-form) column — ⬜ TODO-FREE (optional, needs σ)
- **What**: a single proper score rewarding accuracy AND honest σ; can't be gamed by over-hedging or over-claiming.
- **Now**: not built.
- **Do**: closed-form Gaussian CRPS per feature.
- **Expect**: one number that summarizes the sharpness/calibration tradeoff.
- **Why**: the proper-scoring-rule anchor (optional but strong).

### E10 Bland–Altman plot — ⬜ TODO-FREE
- **What**: per-clip error vs GT magnitude, for 2–3 headline features.
- **Now**: not built.
- **Do**: appendix figure.
- **Expect**: communicates bias vs scatter + range-dependence; reads as statistical maturity.
- **Why**: the method-comparison convention.

### E11 Diebold–Mariano paired test (M3b vs B1) — ⬜ TODO-FREE
- **What**: paired loss-differential significance test on per-feature nMAE.
- **Now**: you report bootstrap CIs.
- **Do**: add the DM-style paired test alongside.
- **Expect**: canonical forecast-comparison significance for "M3b beats B1."
- **Why**: reviewer-standard; light touch.

### E12 Digit-drift metric |emitted − aux_mean| — ⬜ TODO-FREE
- **What**: distance between the number the LM wrote and the aux head's estimate — direct measurement of the digit-dilution problem.
- **Now**: not computed (need aux_mean logged at inference).
- **Do**: log aux_mean in `inference.py`; compute drift from existing/next `inference_results.json`.
- **Expect**: a figure quantifying Problem A before/after slot-decode (G1).
- **Why**: turns "digits are diluted" from a claim into a number.

### E13 WavLM probe-ceiling — ⬜ TODO-FREE (node-light)
- **What**: train a linear/MLP probe directly on frozen WavLM features → per-feature SRCC = the recoverability ceiling (why the topline isn't 1.0).
- **Now**: not built. (This answers investigation Q2.)
- **Do**: probe script over cached WavLM features; report model SRCC vs probe ceiling (and/or as % of it).
- **Expect**: honest "topline = frozen-encoder information ceiling, not 1.0" figure; separates encoder limits from model failure.
- **Why**: ACE "correlation exposes what's recoverable"; pre-empts "why isn't SRCC 1.0."

---

## G — Numeric generation (secondary novelty)

### G1 Verified-slot decode (NBT sentinels + aux splice + variance-gated hedge) — 🟠 RETRAIN (headline generation graft)
- **What**: add `<NUM:feat>` sentinel tokens; verbalizer replaces numerals with sentinels; LM learns placement only; at decode splice `format(aux_mean[f])`, or if the Mondrian map (H4) says abstain, emit the hedge clause.
- **Now**: not built. LM generates digits directly (diluted).
- **Do**: (a) add 11 sentinel tokens (reuse section_head row-mask machinery); (b) verbalizer regex-swap; (c) `train.py` prose-CE supervises structure only; (d) `inference.py` splice; (e) per-feature GATE — slot only features where aux ≥ LM on val SRCC.
- **Expect**: emitted number == aux estimate (no dilution); SRCC/nMAE on slotted features track the aux head; hedging becomes one discrete slot decision (fuses with H6).
- **Why**: removes digit-tokenization dilution + makes numbers faithful-by-construction. HONEST TRADEOFF: SRCC then measures the aux head (report as free-decode vs slot-filled, G3).

### G2 A1 self-context copy (ablation arm) — 🟠 RETRAIN (same batch as G1)
- **What**: prepend a machine-readable measurement block ("measured: snr=… hnr=…"); train values = GT+scheduled-noise, inference values = aux means; LM digit emission becomes a copy task (no new tokens).
- **Now**: not built.
- **Do**: data-prep + inference change; run as the second arm of the G1 retrain.
- **Expect**: near-exact copying without sentinels; the honest ablation baseline for G1.
- **Why**: cheap, no new tokens, isolates "copy" vs "sentinel" gains.

### G3 Free-decode vs slot-filled 2-row main table — ⬜ TODO-FREE (eval design)
- **What**: report both regimes: learned faithfulness (current) vs constructional (G1). Decompose slot eval into (a) value accuracy = aux SRCC/nMAE, (b) selection/hedging quality, (c) digit-drift (E12).
- **Now**: not designed.
- **Do**: define the table + the 3 decompositions.
- **Expect**: owns the faithfulness tradeoff instead of hiding it.
- **Why**: Rudin/Jacovi-Goldberg — the reduction is the point (LM's job relocates to placement + hedging, which IS the novelty).

### G4 Kosmos-style overlap-span tokens — 🟠 RETRAIN (low priority, after G1)
- **What**: `<t_k>` time-bin tokens for overlap spans → exact, IoU-scorable, no digit generation for time.
- **Now**: spans are prose seconds parsed by regex.
- **Do**: add time-bin tokens (bin ≥160 ms = frame period); verbalizer + decode + IoU scoring.
- **Expect**: overlap grounding generated + IoU-checkable by construction.
- **Why**: composes with G1; but bin ≥ frame period, so it's another argument for T1.

### G5 SKIP: xVal / R2L / pix2seq-bins — ⛔ SKIP
- **What**: alternative numeric encodings.
- **Now**: n/a.
- **Do**: nothing. xVal = workshop-only + arch-mismatched to frozen LoRA; R2L = Qwen is already single-digit (the good regime); pix2seq-bins = Shikra's ablation shows numeric text beats learned bins in our regime.
- **Expect**: G1 achieves the faithfulness goal more cleanly.
- **Why**: verified dead ends; don't spend a retrain.

---

## D — Text degeneration (templated-repetition climb)

### D1 Unlikelihood (token-level) — ✅ DONE (keep)
- **What**: push down repeated-token probability; already in the loss (`lambda_unlikelihood=0.1`).
- **Now**: active.
- **Do**: keep token-level; do NOT invest in sequence-level (needs decode-in-loop; D2 supersedes).
- **Expect**: mild anti-repetition.
- **Why**: cheap, already there.

### D2 DITTO — 🟠 RETRAIN (bundle)
- **What**: manufacture pseudo-repetitive data, train P(repeat) to decay exponentially (Xu 2022) — targets greedy sentence loops directly.
- **Now**: not built.
- **Do**: data-side + one loss term in `compute_loss`; bundle into the H2/G1 retrain.
- **Expect**: rep_n_max flat over epochs; no perplexity cost.
- **Why**: best-matched training-time fix for our exact pathology (greedy + late-epoch loops on templated numeric reports).

### D3 Target diversification — 🟠 RETRAIN (data + bundle)
- **What**: 3–5 paraphrases per clip (varied order/phrasing, constrained to the SFS-parser grammar, no colon form), sample one per epoch.
- **Now**: single canonical template — which Repetition-In-Repetition-Out says MANUFACTURES the degeneration.
- **Do**: generate paraphrases with local Ollama; dataloader samples one/epoch.
- **Expect**: higher target entropy → less repetition; matters MORE if G1 sentinels land (numbers no longer anchor CE).
- **Why**: attacks the root cause (low-entropy targets), not just the symptom. NOTE: do NOT use gemma/Ollama for the numeric VALUES — only for surface phrasing of an already-correct target.

### D4 Contrastive search decoding sweep — ⬜ TODO-FREE
- **What**: deterministic decoding (argmax over top-k minus a degeneration penalty) — compatible with reproducible paper numbers; zero-retrain.
- **Now**: greedy only.
- **Do**: add the option to `inference.py`; sweep the penalty α on val, check SRCC doesn't move (templated reports legitimately repeat structure).
- **Expect**: a decoding backstop; adopt only if greedy shows measurable late-clip repetition and SRCC is unharmed.
- **Why**: cheapest experiment on the list; runs on existing checkpoints.

### D5 SKIP: label smoothing / sequence-level UL — ⛔ SKIP
- **What**: label smoothing as a degeneration fix; sequence-level unlikelihood.
- **Now**: n/a.
- **Do**: nothing. Riley & Chiang: LS has ~no effect on repetition; uniform LS is wrong for digits. Seq-level UL needs decode-in-loop; DITTO supersedes.
- **Expect**: no loss from skipping.
- **Why**: verified non-fixes.

---

## C — Feature confounds / decorrelation (SRMR↔f0 0.80; SNR↔temporal)

### C1 RIR + independent-SNR augmentation — ✅ DONE (primary fix)
- **What**: adds reverberation + independent noise to break the anechoic SRMR≈f0 shortcut in the INPUT distribution.
- **Now**: in the pipeline.
- **Do**: nothing (it's the causal fix; C2/C3 are complements).
- **Expect**: SRMR stops being a pure pitch proxy.
- **Why**: Geirhos shortcut-learning's recommended intervention.

### C2 VICReg block-cross-covariance loss — 🟠 RETRAIN (cheap, bundle)
- **What**: penalize `(cross-cov(h_srmr_block, h_f0_block))²` on the pooled adapter vector — the exact "SRMR must not covary with f0" statement.
- **Now**: not built.
- **Do**: ~5 lines in `compute_loss`, `lambda_decorr~1e-2`; bundle into a retrain.
- **Expect**: reduced residual SRMR↔f0 correlation; watch that SRMR's OWN SRCC doesn't drop (a decorrelation that tanks SRMR is a loss).
- **Why**: kills the (mostly linear) dependence augmentation misses in the representation. Cheap to try.

### C3 HSIC escalation — 🟠 RETRAIN (conditional)
- **What**: kernel independence penalty for NONLINEAR dependence (SNR↔temporal).
- **Now**: not built.
- **Do**: ~10-line RBF V-statistic; only if C2 leaves nonlinear residual dependence.
- **Expect**: removes nonlinear coupling covariance misses; needs batch ≥64.
- **Why**: the "linear→nonlinear decorrelation ladder" second rung.

### C4 Concept Whitening — 🟠 RETRAIN (conditional, high ceiling)
- **What**: whiten+rotate the aux-head input so each feature gets an orthogonal axis — SRMR⊥f0 BY CONSTRUCTION.
- **Now**: not built.
- **Do**: replace the norm feeding the aux head with a CW module; one retrain.
- **Expect**: axis-aligned interpretable features; cleaner paper sentence than "added a covariance penalty."
- **Why**: highest ceiling but changes the adapter graph — the "if I get one architectural retrain for decorrelation" bet.

### C5 SKIP: DANN gradient-reversal — ⛔ SKIP
- **What**: adversarially scrub f0 from the SRMR pathway.
- **Do**: nothing. Elazar & Goldberg: adversarial removal is LEAKY (a fresh probe recovers the "removed" info); min-max is unstable under a scarce-retrain budget.
- **Why**: statistical-dependence penalties (C2/C3) are certifiable; adversarial removal isn't. Cite as the reason we chose penalties.

### C6 SKIP: IRM — ⛔ SKIP
- **What**: invariant risk minimization across augmentation "environments."
- **Do**: nothing. Rosenfeld 2021: IRM fails with few/synthetic environments — exactly our regime.
- **Why**: cite as the principled ideal + why we used direct penalties instead.

---

## F — Conditioning / FiLM

### F1 Re-examine the FiLM "bug" diagnosis — ⬜ TODO-FREE (cheap investigation)
- **What**: the "permanent dead gradient" claim is WRONG — the projection weights get gradient at step 0 (∝ audio⊗overlap_embed) and move off zero, so the input-gradient goes live at step 1. Any concat>FiLM gap is a one-step-delay artifact or a DIFFERENT cause.
- **Now**: bug assumed real; `normal_(0,0.01)` patch proposed.
- **Do**: inspect a training log — do the overlap-projection weights (γ/β) actually move early? Is there a real, persistent concat>FiLM gap after the fix?
- **Expect**: either "the gap is real, cause X" or "the gap was a transient" — decides whether F is worth ANY effort.
- **Why**: don't spend a retrain on a mis-diagnosed bug (adversarial verify V4 refuted the adaLN-Zero "fix").

### F2 FiLM init fix — ⛔ SKIP (adaLN-Zero refuted) / keep normal_(0,0.01)
- **What**: adaLN-Zero was claimed to fix the init "with live gradient" — REFUTED: zeroing the output projection severs the input-gradient at init too; it REPRODUCES the bug. identity-at-init vs live-input-gradient-at-init is a TRADEOFF.
- **Now**: current init is the "bug" form; `normal_(0,0.01)` is the only patch that gives a live input gradient at init (at the cost of identity).
- **Do**: keep `normal_(0,0.01)` IF F1 shows a real gap; otherwise leave it. Do NOT adopt adaLN-Zero believing it dominates.
- **Expect**: at most a marginal, one-step-delay effect.
- **Why**: verified; don't oversell in the paper.

### F3 Keep FiLM (don't switch to cross-attention) — 📄 PAPER
- **What**: overlap context is per-frame TIME-ALIGNED, so FiLM is the minimal correct operator; cross-attention/Perceiver are for UNALIGNED context.
- **Now**: FiLM is the design.
- **Do**: framing + cite Personal VAD 2.0 (FiLM>concat, same domain) + SpeakerBeam + DiT (adaLN-Zero beat cross-attention).
- **Expect**: defends FiLM-over-concat (your concat>FiLM result was the init, not architecture).
- **Why**: same-domain evidence; cheap paper win.

---

## T — Token rate

### T1 Conv 8× → 4× (12.5 Hz) — 🟠 RETRAIN (second retrain, counting)
- **What**: halve temporal compression → 12.5 Hz / 80 ms tokens = field consensus (Voxtral/Kimi/Phi-4/Moshi).
- **Now**: 6.25 Hz / 160 ms; counting features weakest (speaking_rate SRCC ~0.26).
- **Do**: one config line (conv stride 8→4); retrain; plot counting/rate accuracy vs frame rate.
- **Expect**: improved speaking_rate / pause SRCC; the FIRST result tying audio-LLM counting to encoder frame rate. Cost ~2× audio-prefix length (NOT 2× full compute).
- **Why**: targets the counting deficit NO loss/decorrelation/FiLM fix can touch. Frame as UNDERSAMPLING + peer consensus, NOT a Nyquist theorem (the "8 Hz floor" was mislabeled; each 160 ms token is a summary so detection survives, counting aliases).

### T2 Token-rate paper framing — 📄 PAPER
- **What**: modulation spectrum peaks ~5 Hz (Ding 2017), intelligibility to ~12 Hz (Elliott & Theunissen 2009); 6.25 Hz undersamples; 12.5 Hz is ASR-validated.
- **Do**: framing paragraph + the SED-resolution split (tagging robust, counting scales with rate).
- **Expect**: principled "why counting is weak" story.
- **Why**: turns a weakness into an analyzed contribution.

---

## X — Cross-domain / AMI

### X1 Head-only deep ensembles (epistemic) — 🟠 RETRAIN-ish (small) / ⬜ code
- **What**: 5 copies of ONLY the ReliabilityHead (different seeds); epistemic = Var over heads' means; total = aleatoric σ² + epistemic → feed the Mondrian threshold (H4).
- **Now**: not built. σ (H1) captures aleatoric only.
- **Do**: instantiate 5 heads; combine uncertainties; report head-disagreement.
- **Expect**: catches OOD "confidently wrong" on AMI that σ misses; ~zero extra compute.
- **Why**: the external-review AMI priority; cheap complement to σ. Limit: shared trunk under-estimates if the adapter itself is OOD.

### X2 AMI eval + Mondrian recalibration — ⏸ BLOCKED (nodes/data)
- **What**: run the pipeline on AMI; recalibrate Mondrian λ on an AMI calibration split (NO retrain).
- **Now**: pending (external-review priority).
- **Do**: AMI features → inference → recalibrate → report coverage/risk under domain shift.
- **Expect**: the cross-domain abstention story; recalibration-without-retrain is itself a selling point.
- **Why**: closes the pending cross-domain claim.

---

## P — Paper framing (no code)

- **P1** ACE Challenge as the same-field anchor (bias/error/ρ vs measured GT) — CAVEAT: ACE uses PEARSON, present SRCC as our rank adaptation.
- **P2** Gneiting "maximize sharpness subject to calibration" — the normative statement of the hedging goal.
- **P3** Thomson & Reiter contrast — the CITED reason we retired the tolerance/band-F1 metric (binary claim-matching → continuous grading removes tolerance arbitrariness).
- **P4** Rudin + Jacovi-Goldberg — the faithfulness-by-construction vs post-hoc tradeoff (frames G3).
- **P5** Novelty as a 4-way COMBINATION (attribute-scoped + inline + measurement-triggered + physics-grounded); cite the 6 nearest neighbors; claim the COMBINATION only.
- **P6** QualiSpeech + SpeechQualityLLM contrasts — novelty rests on PHYSICAL-vs-perceptual GT (SpeechQualityLLM already parses-and-scores, but perceptual).
- **P7** MLLM-IQA novelty-boundary shield (Q-Align/DepictQA emit ratings/classes, not measured numbers).
- **P8** CCC + Bland-Altman "model as measurement instrument vs reference instrument" framing.
- **P9** OmniPred / Decoding-based Regression — the text-as-regressor license.
- **P10** WavLM probe-ceiling framing (topline = information ceiling, not 1.0) — pairs with E13.

---

## O — Ops / cleanup

### O1 src/ subpackage reorg — ✅ DONE
- Committed 3beeca1, pushed; 728 tests collect clean.

### O2 Corrected + verified Fable research — ✅ DONE
- `fable-corrected-2026-07-11.md` + this registry.

### O3 Fix 7 stale voice-patch tests — ⬜ TODO-FREE
- **What**: `test_ntl`, `test_feature_tags`×4, `test_token_init`, `test_ckpt_selection` hardcode the old 12-feat/9-tag/19-tok reality; pre-existing debt (NOT caused by the reorg).
- **Do**: update to the 11-feature reality (careful with test_ckpt_selection — tied to your in-flight edits).
- **Expect**: green unit suite.
- **Why**: hygiene; low risk except ckpt_selection.

### O4 PSC deploy of the reorg — ⏸ BLOCKED (until parked retrain finishes)
- **What**: PSC `cur_train` is still FLAT; deploying the subpackage layout now would break the parked B1/M3c/M3b resume + PSC-only scripts.
- **Do**: after the corrected retrain finishes → sync the whole tree + fix PSC-only scripts' flat imports.
- **Expect**: local/github/PSC reconverged.
- **Why**: protect the in-flight experiment.

---

## The one critical path
`H2 (β-NLL+Stirn σ-head retrain)` is the linchpin: it unlocks `H4 Mondrian`, `E8 ENCE`, `H6 hedge verbalization`, `X1 ensembles`. Bundle `D2 DITTO + D3 diversification + C2 VICReg + B3 hedge-mask` into that SAME retrain. Everything in E (except E8/E9) + D4 + E12 + E13 + B3 + F1 is node-free NOW. T1 is the second retrain. Paper (P) proceeds in parallel.
