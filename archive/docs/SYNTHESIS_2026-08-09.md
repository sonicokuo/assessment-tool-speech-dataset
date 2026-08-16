# SYNTHESIS — 2026-08-09, after three adversarial audits

Read this instead of the 28 sections of `EXPLAINABILITY_2026-08-07.md`. That doc is the
evidence; this is the picture. Every number here was re-derived from primary artifacts by an
auditor who was instructed to try to break it.

---

## 0. THE STATE IN ONE PARAGRAPH

We cannot claim measurement accuracy: a **tuned ridge probe on a single frozen WavLM layer beats
our full pipeline by 0.0134**. What we can claim is an **evaluation** and an **axis**. The
evaluation is instrument-grounded numeric-claim faithfulness, and it is possible here and
structurally impossible in vision/NLP because our targets are formulas over frames and we hold
the clean stems. The axis is **abstention**, where every probe baseline is not weak but
**undefined** — a ridge has 100% coverage by construction and cannot withhold. Contribution 2
survives with a narrower scope than we thought (condition-level, not per-claim). Contribution 3
is not dead but is **unproven and currently confounded**, with a designed replacement evaluation
whose ceilings are already measured and one cheap control still unrun.

---

## 1. WHAT IS SOLID — survived an audit designed to break it

| result | number | status |
|---|---|---|
| readout ordering, identical 6000 clips | ridge 0.6883 < LM free 0.6986 < **verified-slot 0.7246** < aux ~0.73 | reproduced digit-for-digit under ONE identical command |
| the gain is the ESTIMATOR SWAP, not slot structure | free 0.6732 / slot-free 0.6368 (−0.036) / slot-verified 0.7054 (+0.032) | matched 2000 clips |
| film-attn vs attn-concat | **+0.0010 [−0.0017, +0.0037]** | TIE, CI-confirmed |
| attn-concat vs linear-softmax pooling | **+0.0309 [+0.0269, +0.0352]** | retirement-grade |
| layer sweep, 25 layers, full train | L7 **0.7035** vs L24 0.6887 headline; **+0.0650** on ill-posed | sanity drift 0.0004 |
| graded overlap sweep, GT invariant | rho **1.000** vs overlap fraction on 8 of 9 | splice artifacts excluded by design |
| blinded control | **6 of 9 survive** (jitter, shimmer, hnr, speaking_rate, pause_count, pause_rate) | control is CONSERVATIVE — see §3 |
| AURC, continuous risk, arm of record | robust5 **+19.9%**, ill-posed **+12.2%**, srmr +38.3% | recomputed from stored per-feature gains |
| per-frame logvar vs observability | positive on **9 of 11** (hnr 0.849, shimmer 0.789) | stable at n=48, signs all hold |
| `snr_db` is a construction parameter | `rng.uniform` → injected → written as the label | SOURCE-level proof, stronger than correlation |
| `overlap_info` is ORACLE stem-VAD | rebuild bit-identical **45/45**; pyannote **0/45** | settles a 4-month-stale doc |
| twin identity | 9 of 11 features identical on ALL 3000 test + 3000 dev pairs | effective unique GT 13,900/3,000/3,000 |

---

## 2. WHAT IS DEAD — do not write these

1. **Measurement accuracy as a contribution.** Tuned L7 ridge beats us by 0.0134. I recorded
   "+0.0031 over tuned L7" — that was against the UNTUNED ridge. State the loss ourselves.
2. **f0 as the flagship abstention example.** Its graded uncertainty collapses from rho +1.000 to
   **−0.900** when the overlap channel is zeroed. The model was reading its oracle input.
3. **Any conformal / coverage-guarantee framing.** Mondrian certifies **0 of 15 cells**.
4. **"recoverable up to +79%".** That row orders **6 error events in 6000 clips**; the panel mean
   was literally `nan`.
5. **FiLM as contribution 1.** NULL under both decode modes, measured with a fully working FiLM.
6. **Generation-level deletion grounding.** Old protocol zeroed frames (OOD); the correct
   in-distribution re-mix redo is **also NULL** — no feature's win-rate CI excludes 0.5.
7. **"Grounding succeeds for local quantities."** Made before the causal test; not backed.
8. **`overlap_ratio` as a reported predicted feature.** On oracle arms it is a literal echo
   (rho 0.9999). On the audio-only arm its 0.870 is bimodality: **0.5551 on mixtures alone**.
9. **Every abstention statistic emitted by verified-slot decode.** They are a deterministic
   function of a hard-coded oracle rule, not model behaviour.

---

## 3. THE THREE CONTRIBUTIONS, HONESTLY

### C1 — Instrument-grounded numeric-claim faithfulness. **SURVIVES as an EVALUATION.**
The metric (band-free SRCC + nMAE + coverage vs instrument GT, nMAE always against the
constant-predictor floor) is intact and discriminative. What does NOT survive is any claim that
our system measures better than a linear probe. **Frame this as the benchmark contribution and
report the probe skyline ourselves** — a reviewer reproduces it in two hours, and ALLD (ICLR
2025) already prints exactly this row.

### C2 — Observability-driven abstention. **SURVIVES, RE-SCOPED to CONDITION-LEVEL.**
* **Real:** graded, blinded uncertainty response on **6 features**; continuous AURC positive on
  both panels; the hedge decision is inferable from **audio alone** at a cost of ~2-4 points
  (audio-only arm reproduces the policy at 93.9/3.8 vs oracle 95.5/0.0 on matched clips).
* **The blind control is CONSERVATIVE, which strengthens the positives.** Zeroing is not an OOD
  input: **half of training (19,900 clean clips) has all-zero channels**, so blinding says
  "clean" — which should SUPPRESS uncertainty. Six features respond anyway.
* **The narrowing:** sigma separates CONDITIONS at rho 1.000 but ranks errors WITHIN a condition
  at only **0.05-0.26**. So this is a soft overlap detector at the condition level; per-claim
  selectivity is marginal. **Measure within-bin AURC before a reviewer does.**
* **The moat:** on this axis every probe baseline is **structurally undefined**, not merely
  weaker. That is the one comparison we win by construction.

### C3 — Per-claim causal grounding. **UNPROVEN, CONFOUNDED, WITH A DESIGNED FIX.**
* The correlational panel (deviation maps track each clip's overlap layout) is **uncontrolled for
  input echo**: the reference IS model input channel 0, verbatim, frame-aligned, at 20 ms
  rasterization. A model reading only that channel reproduces every observed signature.
* The direct causal test (in-distribution re-mix substitution, n=120) is **NULL**.
* **What contribution 2 needs still holds:** the model knows WHERE IT CANNOT MEASURE (per-frame
  logvar rises under overlap). What fails is that the emitted NUMBERS come from specific regions.
* **The fix is designed and its ceilings are measured** — see §4.

---

## 4. THE ORACLE-MAP EVALUATION — the replacement for C3

Score model attribution against **computable exact contribution maps** derived from each
feature's defining formula, not against the overlap mask.

* **Scope: 10 of 11 features.** Exact and on disk: f0_mean, **f0_sd** (non-uniform sensitivity —
  the first map that is not a mask), overlap_ratio. Exact after a ~20-line event dump: jitter,
  shimmer, hnr, speaking_rate. Exact after a provenance-matched rebuild: srmr.
  Approximate by necessity: pause_count/pause_rate (a COUNT has no unique additive
  decomposition — splitting one 0.65 s pause can RAISE it). **snr EXCLUDED and pre-registered as
  such** — its label is a sampled construction parameter, so the derivative is identically zero.
* **The reference is NOT redundant with overlap** — measured Spearman **−0.389**: the clean-stem
  evidence mask points INTO overlap while the f0-defining mask points OUT of it. Overlap explains
  only ~15% of evidence-mask variance.
* **New null N5:** score the clip's own overlap map as if it were the model map. This measures
  exactly how much alignment is purchasable from overlap alone — the direct guard against the
  echo confound. The old evaluation never had this bar.
* **The clean-clip panel is the sharpest test:** on `_s1clean` twins the overlap reference is
  degenerate (the old evaluation dropped all 200) but evidence maps stay structured and **N5 is
  vacuous**. Alignment there is pure evidence-tracking with the confound structurally absent.
* **Score gradient saliency, not the value map.** The invariance argument condemns only the value
  map; `∂ŷ/∂h_t` is a property of the learned function. Adapter-only, no 8B LM in the graph.
* **Novelty must be NARROWED.** Nearest precedents: Mamalakis et al. (exact per-pixel
  contributions, invented synthetic function), Aldeia & de França (real equations, tabular).
  Claim **"first for real measured targets over raw signal elements"**, citing them.

### The 160 ms ceiling — and why it is a RESULT, not just a caveat
A **perfect** map, block-pooled to our token grid, scores **f0 0.743**, **f0_sd 0.656**,
**overlap 0.932**. Cause: **median voiced-run length is 140 ms and 54% of voiced runs are shorter
than one token.** Every score must be reported as **%-of-ceiling**.
**The paper-relevant consequence:** token rate 12.5 vs 6.25 Hz was measured **NULL for accuracy
(−0.0014)**. So token rate is free on accuracy but **caps groundability** — a resolution ablation
that is null on the usual metric and meaningful on ours. That is a finding, not a limitation.

---

## 5. WHAT IS BLOCKING WHAT

| blocker | blocks | cost |
|---|---|---|
| **U-1**: which audio the corrected f0 GT encodes | ALL oracle-map scoring (the −0.389 confound FLIPS the expected sign) | one login-node command |
| **zeroed-overlap attribution control** | whether C3's correlational panel is real at all | CPU-only, flag already exists |
| **`fw` targets have ZERO clean-clip f0/hnr/shimmer values** (0 of 19,900, parser-verified) | the entire LM-level voice panel, on EVERY arm except the fw2 headline | retrain |
| **graded sweep never run on the audio-only arm** | whether f0 observability is audio-inferable at all, or only shortcut-suppressed | 1 GPU run |
| **within-bin AURC never computed** | whether C2 is per-claim or condition-level | cheap |
| no sigma capture on the attn-concat arm | Mondrian/AURC verified only on fw2 | cheap |

---

## 6. TRAPS — things that look right and are not

1. **`0.7246` and `0.7236` are NOT target-matched.** md5-verified: 0.7246 is film-attn on **fw2**,
   0.7236 is attn-concat on **fw**. The verified-slot comparison survives (aux GT is CSV scalars,
   so fw/fw2 cannot move it) but **the free-decode comparison between those arms is confounded**.
2. **The 0.6986 row is 5,650 clips, not 6,000**, and is the fw2 arm.
3. **Layer 7 is not a clean single knob.** L7 activations are ~20x larger than L24 and **nothing
   normalizes** — single-knob in DATA, not in OPTIMIZATION SCALE.
4. **The aux ceiling 0.7280 has no pinnable source** (recomputes 0.7300 / 0.7346). Quote ~0.73.
5. **The hedge sentence embeds the oracle number verbatim** ("(0.78 overlap ratio)") and the SFS
   parser SCORES it as a prediction — a third oracle path, distinct from input and gate.
6. **The "sigma requirement 0.24-0.30" has no provenance anywhere on disk.** Unverifiable.
7. **`w_cur` is uniform by construction on a mean-pooled arm.** Never quote it as a learned map.
8. **The f0 train/test shift does NOT explain f0 weakness** — matched-n control gives +0.03.
   f0 from pooled WavLM is weak everywhere.

---

## 7. CRITICAL PATH

**Free / CPU, do first:** U-1 provenance · zeroed-overlap control · within-bin AURC ·
`dump_oracle_events.py` + `build_oracle_maps.py` · fix `fw`→`fw2` in the five configs.

**One GPU run each, ranked by what they decide:**
1. graded sweep on the audio-only arm — decides whether C2's mechanism is audio-inferable.
2. gradient saliency vs oracle maps — decides C3.
3. fw2 retrain of the shipping arm — unlocks the LM-level voice panel.
4. verified-slot with a model-internal sigma gate replacing the oracle rule — removes the last
   oracle path; robust5 is provably unchanged, so only the abstention panel can move.
5. layer-7 retrain — the matched comparison the layer sweep made necessary.

**Do not spend runs on:** conformal machinery · standalone Liu · full fine-tuning ·
human metric validation (artifact of the retired band-SFS) · IRM.
