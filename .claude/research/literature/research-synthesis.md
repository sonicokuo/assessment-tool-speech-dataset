# Research Synthesis (the 3 literature reviews behind the redirect)

Three independent literature-review agents were run to decide whether to patch the
fragile special-token section path or replace it. All three pointed the same way:
the degeneration is architectural, and the robust alternatives are (A) token-free
decoupled grounding, with (B) the special-token path being intrinsically weak at our
data scale, and (C) RLVR-on-SFS as a well-supported way to lift faithfulness. arXiv
ids are inline so they can be cited directly.

---

## (A) Token-free 2D grounding

**Problem confirmation.** Special-token grounding degenerates because new tokens are
under-trained and off-manifold -- the "Magikarp" / glitch-token analysis
(**arXiv:2405.05417**) shows under-trained vocabulary entries behave anomalously and
destabilize generation. This matches our finding (c): only token-EMITTING runs
degenerate.

**The robust alternative: DECOUPLED query heads.** Instead of asking the LM to emit
tokens that drive attention, hold a fixed set of learned query vectors that
cross-attend the input directly. Precedents:
- DETR object queries (**arXiv:2005.12872**)
- Slot-Attention (**arXiv:2006.15055**)
- Q-Former / BLIP-2 (**arXiv:2301.12597**)
- LISA / GLaMM (grounded segmentation from a decoupled head; GLaMM = per-phrase
  boxes + IoU, the direct CV parallel to our per-overlap-window IoU).

**The strongest audio precedent: arXiv:2602.10230 (Frame-Level Internal Tool Use).**
It replaced timestamp-TOKEN emission with a DECOUPLED frame head precisely BECAUSE
token-based models "collapse completely" out-of-distribution -- exactly our failure
mode. This is the load-bearing citation for the redirect: a published audio system hit
our exact degeneration and fixed it our exact way.

**Validation suite for the grounding claim** (the maps must be shown faithful, not
just pretty):
- pointing-game / IoU of each per-feature map vs oracle overlap spans;
- deletion / insertion curves (mask the attended region, watch the prediction drop);
- injection faithfulness (synthesize a feature at known T-F coordinates, watch the
  map + number + hedge move -- our unique causal card);
- model-randomization and label-randomization sanity checks;
- Wilcoxon vs random attention.

This is realized in code as `src/decoupled_grounding.py` (learned per-feature queries,
`V.detach()` grounding pool, orthogonality penalty) and the existing
`src/grounding_metrics.py` (time / freq concentration ratios) + faithfulness scripts.

---

## (B) Two-stage special-token alignment (why we are NOT keeping the token path)

If one insisted on structural tokens, the literature says:
- Structural tokens must be content-free and word-DISTINCT: ChatML control tokens,
  ViT registers (**arXiv:2309.16588**), Llama-3 reserved special tokens. Our R10
  semantic-init FAILED for exactly this reason -- it made the tokens word-LIKE, so the
  LM generated prose around them (finding (a)).
- Pause-token evidence (**arXiv:2310.02226**) shows finetune-only structural tokens
  are intrinsically weak. So NO embedding trick (semantic init, textual multi-token
  tags, etc.) fully matches the untagged-0% baseline at our data scale (~14-42k clips).
- The real backstop if you must emit structure is forced / grammar-constrained
  decoding (**arXiv:2305.13971**), which guarantees structure and kills `</sec>`
  degeneration.
- Two-stage warmup (the BLIP-2 pattern) buys STABILITY, not metric gains. Mixing
  untagged data + a KL-to-base leash also help stability.

**Verdict:** the special-token path is a dead end at our scale; the decoupled head (A)
sidesteps it entirely. The token-path findings (R10, lambda sensitivity, forced-decode)
go into the ablation / negative-results section, not the headline.

---

## (C) RL / agentic (RLVR-on-SFS)

**Why we are an ideal RLVR candidate.** SFS is a deterministic, human-validated
verifier (rho=0.69). RLVR / GRPO wants exactly a ground-truth-checked scalar reward,
which SFS is.

**Precedent: R1-AQA (arXiv:2503.11197).** GRPO on a 7-8B audio LLM with ~38k samples;
RL beat SFT; and -- a transferable negative result -- chain-of-thought did NOT help
audio QA. So we run GRPO without CoT scaffolding.

**Reward design** (realized in `src/sfs_reward.py`):
- reward = SFS-**F1** (NOT recall -- recall is trivially hacked by number-spamming
  every plausible value, because recall only asks "was the feature mentioned");
- minus a repetition + non-ASCII / foreign-token degeneration penalty (SFS is
  degeneration-blind, so RL could otherwise drift to high-SFS-but-garbage text);
- with a KL leash to the SFT reference (don't forget fluency while chasing SFS).

**Plan:** SFT cold-start (from v9) -> RAFT best-of-n rejection-sampling de-risk ->
GRPO-on-SFS. RAFT first because it is GRPO without the KL/PPO machinery -- far more
stable and surfaces reward-hacking cheaply.

**Risks:**
- Metric gaming: report BLEU / ROUGE / BERTScore alongside SFS, and hand-audit
  samples, so RL cannot quietly game the verifier.
- f0_mean may not move: RL AMPLIFIES capability the SFT model already has; it does not
  CREATE it. If the model cannot estimate F0 from audio, no reward shaping conjures it.
  (Consistent with the field-wide pitch-is-hard finding.)

---

## Supporting landscape (for positioning / intro, all verified primary sources)

- **arXiv:2501.17202** (ICLR 2025): qualitative adjectives + 1-5 MOS, LLM-verbalized
  NISQA human ratings -- no physical numbers, no interpretability. Closest competitor;
  accepted at ICLR WITH modest numbers -> evidence ICLR fits modest-numbers framing.
- **QualiSpeech arXiv:2503.20290** (ACL 2025 Long): human+GPT qualitative descriptions
  + GPT-judge metric, no MEASURED values anywhere, single-speaker (no overlap). We
  differ on every load-bearing axis (signal-characterization vs perceptual-opinion).
- **TRACE arXiv:2601.13742** (EACL 2026 Findings); **arXiv:2603.10175** (Interspeech
  2026, claims SOTA on QualiSpeech -- contested lane, orthogonal to ours).
- **SonicBench arXiv:2601.11039**: frozen-encoder linear probes hit 60-90% on
  pitch/loudness while end-to-end audio-LLMs are near chance -> the failure lives in
  alignment / decoding, NOT the encoder. **PitchBench arXiv:2605.26176**: pitch
  unreliable across all ALMs -> our weak F0 is field-wide, not our encoder.
- Method verdicts: PiSSA / DoRA ~= 0-1 pt (ablation completeness only; PiSSA mutates
  base weights -> inference conversion needed). Flamingo-style deep cross-attention
  fusion = NO at ~14k clips (prefix beats xattn under LoRA). Top lever = the
  decoupled grounding head into the generation analysis, not connector swaps.

---

## arXiv id quick index

| id | what |
|----|------|
| 2602.10230 | Frame-Level Internal Tool Use -- decoupled frame head replaces timestamp tokens (the key audio precedent) |
| 2405.05417 | Magikarp / under-trained glitch tokens |
| 2005.12872 | DETR object queries |
| 2006.15055 | Slot-Attention |
| 2301.12597 | Q-Former / BLIP-2 |
| 2309.16588 | ViT registers (content-free tokens) |
| 2310.02226 | pause tokens (finetune-only structural tokens are weak) |
| 2305.13971 | grammar-constrained / forced decoding |
| 2503.11197 | R1-AQA -- GRPO on audio LLM, RL>SFT, CoT didn't help |
| 2501.17202 | ICLR 2025 modest-numbers quality-description competitor |
| 2503.20290 | QualiSpeech (ACL 2025) |
| 2601.11039 | SonicBench (bottleneck is alignment/decoding) |
| 2605.26176 | PitchBench (pitch field-wide hard) |
| 2601.13742 | TRACE (EACL 2026 Findings) |
| 2603.10175 | Interspeech 2026 QualiSpeech-SOTA |
