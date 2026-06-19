# Training Run History

This is the authoritative run log for AQUA-NL. Each row is a named checkpoint, what
it changed, and what it showed. "Degeneration" (degen) = fraction of validation
generations that collapse into repetition loops, special-token spam, or
foreign-token (non-ASCII / Chinese) injection. SFS = Signal Faithfulness Score F1
unless noted. The 2D-map / section path is the novel grounding contribution and is
where almost all instability lives; the untagged path is the clean metric leg.

Numbers below are on Libri2Mix unless noted. The val SFS numbers are noisy because
they are computed on a small (32-clip) val slice during training; test numbers
(3000 clips) are the paper numbers.

> CAVEAT carried from the memory: the F0-GT fix (clean-frame F0, see findings doc)
> invalidates v9's *original* headline SFS/grounding numbers when scored against the
> old mixture-F0 answer key. They must be re-scored against clean-F0 CSVs. The
> headline aggregate values below (SFS 0.52, BLEU 31.5) are the pre-fix figures and
> are being re-computed.

---

## The two tracks

- **Untagged track** (LM emits clean prose, no `<sec_*>` / `<f_*>` tokens) — the
  metric + hedging leg. Empirically **0% degeneration** in every run (v7/v8/v9, all
  adapter variants). This is the headline-quality, clean track.
- **Tagged / section track** (use_sections=true: LM emits structural tokens that
  drive per-attribute attention -> the 2D grounding maps). This is the novel
  contribution but is **degeneration-prone** (3-44% depending on recipe).

---

## Run table

| Run | Track | Recipe delta | Degen | SFS | Notes |
|-----|-------|--------------|-------|-----|-------|
| **v9_lora_8b_dur** | untagged | LoRA r=16 Qwen3-8B, film-mamba, no sections | **0%** | **~0.52** test | HEADLINE. BLEU 31.5, ROUGE-L 0.65, BERTScore 0.67, P 0.41, R 0.81. |
| **v11_section_head_lora** | tagged | section_head 2D maps, NO readout loss | ~3.1% (ep9 best) | 0.48 (val) / 0.55 (test) | The ungrounded-but-clean section baseline. Maps exist but are not grounded (predates the readout head). BLEU 8.0 -> figure only, never headline. |
| **v12_readout_derisk** | tagged | added readout grounding lambda=0.5 | **31%** | 0.26 | Grounding at full strength corrupts generation. 10x degen jump vs v11. Confirmed grounding != degeneration-fix; they are orthogonal. |
| **v13_section_warmup** | tagged | readout lambda=0.05 + WARMUP (ramp 0->0.05 over 3 epochs) | settles **~9%** | ~0.48-0.50 | Grounded 2D maps at near-v11 cleanliness. THE working grounding figure. Ramp did not spiral. |
| **v14_aug** | untagged | + noise augmentation (controlled-SNR), clean-F0 targets | **0%** | peaked 0.515 | Augmentation / metric leg. f0_mean recovered 11% -> 23% on clean-F0 scoring. Clean numbers. |
| **v15** | tagged | 2D-map + augmentation FUSION, R10 semantic-init ON, lambda=0.05 | **44%** (ep1) | - | FUSION attempt 1. R10-on is the culprit (see findings). |
| **v15v2_nor10** | tagged | FUSION, R10 OFF (mean-init), lambda=0.05 | 16% / 16% / **41% spike** | 0.403 (ep1) | FUSION attempt 2. R10-off dropped ep1 degen 44->16%, but still spikes to 41% later. |
| **v15v3** | tagged | FUSION, R10 OFF, lambda=0.02 (gentler) | 41% / 31% | - | FUSION attempt 3. Lower lambda, still high-variance / unstable. |

---

## v13 full warmup ramp (the decisive grounding test)

The decisive question for the grounding contribution was whether readout grounding
could be added without the v12 degeneration spiral. v13 answered yes:

| Epoch | effective lambda | degen | SFS |
|-------|------------------|-------|-----|
| 1 | 0.0 (warmup) | 9% (3/32) | 0.372 |
| 2 | 0.0167 | 22% (transient ramp spike) | 0.404 |
| 3 | 0.0333 | 19% | 0.483 |
| 4 | 0.05 (full) | 19% | 0.501 |
| 5 | 0.05 (full) | **9%** | 0.479 |

At full grounding strength it settled to ~9% degen + SFS ~0.48-0.50. Compare:
v11 (ungrounded) 3% / 0.48 ; v12 (lambda=0.5) 31% / 0.26 ; v13 (lambda=0.05 + warmup)
~9% / 0.48 with GROUNDED maps. The degeneration-gated checkpoint guard correctly
withheld the 22% epoch-2 checkpoint despite its higher SFS and saved a clean epoch
as best.pt. Static-query-mode fallback was NOT needed.

---

## Fusion conclusion (why there is no single fused headline yet)

The three v15 runs tried to FUSE the 2D-map (tagged) path with the noise
augmentation in one model. All three swing **16-44% degeneration** on the augmented
tagged section path, versus v13 (2D-map, NO augmentation) which is **stable at 9%**.
Both isolated knobs help (R10 off, lambda down) but neither tames the augmented
section path. The augmented tagged section path is **UNSTABLE**.

Two takeaways that became the redirect:

1. The augmented tagged section path is not currently stable enough to be a single
   fused headline. The earlier fallback was a TWO-MODEL paper (v13 grounding figure +
   v14 augmentation/metric leg, unfused). See `../decisions/decisions.md`.
2. The instability is architectural, not a tuning miss: special-token EMISSION is
   fragile. This motivated the redirect to a token-free decoupled grounding head
   (see `../decisions/decisions.md` and `../literature/research-synthesis.md`).

---

## Verified cross-cutting results (carry into the paper)

- SFS vs human: Spearman rho = 0.69 (Pearson 0.70, p < 1e-4, n=50 single rater).
- Overlap attention grounded on 95% of clips (Wilcoxon p < 1e-6, mean concentration
  ratio 1.10, capped ~1.27 by Libri2Mix's 78% base overlap rate).
- Per-feature SFS-F1 (v9_film_attn, 3000 clips, pre clean-F0 rescore):
  pause_count 0.97-0.99, speaking_rate 0.52-0.56, snr 0.45-0.52, overlap_ratio
  0.39-0.44, pause_rate 0.29-0.76, srmr 0.20-0.23, f0_sd 0.19-0.23, f0_mean 0.11-0.15.
  Low aggregate is DRAGGED by f0_mean (ill-posed mixture-F0 GT); see findings doc.
- Zero-shot LM floor (no audio): SFS 0.0, BLEU 0.04.
