"""ckpt_selection.py — degeneration-aware, lower-variance checkpoint selection.

WHY THIS EXISTS
---------------
`train.py` selects `best.pt` on `val_sfs_f1` (higher = better). That is the right
axis (val_loss is ~88% prose-structural CE and under-weights the digit tokens SFS
scores, so loss-based selection cuts off numeric grounding early). BUT SFS only
parses NUMBERS from the generated text, so it is BLIND to fluency/structural
degeneration: a checkpoint that emits `</sec></sec></sec>...` tag-spam, a
repetition loop, or foreign-token injection can still contain parseable numbers
and therefore score HIGH SFS while its BLEU collapses. This is exactly the v11
section-path failure (val SFS ~0.51 selected, yet BLEU 8.0 vs the clean path's
31.5). An SFS-only argmax can thus actively SELECT a degenerate checkpoint.

Two robustness fixes, both implemented here as pure functions so they can be
unit-tested without a GPU / model:

1. DEGENERATION GUARD — gate the SFS argmax on a degeneration floor computed from
   signals train.py already produces (BLEU on the same val generations) plus a
   cheap repetition (rep-n) and non-ASCII-fraction check on those generations.
   The BLEU floor is RELATIVE (drop vs the running max), so a legitimately
   lower-BLEU-but-clean config is not wrongly rejected; only a collapse is.

2. SEEDED / STABLE VAL SLICE — train.py scores SFS on the FIRST 32 val clips
   (a biased convenience prefix reused every epoch). `seeded_val_indices` draws a
   fixed RANDOM subset once (stable across epochs via a fixed seed) so the
   estimate is representative, optionally stratified by a per-clip key (e.g.
   overlap-ratio bin) so SFS behaviour under Libri2Mix's ~78% overlap is sampled.

None of these functions import torch or any model code — they operate on the
generated strings and scalar metrics train.py already has in hand.
"""
from __future__ import annotations

import random
from collections import Counter

# Per-clip repetition threshold: a single generation whose rep-n exceeds this is
# considered a degenerate repetition loop. A realistic clean templated quality
# description ("The <feature> is <value>." x N) scores rep_n(n=4) ~ 0.01, while a
# looping clip scores ~ 0.83, so 0.5 cleanly separates the two.
REP_CLIP_THRESH: float = 0.5


# ── Degeneration statistics over a batch of generated strings ────────────────
def _word_ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rep_n(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats: 1 - |unique n-grams| / |n-grams|.

    0.0 for non-repetitive text; approaches 1.0 for a tight repetition loop
    (e.g. '</sec> </sec> </sec> ...'). Texts shorter than n tokens have no
    n-grams and score 0.0 (cannot be flagged for repetition).
    """
    toks = text.split()
    grams = _word_ngrams(toks, n)
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def nonascii_frac(text: str) -> float:
    """Fraction of characters that are non-ASCII (catches foreign-token
    injection like the Chinese-character runs seen on the degenerate section
    path). 0.0 for clean English prose."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def degeneration_stats(texts: list[str], n: int = 4,
                       rep_clip_thresh: float = REP_CLIP_THRESH) -> dict:
    """Aggregate degeneration signals over a batch of generations.

    Returns mean rep-n (per text), the overall non-ASCII character fraction
    (concatenated), the worst single-text rep-n (kept for telemetry), and the
    FRACTION of clips whose per-clip rep-n exceeds `rep_clip_thresh` (the
    repetition analogue of `frac_clips_nonascii`). The fraction is what the guard
    gates on, so a few looping clips in an otherwise clean batch (the v14 case)
    do not discard the whole checkpoint while a mostly-looping batch still does.
    """
    texts = [t or "" for t in texts]
    if not texts:
        return {"rep_n_mean": 0.0, "rep_n_max": 0.0, "frac_clips_high_rep": 0.0,
                "nonascii_frac": 0.0, "frac_clips_nonascii": 0.0}
    reps = [rep_n(t, n) for t in texts]
    total_chars = sum(len(t) for t in texts)
    nonascii = sum(1 for t in texts for ch in t if ord(ch) > 127)
    # Fraction of CLIPS containing any non-ASCII char. The aggregate
    # nonascii_frac (chars) misses the section-path failure where foreign tokens
    # are injected into many clips but are a tiny fraction of total characters
    # (v12 epoch 2: ~1% of chars but 34% of clips). This per-clip rate catches it.
    n_nonascii_clips = sum(1 for t in texts if any(ord(ch) > 127 for ch in t))
    # Fraction of CLIPS that are individually repetition-degenerate. The
    # rep_n_max gate was too strict (v14: 3/32 looping clips, max 0.62, withheld
    # an otherwise-clean checkpoint every epoch). This per-clip rate gates instead.
    n_high_rep_clips = sum(1 for r in reps if r > rep_clip_thresh)
    return {
        "rep_n_mean": sum(reps) / len(reps),
        "rep_n_max": max(reps),
        "frac_clips_high_rep": n_high_rep_clips / len(texts),
        "nonascii_frac": (nonascii / total_chars) if total_chars else 0.0,
        "frac_clips_nonascii": n_nonascii_clips / len(texts),
    }


# ── The selection guard ──────────────────────────────────────────────────────
def passes_degeneration_guard(
    bleu: float | None,
    best_bleu: float | None,
    rep_n_max: float,
    nonascii_frac_val: float,
    frac_clips_nonascii: float = 0.0,
    frac_clips_high_rep: float = 0.0,
    *,
    bleu_rel_floor: float = 0.6,
    rep_n_thresh: float = 0.95,
    nonascii_thresh: float = 0.05,
    clip_nonascii_thresh: float = 0.15,
    clip_rep_thresh: float = 0.15,
) -> tuple[bool, str]:
    """Return (is_clean, reason).

    Rejects an epoch's checkpoint from SFS-based selection if its generations
    look degenerate:
      - BLEU collapsed RELATIVE to the best clean BLEU seen so far
        (bleu < bleu_rel_floor * best_bleu). Relative, not absolute, so a config
        that is simply less fluent is not penalised — only a collapse is.
      - the FRACTION of clips that are individually repetition-degenerate
        exceeds clip_rep_thresh. The per-clip-fraction gate (not the single
        worst clip) is the fix for v14: a 32-clip batch with median rep-n 0.0
        but 3 looping clips (max 0.62) used to be withheld EVERY epoch under the
        old `rep_n_max > 0.5` gate, so no best.pt was ever saved. A few bad clips
        are tolerated; a mostly-looping batch is still withheld. Mirrors the
        `frac_clips_nonascii` design.
      - rep_n_max is kept ONLY as a catastrophic backstop at a HIGH threshold
        (0.95): a single clip that is essentially one token repeated is so
        broken it is worth rejecting even if it is alone, but normal looping
        clips (~0.6-0.85) no longer trip it.
      - non-ASCII character fraction exceeds nonascii_thresh (foreign-token
        injection).

    A None BLEU or None best_bleu skips only the BLEU check (the rep-n and
    non-ASCII guards still apply, so a degenerate first epoch is still caught).
    """
    if (
        bleu is not None
        and best_bleu is not None
        and best_bleu > 0.0
        and bleu < bleu_rel_floor * best_bleu
    ):
        return False, f"bleu {bleu:.2f} < {bleu_rel_floor:.2f}*best {best_bleu:.2f}"
    if frac_clips_high_rep > clip_rep_thresh:
        return False, (f"frac_clips_high_rep {frac_clips_high_rep:.3f} > "
                       f"{clip_rep_thresh:.2f} (repetition/tag-spam in many clips)")
    if rep_n_max > rep_n_thresh:
        return False, (f"rep_n_max {rep_n_max:.3f} > {rep_n_thresh:.2f} "
                       f"(catastrophic single-clip repetition)")
    if nonascii_frac_val > nonascii_thresh:
        return False, f"nonascii_frac {nonascii_frac_val:.4f} > {nonascii_thresh:.3f}"
    if frac_clips_nonascii > clip_nonascii_thresh:
        return False, (f"frac_clips_nonascii {frac_clips_nonascii:.3f} > "
                       f"{clip_nonascii_thresh:.2f} (foreign-token injection in many clips)")
    return True, "clean"


def should_save_best(
    sfs_f1: float | None,
    best_sfs_f1: float,
    bleu: float | None,
    best_bleu: float | None,
    gen_texts: list[str],
    *,
    bleu_rel_floor: float = 0.6,
    rep_n_thresh: float = 0.95,
    nonascii_thresh: float = 0.05,
    clip_rep_thresh: float = 0.15,
    n_gram: int = 4,
) -> tuple[bool, str]:
    """Top-level decision: save best.pt iff SFS improved AND the generations are
    not degenerate. Pure; train.py passes the per-epoch generations and metrics.
    """
    if sfs_f1 is None or sfs_f1 <= best_sfs_f1:
        return False, "no sfs improvement"
    stats = degeneration_stats(gen_texts, n=n_gram)
    ok, reason = passes_degeneration_guard(
        bleu, best_bleu, stats["rep_n_max"], stats["nonascii_frac"],
        stats["frac_clips_nonascii"], stats["frac_clips_high_rep"],
        bleu_rel_floor=bleu_rel_floor,
        rep_n_thresh=rep_n_thresh,
        nonascii_thresh=nonascii_thresh,
        clip_rep_thresh=clip_rep_thresh,
    )
    if not ok:
        return False, f"sfs improved but degenerate ({reason})"
    return True, "sfs improved, clean"


# ── Stable / representative validation subset ────────────────────────────────
def seeded_val_indices(
    n_total: int,
    n_sample: int,
    seed: int = 1234,
    strata: list | None = None,
) -> list[int]:
    """A fixed RANDOM subset of validation indices, stable across epochs.

    Replaces train.py's biased `range(n_sample)` prefix slice. With a fixed seed
    the SAME representative subset is scored every epoch (so cross-epoch SFS
    trends are comparable) but it is no longer correlated with dataset ordering.

    If `strata` is given (one hashable key per item, len == n_total, e.g. an
    overlap-ratio bin), the sample is allocated across strata proportionally so
    every condition is represented. Returns sorted indices.
    """
    n_sample = min(n_sample, n_total)
    rng = random.Random(seed)
    if not strata:
        return sorted(rng.sample(range(n_total), n_sample))

    if len(strata) != n_total:
        raise ValueError(f"strata length {len(strata)} != n_total {n_total}")
    by_stratum: dict = {}
    for idx, key in enumerate(strata):
        by_stratum.setdefault(key, []).append(idx)

    chosen: list[int] = []
    keys = sorted(by_stratum.keys(), key=lambda k: str(k))
    for key in keys:
        pool = by_stratum[key]
        take = max(1, round(n_sample * len(pool) / n_total))
        take = min(take, len(pool))
        chosen.extend(rng.sample(pool, take))
    # Trim/pad to exactly n_sample without reintroducing prefix bias.
    if len(chosen) > n_sample:
        chosen = rng.sample(chosen, n_sample)
    elif len(chosen) < n_sample:
        remaining = [i for i in range(n_total) if i not in set(chosen)]
        chosen.extend(rng.sample(remaining, n_sample - len(chosen)))
    return sorted(chosen)
