"""Generation-quality metrics that complement SFS: BLEU-4, ROUGE-L, BERTScore-F1.

SFS measures numerical faithfulness. These three measure surface / semantic overlap
with the reference. Use them together: high SFS + high BLEU/ROUGE = fluent and factual;
high BLEU + low SFS = fluent hallucinations; low BLEU + high SFS = factual but stilted.

Deps (install in env before using):
    pip install sacrebleu rouge-score bert-score

Each metric fails soft — if the dependency is missing, we print a warning and return
None for that field rather than crashing the whole evaluation.
"""

from __future__ import annotations


def _compute_bleu(hyps: list[str], refs: list[str]) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        print("[text_metrics] sacrebleu not installed — skipping BLEU (pip install sacrebleu)")
        return None
    # sacrebleu expects list-of-refs-lists (one inner list per reference set)
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def _compute_rouge_l(hyps: list[str], refs: list[str]) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[text_metrics] rouge-score not installed — skipping ROUGE-L (pip install rouge-score)")
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fs = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return sum(fs) / len(fs) if fs else 0.0


def _compute_bertscore(hyps: list[str], refs: list[str], lang: str = "en") -> float | None:
    """BERTScore-F1 mean. Downloads a ~1 GB model on first use."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("[text_metrics] bert-score not installed — skipping BERTScore (pip install bert-score)")
        return None
    _, _, f1 = bert_score_fn(hyps, refs, lang=lang, rescale_with_baseline=True, verbose=False)
    return float(f1.mean().item())


def compute_generation_metrics(
    hyps: list[str],
    refs: list[str],
    use_bertscore: bool = True,
) -> dict[str, float | None]:
    """Compute BLEU-4, ROUGE-L, and (optionally) BERTScore-F1 over aligned hyp/ref lists.

    Args:
        hyps: generated strings.
        refs: reference strings (same length and order as hyps).
        use_bertscore: set False to skip the expensive BERTScore pass (still runs BLEU+ROUGE).
    Returns:
        dict with keys "bleu", "rouge_l", "bertscore_f1" — values may be None if a dep is
        missing or if the input is empty.
    """
    assert len(hyps) == len(refs), f"hyp/ref length mismatch: {len(hyps)} vs {len(refs)}"

    # Drop pairs where either side is empty (BLEU/ROUGE would NaN/0 on them anyway).
    clean = [(h.strip(), r.strip()) for h, r in zip(hyps, refs) if h.strip() and r.strip()]
    if not clean:
        return {"bleu": None, "rouge_l": None, "bertscore_f1": None}
    hyps_c = [h for h, _ in clean]
    refs_c = [r for _, r in clean]

    out = {
        "bleu": _compute_bleu(hyps_c, refs_c),
        "rouge_l": _compute_rouge_l(hyps_c, refs_c),
        "bertscore_f1": _compute_bertscore(hyps_c, refs_c) if use_bertscore else None,
    }
    return out
