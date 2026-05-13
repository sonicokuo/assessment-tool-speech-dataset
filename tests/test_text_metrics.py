"""Sanity tests for src/text_metrics.compute_generation_metrics.

Four scenarios:
  1. Identical hyp == ref             → BLEU ≈ 100, ROUGE-L ≈ 1.0, BERTScore ≈ 1.0
  2. Paraphrase (same facts, worded differently) → mid BLEU, high BERTScore
  3. Totally unrelated hyp            → BLEU ≈ 0, ROUGE-L low, BERTScore low
  4. Real gemma4 descriptions (reference) vs synthetic model output
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from text_metrics import compute_generation_metrics


def _show(title: str, m: dict) -> None:
    print(f"\n{title}")
    print(f"  BLEU-4:        {m['bleu']}")
    print(f"  ROUGE-L:       {m['rouge_l']}")
    print(f"  BERTScore-F1:  {m['bertscore_f1']}")


def test_identical():
    ref = "The speech sample has a duration of 6.18 s and an SNR of 15.63 dB."
    m = compute_generation_metrics([ref], [ref], use_bertscore=True)
    _show("IDENTICAL (floor=ceiling):", m)
    assert m["bleu"] >= 99.0, f"BLEU on identical should be ~100, got {m['bleu']}"
    assert m["rouge_l"] >= 0.99, f"ROUGE-L on identical should be ~1.0, got {m['rouge_l']}"
    assert m["bertscore_f1"] >= 0.95


def test_paraphrase():
    ref = "The speech sample has a duration of 6.18 s and an SNR of 15.63 dB."
    hyp = "This audio clip lasts 6.18 seconds with a signal-to-noise ratio of 15.63 dB."
    m = compute_generation_metrics([hyp], [ref], use_bertscore=True)
    _show("PARAPHRASE (same facts, different words):", m)
    # BLEU will be low-ish (surface n-gram overlap), BERTScore should stay high (semantic).
    assert m["bertscore_f1"] > m["bleu"] / 100.0, "BERTScore should reward paraphrases over BLEU"


def test_unrelated():
    ref = "The speech sample has a duration of 6.18 s and an SNR of 15.63 dB."
    hyp = "The weather tomorrow is expected to be sunny with a high of 72 degrees."
    m = compute_generation_metrics([hyp], [ref], use_bertscore=True)
    _show("UNRELATED:", m)
    assert m["bleu"] < 10.0, f"BLEU on unrelated should be low, got {m['bleu']}"
    assert m["rouge_l"] < 0.3


def test_corpus_with_real_descriptions():
    """Use two real gemma4 descriptions as reference, one matching + one mismatching prediction.

    Skipped when the local-only fixture isn't present (scratch/ is gitignored,
    so CI / PSC checkouts don't have these CSVs).
    """
    import csv, glob, pytest
    samples = []
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "scratch",
                                          "verb_samples", "train-100_*.csv")))
    if not paths:
        pytest.skip(
            "scratch/verb_samples/train-100_*.csv not present "
            "(scratch/ is gitignored; this test relies on local-only fixtures)."
        )
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                desc = row.get("quality_description", "") or ""
                if desc and not desc.startswith("[ERROR]") and len(desc) > 200:
                    samples.append(desc)
                if len(samples) >= 3:
                    break
        if len(samples) >= 3:
            break
    if len(samples) < 3:
        pytest.skip(f"only {len(samples)} usable descriptions found in scratch/verb_samples")

    # Build 3 hyp/ref pairs:
    # (a) identical   (b) swap target  (c) light paraphrase via a word change
    refs = samples
    hyps = [
        samples[0],                          # identical
        samples[2],                          # totally wrong clip → low scores
        samples[2].replace("The", "This"),   # tiny paraphrase
    ]
    m = compute_generation_metrics(hyps, refs, use_bertscore=True)
    _show("CORPUS (1 ident + 1 wrong + 1 paraphrase):", m)
    # Averages should be between individual extremes.
    assert m["bleu"] > 5.0 and m["bleu"] < 100.0
    assert 0.3 < m["rouge_l"] < 0.95


def test_empty_pairs():
    m = compute_generation_metrics([], [], use_bertscore=False)
    _show("EMPTY INPUT (should return all None):", m)
    assert m == {"bleu": None, "rouge_l": None, "bertscore_f1": None}


def test_skip_bertscore():
    """Verify use_bertscore=False skips the expensive call."""
    ref = "hello world"
    hyp = "hello world"
    m = compute_generation_metrics([hyp], [ref], use_bertscore=False)
    _show("use_bertscore=False:", m)
    assert m["bleu"] is not None
    assert m["rouge_l"] is not None
    assert m["bertscore_f1"] is None


if __name__ == "__main__":
    test_identical()
    test_paraphrase()
    test_unrelated()
    test_corpus_with_real_descriptions()
    test_empty_pairs()
    test_skip_bertscore()
    print("\n✅ all tests passed")
