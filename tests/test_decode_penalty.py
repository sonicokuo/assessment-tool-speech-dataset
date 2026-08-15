"""Tests for D4 deterministic repetition control in src/inference.py's sample_token.

Torch tensors only (no model) -> runs anywhere torch is installed.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import (  # noqa: E402
    _apply_repetition_penalty,
    _block_repeat_ngrams,
    sample_token,
)


def test_repetition_penalty_downweights_positive_logit():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 2.0
    out = _apply_repetition_penalty(logits.clone(), [3], penalty=2.0)
    assert out[0, 3].item() == pytest.approx(1.0)      # 2.0 / 2.0
    assert out[0, 5].item() == 0.0                     # untouched


def test_repetition_penalty_multiplies_negative_logit():
    logits = torch.zeros(1, 10)
    logits[0, 3] = -2.0
    out = _apply_repetition_penalty(logits.clone(), [3], penalty=2.0)
    assert out[0, 3].item() == pytest.approx(-4.0)     # negative -> * penalty (more negative)


def test_no_repeat_ngram_blocks_completion():
    # prev = [1,2,1], n=2: prefix=(1,); token following 1 in prev is 2 -> banned
    logits = torch.zeros(1, 5)
    out = _block_repeat_ngrams(logits.clone(), [1, 2, 1], n=2)
    assert out[0, 2].item() == float("-inf")
    assert out[0, 4].item() == 0.0
    # n=0 is a no-op
    assert torch.equal(_block_repeat_ngrams(logits.clone(), [1, 2, 1], n=0), logits)


def test_greedy_avoids_repeat_under_strong_penalty():
    logits = torch.tensor([[0.0, 0.0, 0.0, 5.0, 4.0]])
    # without penalty greedy picks 3; with a strong penalty on 3 it must pick 4
    plain = sample_token(logits.clone(), top_k=1)
    assert plain.item() == 3
    penalized = sample_token(logits.clone(), top_k=1, prev_ids=[3], repetition_penalty=100.0)
    assert penalized.item() == 4


def test_defaults_are_noop():
    torch.manual_seed(0)
    logits = torch.randn(1, 20)
    a = sample_token(logits.clone(), top_k=1, prev_ids=[1, 2, 3])   # defaults penalty=1, ngram=0
    b = sample_token(logits.clone(), top_k=1)
    assert a.item() == b.item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
