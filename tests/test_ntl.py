"""CPU unit tests for src/ntl.py — the Number Token Loss (NTL-WAS/abs, arXiv 2411.02083).

Covers (all CPU, no model download for the core tests):
  - digit_token_ids: maps '0'..'9' to single distinct ids; rejects multi-token digits.
  - number_token_loss DECREASES as the predicted digit distribution concentrates on the
    correct digit (the whole point: ordinal, not nominal).
  - number_token_loss == 0 when there are no digit-target positions.
  - number_token_loss respects the target_mask (padding excluded).
  - The 8->12 aux head: adapter.N_AUX_FEATURES tracks feature_set.N_FEATURES == 12.

A real-Qwen tokenizer test is included but skipped automatically if the tokenizer
cannot be loaded offline (no network / weights), so the suite stays green on CPU.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ntl import digit_token_ids, number_token_loss


# ── A tiny fake tokenizer where each digit '0'..'9' is its own single id ──────
# Vocab layout: ids 0..9 are the digits, ids 10+ are arbitrary "structural" tokens.
class _FakeDigitTokenizer:
    name_or_path = "fake-digit-tokenizer"

    def encode(self, text, add_special_tokens=False):
        # Single-char digit → its int value as the id. Anything else → a non-digit id.
        if len(text) == 1 and text.isdigit():
            return [int(text)]
        return [100 + (hash(text) % 50)]


# Vocab size for synthetic logits: must exceed the largest digit id (9).
VOCAB = 32


def _digit_ids():
    return digit_token_ids(_FakeDigitTokenizer())


def test_digit_token_ids_are_ten_distinct():
    ids = _digit_ids()
    assert ids.shape == (10,)
    assert ids.dtype == torch.long
    # Fake tokenizer maps digit d -> id d, so they are 0..9 in order.
    assert ids.tolist() == list(range(10))
    assert len(set(ids.tolist())) == 10


def test_digit_token_ids_rejects_multi_token_digit():
    class _BadTokenizer:
        name_or_path = "bad"

        def encode(self, text, add_special_tokens=False):
            return [1, 2]  # every digit splits into two tokens → invalid for NTL

    with pytest.raises(ValueError):
        digit_token_ids(_BadTokenizer())


def _logits_peaked_on(target_digit: int, sharpness: float) -> torch.Tensor:
    """(1,1,VOCAB) logits whose digit-restricted softmax peaks on `target_digit`.

    sharpness scales how concentrated the predicted digit distribution is: 0 = uniform
    over digits, large = a near one-hot on target_digit.
    """
    logits = torch.zeros(1, 1, VOCAB)
    logits[0, 0, target_digit] = sharpness
    return logits


def test_ntl_lower_when_distribution_concentrates_on_correct_digit():
    """The headline property: loss(concentrated-correct) < loss(diffuse) < loss(wrong)."""
    digit_ids = _digit_ids()
    target = torch.tensor([[7]])  # target token is digit '7'

    # 1) Diffuse: uniform over digits → E_pred = 4.5, |4.5 - 7| = 2.5
    diffuse = number_token_loss(_logits_peaked_on(7, 0.0), target, digit_ids)
    # 2) Concentrated on the CORRECT digit 7 → E_pred → 7, loss → 0
    correct = number_token_loss(_logits_peaked_on(7, 12.0), target, digit_ids)
    # 3) Concentrated on a WRONG digit 0 → E_pred → 0, loss → 7
    wrong = number_token_loss(_logits_peaked_on(0, 12.0), target, digit_ids)

    assert correct.item() < diffuse.item(), (correct.item(), diffuse.item())
    assert diffuse.item() < wrong.item(), (diffuse.item(), wrong.item())
    # Concentrating harder on the right digit keeps lowering the loss (monotone).
    soft = number_token_loss(_logits_peaked_on(7, 3.0), target, digit_ids)
    assert correct.item() < soft.item() < diffuse.item()
    # And the correct-concentrated loss is essentially zero.
    assert correct.item() < 1e-2


def test_ntl_zero_when_no_digit_targets():
    """No digit-target positions → exactly 0 (and graph-safe)."""
    digit_ids = _digit_ids()
    # Target is a non-digit token id (>= 10), so no position is a digit.
    target = torch.tensor([[15, 20]])
    logits = torch.randn(1, 2, VOCAB, requires_grad=True)
    loss = number_token_loss(logits, target, digit_ids)
    assert loss.item() == 0.0
    # Graph-connected zero: backward must not error and grad is all-zero.
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_ntl_respects_target_mask():
    """A digit position masked out (padding) must not contribute."""
    digit_ids = _digit_ids()
    # Two positions, both target digit '3'. Mask out the second.
    target = torch.tensor([[3, 3]])
    # First position predicts '3' perfectly; second predicts wrong digit '9'.
    logits = torch.zeros(1, 2, VOCAB)
    logits[0, 0, 3] = 12.0   # correct → ~0 loss
    logits[0, 1, 9] = 12.0   # wrong, but masked out
    mask = torch.tensor([[1, 0]])
    loss = number_token_loss(logits, target, digit_ids, target_mask=mask)
    # Only the correct, unmasked position counts → loss ~ 0.
    assert loss.item() < 1e-2
    # Without the mask, the wrong position drags the mean up.
    loss_unmasked = number_token_loss(logits, target, digit_ids)
    assert loss_unmasked.item() > 1.0


def test_ntl_multi_position_mean():
    """Loss is the MEAN over all digit-target positions."""
    digit_ids = _digit_ids()
    target = torch.tensor([[2, 5]])
    logits = torch.zeros(1, 2, VOCAB)
    logits[0, 0, 2] = 12.0   # correct → ~0
    logits[0, 1, 0] = 12.0   # predicts 0, target 5 → ~5
    loss = number_token_loss(logits, target, digit_ids)
    # mean(~0, ~5) ~ 2.5
    assert 2.0 < loss.item() < 3.0


def test_ntl_is_differentiable():
    digit_ids = _digit_ids()
    target = torch.tensor([[4]])
    logits = torch.zeros(1, 1, VOCAB, requires_grad=True)
    loss = number_token_loss(logits, target, digit_ids)
    loss.backward()
    assert logits.grad is not None
    # Gradient must be non-zero at the digit logits (the loss depends on them).
    assert torch.count_nonzero(logits.grad[0, 0, digit_ids]) > 0


def test_aux_head_dim_tracks_feature_set_12():
    """The aux regression head output dim auto-tracks the 12-feature set.

    adapter.py imports mamba_ssm (a CUDA extension) at module top, which is not
    importable on a CPU/login node, so the full-import check is skipped there. The
    feature_set side (the source of truth) is always asserted; the adapter side is
    asserted whenever the module imports (e.g. on a GPU node / the smoke run).
    """
    import feature_set
    assert feature_set.N_FEATURES == 12
    # adapter.py imports mamba_ssm at module top; its CUDA init can be unstable on a
    # CPU/login node (segfault, not a catchable exception), so gate on CUDA first.
    torch_mod = pytest.importorskip("torch")
    if not torch_mod.cuda.is_available():
        pytest.skip("adapter import needs CUDA/mamba_ssm; skipped on CPU host")
    try:
        import adapter
    except Exception as e:
        pytest.skip(f"adapter not importable here: {e}")
    assert adapter.N_AUX_FEATURES == feature_set.N_FEATURES == 12


# ── Optional: real Qwen tokenizer (skipped if unavailable offline) ────────────
def test_digit_token_ids_on_real_qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as e:  # no network / weights cached → skip, don't fail
        pytest.skip(f"Qwen tokenizer unavailable offline: {e}")
    ids = digit_token_ids(tok)
    assert ids.shape == (10,)
    assert len(set(ids.tolist())) == 10  # Qwen tokenizes digits one-per-token


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
