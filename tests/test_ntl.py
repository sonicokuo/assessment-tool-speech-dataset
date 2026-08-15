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

from model.ntl import digit_token_ids, number_token_loss


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


def test_aux_head_dim_tracks_feature_set_11():
    """The aux regression head output dim auto-tracks the 11-feature set.

    (2026-06-24 voice patch dropped articulation_rate: 12 -> 11 SUPERVISED_FEATURES.)
    adapter.py imports mamba_ssm (a CUDA extension) at module top, which is not
    importable on a CPU/login node, so the full-import check is skipped there. The
    feature_set side (the source of truth) is always asserted; the adapter side is
    asserted whenever the module imports (e.g. on a GPU node / the smoke run).
    """
    import data.feature_set as feature_set
    assert feature_set.N_FEATURES == 11
    # adapter.py imports mamba_ssm at module top; its CUDA init can be unstable on a
    # CPU/login node (segfault, not a catchable exception), so gate on CUDA first.
    torch_mod = pytest.importorskip("torch")
    if not torch_mod.cuda.is_available():
        pytest.skip("adapter import needs CUDA/mamba_ssm; skipped on CPU host")
    try:
        import model.adapter as adapter
    except Exception as e:
        pytest.skip(f"adapter not importable here: {e}")
    assert adapter.N_AUX_FEATURES == feature_set.N_FEATURES == 11


# ── F13 (2026-07-16): NTL-WAS form + nums-channel wiring (src/train.py) ──────
# number_token_loss_was / _ntl_loss / the nums-forward NTL live in train.py
# (model/ntl.py is owned elsewhere). Importing train pulls model.adapter, whose
# module-top mamba_ssm import needs CUDA — gate exactly like the aux-head test.
def _import_train():
    if not torch.cuda.is_available():
        pytest.skip("train.py imports model.adapter (mamba_ssm CUDA ext); skipped on CPU host")
    try:
        import train
    except Exception as e:
        pytest.skip(f"train not importable here: {e}")
    return train


def _peaked(vocab_slot: int, sharpness: float = 12.0) -> torch.Tensor:
    logits = torch.zeros(1, 1, VOCAB)
    logits[0, 0, vocab_slot] = sharpness
    return logits


def test_ntl_was_closer_digit_scores_lower():
    """The F13 acceptance case: predicted mass concentrated on 5 vs true 4 must
    yield LOWER was-loss than concentrated on 9 vs true 4 (ordinal CDF distance:
    W1(one-hot 5, one-hot 4) = 1 < W1(one-hot 9, one-hot 4) = 5)."""
    train = _import_train()
    digit_ids = _digit_ids()
    target = torch.tensor([[4]])
    on5 = train.number_token_loss_was(_peaked(5), target, digit_ids)
    on9 = train.number_token_loss_was(_peaked(9), target, digit_ids)
    assert on5.item() < on9.item(), (on5.item(), on9.item())
    # Near-one-hot predictions → W1 is essentially the digit distance.
    assert abs(on5.item() - 1.0) < 0.05
    assert abs(on9.item() - 5.0) < 0.05
    # Concentrating on the CORRECT digit → ~0.
    correct = train.number_token_loss_was(_peaked(4), target, digit_ids)
    assert correct.item() < 1e-2


def test_ntl_was_penalizes_symmetric_multimodal_mass():
    """WHY the WAS form exists: mass split evenly on y-1 and y+1 fools the
    legacy expected-value penalty (|E - y| = 0) but W1 sees it (= 1)."""
    train = _import_train()
    digit_ids = _digit_ids()
    target = torch.tensor([[4]])
    logits = torch.full((1, 1, VOCAB), -30.0)
    logits[0, 0, 3] = 10.0   # p(3) = p(5) = 0.5
    logits[0, 0, 5] = 10.0
    legacy = number_token_loss(logits, target, digit_ids)          # |E - 4| ~ 0
    was = train.number_token_loss_was(logits, target, digit_ids)   # W1 ~ 1
    assert legacy.item() < 0.05
    assert abs(was.item() - 1.0) < 0.05


def test_ntl_was_masks_and_no_digit_targets():
    train = _import_train()
    digit_ids = _digit_ids()
    # No digit-target positions → graph-connected 0 (same contract as legacy).
    logits = torch.randn(1, 2, VOCAB, requires_grad=True)
    zero = train.number_token_loss_was(logits, torch.tensor([[15, 20]]), digit_ids)
    assert zero.item() == 0.0
    zero.backward()
    assert torch.count_nonzero(logits.grad) == 0
    # target_mask excludes a wrong-digit position → only the correct one counts.
    target = torch.tensor([[3, 3]])
    lg = torch.zeros(1, 2, VOCAB)
    lg[0, 0, 3] = 12.0   # correct
    lg[0, 1, 9] = 12.0   # wrong, but masked out
    masked = train.number_token_loss_was(lg, target, digit_ids,
                                         target_mask=torch.tensor([[1, 0]]))
    assert masked.item() < 1e-2


def test_ntl_form_dispatch():
    """_ntl_loss routes 'was' → number_token_loss_was, 'mse' → legacy
    model.ntl.number_token_loss, and rejects unknown forms."""
    train = _import_train()
    digit_ids = _digit_ids()
    target = torch.tensor([[4]])
    logits = _peaked(9)
    was = train._ntl_loss(logits, target, digit_ids, None, "was")
    mse = train._ntl_loss(logits, target, digit_ids, None, "mse")
    assert torch.isclose(was, train.number_token_loss_was(logits, target, digit_ids))
    assert torch.isclose(mse, number_token_loss(logits, target, digit_ids))
    with pytest.raises(ValueError):
        train._ntl_loss(logits, target, digit_ids, None, "huber")


# ── F13 nums-channel wiring: NTL must fire on the nums forward's logits ──────
class _FakeCharTokenizer:
    """Char-level tokenizer: digits '0'-'9' → ids 0-9 (distinct single tokens,
    the digit_token_ids invariant), other chars → ids 10..28. eos=30, pad=31."""
    name_or_path = "fake-char-tokenizer"
    eos_token_id = 30
    pad_token_id = 31
    unk_token_id = None

    def _char_id(self, ch: str) -> int:
        return int(ch) if ch.isdigit() else 10 + (ord(ch) % 19)

    def encode(self, text, add_special_tokens=False):
        return [self._char_id(c) for c in text]

    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=False):
        from types import SimpleNamespace
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=ids)


class _MockLLM(torch.nn.Module):
    """Stands in for the HF causal LM: returns (loss, logits) shaped like the
    real forward and counts calls, so tests can pin how many forwards ran."""

    def __init__(self, vocab=32, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(dim, vocab)
        self.n_calls = 0

    # attention_mask accepted since 2026-07-27: _ce_against_target now passes an
    # explicit mask so the LM never attends over prefix/target PADDING. The mock
    # must mirror the real HF signature or it hides that call-site change.
    def forward(self, inputs_embeds=None, labels=None, output_hidden_states=False,
                attention_mask=None):
        from types import SimpleNamespace
        self.n_calls += 1
        logits = self.proj(inputs_embeds.float())
        return SimpleNamespace(
            loss=logits.mean(),   # graph-connected stand-in CE
            logits=logits,
            hidden_states=(inputs_embeds,) if output_hidden_states else None,
        )


class _MockAdapter(torch.nn.Module):
    def forward(self, audio_features, overlap_info):
        return torch.zeros(audio_features.shape[0], 3, 8)  # (B, N, dim) prefix


def _mock_loss_setup(train):
    tok = _FakeCharTokenizer()
    llm = _MockLLM()
    embed = torch.nn.Embedding(32, 8)
    adapter = _MockAdapter()
    prompt_ids = torch.tensor([[12, 13]])
    audio = torch.zeros(2, 6, 4)
    overlap = torch.zeros(2, 6, 4)
    return tok, llm, embed, adapter, prompt_ids, audio, overlap


def test_ntl_invoked_on_nums_logits_when_active(monkeypatch):
    """lambda_ntl>0 + nums target present → _ntl_loss fires TWICE (prose then
    nums), the nums call sees the nums target's digit ids, loss_ntl_nums > 0,
    and NO extra LM forward is added (the tensors ride the existing forwards)."""
    train = _import_train()
    tok, llm, embed, adapter, prompt_ids, audio, overlap = _mock_loss_setup(train)
    calls = []
    real_ntl = train._ntl_loss

    def spy(logits, target_ids, digit_ids, target_mask, form):
        calls.append({"target_ids": target_ids.detach().clone(), "form": form})
        return real_ntl(logits, target_ids, digit_ids, target_mask, form)

    monkeypatch.setattr(train, "_ntl_loss", spy)
    config = {"max_target_length": 24, "max_nums_length": 12,
              "lambda_prose": 1.0, "lambda_nums": 1.0, "lambda_mse": 0.0,
              "lambda_ntl": 0.5, "ntl_form": "was"}
    total, metrics = train.compute_loss(
        adapter, llm, embed, tok, audio, overlap,
        ["snr is 15.", "hnr is 8."], prompt_ids, torch.device("cpu"), config,
        target_nums=["snr=15", "hnr=8"],
    )
    assert len(calls) == 2, "NTL must run on BOTH the prose and nums channels"
    assert all(c["form"] == "was" for c in calls)
    # The second (nums) call's targets carry the nums string's digits 1 and 5.
    nums_ids = calls[1]["target_ids"]
    assert bool((nums_ids == 1).any()) and bool((nums_ids == 5).any())
    assert metrics["loss_ntl_nums"] > 0.0
    assert metrics["loss_ntl_prose"] > 0.0
    assert metrics["loss_ntl"] == pytest.approx(
        metrics["loss_ntl_prose"] + metrics["loss_ntl_nums"], rel=1e-5)
    assert llm.n_calls == 2   # prose + nums forwards only — NTL adds none


def test_ntl_nums_silent_when_lambda_zero(monkeypatch):
    """lambda_ntl=0 (and no unlikelihood) → _ntl_loss never called, loss_ntl
    keys are 0.0, and the forward count is unchanged (byte-identical path)."""
    train = _import_train()
    tok, llm, embed, adapter, prompt_ids, audio, overlap = _mock_loss_setup(train)
    calls = []
    monkeypatch.setattr(
        train, "_ntl_loss",
        lambda *a, **k: calls.append(a) or torch.tensor(0.0))
    config = {"max_target_length": 24, "max_nums_length": 12,
              "lambda_prose": 1.0, "lambda_nums": 1.0, "lambda_mse": 0.0,
              "lambda_ntl": 0.0}
    _, metrics = train.compute_loss(
        adapter, llm, embed, tok, audio, overlap,
        ["snr is 15.", "hnr is 8."], prompt_ids, torch.device("cpu"), config,
        target_nums=["snr=15", "hnr=8"],
    )
    assert calls == []
    assert metrics["loss_ntl"] == 0.0
    assert metrics["loss_ntl_prose"] == 0.0
    assert metrics["loss_ntl_nums"] == 0.0
    assert llm.n_calls == 2   # prose + nums


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
