"""Tests for the [ditto] sentence-level anti-repetition term (F14, 2026-07-16).

src/train.py implements a DITTO-style penalty (Xu et al., NeurIPS 2022,
"Learning to Break the Loop"): per clip, ONE sentence from the prose target is
tiled k times (k in [8,16]) into a pseudo-repetitive sequence, one extra LM
forward runs on (prefix + prompt + tiled target), and -log(1-p) is applied ONLY
to the tokens of the copies AFTER the first occurrence. Design constraint from
the 2026-07-15 memo: the real targets INTENTIONALLY repeat the sentence FRAME
with different values, so the penalty must apply only to IDENTICAL repeated
sentences — the tiling construction guarantees that, and the natural targets
are never penalized. lambda_ditto defaults to 0.0 (byte-identical training).

Covers:
  - build_ditto_tiled_batch: identical sentence copies, k within [k_min, k_max],
    penalty mask 0 on the first copy / 1 on every later copy, truncation to
    max_target_length, empty-target rows contribute nothing.
  - ditto_unlikelihood_loss: positive on a repeated sequence; first-occurrence
    (mask=0) positions can NEVER contribute; all-zero mask → exactly 0.
  - compute_loss gating: lambda_ditto=0 (the default) runs NO extra forward and
    logs loss_ditto=0.0; lambda_ditto>0 adds exactly ONE forward and a positive
    loss_ditto.

Importing train pulls model.adapter (module-top mamba_ssm import → CUDA), so
every test gates through _import_train() and skips on a CPU host — same
convention as tests/test_ntl.py. Run on a PSC GPU node.
"""

import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _import_train():
    if not torch.cuda.is_available():
        pytest.skip("train.py imports model.adapter (mamba_ssm CUDA ext); skipped on CPU host")
    try:
        import train
    except Exception as e:
        pytest.skip(f"train not importable here: {e}")
    return train


# ── Fakes (mirror tests/test_ntl.py so either file runs standalone) ───────────
class _FakeCharTokenizer:
    """Char-level tokenizer: digits '0'-'9' → ids 0-9, other chars → ids 10..28.
    eos=30, pad=31 — vocab 32 matches the mock LM head below."""
    name_or_path = "fake-char-tokenizer"
    eos_token_id = 30
    pad_token_id = 31
    unk_token_id = None

    def _char_id(self, ch: str) -> int:
        return int(ch) if ch.isdigit() else 10 + (ord(ch) % 19)

    def encode(self, text, add_special_tokens=False):
        return [self._char_id(c) for c in text]

    def __call__(self, text, truncation=True, max_length=None, add_special_tokens=False):
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=ids)


class _MockLLM(torch.nn.Module):
    """Counts forwards so tests can pin exactly how many the ditto path adds."""

    def __init__(self, vocab=32, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(dim, vocab)
        self.n_calls = 0

    # attention_mask accepted since 2026-07-27: _ce_against_target now passes an
    # explicit mask so the LM never attends over prefix/target PADDING. The mock
    # must mirror the real HF signature or it hides that call-site change.
    def forward(self, inputs_embeds=None, labels=None, output_hidden_states=False,
                attention_mask=None):
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


SENT = "The SNR is 15 dB."
CPU = torch.device("cpu")


# ── _split_sentences ──────────────────────────────────────────────────────────
def test_split_sentences_basic():
    train = _import_train()
    text = "The SNR is 15 dB. The HNR is 8 dB! Is it clean? Yes."
    sents = train._split_sentences(text)
    assert sents == ["The SNR is 15 dB.", "The HNR is 8 dB!", "Is it clean?", "Yes."]
    assert train._split_sentences("") == []
    assert train._split_sentences(None) == []


# ── build_ditto_tiled_batch ───────────────────────────────────────────────────
def test_tiled_construction_identical_sentences_k_in_range():
    """A single-sentence target → the sampled sentence is deterministic. The
    tiling must be: first copy verbatim, then k-1 IDENTICAL leading-space
    copies, k in [8,16]; penalty mask 0 exactly on the first copy, 1 on every
    repetition token."""
    train = _import_train()
    tok = _FakeCharTokenizer()
    tiled, pen = train.build_ditto_tiled_batch(
        tok, [SENT], max_length=100_000, device=CPU, rng=random.Random(0))
    first = tok.encode(SENT)
    rep = tok.encode(" " + SENT)
    L = tiled.shape[1]
    # Recover k from the lengths: L = len(first) + (k-1) * len(rep), k in [8,16].
    assert (L - len(first)) % len(rep) == 0
    k = 1 + (L - len(first)) // len(rep)
    assert 8 <= k <= 16, k
    # First copy verbatim, then identical repetition blocks.
    assert tiled[0, : len(first)].tolist() == first
    for i in range(k - 1):
        lo = len(first) + i * len(rep)
        assert tiled[0, lo: lo + len(rep)].tolist() == rep
    # Penalty mask: 0 on the first occurrence, 1 on ALL later copies.
    assert pen[0, : len(first)].sum().item() == 0
    assert pen[0, len(first):].min().item() == 1
    assert int(pen.sum().item()) == (k - 1) * len(rep)


def test_tiled_k_bounds_hold_across_seeds():
    train = _import_train()
    tok = _FakeCharTokenizer()
    first = tok.encode(SENT)
    rep = tok.encode(" " + SENT)
    for seed in range(8):
        tiled, _ = train.build_ditto_tiled_batch(
            tok, [SENT], max_length=100_000, device=CPU, rng=random.Random(seed))
        k = 1 + (tiled.shape[1] - len(first)) // len(rep)
        assert 8 <= k <= 16, (seed, k)


def test_tiled_truncates_to_max_length():
    train = _import_train()
    tok = _FakeCharTokenizer()
    max_len = 40   # SENT is 17 chars → k>=8 tiling far exceeds this
    tiled, pen = train.build_ditto_tiled_batch(
        tok, [SENT], max_length=max_len, device=CPU, rng=random.Random(0))
    assert tiled.shape[1] == max_len
    assert pen.shape == tiled.shape
    first = tok.encode(SENT)
    # First copy still unpenalized; everything after it (up to the cut) is.
    assert pen[0, : len(first)].sum().item() == 0
    assert pen[0, len(first):].min().item() == 1


def test_tiled_empty_target_contributes_nothing():
    train = _import_train()
    tok = _FakeCharTokenizer()
    tiled, pen = train.build_ditto_tiled_batch(
        tok, ["", SENT], max_length=200, device=CPU, rng=random.Random(0))
    assert pen[0].sum().item() == 0          # empty clip: all-zero penalty row
    assert pen[1].sum().item() > 0           # real clip unaffected
    assert tiled.shape == pen.shape


# ── ditto_unlikelihood_loss ───────────────────────────────────────────────────
def test_ditto_loss_positive_on_repeats_and_zero_when_unmasked():
    train = _import_train()
    B, L, V = 1, 6, 12
    target = torch.tensor([[3, 4, 5, 3, 4, 5]])        # sentence "345" tiled twice
    pen = torch.tensor([[0, 0, 0, 1, 1, 1]])           # penalize only the 2nd copy
    uniform = torch.zeros(B, L, V)
    loss = train.ditto_unlikelihood_loss(uniform, target, pen)
    assert loss.item() > 0.0                            # -log(1 - 1/V) per token
    # All-zero mask (lambda_ditto path off / empty targets) → exactly 0.
    zero = train.ditto_unlikelihood_loss(uniform, target, torch.zeros_like(pen))
    assert zero.item() == 0.0


def test_ditto_penalizes_only_repetitions_after_first():
    """Boosting p(target) at FIRST-OCCURRENCE (mask=0) positions must not move
    the loss; boosting it at penalized positions must increase it."""
    train = _import_train()
    B, L, V = 1, 6, 12
    target = torch.tensor([[3, 4, 5, 3, 4, 5]])
    pen = torch.tensor([[0, 0, 0, 1, 1, 1]])
    base = torch.zeros(B, L, V)
    loss_base = train.ditto_unlikelihood_loss(base, target, pen)
    # Near-certain re-emission on the FIRST copy only → identical loss.
    first_hot = base.clone()
    for j in range(3):
        first_hot[0, j, target[0, j]] = 20.0
    loss_first = train.ditto_unlikelihood_loss(first_hot, target, pen)
    assert torch.isclose(loss_first, loss_base, atol=1e-6)
    # Near-certain re-emission on the REPEATED copy → loss blows up.
    rep_hot = base.clone()
    for j in range(3, 6):
        rep_hot[0, j, target[0, j]] = 20.0
    loss_rep = train.ditto_unlikelihood_loss(rep_hot, target, pen)
    assert loss_rep.item() > loss_base.item() + 1.0


# ── compute_loss gating (lambda_ditto) ────────────────────────────────────────
def _mock_loss_setup():
    tok = _FakeCharTokenizer()
    llm = _MockLLM()
    embed = torch.nn.Embedding(32, 8)
    adapter = _MockAdapter()
    prompt_ids = torch.tensor([[12, 13]])
    audio = torch.zeros(2, 6, 4)
    overlap = torch.zeros(2, 6, 4)
    return tok, llm, embed, adapter, prompt_ids, audio, overlap


def test_compute_loss_ditto_off_by_default():
    """No lambda_ditto in config (the default) → loss_ditto 0.0 and NO extra
    forward: the training path is byte-identical to pre-F14."""
    train = _import_train()
    tok, llm, embed, adapter, prompt_ids, audio, overlap = _mock_loss_setup()
    config = {"max_target_length": 24, "lambda_prose": 1.0}
    _, metrics = train.compute_loss(
        adapter, llm, embed, tok, audio, overlap,
        [SENT, "The HNR is 8 dB."], prompt_ids, CPU, config,
    )
    assert metrics["loss_ditto"] == 0.0
    assert llm.n_calls == 1     # prose forward only


def test_compute_loss_ditto_on_adds_one_forward_and_positive_loss():
    train = _import_train()
    tok, llm, embed, adapter, prompt_ids, audio, overlap = _mock_loss_setup()
    config = {"max_target_length": 24, "lambda_prose": 1.0, "lambda_ditto": 0.5}
    _, metrics = train.compute_loss(
        adapter, llm, embed, tok, audio, overlap,
        [SENT, "The HNR is 8 dB."], prompt_ids, CPU, config,
    )
    assert metrics["loss_ditto"] > 0.0
    assert llm.n_calls == 2     # prose + ONE ditto forward, no more


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
