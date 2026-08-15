"""F1 (2026-07-16): the decode stop-set must cover BOTH Qwen enders.

Post-trained Qwen3 has tokenizer.eos_token = <|im_end|> (151645) and the
base/document ender <|endoftext|> (151643); the shipped generation_config
stops on both. A single-id stop set misses an emitted <|endoftext|> and the
generation continues as off-task boilerplate (the degeneration class the
2026-07-15 root-cause memo attributes to the swallowed document ender).

These tests pin the helper's contract without loading a real model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import _val_eos_ids  # noqa: E402


class _QwenLikeTokenizer:
    """Post-trained Qwen3 shape: eos=<|im_end|>, distinct <|endoftext|>."""
    eos_token_id = 151645
    unk_token_id = None

    def convert_tokens_to_ids(self, tok):
        return {"<|endoftext|>": 151643, "<|im_end|>": 151645}.get(tok)


class _BaseLikeTokenizer:
    """Base-model shape: eos IS <|endoftext|> — no duplicate in the stop set."""
    eos_token_id = 151643
    unk_token_id = None

    def convert_tokens_to_ids(self, tok):
        return {"<|endoftext|>": 151643}.get(tok)


class _NoEotTokenizer:
    """Tokenizer without <|endoftext|> (unk fallback) — degrade to [eos]."""
    eos_token_id = 2
    unk_token_id = 0

    def convert_tokens_to_ids(self, tok):
        return 0  # unk


def test_post_trained_qwen_gets_both_enders():
    assert _val_eos_ids(_QwenLikeTokenizer()) == [151645, 151643]


def test_base_model_no_duplicates():
    assert _val_eos_ids(_BaseLikeTokenizer()) == [151643]


def test_missing_endoftext_degrades_to_eos_only():
    assert _val_eos_ids(_NoEotTokenizer()) == [2]


def test_inference_stop_set_matches_helper():
    """inference.py builds its own eos_ids set inline; keep the two paths in
    lockstep by asserting the inline construction (replicated here from
    inference.py) yields the same ids as _val_eos_ids."""
    tok = _QwenLikeTokenizer()
    eos_ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    if eot is not None and eot != tok.unk_token_id:
        eos_ids.add(eot)
    assert eos_ids == set(_val_eos_ids(tok))
