"""Tests for src/token_init.py — semantic warm-start of added special tokens."""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from token_init import (  # noqa: E402
    tag_descriptions,
    build_semantic_tag_init,
    semantic_init_new_rows,
)
from section_tags import SECTION_TAGS, FEATURE_TAGS, SPECIAL_TOKENS  # noqa: E402

N_OPEN_TAGS = len(SECTION_TAGS) + len(FEATURE_TAGS)  # 15 open tags carry phrases


# ── catalog → descriptions ───────────────────────────────────────────────────
def test_tag_descriptions_cover_all_open_tags():
    d = tag_descriptions()
    for s in SECTION_TAGS:
        assert d[s.tag] == s.display_name
    for f in FEATURE_TAGS:
        assert d[f.tag] == f.display_name
    # closing/range markers are intentionally absent (mean-init fallback)
    assert "</sec>" not in d and "<r>" not in d


# ── the pure init core ───────────────────────────────────────────────────────
def test_semantic_init_sets_rows_to_word_mean_and_distinct():
    old_vocab = 10
    dim = 4
    emb = torch.nn.Embedding(13, dim)
    torch.nn.init.normal_(emb.weight, std=1.0)
    # 3 new rows (10,11,12) each from different existing subwords
    row_to_sub = {10: [0, 1], 11: [5], 12: [2, 3, 4]}
    before = emb.weight.detach().clone()

    n = semantic_init_new_rows(emb.weight, None, old_vocab, row_to_sub)
    assert n == 3

    # each new row == mean of its source rows (sources unchanged: < old_vocab)
    assert torch.allclose(emb.weight[10], before[[0, 1]].mean(0), atol=1e-6)
    assert torch.allclose(emb.weight[11], before[5], atol=1e-6)
    assert torch.allclose(emb.weight[12], before[[2, 3, 4]].mean(0), atol=1e-6)
    # the three new rows are DISTINCT (the whole point vs identical mean-init)
    assert not torch.allclose(emb.weight[10], emb.weight[11])
    assert not torch.allclose(emb.weight[11], emb.weight[12])
    # old rows untouched
    assert torch.allclose(emb.weight[:old_vocab], before[:old_vocab])


def test_semantic_init_writes_output_embeddings_when_untied():
    old_vocab = 8
    in_w = torch.randn(11, 3)
    out_w = torch.randn(11, 3)
    in_before, out_before = in_w.clone(), out_w.clone()
    semantic_init_new_rows(in_w, out_w, old_vocab, {8: [0, 1], 9: [2], 10: [3, 4]})
    assert torch.allclose(in_w[8], in_before[[0, 1]].mean(0), atol=1e-6)
    assert torch.allclose(out_w[8], out_before[[0, 1]].mean(0), atol=1e-6)


def test_semantic_init_skips_bad_rows():
    old_vocab = 5
    w = torch.randn(7, 2)
    before = w.clone()
    # row 3 is < old_vocab (not new) → skip; row 99 out of range → skip;
    # subwords referencing new rows (>=old_vocab) are filtered out → empty → skip
    n = semantic_init_new_rows(w, None, old_vocab, {3: [0], 99: [0], 6: [5, 6]})
    assert n == 0
    assert torch.allclose(w, before)


# ── catalog builder with a tiny fake tokenizer ───────────────────────────────
class _FakeTok:
    """Maps each known tag to a fixed new id; words to deterministic subword ids."""
    def __init__(self, old_vocab):
        self.old_vocab = old_vocab
        self._tag_ids = {}
        nxt = old_vocab
        for t in [s.tag for s in SECTION_TAGS] + [f.tag for f in FEATURE_TAGS]:
            self._tag_ids[t] = nxt
            nxt += 1

    def convert_tokens_to_ids(self, tag):
        return self._tag_ids.get(tag)

    def __call__(self, phrase, add_special_tokens=False):
        # one subword per character, mapped into [0, old_vocab)
        ids = [(ord(c) % (self.old_vocab - 1)) + 1 for c in phrase if not c.isspace()]
        return types.SimpleNamespace(input_ids=ids)


def test_build_semantic_tag_init_from_catalog():
    old_vocab = 100
    tok = _FakeTok(old_vocab)
    m = build_semantic_tag_init(tok, old_vocab)
    # every section + feature open tag gets a row with non-empty in-vocab subwords
    assert len(m) == len(SECTION_TAGS) + len(FEATURE_TAGS)
    for rid, subs in m.items():
        assert rid >= old_vocab
        assert subs and all(0 <= s < old_vocab for s in subs)


# ── regression: Qwen-style padded embedding vs tokenizer length (the R10 bug) ──
class _PaddedVocabTok:
    """Real-tokenizer-like with a Qwen3 padding gap.

    The tokenizer assigns new-token ids starting at `len(tokenizer)` (the REAL
    vocab length), but the model's embedding matrix is padded to a LARGER row
    count (e.g. Qwen3-8B: 151669 real tokens, 151936 embedding rows). So the new
    <sec_*>/<f_*> ids land in [len_tok, len_tok + n_added) — BELOW the padded
    embedding-matrix row count. Passing that padded count as the "new token"
    boundary rejects every open tag (the R10 `0 open tags` log).
    """

    def __init__(self, len_tok: int):
        self._len = len_tok
        self._ids = {}

    def add_tokens(self, tokens):
        added = 0
        for t in tokens:
            if t not in self._ids:
                self._ids[t] = self._len
                self._len += 1
                added += 1
        return added

    def __len__(self):
        return self._len

    def convert_tokens_to_ids(self, tag):
        return self._ids.get(tag)

    def __call__(self, phrase, add_special_tokens=False):
        # one in-vocab subword per non-space char (deterministic, < first new id)
        first_new = min(self._ids.values()) if self._ids else self._len
        ids = [(ord(c) % (first_new - 1)) + 1 for c in phrase if not c.isspace()]
        return types.SimpleNamespace(input_ids=ids)


def test_padded_embedding_uses_tokenizer_boundary_not_emb_rows():
    """Reproduce + guard the R10 bug: new-token boundary is len(tokenizer)-added,
    NOT the padded embedding-matrix row count.

    With the padded embedding size as the boundary (the bug) the warm-start
    matches ZERO open tags. With the pre-add tokenizer length (the fix) it
    matches all 15, and the initialized rows are distinct.
    """
    # Mirror Qwen3-8B: padding gap (emb_pad - len_tok) LARGER than tokens added,
    # so ALL new ids fall inside the gap (below emb_pad) and the bug drops them all.
    len_tok = 100        # real tokenizer length before add
    emb_pad = 150        # embedding matrix rows before add (gap = 50 > 19 added)
    dim = 8

    tok = _PaddedVocabTok(len_tok)
    added = tok.add_tokens(SPECIAL_TOKENS)   # mirrors train.py add_tokens(...)
    assert added == len(SPECIAL_TOKENS) == 19
    # EVERY new id sits inside the padding gap (below emb_pad) → triggers the bug
    new_ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    assert min(new_ids) == len_tok and max(new_ids) < emb_pad

    # (bug) padded embedding row count as the boundary → 0 open tags matched
    assert len(build_semantic_tag_init(tok, emb_pad)) == 0

    # (fix) pre-add tokenizer length as the boundary → all 15 open tags matched
    new_token_start = len(tok) - added       # == len_tok
    row_map = build_semantic_tag_init(tok, new_token_start)
    assert len(row_map) == N_OPEN_TAGS == 15

    # Build a padded+resized embedding table and mean-init the tail rows exactly
    # like train.py (rows >= emb_pad get the SAME pretrained-mean vector).
    emb = torch.nn.Embedding(emb_pad + added, dim)
    torch.nn.init.normal_(emb.weight, std=1.0)
    with torch.no_grad():
        mean_in = emb.weight[:emb_pad].mean(0)
        emb.weight[emb_pad:].copy_(mean_in)
    mean_row = emb.weight[emb_pad].clone()   # the identical mean-init vector

    n = semantic_init_new_rows(emb.weight, None, new_token_start, row_map)
    assert n == N_OPEN_TAGS == 15            # (a) 15 initialized, not 0

    # (b) the initialized open-tag rows are NOT all identical, and each differs
    #     from the plain mean-init vector (semantic, not mean, init).
    init_rows = [emb.weight[r] for r in row_map]
    assert not all(torch.allclose(init_rows[0], r, atol=1e-6) for r in init_rows)
    assert all(not torch.allclose(r, mean_row, atol=1e-6) for r in init_rows)
    # each row equals the mean of its phrase subwords (semantic derivation)
    for rid, subs in row_map.items():
        assert torch.allclose(
            emb.weight[rid],
            emb.weight[torch.tensor(subs)].mean(0),
            atol=1e-6,
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
