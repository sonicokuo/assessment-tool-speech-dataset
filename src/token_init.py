"""token_init.py — semantic warm-start for the added <sec_*>/<f_*> special tokens.

PROBLEM
-------
train.py registers 19 new special tokens (6 section opens + </sec> + 9 feature
opens + </f> + <r>/</r>) and MEAN-initializes every new embedding/lm_head row to
the SAME vector (the average of the pretrained vocab). So at step 0 the model
cannot tell <sec_noise> from <f_f0_sd> — every bit of differentiation must be
learned from 14k clips, and on the small/section path that fragility shows up as
tag-spam / foreign-token degeneration (the v11 failure).

FIX
---
Initialize each NEW token's row from the mean embedding of its descriptive
words (its catalog display_name), e.g. <sec_pitch> from the subwords of "pitch",
<f_snr> from "signal-to-noise ratio". The model starts with tokens that already
point in a sensible direction (this is the same trick used to warm-start added
tokens in the literature). Closing/range markers (</sec>, </f>, <r>, </r>) have
no natural phrase and keep the mean-init fallback.

The matrix-writing core is pure (takes tensors + a row->subword-id map) so it is
unit-testable without a real LM; build_semantic_tag_init derives the map from the
SECTION_TAGS / FEATURE_TAGS catalog + the tokenizer.
"""
from __future__ import annotations

import torch

from section_tags import SECTION_TAGS, FEATURE_TAGS


def tag_descriptions() -> dict[str, str]:
    """{open-tag string -> descriptive phrase} from the catalog display names."""
    d: dict[str, str] = {}
    for s in SECTION_TAGS:
        d[s.tag] = s.display_name
    for f in FEATURE_TAGS:
        d[f.tag] = f.display_name
    return d


def build_semantic_tag_init(tokenizer, old_vocab_size: int) -> dict[int, list[int]]:
    """row_id (the NEW token's id) -> existing subword ids of its phrase.

    Only includes tags whose id is >= old_vocab_size (genuinely new) and whose
    phrase tokenizes to at least one in-vocab subword. Closing/range markers are
    absent from tag_descriptions() and so keep the caller's mean-init.
    """
    out: dict[int, list[int]] = {}
    for tag, phrase in tag_descriptions().items():
        rid = tokenizer.convert_tokens_to_ids(tag)
        if rid is None or rid < old_vocab_size:
            continue
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        ids = [int(i) for i in ids if 0 <= int(i) < old_vocab_size]
        if ids:
            out[rid] = ids
    return out


@torch.no_grad()
def semantic_init_new_rows(
    in_emb_weight: torch.Tensor,
    out_emb_weight: torch.Tensor | None,
    old_vocab_size: int,
    row_to_subwords: dict[int, list[int]],
) -> int:
    """Set each new-token row to the mean of its descriptive-word subword rows.

    in_emb_weight / out_emb_weight: the (vocab, dim) input and (untied) output
    embedding weights. old_vocab_size guards against using new rows as sources.
    Returns the number of rows initialized. Pure / in-place.
    """
    n = 0
    for row_id, subwords in row_to_subwords.items():
        src = [i for i in subwords if 0 <= i < old_vocab_size]
        if not src or row_id < old_vocab_size or row_id >= in_emb_weight.shape[0]:
            continue
        idx = torch.tensor(src, dtype=torch.long, device=in_emb_weight.device)
        in_emb_weight[row_id] = in_emb_weight[idx].mean(dim=0).to(in_emb_weight.dtype)
        if out_emb_weight is not None and out_emb_weight.shape[0] > row_id:
            out_emb_weight[row_id] = out_emb_weight[idx].mean(dim=0).to(out_emb_weight.dtype)
        n += 1
    return n
