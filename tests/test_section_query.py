"""Tests for the section-query path: SpecEncoder + SectionQueryHead + injection.

Heavy tests (full BEATs forward) are gated behind a `RUN_BEATS_TEST=1` env var
so they don't slow down the default test sweep. The lighter unit tests on
SectionQueryHead's math run unconditionally.

To run the BEATs integration test:
    RUN_BEATS_TEST=1 conda run -n idl python -m pytest tests/test_section_query.py -v
"""

import importlib.machinery
import os
import sys
import types

import pytest
import torch

# Stub mamba_ssm so train.py can be imported on machines without the CUDA-only
# mamba build (every Mac dev box). Mirrors the trick used in test_compute_loss_b_full.py.
if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    stub.Mamba = object
    sys.modules["mamba_ssm"] = stub


# ─── Lightweight unit tests on SectionQueryHead ──────────────────────────────


class TestSectionQueryHead:
    def setup_method(self):
        from section_query import SectionQueryHead
        self.head = SectionQueryHead(
            n_sections=6, d_patch=768, d_lm=2048, d_k=256, d_v=256,
        )

    def test_param_count_is_small(self):
        # Sanity check: section head should be <2M params (negligible vs LM).
        n_params = sum(p.numel() for p in self.head.parameters())
        assert n_params < 2_000_000, f"section_head has {n_params} params, expected <2M"

    def test_precompute_kv_shapes(self):
        patches = torch.randn(3, 248, 768)
        K, V = self.head.precompute_kv(patches)
        assert K.shape == (3, 248, 256)
        assert V.shape == (3, 248, 256)

    def test_forward_single_section(self):
        patches = torch.randn(2, 100, 768)
        K, V = self.head.precompute_kv(patches)
        section_idx = torch.tensor([0, 3])  # noise for clip 0, tempo for clip 1
        e_t, alpha = self.head(section_idx, K, V)
        assert e_t.shape == (2, 2048)
        assert alpha.shape == (2, 100)
        # Attention rows sum to 1
        assert torch.allclose(alpha.sum(-1), torch.ones(2), atol=1e-5)
        # All weights non-negative
        assert (alpha >= 0).all()

    def test_forward_all_sections(self):
        patches = torch.randn(2, 100, 768)
        K, V = self.head.precompute_kv(patches)
        e_all, alpha_all = self.head.forward_all_sections(K, V)
        assert e_all.shape == (2, 6, 2048)
        assert alpha_all.shape == (2, 6, 100)
        assert torch.allclose(alpha_all.sum(-1), torch.ones(2, 6), atol=1e-5)

    def test_queries_differ_per_section(self):
        # Each section's static query should be distinct after init (randn scaled by 0.02).
        q = self.head.queries
        assert q.shape == (6, 256)
        # Distances between every pair > 0
        for i in range(6):
            for j in range(i + 1, 6):
                assert not torch.allclose(q[i], q[j])

    def test_attention_maps_differ_per_section_with_same_input(self):
        # Given the same K/V, different sections should produce different α
        # (because the static queries are different).
        torch.manual_seed(0)
        patches = torch.randn(1, 50, 768)
        K, V = self.head.precompute_kv(patches)
        e_all, alpha_all = self.head.forward_all_sections(K, V)
        # at least one pair of sections should have noticeably different alphas
        max_diff = 0.0
        for i in range(6):
            for j in range(i + 1, 6):
                d = (alpha_all[0, i] - alpha_all[0, j]).abs().max().item()
                max_diff = max(max_diff, d)
        assert max_diff > 1e-4, f"all section alphas suspiciously equal (max pairwise diff = {max_diff})"

    def test_gradient_flows_through_queries_and_W(self):
        patches = torch.randn(2, 50, 768, requires_grad=False)
        K, V = self.head.precompute_kv(patches)
        e_t, alpha = self.head(torch.tensor([0, 0]), K, V)
        loss = e_t.sum() + alpha.sum()
        loss.backward()
        # Gradient through the alpha (attention-weight) path: K (via scores),
        # queries (via scores). These don't depend on the tanh(gate) so they
        # flow regardless of gate state.
        assert self.head.queries.grad is not None
        assert self.head.queries.grad.abs().sum() > 0
        assert self.head.W_k.weight.grad.abs().sum() > 0

        # Flamingo zero-init property: at gate=0 the e_t pathway is multiplied
        # by tanh(0)=0, so W_o and W_v receive NO gradient at the very first
        # step. They unblock once the gate moves slightly (see
        # test_W_o_W_v_gradient_unblocks_after_gate_moves below).
        assert self.head.W_o.weight.grad is None or self.head.W_o.weight.grad.abs().sum() == 0
        assert self.head.W_v.weight.grad is None or self.head.W_v.weight.grad.abs().sum() == 0

    def test_zero_init_contribution_is_exactly_zero(self):
        """Flamingo property #1: at init the cross-attn contribution is exactly 0.

        Ensures adding the section_head to a model is byte-identical to the
        base model at step 0. Without this property the section_head injects
        random noise into the LM from the very first batch, destabilising
        training (this was the actual bug in v3 that collapsed val_sfs_f1 to 0
        by epoch 4).
        """
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        e_t, _ = self.head(torch.tensor([0, 0]), K, V)
        assert e_t.abs().max().item() == 0.0, \
            f"e_t at init must be EXACTLY zero, got max |e_t| = {e_t.abs().max().item()}"

    def test_gate_gets_gradient_at_init(self):
        """Flamingo property #2: gate gets non-zero gradient at init.

        sech^2(0) = 1, so d(tanh(alpha))/d(alpha) at alpha=0 is 1 (not 0).
        This is what lets the gate learn from step 1 even though its current
        value zeros out the e_t pathway. Without this property the gate
        would be stuck at 0 forever and the cross-attention path would never
        activate.
        """
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        e_t, _ = self.head(torch.tensor([0, 0]), K, V)
        loss = e_t.sum()
        loss.backward()
        assert self.head.injection_gate.grad is not None
        assert self.head.injection_gate.grad.abs().sum() > 0, \
            "gate must receive gradient at init (sech^2(0)=1)"

    def test_W_o_W_v_gradient_unblocks_after_gate_moves(self):
        """Once the gate moves slightly off zero, W_o and W_v start training."""
        # Simulate a few optimisation steps having moved the gate slightly.
        with torch.no_grad():
            self.head.injection_gate.fill_(0.1)
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        e_t, _ = self.head(torch.tensor([0, 0]), K, V)
        loss = e_t.sum()
        loss.backward()
        assert self.head.W_o.weight.grad.abs().sum() > 0, \
            "W_o must receive gradient once gate is nonzero"
        assert self.head.W_v.weight.grad.abs().sum() > 0, \
            "W_v must receive gradient once gate is nonzero"


class TestSectionQueryHeadDynamic:
    """Tests for the dynamic-query forward used by section_query_mode='dynamic'."""

    def setup_method(self):
        from section_query import SectionQueryHead
        self.head = SectionQueryHead(n_sections=6, d_patch=768, d_lm=2048, d_k=256, d_v=256)

    def test_forward_dynamic_shape_a_with_batch_idx(self):
        # N queries; K, V are per-clip; batch_idx maps query → clip.
        patches = torch.randn(3, 100, 768)
        K, V = self.head.precompute_kv(patches)
        h_t = torch.randn(5, 2048)
        batch_idx = torch.tensor([0, 0, 1, 1, 2])
        e_t, alpha = self.head.forward_dynamic(h_t, K, V, batch_idx=batch_idx)
        assert e_t.shape == (5, 2048)
        assert alpha.shape == (5, 100)
        assert torch.allclose(alpha.sum(-1), torch.ones(5), atol=1e-5)

    def test_forward_dynamic_shape_b_one_per_clip(self):
        # Per-clip query: B queries, no batch_idx needed.
        patches = torch.randn(3, 100, 768)
        K, V = self.head.precompute_kv(patches)
        h_t = torch.randn(3, 2048)
        e_t, alpha = self.head.forward_dynamic(h_t, K, V)
        assert e_t.shape == (3, 2048)
        assert alpha.shape == (3, 100)
        assert torch.allclose(alpha.sum(-1), torch.ones(3), atol=1e-5)

    def test_dynamic_query_differs_when_h_t_differs(self):
        # Same K/V, two different h_t → different attention maps.
        torch.manual_seed(0)
        patches = torch.randn(1, 50, 768)
        K, V = self.head.precompute_kv(patches)
        # Bigger init on W_q so the two queries are distinguishable at random init
        torch.nn.init.normal_(self.head.W_q.weight, mean=0.0, std=0.5)
        h_a = torch.randn(1, 2048)
        h_b = torch.randn(1, 2048)
        _, alpha_a = self.head.forward_dynamic(h_a, K, V)
        _, alpha_b = self.head.forward_dynamic(h_b, K, V)
        # The attention vectors should be meaningfully different
        diff = (alpha_a - alpha_b).abs().max().item()
        assert diff > 1e-4

    def test_dynamic_gradient_flows_to_Wq(self):
        # Move the gate slightly off zero to test the post-init behavior.
        # At gate=0 (Flamingo init), e_t = tanh(0) * W_o(z) = 0 and
        # alpha.sum() per row is constant (softmax sums to 1), so both
        # paths give zero gradient. The interesting question — does the
        # cross-attention propagate gradient to W_q and h_t once the gate
        # has moved? — needs gate != 0.
        with torch.no_grad():
            self.head.injection_gate.fill_(0.1)
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        h_t = torch.randn(3, 2048, requires_grad=True)
        e_t, alpha = self.head.forward_dynamic(
            h_t, K, V, batch_idx=torch.tensor([0, 1, 1]),
        )
        loss = e_t.sum() + alpha.sum()
        loss.backward()
        # W_q gets grad through the e_t pathway
        assert self.head.W_q.weight.grad.abs().sum() > 0
        # h_t (upstream) gets grad — this is what closes the loop back to the LM
        assert h_t.grad is not None
        assert h_t.grad.abs().sum() > 0
        # Static queries are unused on this path
        assert self.head.queries.grad is None

    def test_dynamic_zero_init_property(self):
        """Same Flamingo property as static path: at gate=0, e_t is exactly 0
        and only the gate itself receives gradient."""
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        h_t = torch.randn(3, 2048)
        e_t, _ = self.head.forward_dynamic(
            h_t, K, V, batch_idx=torch.tensor([0, 1, 1]),
        )
        assert e_t.abs().max().item() == 0.0, \
            "dynamic e_t at init must also be exactly 0"
        e_t.sum().backward()
        # gate receives gradient even though e_t is 0
        assert self.head.injection_gate.grad is not None
        assert self.head.injection_gate.grad.abs().sum() > 0
        # W_o / W_v / W_q frozen at this step (gate=0 zeroes their pathway)
        assert self.head.W_o.weight.grad is None or self.head.W_o.weight.grad.abs().sum() == 0
        assert self.head.W_v.weight.grad is None or self.head.W_v.weight.grad.abs().sum() == 0
        # W_q technically gets gradient via alpha.sum() if loss included that,
        # but for e_t-only loss it's 0 (alpha doesn't appear in the loss).
        assert self.head.W_q.weight.grad is None or self.head.W_q.weight.grad.abs().sum() == 0


class TestEosSupervision:
    """Verify training targets end with EOS AND that EOS appears in the loss
    (not masked to -100) — even when pad_token == eos_token, which is the
    common fallback for Qwen-style tokenizers."""

    def _make_tokenizer(self, pad_equals_eos: bool):
        """A minimal stand-in tokenizer that mimics HF's __call__ contract."""
        class TinyTok:
            def __init__(self, pad_equals_eos: bool):
                self.eos_token_id = 99
                self.pad_token_id = 99 if pad_equals_eos else 100
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"

            def __call__(self, text, truncation=True, max_length=None,
                         add_special_tokens=False, return_tensors=None,
                         padding=False):
                # Each "word" → ord('a') + offset, e.g. "abc" → [97, 98, 99 capped].
                # We just emit 1 id per character, capped at max_length.
                if isinstance(text, str):
                    ids = [ord(c) % 50 + 1 for c in text][: max_length or 10**9]
                    from types import SimpleNamespace
                    return SimpleNamespace(input_ids=ids)
                raise NotImplementedError

        return TinyTok(pad_equals_eos)

    def _expected_last_content_idx(self, ids_row, attn_row):
        # Find the last position with attention_mask=1 (i.e. EOS slot).
        nonzero = (attn_row == 1).nonzero(as_tuple=True)[0]
        return nonzero[-1].item() if nonzero.numel() else -1

    def test_eos_appended_when_pad_distinct_from_eos(self):
        from train import _tokenize_with_eos
        tok = self._make_tokenizer(pad_equals_eos=False)
        ids, attn = _tokenize_with_eos(tok, ["hello", "hi"], max_length=8,
                                       device=torch.device("cpu"))
        for row_ids, row_attn in zip(ids, attn):
            last = self._expected_last_content_idx(row_ids, row_attn)
            assert last >= 0, "no content positions"
            assert row_ids[last].item() == tok.eos_token_id, (
                "last content position should be EOS"
            )

    def test_eos_appended_when_pad_equals_eos(self):
        # The collision case — most important: even when pad_id == eos_id,
        # _tokenize_with_eos still correctly puts EOS at content end and the
        # attention_mask marks it as content.
        from train import _tokenize_with_eos
        tok = self._make_tokenizer(pad_equals_eos=True)
        ids, attn = _tokenize_with_eos(tok, ["abc", "hello world"],
                                       max_length=20, device=torch.device("cpu"))
        for row_ids, row_attn in zip(ids, attn):
            last = self._expected_last_content_idx(row_ids, row_attn)
            assert row_ids[last].item() == tok.eos_token_id
            # And it's marked as content (1) in attn, so label masking
            # (-100 where attn==0) will KEEP this EOS in the loss.
            assert row_attn[last].item() == 1

    def test_label_masking_keeps_eos_under_pad_equals_eos(self):
        # Simulate the train.py label construction. Both pad and eos = 99.
        # After tokenize-with-eos, ids = [...content..., 99, 99 (pad), 99 (pad)]
        # but attn = [..., 1, 0, 0]. Label masking by attn=0 keeps the FIRST 99
        # (the genuine EOS) in the loss, masks the pad 99s out.
        from train import _tokenize_with_eos
        tok = self._make_tokenizer(pad_equals_eos=True)
        ids, attn = _tokenize_with_eos(tok, ["a", "ab"], max_length=10,
                                       device=torch.device("cpu"))

        labels = ids.clone()
        labels[attn == 0] = -100

        # Row 0 ("a" + EOS, padded to len 3): content = [t_a, 99]; pad = [99]
        # Labels: [t_a, 99, -100]
        assert labels[0, 0].item() != -100
        assert labels[0, 1].item() == 99, "EOS at content end must NOT be masked"
        assert labels[0, 2].item() == -100, "pad past content must be masked"

        # Row 1 ("ab" + EOS, no padding needed): content = [t_a, t_b, 99]
        # Labels: [t_a, t_b, 99]
        assert (labels[1] != -100).all(), "row 1 has no pad and shouldn't be masked anywhere"
        assert labels[1, -1].item() == 99, "row 1 must end with EOS"

    def test_eos_survives_truncation(self):
        # When the natural target is too long, _tokenize_with_eos truncates
        # to max_length-1, then appends EOS, so the final length is exactly
        # max_length and EOS is always present.
        from train import _tokenize_with_eos
        tok = self._make_tokenizer(pad_equals_eos=False)
        long_text = "a" * 200
        ids, attn = _tokenize_with_eos(tok, [long_text], max_length=20,
                                       device=torch.device("cpu"))
        assert ids.shape[1] == 20
        assert ids[0, -1].item() == tok.eos_token_id, (
            "even after truncation, EOS must be the last token"
        )
        assert attn[0, -1].item() == 1


class TestVariableLengthBatching:
    """Libri2Mix clips have variable durations -> BEATs patch counts vary
    per clip. collate_fn pads to the batch max and emits a mask;
    SectionQueryHead must respect that mask so attention never lands on
    padded positions.
    """

    def test_collate_pads_and_emits_mask(self):
        from dataset import collate_fn
        # Two synthetic batch items with different "patch counts"
        a_patches = torch.randn(100, 768)
        b_patches = torch.randn(248, 768)
        batch = [
            {
                "audio_features": torch.zeros(10, 1024),
                "overlap_info":   torch.zeros(10, 4),
                "target_text":    "a",
                "filename":       "a.wav",
                "beats_patches":  a_patches,
            },
            {
                "audio_features": torch.zeros(10, 1024),
                "overlap_info":   torch.zeros(10, 4),
                "target_text":    "b",
                "filename":       "b.wav",
                "beats_patches":  b_patches,
            },
        ]
        out = collate_fn(batch)
        assert out["beats_patches"].shape == (2, 248, 768)
        assert out["beats_patches_mask"].shape == (2, 248)
        # Row 0 has 100 real patches then 148 padded → mask True from 100 onward
        assert (out["beats_patches_mask"][0, :100] == False).all()
        assert (out["beats_patches_mask"][0, 100:] == True).all()
        # Row 1 has 248 real patches → mask is all False
        assert (out["beats_patches_mask"][1] == False).all()
        # Padded rows in beats_patches are zero
        assert (out["beats_patches"][0, 100:] == 0).all()

    def test_attention_ignores_padded_positions(self):
        from section_query import SectionQueryHead
        head = SectionQueryHead(n_sections=6, d_patch=768, d_lm=2048, d_k=256, d_v=256)
        B, P_max = 2, 100
        patches = torch.randn(B, P_max, 768)
        key_padding_mask = torch.zeros(B, P_max, dtype=torch.bool)
        # First clip has 60 real patches; second has 100 real patches
        key_padding_mask[0, 60:] = True
        K, V = head.precompute_kv(patches)
        # forward_all_sections returns alpha of shape (B, n_sections, P_max)
        _, alpha = head.forward_all_sections(K, V, key_padding_mask=key_padding_mask)
        # For row 0, attention at padded positions (60:) must be 0 after softmax
        assert torch.allclose(alpha[0, :, 60:].sum(-1), torch.zeros(6), atol=1e-6), \
            "attention leaked to padded positions"
        # Row 0's attention over real positions should sum to 1
        assert torch.allclose(alpha[0, :, :60].sum(-1), torch.ones(6), atol=1e-5)
        # Row 1 is unmasked → full sum to 1 across all positions
        assert torch.allclose(alpha[1].sum(-1), torch.ones(6), atol=1e-5)

    def test_dynamic_attention_ignores_padded_positions(self):
        from section_query import SectionQueryHead
        head = SectionQueryHead(n_sections=6, d_patch=768, d_lm=2048, d_k=256, d_v=256)
        B, P_max = 3, 80
        patches = torch.randn(B, P_max, 768)
        key_padding_mask = torch.zeros(B, P_max, dtype=torch.bool)
        key_padding_mask[0, 40:] = True   # clip 0: 40 real, 40 pad
        key_padding_mask[1, 70:] = True   # clip 1: 70 real, 10 pad
        K, V = head.precompute_kv(patches)
        h_t = torch.randn(5, 2048)
        batch_idx = torch.tensor([0, 0, 1, 1, 2])
        _, alpha = head.forward_dynamic(h_t, K, V, batch_idx=batch_idx,
                                        key_padding_mask=key_padding_mask)
        # Queries 0, 1 are from clip 0 → their alpha must be 0 at pad positions
        assert torch.allclose(alpha[0, 40:].sum(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(alpha[1, 40:].sum(), torch.tensor(0.0), atol=1e-6)
        # Queries 2, 3 from clip 1 → 0 at positions 70+
        assert torch.allclose(alpha[2, 70:].sum(), torch.tensor(0.0), atol=1e-6)
        # Each row sums to 1 over the unmasked positions
        assert torch.allclose(alpha.sum(-1), torch.ones(5), atol=1e-5)


class TestRangeMarkers:
    """Verify <r>...</r> markers integrate cleanly with the rest of the pipeline.

    The markers are added inside <f_overlap_segments> spans so multi-range
    overlap clips get one attention map per range while keeping the visible
    prose clean (strip_all_tags removes them).
    """

    def test_special_tokens_includes_range_markers(self):
        from section_tags import RANGE_OPEN_TAG, RANGE_CLOSE_TAG, SPECIAL_TOKENS
        assert RANGE_OPEN_TAG in SPECIAL_TOKENS
        assert RANGE_CLOSE_TAG in SPECIAL_TOKENS

    def test_strip_removes_range_markers(self):
        from section_tags import strip_all_tags
        text = (
            "<sec_overlap><f_overlap_segments>overlap at "
            "<r>0.5-1.0s</r>, <r>3.0-4.5s</r></f></sec>"
        )
        assert strip_all_tags(text) == "overlap at 0.5-1.0s, 3.0-4.5s"

    def test_sfs_parser_handles_range_markers_transparently(self):
        # The TaggedClaimParser regex doesn't care about <r>...</r> wrappers
        # because extract_overlap_segments scans the whole feature-span body.
        from sfs import TaggedClaimParser
        text = (
            "<f_overlap_segments>overlap at <r>0.5-1.0s</r>, "
            "<r>3.0-4.5s</r>, <r>7.0-9.0s</r></f>"
        )
        claims = TaggedClaimParser().parse(text)
        starts = [c.value for c in claims if c.feature == "overlap_start"]
        ends = [c.value for c in claims if c.feature == "overlap_end"]
        assert starts == [0.5, 3.0, 7.0]
        assert ends == [1.0, 4.5, 9.0]

    def test_range_attention_key_from_body(self):
        # The inference-time helper that maps a parsed <r> body to its
        # attention-map key.
        from inference import _range_attention_key
        assert _range_attention_key("0.5-1.0s") == "overlap@0.5-1.0s"
        assert _range_attention_key("0.5 to 1.0 s") == "overlap@0.5-1.0s"
        # Unparseable body → fallback key with truncated raw text
        assert _range_attention_key("around half a second").startswith("overlap@malformed:")

    def test_verbalizer_wraps_ranges_in_marker_tags(self):
        # Section body from the verbalizer should contain <r>...</r> per range,
        # not the legacy comma-joined single-string value.
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from feature_verbalization import _build_section_bodies
        row = {
            "duration_sec": "10.0", "snr_db": "12.5", "srmr": "4.8",
            "f0_mean_hz": "188.0", "f0_sd_hz": "42.0",
            "praat_speaking_rate_syl_sec": "4.9",
            "praat_pause_count": "4", "praat_pause_rate_per_min": "24.0",
            "overlap_ratio": "0.65",
            "overlap_segments": "8000-16000;48000-72000;112000-144000",
        }
        body = _build_section_bodies(row)["overlap"]
        # Three ranges, each in its own <r>...</r>
        assert body.count("<r>") == 3
        assert body.count("</r>") == 3
        assert "<r>0.5-1.0s</r>" in body
        assert "<r>3.0-4.5s</r>" in body
        assert "<r>7.0-9.0s</r>" in body


class TestDynamicInjectionHelper:
    """End-to-end test of train.py::_inject_section_summaries_dynamic with a mock LM."""

    def test_dynamic_injection_runs_and_writes_at_section_positions(self):
        import torch.nn as nn
        from train import _inject_section_summaries_dynamic
        from section_query import SectionQueryHead

        # Tiny mock LM that returns hidden_states[-1] of the same shape as inputs.
        class MockLM(nn.Module):
            def forward(self, inputs_embeds=None, output_hidden_states=False, **kw):
                import types
                hs = inputs_embeds * 0.5  # arbitrary deterministic transform
                return types.SimpleNamespace(
                    hidden_states=(hs,) if output_hidden_states else None,
                )

        B, L, d_lm = 2, 8, 16
        prefix_embeds = torch.randn(B, 3, d_lm)
        prompt_embeds = torch.randn(B, 4, d_lm)
        target_embeds = torch.randn(B, L, d_lm)
        # target_ids: section opens at (0, 2), (0, 5), (1, 0)
        target_ids = torch.tensor([
            [9, 9, 100, 9, 9, 101, 9, 9],
            [100, 9, 9, 9, 9, 9, 9, 9],
        ])

        # Build a section head + cached K/V
        head = SectionQueryHead(n_sections=2, d_patch=4, d_lm=d_lm, d_k=4, d_v=4)
        patches = torch.randn(B, 10, 4)
        K, V = head.precompute_kv(patches)

        ctx = {
            "mode": "dynamic",
            "head": head,
            "K": K, "V": V,
            "section_id_to_idx": {100: 0, 101: 1},
        }
        out = _inject_section_summaries_dynamic(
            MockLM(), prefix_embeds, prompt_embeds, target_embeds, target_ids, ctx,
        )
        assert out.shape == target_embeds.shape

        # Non-section positions must be unchanged
        unchanged_positions = [
            (0, 0), (0, 1), (0, 3), (0, 4), (0, 6), (0, 7),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        ]
        for b, l in unchanged_positions:
            assert torch.allclose(out[b, l], target_embeds[b, l]), \
                f"non-section position ({b}, {l}) unexpectedly modified"

        # Section positions must have been residually augmented
        section_positions = [(0, 2), (0, 5), (1, 0)]
        for b, l in section_positions:
            diff = (out[b, l] - target_embeds[b, l]).abs().sum().item()
            assert diff > 0, f"section position ({b}, {l}) was not injected"

    def test_dynamic_injection_no_sections_skips_pass1(self):
        # If no section tokens are in the batch, no LM forward should happen.
        import torch.nn as nn
        from train import _inject_section_summaries_dynamic
        from section_query import SectionQueryHead

        # If the LM is called we want a noticeable side-effect to detect it.
        class WatchdogLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False
            def forward(self, **kw):
                self.called = True
                raise RuntimeError("should not be called when there are no sections")

        head = SectionQueryHead(n_sections=2, d_patch=4, d_lm=8, d_k=4, d_v=4)
        target_embeds = torch.randn(2, 5, 8)
        target_ids = torch.tensor([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
        ])
        K, V = head.precompute_kv(torch.randn(2, 10, 4))
        ctx = {
            "mode": "dynamic", "head": head, "K": K, "V": V,
            "section_id_to_idx": {100: 0, 101: 1},
        }
        out = _inject_section_summaries_dynamic(
            WatchdogLM(),
            torch.randn(2, 3, 8), torch.randn(2, 4, 8),
            target_embeds, target_ids, ctx,
        )
        # Returns target_embeds unchanged
        assert torch.allclose(out, target_embeds)


# ─── PatchGrid reshape sanity ──────────────────────────────────────


class TestPatchGridReshape:
    def test_reshape_attention_to_2d(self):
        from spec_encoder import PatchGrid
        grid = PatchGrid(n_patches=248, d_patch=768, time_dim=31, freq_dim=8, backend="beats")
        alpha = torch.rand(2, 248).softmax(dim=-1)
        alpha_2d = grid.reshape_attention(alpha)
        assert alpha_2d.shape == (2, 31, 8)
        # Per-batch sum still 1
        assert torch.allclose(alpha_2d.sum(dim=(1, 2)), torch.ones(2), atol=1e-5)

    def test_reshape_attention_preserves_row_major_order(self):
        from spec_encoder import PatchGrid
        # Construct an attention vector where patch i has value i (so we can
        # verify the reshape lays out time-bins as rows, freq-bins as cols).
        grid = PatchGrid(n_patches=12, d_patch=4, time_dim=3, freq_dim=4, backend="beats")
        alpha = torch.arange(12, dtype=torch.float32).unsqueeze(0)
        alpha_2d = grid.reshape_attention(alpha)
        # Row 0 should be patches 0..3, row 1 is 4..7, row 2 is 8..11.
        assert alpha_2d[0, 0].tolist() == [0., 1., 2., 3.]
        assert alpha_2d[0, 1].tolist() == [4., 5., 6., 7.]
        assert alpha_2d[0, 2].tolist() == [8., 9., 10., 11.]


# ─── Injection helper from train.py ──────────────────────────────────────


class TestInjectionHelper:
    def test_section_summaries_are_added_at_section_positions(self):
        # Stand-alone test of train.py's _inject_section_summaries.
        from train import _inject_section_summaries

        # Fake setup: B=2, L=8, d_lm=4, n_sections=3.
        # target_ids has section open ids 100, 101, 102. Section idx 0 -> 100, 1 -> 101, 2 -> 102.
        target_ids = torch.tensor([
            [100, 5, 6, 101, 7, 8, 9, 102],     # opens at pos 0, 3, 7
            [5,   6, 100, 7,  8, 101, 9, 10],   # opens at pos 2, 5
        ])
        target_embeds = torch.zeros(2, 8, 4)
        e_all = torch.tensor([
            [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
            [[10., 10., 10., 10.], [20., 20., 20., 20.], [30., 30., 30., 30.]],
        ])  # (B=2, n_sections=3, d_lm=4)
        section_id_to_idx = {100: 0, 101: 1, 102: 2}

        ctx = {"e_all": e_all, "section_id_to_idx": section_id_to_idx}
        modified = _inject_section_summaries(target_ids, target_embeds, ctx)
        # Row 0: pos 0 gets section 0's summary (1's), pos 3 gets section 1 (2's), pos 7 gets section 2 (3's).
        assert modified[0, 0].tolist() == [1., 1., 1., 1.]
        assert modified[0, 3].tolist() == [2., 2., 2., 2.]
        assert modified[0, 7].tolist() == [3., 3., 3., 3.]
        # Row 1: pos 2 gets section 0 (10's), pos 5 gets section 1 (20's).
        assert modified[1, 2].tolist() == [10., 10., 10., 10.]
        assert modified[1, 5].tolist() == [20., 20., 20., 20.]
        # Non-section positions unchanged
        assert modified[0, 1].abs().sum() == 0
        assert modified[1, 0].abs().sum() == 0

    def test_no_section_tokens_returns_unchanged(self):
        from train import _inject_section_summaries
        target_ids = torch.tensor([[1, 2, 3, 4]])
        target_embeds = torch.randn(1, 4, 8)
        e_all = torch.randn(1, 3, 8)
        ctx = {"e_all": e_all, "section_id_to_idx": {100: 0, 101: 1, 102: 2}}
        out = _inject_section_summaries(target_ids, target_embeds, ctx)
        # No section opens in the batch — embeds unchanged.
        assert torch.allclose(out, target_embeds)


# ─── Heavy: full BEATs forward (gated) ──────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("RUN_BEATS_TEST"),
    reason="Set RUN_BEATS_TEST=1 to run BEATs integration (downloads ~370 MB ckpt).",
)
class TestBEATsIntegration:
    def test_beats_loads_and_runs(self):
        from spec_encoder import SpecEncoder
        enc = SpecEncoder(model_name="beats", freeze=True)
        assert enc.backend == "beats"
        assert enc.d_out == 768
        waveform = torch.randn(2, 80000) * 0.1  # 5 s clip, 16 kHz
        patches, grid = enc(waveform)
        assert patches.shape[0] == 2
        assert patches.shape[2] == 768
        assert grid.time_dim * grid.freq_dim == patches.shape[1]
        assert grid.freq_dim == 8  # 128 mel bins / 16 patch size

    def test_beats_into_section_head(self):
        from section_query import SectionQueryHead
        from spec_encoder import SpecEncoder
        enc = SpecEncoder(model_name="beats", freeze=True)
        head = SectionQueryHead(n_sections=6, d_patch=enc.d_out, d_lm=2048, d_k=256, d_v=256)
        waveform = torch.randn(2, 80000) * 0.1
        patches, grid = enc(waveform)
        K, V = head.precompute_kv(patches)
        e_all, alpha_all = head.forward_all_sections(K, V)
        assert e_all.shape == (2, 6, 2048)
        assert alpha_all.shape == (2, 6, grid.n_patches)
        # 2D reshape
        alpha_2d = grid.reshape_attention(alpha_all)
        assert alpha_2d.shape == (2, 6, grid.time_dim, grid.freq_dim)
