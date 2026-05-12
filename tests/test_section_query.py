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
        assert self.head.queries.grad is not None
        assert self.head.queries.grad.abs().sum() > 0
        assert self.head.W_o.weight.grad.abs().sum() > 0
        assert self.head.W_k.weight.grad.abs().sum() > 0
        assert self.head.W_v.weight.grad.abs().sum() > 0


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
        patches = torch.randn(2, 50, 768)
        K, V = self.head.precompute_kv(patches)
        h_t = torch.randn(3, 2048, requires_grad=True)
        e_t, alpha = self.head.forward_dynamic(
            h_t, K, V, batch_idx=torch.tensor([0, 1, 1]),
        )
        loss = e_t.sum() + alpha.sum()
        loss.backward()
        # W_q gets grad
        assert self.head.W_q.weight.grad.abs().sum() > 0
        # h_t (upstream) gets grad — this is what closes the loop back to the LM
        assert h_t.grad is not None
        assert h_t.grad.abs().sum() > 0
        # Static queries are unused on this path
        assert self.head.queries.grad is None


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
