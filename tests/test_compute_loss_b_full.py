"""Test the B-full + aux-head compute_loss in src/train.py.

We mock the LM so we don't need transformers/peft locally. The mock LM's forward returns
a tiny output with a real `.loss` attribute that depends on the inputs (so backprop works).

Tests:
  - compute_loss runs both LM forwards (prose + nums) when target_nums is provided.
  - MSE term is computed and masked correctly when gt_scalars + gt_mask are provided.
  - Loss weights are applied per-term.
  - Gradients flow back into adapter + aux_head parameters via .backward().
  - When B-full inputs are absent, function silently degrades to prose-only loss.
"""

import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Only mamba_ssm needs a stub (it's a CUDA-only build, can't be installed on Mac).
# peft / transformers / wandb are installed in the local idl env.
# Important: stubbed modules need a real __spec__ so tools like transformers' introspection
# (`importlib.util.find_spec("X")`) don't crash with "X.__spec__ is None".
import importlib.machinery

if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    stub.Mamba = object
    sys.modules["mamba_ssm"] = stub

from feature_set import N_FEATURES


# ─── Mocks ─────────────────────────────────────────────────────────

class MockTokenizer:
    """Tiny tokenizer: each character → ord() token id, padded to max_length."""
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        if isinstance(text, str):
            text = [text]
        max_len = max_length or max(len(s) for s in text)
        ids = []
        for s in text:
            row = [ord(c) % 1000 + 1 for c in s][:max_len]   # +1 to avoid pad_id collision
            row += [self.pad_token_id] * (max_len - len(row))
            ids.append(row)
        return types.SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.long))


class MockLM(nn.Module):
    """Mock LM: returns a fake output with a .loss that's the mean of inputs_embeds (so
    gradient flows back). Different label tensors produce different losses."""

    def __init__(self):
        super().__init__()
        self.W = nn.Linear(8, 8)

    def __call__(self, inputs_embeds=None, labels=None, **kw):
        # Make loss depend on both inputs_embeds and labels — mean of products
        h = self.W(inputs_embeds)                           # (B, T, 8)
        # Some pseudo-CE that's a function of inputs and (non-pad) labels
        valid = (labels != -100).to(inputs_embeds.dtype)    # (B, T)
        pseudo_ce = (h.sum(dim=-1) * valid).mean()
        return types.SimpleNamespace(loss=pseudo_ce.abs())


class MockEmbed(nn.Module):
    """Embedding lookup: integer id → 8-dim vector."""
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(2000, 8)

    def __call__(self, ids):
        return self.E(ids)


class MockAdapterWithAux(nn.Module):
    """Mock adapter: returns (prefix, scalar_pred). prefix shape (B, N=4, 8)."""

    def __init__(self, n_features: int = N_FEATURES, lm_dim: int = 8):
        super().__init__()
        self.audio_proj = nn.Linear(4, lm_dim)
        self.regress_head = nn.Linear(lm_dim, n_features)

    def forward(self, audio, overlap):
        # audio: (B, T, 4)  →  prefix (B, T//... = 4, lm_dim=8)
        # Just compress audio to 4 prefix tokens via mean-pool over groups
        B, T, _ = audio.shape
        prefix = self.audio_proj(audio[:, :4])              # (B, 4, 8)
        scalar_pred = self.regress_head(prefix.mean(dim=1)) # (B, n_features)
        return prefix, scalar_pred


def _make_components():
    # compute_loss casts inputs to bfloat16; mocks must match.
    return {
        "adapter": MockAdapterWithAux(n_features=N_FEATURES, lm_dim=8).to(torch.bfloat16),
        "llm": MockLM().to(torch.bfloat16),
        "embed_layer": MockEmbed().to(torch.bfloat16),
        "tokenizer": MockTokenizer(),
    }


def _make_batch(B: int = 2, T: int = 16):
    audio = torch.randn(B, T, 4)
    overlap = torch.randn(B, T, 4)
    target_text = ["The duration is 10.5 s." for _ in range(B)]
    target_nums = ["snr=15.66 hnr=8.34" for _ in range(B)]
    gt_scalars = torch.randn(B, N_FEATURES)
    gt_mask = torch.ones(B, N_FEATURES, dtype=torch.bool)
    return audio, overlap, target_text, target_nums, gt_scalars, gt_mask


def _config(lambda_prose=1.0, lambda_nums=0.0, lambda_mse=0.0):
    return {
        "max_target_length": 64,
        "max_nums_length": 32,
        "lambda_prose": lambda_prose,
        "lambda_nums": lambda_nums,
        "lambda_mse": lambda_mse,
    }


# ─── Tests ─────────────────────────────────────────────────────────


def test_prose_only_path_runs():
    """Smoke test: with all B-full inputs absent, compute_loss returns prose-only CE."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, _, _, _ = _make_batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(),
    )

    assert m["loss_lm_prose"] > 0.0
    assert m["loss_lm_nums"] == 0.0
    assert m["loss_mse"] == 0.0
    assert m["loss_total"] == pytest.approx(m["loss_lm_prose"], rel=1e-4)


def test_b_full_runs_both_forwards():
    """B-full path: both prose CE and nums CE are non-zero, MSE is zero."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, target_nums, _, _ = _make_batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=0.3, lambda_nums=1.0, lambda_mse=0.0),
        target_nums=target_nums,
    )

    assert m["loss_lm_prose"] > 0.0
    assert m["loss_lm_nums"] > 0.0     # second forward fired
    assert m["loss_mse"] == 0.0


def test_aux_head_mse_path_with_full_mask():
    """When gt_mask is all True, MSE loss is per_feat_mse averaged over all 13 slots."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, _, gt_scalars, gt_mask = _make_batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    # MSE only — set lambda_prose tiny so total is dominated by mse term
    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=0.0, lambda_nums=0.0, lambda_mse=1.0),
        gt_scalars=gt_scalars,
        gt_mask=gt_mask,
    )

    assert m["loss_mse"] > 0.0
    # total should equal lambda_mse * mse_loss
    assert m["loss_total"] == pytest.approx(m["loss_mse"], rel=1e-4)


def test_aux_head_mse_masking_zeros_missing_slots():
    """With all-False mask, MSE contribution is zero (nothing to score)."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, _, gt_scalars, _ = _make_batch()
    gt_mask_zeros = torch.zeros_like(gt_scalars, dtype=torch.bool)
    prompt_ids = torch.tensor([[1, 2, 3]])

    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=1.0, lambda_nums=0.0, lambda_mse=1.0),
        gt_scalars=gt_scalars,
        gt_mask=gt_mask_zeros,
    )

    assert m["loss_mse"] == 0.0


def test_b_full_gradients_reach_adapter_and_aux_head():
    """End-to-end: total_loss.backward() puts gradients into adapter.audio_proj and adapter.regress_head."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, target_nums, gt_scalars, gt_mask = _make_batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    # Zero existing grads
    for p in cmp["adapter"].parameters():
        if p.grad is not None:
            p.grad.zero_()

    total, _ = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=1.0, lambda_nums=1.0, lambda_mse=0.5),
        target_nums=target_nums,
        gt_scalars=gt_scalars,
        gt_mask=gt_mask,
    )
    total.backward()

    # Both parts of the adapter must have nonzero gradients
    audio_proj_grad = cmp["adapter"].audio_proj.weight.grad.abs().max().item()
    regress_grad = cmp["adapter"].regress_head.weight.grad.abs().max().item()
    assert audio_proj_grad > 0.0, "audio_proj got no gradient — CE path not flowing"
    assert regress_grad > 0.0, "regress_head got no gradient — MSE path not flowing"
    print(f"\n[grad] audio_proj: {audio_proj_grad:.4f}, regress_head: {regress_grad:.4f}")


def test_lambda_weights_scale_terms():
    """Doubling lambda_prose roughly doubles the prose contribution to total."""
    from train import compute_loss

    cmp = _make_components()
    audio, overlap, target_text, target_nums, gt_scalars, gt_mask = _make_batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    torch.manual_seed(0)
    _, m_a = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=1.0, lambda_nums=0.0, lambda_mse=0.0),
        target_nums=target_nums,
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )

    torch.manual_seed(0)
    _, m_b = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=2.0, lambda_nums=0.0, lambda_mse=0.0),
        target_nums=target_nums,
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )

    # The base loss values should be the same; the totals should differ by 2x
    assert m_a["loss_lm_prose"] == pytest.approx(m_b["loss_lm_prose"], rel=1e-4)
    assert m_b["loss_total"] == pytest.approx(2.0 * m_a["loss_total"], rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
