"""Integration tests: the reliability head + NLL term wired into train.compute_loss,
plus the default-off no-op guarantee (deliverable requirements (a) wiring + (d)).

Mocks the LM/tokenizer/embed exactly like tests/test_compute_loss_b_full.py so no
transformers/peft/CUDA is needed. Two adapter mocks are used:
  - MockAdapterPlainAux:      returns (prefix, scalar_pred tensor)         [plain MSE]
  - MockAdapterReliability:   returns (prefix, (mean, log_var))            [reliability]
"""

import os
import sys
import types
import importlib.machinery

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "mamba_ssm" not in sys.modules:
    stub = types.ModuleType("mamba_ssm")
    stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    stub.Mamba = object
    sys.modules["mamba_ssm"] = stub

from feature_set import N_FEATURES  # noqa: E402


# ─── Mocks (mirror test_compute_loss_b_full.py) ──────────────────────

class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1001

    def __call__(self, text, return_tensors=None, padding=None,
                 truncation=None, max_length=None, add_special_tokens=True):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        rows = []
        for s in texts:
            row = [ord(c) % 1000 + 1 for c in s]
            if max_length is not None:
                row = row[:max_length]
            rows.append(row)
        if return_tensors == "pt" or padding:
            max_len = max((len(r) for r in rows), default=0)
            padded = [r + [self.pad_token_id] * (max_len - len(r)) for r in rows]
            return types.SimpleNamespace(input_ids=torch.tensor(padded, dtype=torch.long))
        return types.SimpleNamespace(input_ids=(rows[0] if single else rows))


class MockLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(8, 8)

    def __call__(self, inputs_embeds=None, labels=None, **kw):
        h = self.W(inputs_embeds)
        valid = (labels != -100).to(inputs_embeds.dtype)
        pseudo_ce = (h.sum(dim=-1) * valid).mean()
        return types.SimpleNamespace(loss=pseudo_ce.abs())


class MockEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(2000, 8)

    def __call__(self, ids):
        return self.E(ids)


class MockAdapterPlainAux(nn.Module):
    """Returns (prefix, scalar_pred) — the plain mean head."""
    def __init__(self, n_features=N_FEATURES, lm_dim=8):
        super().__init__()
        self.audio_proj = nn.Linear(4, lm_dim)
        self.regress_head = nn.Linear(lm_dim, n_features)

    def forward(self, audio, overlap):
        prefix = self.audio_proj(audio[:, :4])
        scalar_pred = self.regress_head(prefix.mean(dim=1))
        return prefix, scalar_pred


class MockAdapterReliability(nn.Module):
    """Returns (prefix, (mean, log_var)) — the heteroscedastic head."""
    def __init__(self, n_features=N_FEATURES, lm_dim=8):
        super().__init__()
        self.audio_proj = nn.Linear(4, lm_dim)
        self.regress_head = nn.Linear(lm_dim, 2 * n_features)
        self.n_features = n_features

    def forward(self, audio, overlap):
        prefix = self.audio_proj(audio[:, :4])
        out = self.regress_head(prefix.mean(dim=1))
        mean = out[..., : self.n_features]
        log_var = out[..., self.n_features:]
        return prefix, (mean, log_var)


def _components(adapter):
    return {
        "adapter": adapter.to(torch.bfloat16),
        "llm": MockLM().to(torch.bfloat16),
        "embed_layer": MockEmbed().to(torch.bfloat16),
        "tokenizer": MockTokenizer(),
    }


def _batch(B=2, T=16):
    audio = torch.randn(B, T, 4)
    overlap = torch.randn(B, T, 4)
    target_text = ["The duration is 10.5 s." for _ in range(B)]
    gt_scalars = torch.randn(B, N_FEATURES)
    gt_mask = torch.ones(B, N_FEATURES, dtype=torch.bool)
    return audio, overlap, target_text, gt_scalars, gt_mask


def _config(**kw):
    base = {
        "max_target_length": 64,
        "max_nums_length": 32,
        "lambda_prose": 1.0,
        "lambda_nums": 0.0,
        "lambda_mse": 0.0,
        "lambda_nll": 0.0,
    }
    base.update(kw)
    return base


def test_nll_term_fires_with_reliability_head():
    from train import compute_loss
    torch.manual_seed(0)
    cmp = _components(MockAdapterReliability())
    audio, overlap, target_text, gt_scalars, gt_mask = _batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=0.0, lambda_nll=1.0),
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )
    assert m["loss_nll"] > 0.0, "NLL term should fire with the reliability head"
    assert "reliability_sigma_mean" in m
    assert m["reliability_sigma_mean"] > 0.0
    # total == lambda_nll * nll (prose off, others off).
    assert m["loss_total"] == pytest.approx(m["loss_nll"], rel=1e-3)


def test_nll_masked_zero_when_no_gt_present():
    from train import compute_loss
    cmp = _components(MockAdapterReliability())
    audio, overlap, target_text, gt_scalars, _ = _batch()
    zero_mask = torch.zeros(gt_scalars.shape, dtype=torch.bool)
    prompt_ids = torch.tensor([[1, 2, 3]])
    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=1.0, lambda_nll=1.0),
        gt_scalars=gt_scalars, gt_mask=zero_mask,
    )
    assert m["loss_nll"] == 0.0


def test_reliability_head_mse_path_uses_mean():
    """With the reliability head AND lambda_mse>0, the MSE term still works (it uses the
    predicted MEAN, not the tuple)."""
    from train import compute_loss
    cmp = _components(MockAdapterReliability())
    audio, overlap, target_text, gt_scalars, gt_mask = _batch()
    prompt_ids = torch.tensor([[1, 2, 3]])
    total, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=0.0, lambda_mse=1.0, lambda_nll=0.0),
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )
    assert m["loss_mse"] > 0.0
    assert m["loss_nll"] == 0.0


def test_nll_gradients_reach_adapter_head():
    from train import compute_loss
    cmp = _components(MockAdapterReliability())
    audio, overlap, target_text, gt_scalars, gt_mask = _batch()
    prompt_ids = torch.tensor([[1, 2, 3]])
    for p in cmp["adapter"].parameters():
        if p.grad is not None:
            p.grad.zero_()
    total, _ = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=0.0, lambda_nll=1.0),
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )
    total.backward()
    g = cmp["adapter"].regress_head.weight.grad
    assert g is not None and g.abs().max().item() > 0.0


# ─── Default-off no-op: byte-identical to the plain aux head ──────────

def test_default_off_no_op_plain_head_unchanged():
    """With lambda_nll absent/0 and the PLAIN aux head, compute_loss returns exactly
    the same metrics as a config without any reliability keys — the additive code is a
    no-op. We assert the prose+mse path is bit-identical with vs without lambda_nll=0."""
    from train import compute_loss

    audio, overlap, target_text, gt_scalars, gt_mask = _batch()
    prompt_ids = torch.tensor([[1, 2, 3]])

    # Same seed → same adapter weights → same forward.
    torch.manual_seed(123)
    cmp_a = _components(MockAdapterPlainAux())
    torch.manual_seed(123)
    cmp_b = _components(MockAdapterPlainAux())

    cfg_without = {
        "max_target_length": 64, "max_nums_length": 32,
        "lambda_prose": 1.0, "lambda_nums": 0.0, "lambda_mse": 0.5,
    }  # no lambda_nll / reliability keys at all
    cfg_with_zero = dict(cfg_without, lambda_nll=0.0)  # explicit zero

    _, m_a = compute_loss(
        cmp_a["adapter"], cmp_a["llm"], cmp_a["embed_layer"], cmp_a["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"), config=cfg_without,
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )
    _, m_b = compute_loss(
        cmp_b["adapter"], cmp_b["llm"], cmp_b["embed_layer"], cmp_b["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"), config=cfg_with_zero,
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )

    # Identical core terms.
    assert m_a["loss_lm_prose"] == pytest.approx(m_b["loss_lm_prose"], rel=0, abs=0)
    assert m_a["loss_mse"] == pytest.approx(m_b["loss_mse"], rel=0, abs=0)
    assert m_a["loss_total"] == pytest.approx(m_b["loss_total"], rel=0, abs=0)
    # The NLL term is a hard 0 (plain head → reliability_log_var is None).
    assert m_b["loss_nll"] == 0.0
    # No reliability_sigma_mean is emitted on the plain head path.
    assert "reliability_sigma_mean" not in m_b


def test_plain_head_never_emits_nll_even_if_lambda_set():
    """If someone sets lambda_nll>0 but the adapter is the PLAIN head (no log_var), the
    NLL term must stay 0 — it can't fire without a reliability head."""
    from train import compute_loss
    cmp = _components(MockAdapterPlainAux())
    audio, overlap, target_text, gt_scalars, gt_mask = _batch()
    prompt_ids = torch.tensor([[1, 2, 3]])
    _, m = compute_loss(
        cmp["adapter"], cmp["llm"], cmp["embed_layer"], cmp["tokenizer"],
        audio, overlap, target_text, prompt_ids,
        device=torch.device("cpu"),
        config=_config(lambda_prose=1.0, lambda_nll=1.0),
        gt_scalars=gt_scalars, gt_mask=gt_mask,
    )
    assert m["loss_nll"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
