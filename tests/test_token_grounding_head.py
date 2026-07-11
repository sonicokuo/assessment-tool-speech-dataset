"""CPU unit tests for the token-coupled grounding head (M1).

Deterministic, no GPU, no model download. Verifies shapes, prefix masking, the two
losses (pooled-scalar Huber + Liu-2017 attention correctness), the loss-term no-op
paths, and that the head is learnable (a few SGD steps reduce the pooled loss).
"""

import torch

from src.model.token_grounding_head import TokenGroundingHead, token_grounding_loss_term

torch.manual_seed(0)

B, Q, N, LM, PD, H = 3, 2, 7, 16, 12, 8


def _head():
    return TokenGroundingHead(lm_dim=LM, prefix_dim=PD, hidden=H, huber_delta=1.0)


def test_forward_shapes_and_alpha_normalised():
    head = _head()
    th = torch.randn(B, Q, LM)
    pf = torch.randn(B, N, PD)
    pooled, alpha, values = head(th, pf)
    assert pooled.shape == (B, Q)
    assert alpha.shape == (B, Q, N)
    assert values.shape == (B, N)
    # alpha rows are a distribution over prefix tokens.
    assert torch.allclose(alpha.sum(-1), torch.ones(B, Q), atol=1e-5)


def test_prefix_mask_zeros_invalid_positions():
    head = _head()
    th = torch.randn(B, Q, LM)
    pf = torch.randn(B, N, PD)
    mask = torch.ones(B, N)
    mask[:, N - 2:] = 0  # mask out last two prefix tokens
    _, alpha, _ = head(th, pf, prefix_mask=mask)
    # masked positions receive ~0 attention; valid rows still sum to 1.
    assert alpha[:, :, N - 2:].abs().max() < 1e-6
    assert torch.allclose(alpha.sum(-1), torch.ones(B, Q), atol=1e-5)


def test_fully_masked_row_is_finite():
    head = _head()
    th = torch.randn(B, Q, LM)
    pf = torch.randn(B, N, PD)
    mask = torch.zeros(B, N)  # nothing valid -> softmax of all -inf
    pooled, alpha, _ = head(th, pf, prefix_mask=mask)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(alpha).all()  # nan_to_num guard


def test_pooled_loss_gated_by_mask():
    head = _head()
    pooled = torch.tensor([[1.0, 5.0], [2.0, 9.0], [0.0, 0.0]])
    gt = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])  # only col0 of rows0,1 count
    loss = head.pooled_loss(pooled, gt, mask)
    # the two scored slots match exactly -> zero loss despite huge col1 errors.
    assert loss.item() == 0.0


def test_liu_loss_minimised_when_alpha_matches_beta():
    head = _head()
    # beta concentrated on prefix index 0
    beta = torch.zeros(B, Q, N)
    beta[:, :, 0] = 1.0
    alpha_match = beta.clone().clamp(min=1e-8)
    alpha_match = alpha_match / alpha_match.sum(-1, keepdim=True)
    alpha_uniform = torch.full((B, Q, N), 1.0 / N)
    l_match = head.liu_loss(alpha_match, beta)
    l_uniform = head.liu_loss(alpha_uniform, beta)
    # pointing exactly at the oracle region beats spreading attention uniformly.
    assert l_match < l_uniform


def test_loss_term_noops_without_head_or_tensors():
    # head None
    loss, m = token_grounding_loss_term(None, {}, 0.5)
    assert loss is None and m == {}
    # lambda 0
    loss, m = token_grounding_loss_term(_head(), {}, 0.0)
    assert loss is None
    # head present but coupled tensors absent
    loss, m = token_grounding_loss_term(_head(), {}, 0.5)
    assert loss is None and m["loss_token_grounding"] == 0.0


def test_loss_term_full_path_finite_and_keyed():
    head = _head()
    batch = {
        "tg_token_hidden": torch.randn(B, Q, LM),
        "tg_prefix": torch.randn(B, N, PD),
        "tg_prefix_mask": torch.ones(B, N),
        "tg_gt_scalar": torch.randn(B, Q),
        "tg_gt_mask": torch.ones(B, Q),
        "tg_beta": torch.softmax(torch.randn(B, Q, N), dim=-1),
        "tg_region_mask": torch.ones(B, Q),
    }
    loss, m = token_grounding_loss_term(head, batch, 1.0, lambda_liu=0.5)
    assert loss is not None and torch.isfinite(loss)
    assert "loss_tg_pooled" in m and "loss_tg_liu" in m
    assert "loss_token_grounding" in m
    # gradient flows to head params
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_head_is_learnable():
    head = _head()
    th = torch.randn(B, Q, LM)
    pf = torch.randn(B, N, PD)
    gt = torch.randn(B, Q)
    mask = torch.ones(B, Q)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = None
    for _ in range(150):
        opt.zero_grad()
        pooled, _, _ = head(th, pf)
        loss = head.pooled_loss(pooled, gt, mask)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first  # pooled scalar fits the toy target over steps
