"""CPU unit tests for src/token_grounding_validate.py (M1 head deletion validation).

The key property: on a head whose attention concentrates on the frames that actually
drive ``pooled``, deleting the top-alpha frames moves ``pooled`` much more than deleting
random frames (high win-rate); a random-weights head collapses to ~chance. This is the
Adebayo model-randomisation sanity that makes the deletion result trustworthy.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.token_grounding_head import TokenGroundingHead  # noqa: E402
import experiments.token_grounding_validate as tgv  # noqa: E402


def _grounded_head(D=8, hidden=8):
    """A head where logit ∝ prefix[:,1] (attention) and value = prefix[:,0]."""
    head = TokenGroundingHead(lm_dim=8, prefix_dim=D, hidden=hidden)
    with torch.no_grad():
        for m in (head.q_proj, head.k_proj, head.v_proj):
            m.weight.zero_(); m.bias.zero_()
        head.q_proj.weight[0, 0] = 1.0      # q0 = token_hidden[...,0]
        head.k_proj.weight[0, 1] = 4.0      # k0 = 4 * prefix[...,1]  -> logit ∝ prefix[:,1]
        head.v_proj.weight[0, 0] = 1.0      # value = prefix[...,0]
    return head.eval()


def _make_clips(n_clips=40, N=25, D=8, n_signal=3, seed=0):
    """Each clip: n_signal frames get high value (dim0) AND high attention (dim1)."""
    g = torch.Generator().manual_seed(seed)
    th = torch.zeros(1, 1, 8)
    th[0, 0, 0] = 1.0                        # activate the query direction
    clips = []
    for c in range(n_clips):
        pf = 0.1 * torch.randn(1, N, D, generator=g)
        sig = torch.randperm(N, generator=g)[:n_signal]
        pf[0, sig, 0] = 6.0                  # high value on signal frames
        pf[0, sig, 1] = 3.0                  # high attention on signal frames
        clips.append((th, pf, torch.ones(1, N, dtype=torch.bool)))
    return clips


def test_head_level_deletion_structure():
    head = _grounded_head()
    th, pf, pm = _make_clips(1)[0]
    r = tgv.head_level_deletion(head, th, pf, pm, frac=0.2,
                               generator=torch.Generator().manual_seed(1))
    assert set(r) >= {"delta_top", "delta_rand", "win", "nvalid"}
    assert r["nvalid"] == pf.shape[1]
    assert r["delta_top"] >= 0.0 and r["delta_rand"] >= 0.0


def test_grounded_head_wins_deletion():
    """On a grounded head, top-alpha masking must move pooled more than random."""
    head = _grounded_head()
    clips = _make_clips(40)
    s = tgv.summarize_deletion(
        tgv.head_level_deletion(head, th, pf, pm, frac=0.2,
                               generator=torch.Generator().manual_seed(i))
        for i, (th, pf, pm) in enumerate(clips)
    )
    assert s["win_rate"] >= 0.9, s
    assert s["mean_delta_top"] > 2.0 * s["mean_delta_rand"], s


def test_head_level_deletion_is_circular_confound():
    """DOCUMENTED NEGATIVE RESULT (why we need the LM-level test).

    Head-level deletion is *circular*: pooled = sum(alpha * v), so masking the top-alpha
    frames perturbs that alpha-weighted sum by construction — high-alpha frames dominate the
    pool regardless of whether the map is grounded. Consequently a RANDOM-weights head also
    "wins" the deletion (its own high-alpha frames dominate its own pool), so the model_rand
    sanity does NOT collapse for head-level deletion.

    The takeaway encoded here: head-level deletion is a plumbing/smoke check only; the valid
    causal grounding test is LM-level (mask audio -> the *emitted* number changes, measured on
    the LM output which does not mechanically depend on the head's alpha). This test asserts
    the confound so it is documented, not hidden.
    """
    head = _grounded_head()
    clips = _make_clips(60)
    out = tgv.run_head_level(head, clips, frac=0.2, seed=0, with_sanity=True)
    # Both trained AND random heads win most clips => head-level cannot discriminate grounding.
    assert out["trained"]["win_rate"] >= 0.8, out
    assert out["model_rand"]["win_rate"] >= 0.6, out   # random head ALSO wins == circular


def test_load_head_roundtrip(tmp_path):
    head = _grounded_head()
    ckpt = {"token_grounding_head_state_dict": head.state_dict(), "config": {}}
    p = tmp_path / "m3b_like.pt"
    torch.save(ckpt, p)
    loaded = tgv.load_token_grounding_head(str(p), device="cpu")
    assert loaded.q_proj.in_features == 8
    assert loaded.k_proj.in_features == 8
    th, pf, pm = _make_clips(1)[0]
    with torch.no_grad():
        a = head(th, pf, pm)[0]
        b = loaded(th, pf, pm)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_load_head_rejects_wrong_checkpoint(tmp_path):
    p = tmp_path / "decoupled.pt"
    torch.save({"decoupled_head_state_dict": {}, "config": {}}, p)
    try:
        tgv.load_token_grounding_head(str(p))
        assert False, "should have raised on a non-M1 checkpoint"
    except ValueError as e:
        assert "token_grounding_head_state_dict" in str(e)


def test_lm_level_deletion_with_stub_generator():
    """LM-level path with a stub generator (pooled as the 'emitted number' proxy)."""
    head = _grounded_head()
    th, pf, pm = _make_clips(1, seed=3)[0]

    def gen(prefix, mask):
        with torch.no_grad():
            return float(head(th, prefix, mask)[0].item())

    r = tgv.lm_level_deletion(gen, head, th, pf, pm, frac=0.2,
                             generator=torch.Generator().manual_seed(2))
    assert set(r) >= {"delta_top", "delta_rand", "win", "nvalid"}
    # grounded setup: deleting top-alpha frames changes the emitted proxy more than random
    assert r["delta_top"] >= r["delta_rand"]
