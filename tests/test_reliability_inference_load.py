"""FAILING TEST (adversarial verifier): a reliability-head checkpoint cannot be
loaded by the inference adapter-build path.

Defect (src/inference.py):
  - line 442 builds the adapter with build_adapter(...) WITHOUT reliability_head=True,
    so regress_head is a plain nn.Linear(d, F).
  - `reliability_head` is NOT in inference._STRUCTURAL_KEYS, so it is never synced from
    the checkpoint config (unlike use_sections/grounding_mode/etc.).
  - line 446 does adapter.load_state_dict(...) with strict=True (default).

A checkpoint trained with reliability_head=true saves a ReliabilityHead whose
parameters are regress_head.proj.{weight,bias} with shape (2F, d)/(2F,). Loading that
state_dict into a plain-head adapter raises RuntimeError (missing/unexpected/shape
mismatch). So the standard eval/generation script crashes on the planned headline
reliability checkpoint.

This test reproduces the failure at the adapter level (no transformers/LM needed):
it builds the plain adapter the way inference does, then strict-loads a reliability
adapter's state_dict — which must raise. When inference is fixed (sync the flag +
build with reliability_head + strict=False), update this test to assert a clean load.
"""

import os
import sys
import types
import importlib.machinery

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "mamba_ssm" not in sys.modules:
    _stub = types.ModuleType("mamba_ssm")
    _stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    _stub.Mamba = object
    sys.modules["mamba_ssm"] = _stub

from adapter import build_adapter  # noqa: E402
from feature_set import N_FEATURES  # noqa: E402


def _build_like_inference(reliability_head: bool):
    # inference.py:442 builds WITHOUT reliability_head (always plain).
    return build_adapter(
        "concat-only", lm_dim=8, n_aux_features=N_FEATURES,
        reliability_head=reliability_head,
    )


def test_reliability_ckpt_strict_load_into_plain_adapter_fails():
    """Reproduces the inference defect: strict-loading a reliability-head state_dict
    into the plainly-built inference adapter raises (shape/key mismatch)."""
    trained = _build_like_inference(reliability_head=True)   # what train.py saved
    reliability_sd = trained.state_dict()

    # What inference.py builds (plain head, flag never synced from the ckpt).
    inference_adapter = _build_like_inference(reliability_head=False)

    with pytest.raises(RuntimeError):
        inference_adapter.load_state_dict(reliability_sd)  # strict=True (inference:446)


def test_fix_would_load_clean_when_flag_synced():
    """The fix: inference must sync reliability_head from the ckpt config and build the
    head accordingly; then the SAME-shaped head loads with no error. This documents the
    expected post-fix behaviour."""
    trained = _build_like_inference(reliability_head=True)
    reliability_sd = trained.state_dict()
    fixed_inference_adapter = _build_like_inference(reliability_head=True)  # flag synced
    # Should load cleanly (no raise) once the flag is threaded.
    fixed_inference_adapter.load_state_dict(reliability_sd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
