"""Tests for src/peft_config.py — pure (no peft import needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from peft_config import lora_config_kwargs, uses_pissa  # noqa: E402


def _base(**over):
    c = {"lora_rank": 16, "lora_alpha": 32,
         "lora_targets": ["q_proj", "v_proj"], "lora_dropout": 0.05}
    c.update(over)
    return c


def test_defaults_are_vanilla_lora():
    k = lora_config_kwargs(_base())
    assert k["r"] == 16 and k["lora_alpha"] == 32
    assert k["use_dora"] is False
    assert k["init_lora_weights"] is True
    assert k["bias"] == "none" and k["task_type"] == "CAUSAL_LM"


def test_dora_flag_threaded():
    k = lora_config_kwargs(_base(use_dora=True))
    assert k["use_dora"] is True


def test_pissa_init_threaded():
    k = lora_config_kwargs(_base(init_lora_weights="pissa_niter_16"))
    assert k["init_lora_weights"] == "pissa_niter_16"


def test_uses_pissa_detection():
    assert uses_pissa(_base(init_lora_weights="pissa")) is True
    assert uses_pissa(_base(init_lora_weights="pissa_niter_16")) is True
    assert uses_pissa(_base()) is False                 # default True (not a str)
    assert uses_pissa(_base(init_lora_weights=True)) is False


def test_missing_required_field_raises():
    bad = {"lora_alpha": 32, "lora_targets": ["q_proj"]}
    with pytest.raises(KeyError):
        lora_config_kwargs(bad)


def test_dropout_defaults_to_zero_when_absent():
    c = {"lora_rank": 8, "lora_alpha": 16, "lora_targets": ["q_proj"]}
    assert lora_config_kwargs(c)["lora_dropout"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
