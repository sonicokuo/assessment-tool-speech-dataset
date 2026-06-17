"""peft_config.py — build LoRA/PiSSA/DoRA config kwargs from the run config.

One source of truth for the peft.LoraConfig kwargs so train.py and inference.py
stay in lock-step (a PiSSA/DoRA adapter trained with one init must be loaded with
the same init, or the delta is wrong). Two optional knobs beyond vanilla LoRA:

  init_lora_weights:  True (standard LoRA) | "pissa" | "pissa_niter_16" (PiSSA —
                      init A,B from the principal SVD components of W, faster
                      convergence at zero extra cost).
  use_dora:           False | True (DoRA — weight magnitude/direction decomposition,
                      a small accuracy bump, ~1.2-1.4x step time).

PiSSA CAVEAT: PiSSA MUTATES the frozen base weights (it subtracts the principal
component into the residual). inference.py rebuilds the model from the vanilla HF
checkpoint, so a PiSSA adapter will load against the WRONG base unless the
pissa-residual base is saved at train time and restored at load time (NOT yet
implemented here). DoRA has no such issue and is the safe one to adopt now; PiSSA
is plumbed but should not be used until the base-restore path exists.
"""
from __future__ import annotations


def lora_config_kwargs(config: dict) -> dict:
    """Return the kwargs dict for peft.LoraConfig from the run config.

    Raises KeyError if a required LoRA field is missing (lora_rank/alpha/targets),
    matching the existing explicit-kwargs behavior.
    """
    return dict(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_targets"],
        lora_dropout=config.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=bool(config.get("use_dora", False)),
        init_lora_weights=config.get("init_lora_weights", True),
    )


def uses_pissa(config: dict) -> bool:
    """True if the config requests a PiSSA init (so callers can warn / guard the
    base-restore requirement)."""
    v = config.get("init_lora_weights", True)
    return isinstance(v, str) and v.lower().startswith("pissa")
