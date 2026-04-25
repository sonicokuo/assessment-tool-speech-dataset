"""Raw-LM baseline demo — Qwen3-8B alone, no LoRA, no adapter, no audio.

Useful for:
- "What would a stock LLM produce for this prompt?" — a paper baseline showing the
  audio path adds value.
- Sanity check that the LLM + prompt format is sane.
- Comparing adapter-vs-raw on identical prompts.

Usage:
    python -i src/demo_raw.py --config configs/config.psc.yaml --lm_name Qwen/Qwen3-8B

The `-i` flag drops you into a Python REPL after loading. From there:

    >>> raw_gen()                          # generate from the standard prompt
    >>> raw_gen(max_new_tokens=512)        # bump cap
    >>> raw_gen(prefix="Here's a noisy 5-second clip with two speakers overlapping.")
                                           # prepend custom context to the prompt
    >>> raw_gen(top_k=0, top_p=0.9, temperature=0.7)   # sampling

No audio. No adapter. No LoRA. The model only sees text.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import _pick_device


def _load() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.psc.yaml",
                        help="Only the 'prompt' field is consumed.")
    parser.add_argument("--lm_name", type=str, default=None,
                        help="HF model id. Defaults to config['lm_name'].")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    lm_name = args.lm_name or config["lm_name"]

    device = _pick_device()
    print(f"[device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] {lm_name} (raw — no LoRA, no adapter, no checkpoint) …")
    llm = AutoModelForCausalLM.from_pretrained(
        lm_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    llm.eval()
    print(f"[load] {lm_name} ready.")
    print(f"[prompt] {config['prompt']!r}")
    print(f"[ready] try raw_gen() or raw_gen(prefix='custom context')")

    return {"tokenizer": tokenizer, "llm": llm, "device": device,
            "lm_name": lm_name, "config": config}


_S = _load()


@torch.no_grad()
def raw_gen(
    prefix: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
) -> None:
    """Generate from the configured prompt, optionally prepended with `prefix`."""
    base_prompt = _S["config"]["prompt"]
    full_prompt = (prefix.rstrip() + " " if prefix else "") + base_prompt

    input_ids = _S["tokenizer"](full_prompt, return_tensors="pt").input_ids.to(_S["device"])
    n_prompt = input_ids.shape[1]

    out = _S["llm"].generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(top_k != 1 or temperature != 1.0 or top_p != 1.0),
        top_k=top_k if top_k > 0 else 50,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=_S["tokenizer"].pad_token_id,
    )
    gen_ids = out[0, n_prompt:]
    text = _S["tokenizer"].decode(gen_ids, skip_special_tokens=True)

    print(f"\n── raw {_S['lm_name']} generation ──")
    if prefix:
        print(f"[prefix] {prefix}")
    print(f"[prompt] {base_prompt}")
    print(f"\n{text}")
    print(f"\n[meta] {len(text.split())} words, ends in period: {text.rstrip().endswith('.')}")


# Convenience handles
llm = _S["llm"]
tokenizer = _S["tokenizer"]
config = _S["config"]
