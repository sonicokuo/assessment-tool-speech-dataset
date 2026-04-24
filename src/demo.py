"""Interactive demo: load adapter + LLM once, generate from arbitrary test clips on demand.

Usage:
    python -i src/demo.py --config configs/config.psc.yaml \
        --checkpoint $SHARED/checkpoints/q3_8b_qformer/best.pt \
        --test_dir   $SHARED/data/processed/test

The `-i` flag drops you into a Python REPL after loading. From there:

    >>> gen(0)                          # first test clip, default decoding
    >>> gen(0, max_new_tokens=512)      # bump cap
    >>> gen(0, top_k=1)                 # greedy
    >>> gen(0, temperature=0.7, top_p=0.9)   # sampling
    >>> compare(0)                      # generation vs target side-by-side, plus SFS
    >>> info(0)                         # overlap segments + audio length, no generation

    >>> gen_from_wav("path/to/some.wav")          # raw audio in (zero overlap context — see docstring)
    >>> gen_from_wav("path/to/s1_s2_mix.wav", overlap_segs=[(0.5, 2.3), (3.1, 4.0)])
                                        # if you have ground-truth overlap intervals (Libri2Mix)

Everything lives in the namespace; tweak decoding params, run again, repeat.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from adapter import build_adapter
from dataset import PreprocessedDataset
from inference import _pick_device, _sync_config_with_checkpoint, generate
from preprocess import build_overlap_info
from sfs import ClaimParser, SFSScorer


def _load() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = _sync_config_with_checkpoint(config, args.checkpoint)

    device = _pick_device()
    print(f"[device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["lm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] {config['lm_name']} (this is the slow step, ~10 min on PSC) …")
    llm = AutoModelForCausalLM.from_pretrained(
        config["lm_name"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    llm = get_peft_model(
        llm,
        LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            target_modules=config["lora_targets"], lora_dropout=config["lora_dropout"],
            bias="none", task_type="CAUSAL_LM",
        ),
    )

    print(f"[load] checkpoint {args.checkpoint}")
    ck = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    adapter = build_adapter(config["adapter_variant"], lm_dim=llm.config.hidden_size).to(device).to(torch.bfloat16)
    adapter.load_state_dict(ck["adapter_state_dict"])
    llm.load_state_dict(ck["lora_state_dict"])
    adapter.eval(); llm.eval()
    print(f"[load] adapter+LoRA loaded; epoch={ck.get('epoch')}, val_loss={ck.get('best_val_loss', 0):.4f}")

    test_set = PreprocessedDataset(args.test_dir, config.get("descriptions_path"))
    prompt_ids = tokenizer(config["prompt"], return_tensors="pt").input_ids.to(device)
    print(f"[load] {len(test_set)} test clips at {args.test_dir}")
    print(f"[ready] type gen(0), compare(3), info(7), help(gen) …")

    return {
        "adapter": adapter, "llm": llm, "tokenizer": tokenizer,
        "device": device, "test_set": test_set, "prompt_ids": prompt_ids,
        "config": config, "parser": ClaimParser(), "scorer": SFSScorer(),
    }


_S = _load()


def gen(idx: int, max_new_tokens: int = 512, temperature: float = 1.0,
        top_k: int = 1, top_p: float = 1.0) -> str:
    """Generate a description for test clip `idx` and return the text."""
    sample = _S["test_set"][idx]
    text = generate(
        _S["adapter"], _S["llm"], _S["tokenizer"],
        sample["audio_features"], sample["overlap_info"],
        _S["prompt_ids"], _S["device"],
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    print(f"\n── clip {idx}: {sample['filename']} ──")
    print(text)
    print(f"\n[meta] {len(text.split())} words, ends in period: {text.rstrip().endswith('.')}")
    return text


def compare(idx: int, **kwargs) -> dict:
    """Generate, print side-by-side with target, and SFS-score it."""
    sample = _S["test_set"][idx]
    text = generate(
        _S["adapter"], _S["llm"], _S["tokenizer"],
        sample["audio_features"], sample["overlap_info"],
        _S["prompt_ids"], _S["device"],
        max_new_tokens=kwargs.get("max_new_tokens", 512),
        temperature=kwargs.get("temperature", 1.0),
        top_k=kwargs.get("top_k", 1),
        top_p=kwargs.get("top_p", 1.0),
    )
    target = sample.get("target_text", "") or ""
    print(f"\n── clip {idx}: {sample['filename']} ──")
    print(f"\nGENERATED:\n{text}")
    print(f"\nTARGET:\n{target}")
    if target:
        gt_claims = _S["parser"].parse(target)
        gt = {c.feature: c.value for c in gt_claims}
        if sample.get("overlap_segments"):
            gt["overlap_segments"] = sample["overlap_segments"]
        gen_claims = _S["parser"].parse(text)
        result = _S["scorer"].score(gen_claims, gt)
        print(f"\nSFS  P={result['precision']:.2f}  R={result['recall']:.2f}  F1={result['f1']:.2f}  "
              f"({result['n_correct']}/{result['n_claims']} claims correct)")
        return result
    return {}


def info(idx: int) -> None:
    """Print audio features shape + overlap segments for clip `idx` without generating."""
    sample = _S["test_set"][idx]
    af = sample["audio_features"]
    oi = sample["overlap_info"]
    print(f"\n── clip {idx}: {sample['filename']} ──")
    print(f"audio_features: {tuple(af.shape)}  (~{af.shape[0] * 0.02:.2f}s of speech)")
    print(f"overlap_info:   {tuple(oi.shape)}")
    print(f"overlap_segments ({len(sample.get('overlap_segments', []))}): {sample.get('overlap_segments')}")
    if "target_text" in sample:
        t = sample["target_text"]
        print(f"target ({len(t.split())} words): {t[:300]}{'...' if len(t)>300 else ''}")


# ── Raw-wav-in path ────────────────────────────────────────────
# Lazy-loaded WavLM (separate model from the LM); first call to gen_from_wav()
# pays a one-time ~30 sec load + a small download if not cached.
_S["wavlm"] = None


def _ensure_wavlm():
    if _S["wavlm"] is not None:
        return _S["wavlm"]
    from transformers import WavLMModel
    print("[load] microsoft/wavlm-large (one-time, ~30 sec; cached after) …")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(_S["device"]).eval()
    for p in wavlm.parameters():
        p.requires_grad = False
    _S["wavlm"] = wavlm
    print("[load] WavLM ready.")
    return wavlm


def gen_from_wav(
    wav_path: str,
    overlap_segs: list | None = None,
    overlap_ratio: float | None = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 1.0,
):
    """Generate a quality description for an arbitrary .wav file.

    Args:
        wav_path: path to a wav/flac/mp3 file (mono or stereo; auto-resampled to 16 kHz).
        overlap_segs: optional list of (start_sec, end_sec) overlap intervals, e.g. from
            Pyannote or VAD on s1/s2 stems. If None, the model sees zero overlap signal.
            ⚠ Without overlap_segs the model has *no information* about where speakers
            overlap, so its overlap-related outputs (overlap_ratio, overlap_segments,
            "F0 unreliable" hedges) will be ungrounded. Use only when overlap context
            is unavailable, or as a "what does it produce on a clean clip?" sanity check.
        overlap_ratio: optional clip-wide overlap fraction (0–1). Defaults to the sum of
            overlap_segs durations / clip duration; or 0 if no segs.
        max_new_tokens, temperature, top_k, top_p: same as gen().
    """
    import torchaudio
    wavlm = _ensure_wavlm()

    # Load + resample to 16 kHz mono (WavLM input contract).
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        waveform = torch.from_numpy(data).float()
        waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0)
    duration_s = waveform.shape[0] / 16000.0

    # WavLM forward pass → (T, 1024).
    with torch.no_grad():
        af = wavlm(waveform.unsqueeze(0).to(_S["device"])).last_hidden_state.squeeze(0).cpu()
    T = af.shape[0]

    # Build (T, 5) overlap_info from the optional overlap_segs argument. preprocess.py
    # expects a "start_sample-end_sample;..." string, so format it here.
    if overlap_segs:
        segs_str = ";".join(f"{int(s*16000)}-{int(e*16000)}" for s, e in overlap_segs)
        if overlap_ratio is None:
            overlap_ratio = sum(e - s for s, e in overlap_segs) / max(duration_s, 1e-6)
    else:
        segs_str = ""
        overlap_ratio = 0.0
        print("[warn] no overlap_segs provided — overlap_info is all-zeros. "
              "Overlap-related model outputs will be ungrounded.")

    overlap_info, segments_sec = build_overlap_info(segs_str, float(overlap_ratio), T, sample_rate=16000)

    # Generate.
    text = generate(
        _S["adapter"], _S["llm"], _S["tokenizer"],
        af, overlap_info,
        _S["prompt_ids"], _S["device"],
        max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    print(f"\n── wav: {wav_path} ──")
    print(f"[meta] duration={duration_s:.2f}s, T={T} frames, "
          f"overlap_segments={segments_sec}, overlap_ratio={overlap_ratio:.3f}")
    print(text)
    print(f"\n[meta] {len(text.split())} words, ends in period: {text.rstrip().endswith('.')}")


# Convenience handles for the REPL
adapter = _S["adapter"]
llm = _S["llm"]
tokenizer = _S["tokenizer"]
test_set = _S["test_set"]
config = _S["config"]
