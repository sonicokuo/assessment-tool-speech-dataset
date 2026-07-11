"""run_external_baseline.py — zero-shot external audio-LLM baseline for the faithfulness table.

Prompts an off-the-shelf audio LLM (default Qwen2-Audio-7B-Instruct) for the SAME numeric quality
report on the SAME test clips, writes output in the SAME inference_results.json format so
scripts/score_matched_test.py scores it against the SAME GT with the SAME parser — a drop-in row
next to M3b (token-grounding) and B1 (baseline).

Steelman: the prompt gives the model the EXACT output template our parser accepts, so a low score
reflects the model's inability to ESTIMATE the numbers from audio, not a parsing artifact.

External models need raw waveforms (not our WavLM .pt features), so --audio_dir points at the
Libri2Mix test mix_clean wavs; the test clip list comes from the GT json keys (stem, no .wav).

Usage:
  python src/run_external_baseline.py --model Qwen/Qwen2-Audio-7B-Instruct \
    --gt data/descriptions_observability_test.json --audio_dir <.../test/mix_clean> \
    --out .../qwen2audio_results.json --max_clips 3000 --device cuda
"""
import argparse
import json
import os
from pathlib import Path

import torch

PROMPT = (
    "You are an expert speech-quality analyst. Listen carefully to the recording and estimate the "
    "following five acoustic measurements from the audio. Give a specific numeric value for each, based "
    "on what you actually hear. Do NOT output a placeholder, blank, variable, or the words 'estimate'/"
    "'n' — write a real number in every sentence. Respond with exactly these five sentences:\n"
    "The SNR is (your dB number) dB.\n"
    "The SRMR is (your number).\n"
    "The speaking rate is (your number) syllables per second.\n"
    "The recording contains (your whole number) pauses.\n"
    "The pause rate is (your number) per minute.\n"
    "Replace each parenthesized phrase with the actual number you estimate for this recording."
)


def load_audio(path, sr):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def atomic_write(path, obj):
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(obj))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--gt", required=True, help="GT json; its keys are the clip stems to score")
    ap.add_argument("--audio_dir", required=True, help="dir of raw <stem>.wav (Libri2Mix test mix_clean)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_clips", type=int, default=3000)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map={"": a.device}).eval()
    sr = processor.feature_extractor.sampling_rate

    stems = list(json.load(open(a.gt)).keys())[: a.max_clips]
    # resume-safe: keep whatever was already scored
    done = {}
    if os.path.exists(a.out):
        try:
            done = {r["filename"]: r for r in json.load(open(a.out))}
        except Exception:
            done = {}
    results = list(done.values())
    missing_audio = 0

    for i, stem in enumerate(stems):
        fname = f"{stem}.wav"
        if fname in done:
            continue
        wav_path = os.path.join(a.audio_dir, fname)
        if not os.path.exists(wav_path):
            missing_audio += 1
            continue
        try:
            conv = [{"role": "user", "content": [
                {"type": "audio", "audio_url": wav_path}, {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios = [load_audio(wav_path, sr)]
            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to(a.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            gen = gen[:, inputs.input_ids.size(1):]
            resp = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            results.append({"filename": fname, "generated": resp})
        except Exception as e:
            print(f"  [skip] {stem}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            atomic_write(a.out, results)
            print(f"  {i+1}/{len(stems)} ({len(results)} done, {missing_audio} missing audio)", flush=True)
    atomic_write(a.out, results)
    print(f"DONE: {len(results)} scored, {missing_audio} missing audio -> {a.out}")


if __name__ == "__main__":
    main()
