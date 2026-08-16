"""P0 stage 3 — rebuild the preprocessed cache from WavLM LAYER 7 instead of layer 24.

WHY. The full-train sweep (sanity-passed, layer 24 reproduces the recorded 0.6887 to 4 decimals)
shows layer 7 beats layer 24 by +0.0148 robust5 and +0.0650 on the ill-posed panel, and the
STRONGEST probe we could build (tuned ridge on layer 7) reaches 0.7200 -- above our audio-only
aux head's 0.7066. Our system reads layer 24; the baseline now reads layer 7. That comparison
is unfair TO US and cannot be reported as-is, so the layer-7 retrain is the critical run.

SINGLE-KNOB DISCIPLINE. This script COPIES `overlap_info`, `overlap_segments` and `filename`
verbatim from the existing layer-24 .pt and replaces ONLY `audio_features`. Nothing else can
drift -- not the VAD-derived overlap channels, not the segment lists, not the clip ordering.
Any difference between a layer-24 run and a layer-7 run is therefore attributable to the layer
alone.

dtype is kept float32 to match the existing cache exactly; fp16 would be half the disk but
risks silent dtype mismatches in the collate/padding path, which is not a trade worth making
for a comparison whose whole point is that only ONE thing changed.

Usage: extract_layer7.py <split:train|val|test> [layer]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from transformers import WavLMModel

# torchaudio.load dispatches through torchcodec, which on this cluster intermittently fails to
# load libavutil (FFmpeg). soundfile has no such dependency and is what src/preprocess.py
# already falls back to, so read with soundfile and resample with torchaudio's FUNCTIONAL API
# (which does not touch the codec path).
import soundfile as sf
import torchaudio.functional as AF


def read_wav(path: str):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(np.asarray(data)).T          # (C, N)
    return w.mean(dim=0), sr

SH = "/ocean/projects/cis260125p/shared"
SPLIT = sys.argv[1]
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 7
SR = 16000

SRC_PT = f"{SH}/data/processed_corrected/{SPLIT}"
DST_PT = f"{SH}/data/processed_layer{LAYER}/{SPLIT}"
AUDIO_DIRS = {
    "train": [f"{SH}/data/audio_corrected/train-100", f"{SH}/data/audio_corrected/train-100-s1clean"],
    "test": [f"{SH}/data/audio_corrected/test", f"{SH}/data/audio_corrected/test-s1clean"],
    "val": [f"{SH}/data/audio_corrected/dev", f"{SH}/data/audio_corrected/dev-s1clean"],
}[SPLIT]

os.makedirs(DST_PT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().half()
for p in wavlm.parameters():
    p.requires_grad = False

# stem -> audio path
apath = {}
for d in AUDIO_DIRS:
    for f in glob.glob(os.path.join(d, "*.wav")):
        apath[os.path.splitext(os.path.basename(f))[0]] = f

src = sorted(glob.glob(os.path.join(SRC_PT, "*.pt")))
print(f"[init] split={SPLIT} layer={LAYER} src={len(src)} audio={len(apath)} dev={dev}", flush=True)

done = skipped = missing = 0
with torch.no_grad():
    for i, sp in enumerate(src):
        stem = os.path.splitext(os.path.basename(sp))[0]
        dp = os.path.join(DST_PT, os.path.basename(sp))
        if os.path.exists(dp):
            skipped += 1
            continue
        wav_path = apath.get(stem)
        if wav_path is None:
            missing += 1
            continue
        d = torch.load(sp, map_location="cpu", weights_only=False)
        w, sr = read_wav(wav_path)
        if sr != SR:
            w = AF.resample(w, sr, SR)
        hs = wavlm(w.unsqueeze(0).to(dev).half(), output_hidden_states=True).hidden_states
        feat = hs[LAYER][0].float().cpu()                      # (T, 1024)

        # The layer-24 cache defines T. Layer indices share the same frame rate, so lengths
        # match; clamp defensively rather than silently emitting a different-length tensor,
        # because overlap_info is copied verbatim and MUST stay aligned to audio_features.
        T_ref = d["audio_features"].shape[0]
        if feat.shape[0] != T_ref:
            feat = feat[:T_ref] if feat.shape[0] > T_ref else torch.cat(
                [feat, feat[-1:].expand(T_ref - feat.shape[0], -1)], dim=0)

        out = dict(d)                                          # copy EVERYTHING else verbatim
        out["audio_features"] = feat
        # PID-unique tmp: two nodes may work the same split concurrently (they skip existing
        # files, but both can start the SAME file before either finishes). A shared .tmp name
        # would let their writes interleave and os.replace would then publish a CORRUPT .pt.
        tmp = f"{dp}.{os.getpid()}.tmp"
        torch.save(out, tmp)
        os.replace(tmp, dp)                                    # atomic: no half-written .pt
        done += 1
        if (done + skipped) % 2000 == 0:
            print(f"  {done+skipped}/{len(src)} (new={done} skip={skipped} miss={missing})",
                  flush=True)

print(f"[done] split={SPLIT} layer={LAYER} written={done} skipped={skipped} missing={missing}")
print(f"       -> {DST_PT}")
