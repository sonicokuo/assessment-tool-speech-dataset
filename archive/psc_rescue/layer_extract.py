"""P0 stage 1 — extract mean+std pooled WavLM features for ALL 25 layers.

WHY. `src/preprocess.py:196` caches only `last_hidden_state` (layer 24), so BOTH the ridge
skyline and our adapter sit behind the same final-layer bottleneck. That layer is the worst
plausible choice for our targets on two independent grounds:
  * WavLM pretraining mixes utterances with an interfering speaker / DNS noise and predicts the
    CLEAN primary speaker's pseudo-labels, so the top of the stack is explicitly optimised to be
    INVARIANT to interferer and noise properties -- exactly what we measure (arXiv:2110.13900).
  * Acoustic/prosodic content concentrates EARLY in SSL stacks; best MOS-regression layers are
    reported at 3-5 with last-layer penalties up to ~0.16 LCC (arXiv:2508.08962, preprint), and
    linear probes on early layers beat the REAPER pitch tracker on F0 (SLT 2022, arXiv:2210.07185).

We store ONLY mean+std pooled statistics per layer (25 x 2048 per clip), which is all a
SUPERB-style linear probe needs. Full sequences would be ~1 TB; this is a few GB.

MATCHING. Layer 24 is re-fit here on the SAME subsample as every other layer, so the sweep is
internally consistent AND we can sanity-check that layer 24 reproduces the recorded 0.6883.

Usage: layer_extract.py <out.npz> <split:train|test> [n_clips]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torchaudio
from transformers import WavLMModel

SH = "/ocean/projects/cis260125p/shared"
OUT, SPLIT = sys.argv[1], sys.argv[2]
NCLIP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SR = 16000

AUDIO = {
    "train": [f"{SH}/data/audio_corrected/train-100", f"{SH}/data/audio_corrected/train-100-s1clean"],
    "test": [f"{SH}/data/audio_corrected/test", f"{SH}/data/audio_corrected/test-s1clean"],
}[SPLIT]

dev = "cuda" if torch.cuda.is_available() else "cpu"
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().half()
for p in wavlm.parameters():
    p.requires_grad = False

files = []
for d in AUDIO:
    files.extend(sorted(glob.glob(os.path.join(d, "*.wav"))))
files.sort()
if NCLIP:
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))[:NCLIP]
    files = [files[i] for i in sorted(idx)]
print(f"[init] split={SPLIT} clips={len(files)} device={dev}", flush=True)

feats, names = [], []
with torch.no_grad():
    for i, f in enumerate(files):
        try:
            w, sr = torchaudio.load(f)
        except Exception:
            continue
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
        w = w.mean(dim=0)
        if w.numel() < SR // 2:
            continue
        hs = wavlm(w.unsqueeze(0).to(dev).half(), output_hidden_states=True).hidden_states
        # hidden_states = (conv-proj output, layer_1 ... layer_24) -> 25 tensors
        pooled = np.stack(
            [np.concatenate([h[0].float().mean(0).cpu().numpy(),
                             h[0].float().std(0).cpu().numpy()]) for h in hs]
        ).astype(np.float16)                                    # (25, 2048)
        feats.append(pooled)
        names.append(os.path.splitext(os.path.basename(f))[0])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

X = np.stack(feats)                                             # (N, 25, 2048)
np.savez_compressed(OUT, X=X, names=np.array(names))
print(f"wrote {OUT}  X={X.shape} ({X.nbytes/1e9:.2f} GB uncompressed)")
