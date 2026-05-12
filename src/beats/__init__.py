"""Vendored BEATs (Microsoft, 2022) — audio pretrained encoder.

Source: https://github.com/microsoft/unilm/tree/master/beats
Paper:  Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers"
        https://arxiv.org/abs/2212.09058
License: MIT (see headers in each .py file).

We vendor BEATs.py, modules.py, and backbone.py verbatim from Microsoft's
unilm/beats directory (only the two `from X import` statements at the top
of BEATs.py and backbone.py were patched to relative imports) because
BEATs is not available via HuggingFace's `AutoModel.from_pretrained` — the
official release is a fairseq-style checkpoint loaded with a custom model
class. Vendoring is the cleanest solution; the alternative would be a
fragile community HF port.

We did NOT vendor Tokenizers.py or quantizer.py — those are for the
self-supervised pretraining objective (acoustic-token prediction). For
downstream feature extraction (our use case) only the BEATs encoder is
needed.

Loading a pretrained checkpoint (rehosted on HF Hub by lpepino):

    from huggingface_hub import hf_hub_download
    from beats import BEATs, BEATsConfig
    import torch

    ckpt_path = hf_hub_download("lpepino/beats_ckpts", "BEATs_iter3_plus_AS2M.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # waveform: (B, n_samples) float at 16 kHz, in [-1, 1]
    features, _ = model.extract_features(waveform, padding_mask=None)
    # features: (B, T_p, 768) patch embeddings
"""

from .BEATs import BEATs, BEATsConfig  # noqa: F401
