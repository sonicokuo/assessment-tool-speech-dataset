#!/usr/bin/env python3
"""Plot per-section attention overlays on a log-mel spectrogram.

Consumes the JSON produced by scripts/extract_attention.py and renders the
paper figure: one log-mel spectrogram per clip with N panels showing the
per-section attention vector overlaid as a heatmap above the spectrogram.

Usage
-----
    python scripts/plot_attention.py \
      --attention_json $SHARED/checkpoints/v7_lora_8b/attention/1089-..._attention.json \
      --audio_dir      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
      --output_pdf     docs/figures/clip_1089-..._attention.pdf

Or batch over a directory:

    python scripts/plot_attention.py \
      --attention_dir  $SHARED/checkpoints/v7_lora_8b/attention/ \
      --audio_dir      $SHARED/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
      --output_dir     docs/figures/

Layout
------
For each clip the figure has 1 spectrogram row + len(section_attentions)
overlay rows, all sharing the same time axis. The spectrogram is in
greyscale, the attention overlays are 'hot' colormap normalized per row.

The generated text is printed below the figure as the caption, with each
section's sentence color-coded to match its overlay row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SECTION_ORDER = ["noise", "reverb", "pitch", "tempo", "pauses", "overlap"]
SECTION_COLORS = {
    "noise":   "#E45756",   # red
    "reverb":  "#F58518",   # orange
    "pitch":   "#54A24B",   # green
    "tempo":   "#4C78A8",   # blue
    "pauses":  "#B279A2",   # purple
    "overlap": "#9D755D",   # brown
}


def compute_log_mel(wav_path: Path, sr_target: int = 16000,
                    n_mels: int = 80, hop_length: int = 160):
    """Compute log-mel spectrogram with librosa (preferred) or torchaudio."""
    try:
        import librosa
        wav, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=n_mels, hop_length=hop_length,
            n_fft=400, fmin=20, fmax=sr // 2,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        times = librosa.frames_to_time(np.arange(log_mel.shape[1]),
                                       sr=sr, hop_length=hop_length)
        return log_mel, times, sr
    except ImportError:
        # Fallback to torchaudio
        import torch
        import torchaudio
        import torchaudio.transforms as T
        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sr_target:
            wav = T.Resample(sr, sr_target)(wav)
            sr = sr_target
        mel_t = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels,
                                 hop_length=hop_length, n_fft=400)(wav)
        log_mel = (mel_t.clamp(min=1e-10).log10() * 10).squeeze(0).numpy()
        times = np.arange(log_mel.shape[1]) * hop_length / sr
        return log_mel, times, sr


def plot_clip(attention_data: dict, wav_path: Path, output_pdf: Path) -> None:
    """Render one clip's figure: spectrogram + per-section overlays."""
    log_mel, mel_times, sr = compute_log_mel(wav_path)

    sec_attn = attention_data["section_attentions"]
    present_sections = [s for s in SECTION_ORDER if s in sec_attn]
    n_overlays = len(present_sections)

    if n_overlays == 0:
        print(f"  [skip] no sections captured for {wav_path.name}")
        return

    # Time axis for attention: each prefix token covers prefix_token_stride_sec
    stride = attention_data["prefix_token_stride_sec"]
    P = attention_data["n_prefix_tokens"]
    attn_times = np.arange(P) * stride + stride / 2

    fig_height = 2.5 + 0.9 * n_overlays
    fig, axes = plt.subplots(
        n_overlays + 1, 1,
        figsize=(10, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * n_overlays, "hspace": 0.15},
    )
    if n_overlays + 1 == 1:
        axes = [axes]

    # Spectrogram (top)
    ax = axes[0]
    extent = [mel_times[0], mel_times[-1], 0, log_mel.shape[0]]
    ax.imshow(log_mel, aspect="auto", origin="lower", cmap="gray_r",
              extent=extent, interpolation="nearest")
    ax.set_ylabel("Mel bin")
    ax.set_title(f"{wav_path.name}    (duration={attention_data['audio_duration_sec']:.2f}s, "
                 f"P={P} prefix tokens × {stride:.2f}s stride)",
                 fontsize=10)

    # Per-section attention overlays
    for i, section in enumerate(present_sections):
        ax = axes[i + 1]
        vec = np.array(sec_attn[section])
        # Normalize for visual clarity (each row max=1)
        vec_norm = vec / max(vec.max(), 1e-12)
        ax.fill_between(attn_times, 0, vec_norm,
                        color=SECTION_COLORS[section], alpha=0.45,
                        edgecolor=SECTION_COLORS[section], linewidth=1.2)
        ax.set_xlim(mel_times[0], mel_times[-1])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(section, rotation=0, ha="right", va="center",
                      fontsize=10, color=SECTION_COLORS[section],
                      fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")

    # Caption (generated text underneath)
    fig.suptitle("Per-section attention attribution", fontsize=11, y=0.98)
    plt.figtext(
        0.02, 0.005,
        f"Generated: {attention_data['generated'][:280]}{'...' if len(attention_data['generated']) > 280 else ''}",
        fontsize=7, ha="left", wrap=True, color="#444444",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved → {output_pdf}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--attention_json", type=Path,
                     help="Single attention JSON to plot")
    grp.add_argument("--attention_dir", type=Path,
                     help="Directory containing many *_attention.json files")
    p.add_argument("--audio_dir", type=Path, required=True,
                   help="Directory containing the source .wav files (mix_clean/)")
    p.add_argument("--output_pdf", type=Path, default=None,
                   help="Output PDF path (only valid with --attention_json)")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Output directory (used with --attention_dir)")
    p.add_argument("--format", default="pdf", choices=["pdf", "png"],
                   help="Image format for batch output")
    args = p.parse_args()

    if args.attention_json:
        data = json.loads(args.attention_json.read_text())
        wav_path = args.audio_dir / data["filename"]
        if not wav_path.exists():
            print(f"ERROR: wav not found at {wav_path}", file=sys.stderr)
            return 2
        out = args.output_pdf or args.attention_json.with_suffix(f".{args.format}")
        plot_clip(data, wav_path, out)
        return 0

    # Batch mode
    if not args.output_dir:
        print("ERROR: --output_dir required when using --attention_dir", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsons = sorted(args.attention_dir.glob("*_attention.json"))
    print(f"Found {len(jsons)} attention JSONs in {args.attention_dir}")
    for jp in jsons:
        data = json.loads(jp.read_text())
        wav_path = args.audio_dir / data["filename"]
        if not wav_path.exists():
            print(f"  [skip] wav not found: {wav_path}")
            continue
        stem = jp.stem.replace("_attention", "")
        out = args.output_dir / f"{stem}.{args.format}"
        plot_clip(data, wav_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
