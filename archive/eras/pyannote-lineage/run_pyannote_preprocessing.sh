#!/usr/bin/env bash
# AQUA-NL one-time preprocessing for Pyannote-only training.
#
# Re-extracts overlap features using Pyannote on the *mix* (no s1/s2 stems involved)
# for all three splits, then re-preprocesses .pt files using the new feature CSV.
#
# Total wall-clock on PSC H100: ~3-5 hours for ~20k clips across 3 splits.
#
# Prereqs (one-time):
#   1. HF_TOKEN exported with access to pyannote/segmentation-3.0 and
#      pyannote/voice-activity-detection (gated repos).
#   2. Conda env activated, $SHARED set, repo at $SHARED/assessment-tool-speech-dataset.
#
# Usage on PSC:
#   export HF_TOKEN=hf_...
#   bash scripts/run_pyannote_preprocessing.sh
#
# This script does NOT touch the existing features/ or processed/ directories — outputs
# go to features_pyannote/ and processed_pyannote/ so you can roll back by changing
# config paths.

set -euo pipefail

: "${SHARED:?SHARED env var must be set, e.g. export SHARED=/ocean/projects/cis260125p/shared}"
: "${HF_TOKEN:?HF_TOKEN env var must be set — Pyannote repos are gated}"

REPO=$SHARED/assessment-tool-speech-dataset
FEATURES_DIR=$SHARED/data/features_pyannote
PROCESSED_DIR=$SHARED/data/processed_pyannote
LIBRI_ROOT=$SHARED/data/Libri2Mix/Libri2Mix/wav16k/min

mkdir -p "$FEATURES_DIR" "$PROCESSED_DIR"/{train,val,test}

echo "==[1/2] Pyannote feature extraction =="
for split in train-100 dev test; do
  out="$FEATURES_DIR/${split}.csv"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists; delete it manually if you want to re-extract."
    continue
  fi
  echo "→ extracting $split → $out"
  python "$REPO/src/feature_extractor_mix.py" \
    --audio_dir      "$LIBRI_ROOT/$split/mix_clean" \
    --libri2mix_root "$LIBRI_ROOT/$split" \
    --overlap        pyannote \
    --hf_token       "$HF_TOKEN" \
    --output         "$out"
done

echo "==[2/2] Audio preprocessing (4-channel overlap_info) =="
# Map split → output dir name (train.py expects val/, not dev/).
declare -A DIRMAP=( [train-100]=train [dev]=val [test]=test )
for split in train-100 dev test; do
  out_dir="$PROCESSED_DIR/${DIRMAP[$split]}"
  echo "→ preprocess $split → $out_dir"
  python "$REPO/src/preprocess.py" \
    --audio_dir    "$LIBRI_ROOT/$split/mix_clean" \
    --features_csv "$FEATURES_DIR/${split}.csv" \
    --output_dir   "$out_dir"
done

echo "==Done=="
echo "Set in configs/config.psc.yaml:"
echo "  data_dir:     $PROCESSED_DIR"
echo "  features_csv: { train: $FEATURES_DIR/train-100.csv, val: $FEATURES_DIR/dev.csv }"
