#!/bin/bash
cd /ocean/projects/cis260125p/shared/assessment-tool-redirect
export HF_HOME=/ocean/projects/cis260125p/shared/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
echo "=== v9 inference START $(hostname) $(date) ==="
exec /ocean/projects/cis260125p/shared/envs/project/bin/python -u src/inference.py \
  --config /ocean/projects/cis260125p/shared/config.v9_rescore.yaml \
  --checkpoint /ocean/projects/cis260125p/shared/checkpoints/v9_rescore_cleanf0/best.pt \
  --test_dir /ocean/projects/cis260125p/shared/data/processed_pyannote/test \
  --top_k 1 \
  --max_new_tokens 512
