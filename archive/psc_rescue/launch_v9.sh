#!/bin/bash
set -e
cd /ocean/projects/cis260125p/shared/assessment-tool-redirect
LOG=/ocean/projects/cis260125p/shared/logs/v9_infer_full.log
echo "=== prewarm 8B blobs on $(hostname) $(date) ===" 
cat /ocean/projects/cis260125p/shared/hf_cache/hub/models--Qwen--Qwen3-8B/blobs/* > /dev/null 2>&1
echo "=== prewarm done $(date) ==="
export HF_HOME=/ocean/projects/cis260125p/shared/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
# detach the actual inference so it survives our srun step ending
setsid /ocean/projects/cis260125p/shared/envs/project/bin/python -u src/inference.py \
  --config /ocean/projects/cis260125p/shared/config.v9_rescore.yaml \
  --checkpoint /ocean/projects/cis260125p/shared/checkpoints/v9_rescore_cleanf0/best.pt \
  --test_dir /ocean/projects/cis260125p/shared/data/processed_pyannote/test \
  --top_k 1 \
  --max_new_tokens 512 \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "INFERENCE_PID=$PID launched $(date), log=$LOG"
# give it a moment to start writing
sleep 8
echo "--- log head ---"
head -40 "$LOG" 2>/dev/null || true
