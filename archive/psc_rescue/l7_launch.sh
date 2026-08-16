#!/bin/bash
SH=/ocean/projects/cis260125p/shared
# Gate the retrain on the integrity check PASSING. Training 5 hours on a silently-misaligned
# cache costs far more than waiting; a FAIL must stop, not warn.
for i in $(seq 1 120); do
  grep -q "PASS — layer-7 cache is safe to train on" $SH/logs/verify_l7.log 2>/dev/null && break
  if grep -q "TOTAL CORRUPT: [1-9]" $SH/logs/verify_l7.log 2>/dev/null; then
    echo "INTEGRITY FAILED — refusing to launch layer-7 retrain"; exit 1
  fi
  sleep 30
done
grep -q "PASS — layer-7 cache is safe to train on" $SH/logs/verify_l7.log || { echo "TIMEOUT waiting for integrity"; exit 1; }
echo "INTEGRITY PASSED — launching layer-7 + audio-only retrain"
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1 WANDB_DATA_DIR=$SH/wandb_data WANDB_CACHE_DIR=$SH/wandb_cache
cd $SH/repo_verify
$SH/envs/project/bin/python -u src/train.py --config configs/config.l7audioonly.s73.yaml
