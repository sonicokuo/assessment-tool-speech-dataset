#!/bin/bash
SH=/ocean/projects/cis260125p/shared
# wait for BOTH extractions (test runs first, then train via &&)
for i in $(seq 1 240); do
  [ -f "$SH/layers_train.npz" ] && break
  if grep -qE "Traceback|Error" $SH/logs/lx_test.log $SH/logs/lx_train.log 2>/dev/null; then
    echo "EXTRACTION FAILED"; exit 1
  fi
  sleep 30
done
[ -f "$SH/layers_train.npz" ] || { echo "TIMEOUT waiting for extraction"; exit 1; }
cd $SH/repo_verify
$SH/envs/project/bin/python -u $SH/layer_ridge.py \
  $SH/layers_train.npz $SH/layers_test.npz \
  $SH/data/features_corrected_merged/train-100.csv \
  $SH/data/features_corrected_merged/test.csv \
  $SH/layer_sweep.json
