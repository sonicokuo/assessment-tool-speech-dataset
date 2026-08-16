#!/bin/bash
# === FINAL v9-vs-v17 trustworthy eval ===
# Run AFTER v9 inference reaches 3000 clips (and ideally v17 too).
# Resumes are automatic; just rerun run_v9_fg.sh if the inference step died.
set -e
SH=/ocean/projects/cis260125p/shared
PY=$SH/envs/project/bin/python
$PY $SH/eval_v9_v17.py \
  --v9  $SH/checkpoints/v9_rescore_cleanf0/inference_results.json \
  --v17 $SH/checkpoints/v17_decoupled/inference_results.json \
  --out $SH/checkpoints/v9_rescore_cleanf0/trustworthy_v9_vs_v17.json \
  --bootstrap 2000 --seed 0
