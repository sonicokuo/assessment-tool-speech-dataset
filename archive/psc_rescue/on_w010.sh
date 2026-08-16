#!/bin/bash
# Routes 1+2 on the user's interactive allocation. Does NOT scancel: the node is
# user-owned, so this only borrows it via --overlap.
SH=/ocean/projects/cis260125p/shared
JOB=43530605
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
echo "=== ROUTE 2 (value-dependent voice maps) + ROUTE 1 (boundary pause variant) ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/build_oracle_maps_voice.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --clean_dir $SH/data/audio_corrected/test-s1clean \
  --out $SH/oracle_voice_test.npz || { echo "[fatal] voice map build failed"; exit 1; }
echo "=== pricing all four against the trivial nulls ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/nulls_voice.py
echo "ROUTES_DONE"
