#!/bin/bash
# D2: end-to-end positive control. Borrows the user's node via --overlap; never scancels.
SH=/ocean/projects/cis260125p/shared
JOB=43533677
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
echo "=== D2 white-box control: train + capture through the SAME extractor ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/whitebox_control.py \
  --oracle $SH/oracle_maps_test.npz --test_dir $SH/data/processed_corrected/test \
  --out $SH/whitebox_f0sd.npz --train_n 1200 --epochs 6 || exit 1
echo "=== scoring the white-box map on the SAME clean panel ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/score_whitebox.py
echo "WHITEBOX_DONE"
