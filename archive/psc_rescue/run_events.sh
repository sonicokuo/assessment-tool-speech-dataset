#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 3:00:00
#SBATCH -A cis260125p
#SBATCH -J events
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
$SH/envs/project/bin/python -u scripts/build_oracle_maps_events.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --clean_dir $SH/data/audio_corrected/test-s1clean \
  --out $SH/oracle_events_test.npz || exit 1
echo "EVENTS_DONE"
