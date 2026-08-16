#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 2:00:00
#SBATCH -A cis260125p
#SBATCH -J snrpanel
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
$SH/envs/project/bin/python -u scripts/snr_triviality_panel.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --audio_dir $SH/data/audio_corrected/test --n 3000 || exit 1
echo "SNRPANEL_DONE"
