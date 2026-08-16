#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 6:00:00
#SBATCH -A cis260125p
#SBATCH -J dose
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
$SH/envs/project/bin/python -u scripts/causal_dose.py \
  --checkpoint $SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --audio_dir $SH/data/audio_corrected/test \
  --n 150 --out $SH/causal_dose.json || exit 1
echo "DOSE_DONE"
