#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 4:00:00
#SBATCH -A cis260125p
#SBATCH -J causal
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
$SH/envs/project/bin/python -u scripts/causal_remix.py \
  --checkpoint $SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --audio_dir $SH/data/audio_corrected/test \
  --n 200 --seconds 1.5 --out $SH/causal_remix.json || exit 1
echo "CAUSAL_REMIX_DONE"
