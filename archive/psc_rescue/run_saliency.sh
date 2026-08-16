#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 3:00:00
#SBATCH -A cis260125p
#SBATCH -J saliency
SH=/ocean/projects/cis260125p/shared
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
echo "=== NORMAL (oracle overlap input present) ==="
$SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK \
  --test_dir $SH/data/processed_corrected/test --out $SH/saliency_test.npz \
  --features f0_sd,f0_mean,overlap_ratio
echo "=== ZEROED OVERLAP (item 0.2 control) ==="
$SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK \
  --test_dir $SH/data/processed_corrected/test --out $SH/saliency_test_zeroovl.npz \
  --features f0_sd,f0_mean,overlap_ratio --zero_overlap
echo "SALIENCY_ALL_DONE"
