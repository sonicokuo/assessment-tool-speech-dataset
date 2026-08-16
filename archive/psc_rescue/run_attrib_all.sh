#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 6:00:00
#SBATCH -A cis260125p
#SBATCH -J attriball
SH=/ocean/projects/cis260125p/shared
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
P="$SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test"
F="--features f0_sd,f0_mean,overlap_ratio"
# STRICT loader aborts on any state_dict mismatch, so a wrong head fails here in seconds.
$P $F --method grad        --out $SH/v2_grad.npz            || exit 1
$P $F --method grad        --out $SH/v2_grad_zero.npz       --zero_overlap || exit 1
$P $F --method gradxinput  --out $SH/v2_gxi.npz             || exit 1
$P $F --method gradxinput  --out $SH/v2_gxi_zero.npz        --zero_overlap || exit 1
# IG is 8x the backward passes; run it on a subset as a consistency check.
$P --features f0_sd,f0_mean --method ig --ig_steps 8 --limit 1500 --out $SH/v2_ig.npz      || exit 1
$P --features f0_sd,f0_mean --method ig --ig_steps 8 --limit 1500 --out $SH/v2_ig_zero.npz --zero_overlap || exit 1
echo "ATTRIB_V2_DONE"
