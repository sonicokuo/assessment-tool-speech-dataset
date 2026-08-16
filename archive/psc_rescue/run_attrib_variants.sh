#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 5:00:00
#SBATCH -A cis260125p
#SBATCH -J attribvar
SH=/ocean/projects/cis260125p/shared
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
P="$SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test --features f0_sd,f0_mean,overlap_ratio"
$P --method gradxinput --out $SH/sal_gxi.npz
$P --method gradxinput --out $SH/sal_gxi_zeroovl.npz --zero_overlap
$P --method ig --ig_steps 16 --out $SH/sal_ig.npz
$P --method ig --ig_steps 16 --out $SH/sal_ig_zeroovl.npz --zero_overlap
echo "ATTRIB_VARIANTS_DONE"
