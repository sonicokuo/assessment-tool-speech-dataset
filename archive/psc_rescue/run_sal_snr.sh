#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 4:00:00
#SBATCH -A cis260125p
#SBATCH -J salsnr
SH=/ocean/projects/cis260125p/shared
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
P="$SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test --features snr"
$P --method grad       --out $SH/v2_grad_snr.npz            || exit 1
$P --method gradxinput --out $SH/v2_gxi_snr.npz             || exit 1
$P --method grad       --out $SH/v2_grad_snr_zero.npz --zero_overlap || exit 1
echo "SAL_SNR_DONE"
