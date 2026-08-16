#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 5:00:00
#SBATCH -A cis260125p
#SBATCH -J snrslope
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
$SH/envs/project/bin/python -u scripts/snr_gain_slopes.py \
  --checkpoint $SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt \
  --mix_dir  $SH/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --wham_dir $SH/data/wham_noise/tr \
  --rir_glob "$SH/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*.wav" \
  --n 200 --out $SH/snr_gain_slopes_v2.json || exit 1
echo "SNRSLOPE_DONE"
