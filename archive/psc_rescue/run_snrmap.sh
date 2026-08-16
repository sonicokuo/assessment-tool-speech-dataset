#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 4:00:00
#SBATCH -A cis260125p
#SBATCH -J snrmap
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
$SH/envs/project/bin/python -u scripts/build_snr_map.py \
  --mix_dir  $SH/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --live_dir $SH/data/audio_corrected/test \
  --wham_dir $SH/data/wham_noise/tr \
  --rir_glob "$SH/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*.wav" \
  --out $SH/snr_maps_test.npz || exit 1
echo "SNRMAP_DONE"
