#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 4:00:00
#SBATCH -A cis260125p
#SBATCH -J oraclemaps
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
for split in test; do
  $SH/envs/project/bin/python -u scripts/build_oracle_maps.py \
    --features_csv $SH/data/features_corrected_merged/${split}.csv \
    --audio_dir    $SH/data/audio_corrected/${split} \
    --out          $SH/oracle_maps_${split}.npz \
    --rates 6.25,12.5,25,50,100
done
echo "ALL_SPLITS_DONE"
