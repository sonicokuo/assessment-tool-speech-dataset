#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 6:00:00
#SBATCH -A cis260125p
#SBATCH -J dosectrl
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
B="$SH/envs/project/bin/python -u scripts/causal_dose.py --checkpoint $SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt --features_csv $SH/data/features_corrected_merged/test.csv --audio_dir $SH/data/audio_corrected/test --n 150"
# HELD: overlap channel frozen at alpha=0 -> only the AUDIO changes
$B --hold_input --out $SH/causal_dose_held.json || exit 1
# ZEROED: no overlap information at all -> the audio-only condition
$B --zero_input --out $SH/causal_dose_zero.json || exit 1
echo "DOSE_CTRL_DONE"
