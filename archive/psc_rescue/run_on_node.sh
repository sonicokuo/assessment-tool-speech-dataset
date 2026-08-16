#!/bin/bash
# Wait for the interactive allocation, then run BOTH analyses inside it via --overlap.
SH=/ocean/projects/cis260125p/shared
JOB=$1
for i in $(seq 1 480); do
  st=$(squeue -j $JOB -h -o "%T" 2>/dev/null | tr -d " ")
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "ALLOCATION_LOST"; exit 1; }
  sleep 30
done
echo "NODE_READY $(squeue -j $JOB -h -o %N)"
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
echo "=== [1/2] trivial-information nulls ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/trivial_nulls.py
echo "=== [2/2] snr gain slopes (fixed normalisation) ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/snr_gain_slopes.py \
  --checkpoint $SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt \
  --mix_dir  $SH/data/Libri2Mix/Libri2Mix/wav16k/min/test/mix_clean \
  --wham_dir $SH/data/wham_noise/tr \
  --rir_glob "$SH/data/rirs/RIRS_NOISES/real_rirs_isotropic_noises/*.wav" \
  --n 200 --out $SH/snr_gain_slopes_v2.json
echo "BOTH_ANALYSES_DONE"
