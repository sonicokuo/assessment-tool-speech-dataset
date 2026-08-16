#!/bin/bash
# Routes 1+2 in ONE chain with ONE trap. The previous attempt used two chains on the same
# allocation, each with `trap release EXIT` -- the first to finish cancelled the node while
# the second was still waiting, and srun reported "job has expired". One allocation must
# have exactly one owner.
SH=/ocean/projects/cis260125p/shared
JOB=$1
release() { echo "RELEASING NODE $JOB"; scancel $JOB 2>/dev/null; echo "NODE_RELEASED"; }
trap release EXIT

for i in $(seq 1 960); do
  st=$(squeue -j $JOB -h -o "%T" 2>/dev/null | tr -d " ")
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "ALLOCATION_LOST"; exit 1; }
  sleep 30
done
echo "NODE_READY"
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1

echo "=== ROUTE 2: value-dependent voice maps + ROUTE 1: boundary pause variant ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/build_oracle_maps_voice.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --clean_dir $SH/data/audio_corrected/test-s1clean \
  --out $SH/oracle_voice_test.npz || { echo "[fatal] voice map build failed"; exit 1; }

echo "=== pricing all four against the trivial nulls ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/nulls_voice.py
echo "ROUTES_DONE"
