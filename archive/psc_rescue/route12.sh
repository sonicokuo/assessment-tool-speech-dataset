#!/bin/bash
# Routes 1+2: boundary pause_count variant and the VALUE-DEPENDENT voice maps, then price
# all of them against the same trivial nulls. Runs after the signed re-capture on the same
# allocation; releases the node on every exit path.
SH=/ocean/projects/cis260125p/shared
JOB=$1
release() { echo "RELEASING NODE $JOB"; scancel $JOB 2>/dev/null; echo "NODE_RELEASED"; }
trap release EXIT
for i in $(seq 1 960); do
  grep -q "V4_SIGNED_DONE" $SH/logs/v4.log 2>/dev/null && break
  grep -qE "\[fatal\]|ALLOCATION_LOST" $SH/logs/v4.log 2>/dev/null && { echo "upstream failed"; exit 1; }
  [ -z "$(squeue -j $JOB -h -o '%T' 2>/dev/null)" ] && { echo "job gone"; exit 1; }
  sleep 30
done
cd $SH/repo_verify
echo "=== ROUTE 1+2: building voice + boundary maps ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/build_oracle_maps_voice.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --clean_dir $SH/data/audio_corrected/test-s1clean \
  --out $SH/oracle_voice_test.npz
echo "=== pricing them against the trivial nulls ==="
setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/nulls_voice.py
echo "ROUTE12_DONE"
