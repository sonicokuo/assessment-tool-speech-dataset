#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 3:00:00
#SBATCH -A cis260125p
#SBATCH -J routes
SH=/ocean/projects/cis260125p/shared
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
# Guard: the file this job will execute must contain the value-dependent map builder.
grep -q "hnr deviation map zero-sum" scripts/build_oracle_maps_voice.py || {
  echo "[fatal] build_oracle_maps_voice.py missing or stale on PSC"; exit 1; }
echo "GUARD_PASSED"
echo "=== ROUTE 2 (voice maps) + ROUTE 1 (boundary pause variant) ==="
$SH/envs/project/bin/python -u scripts/build_oracle_maps_voice.py \
  --features_csv $SH/data/features_corrected_merged/test.csv \
  --clean_dir $SH/data/audio_corrected/test-s1clean \
  --out $SH/oracle_voice_test.npz || exit 1
echo "=== pricing against the trivial nulls ==="
$SH/envs/project/bin/python -u $SH/nulls_voice.py || exit 1
echo "ROUTES_DONE"
