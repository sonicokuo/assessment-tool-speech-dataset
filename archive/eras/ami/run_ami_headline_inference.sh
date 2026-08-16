#!/usr/bin/env bash
# Run OR RESUME the AMI headline inference (v9_lora_8b_dur) as 2 concurrent shards
# on one H100, then merge. Resumable: each shard skips clips already in its
# inference_results.json, so rerunning after a session timeout finishes only the
# remainder. The token cap does not affect scored output (valid descriptions end
# at EOS ~130 tokens; mode-collapse clips score 0 regardless), so a tighter cap
# on resume is safe and faster.
#
# Usage:   bash scripts/run_ami_headline_inference.sh <SLURM_JOBID> [max_new_tokens=160]
# Example: bash scripts/run_ami_headline_inference.sh 40951996 160
set -euo pipefail
JOBID=${1:?need the SLURM jobid of your interact GPU session (squeue -u $USER)}
CAP=${2:-160}
SHARED=/ocean/projects/cis260125p/shared
PY=$SHARED/envs/project/bin/python
REPO=$SHARED/assessment-tool-speech-dataset
TEST=$SHARED/data/processed_ami/test
CKPT=$SHARED/checkpoints/v9_lora_8b_dur/best.pt
D1=$SHARED/checkpoints_ami_eval/v9_lora_8b_dur
D2=$SHARED/checkpoints_ami_eval/v9_lora_8b_dur_sh2
mkdir -p "$D1" "$D2"
ln -sf "$CKPT" "$D1/best.pt"          # symlink keeps outputs out of the real ckpt dir
ln -sf "$CKPT" "$D2/best.pt"
ENV="export PYTHONNOUSERSITE=1 HF_HOME=$SHARED/hf_cache HF_HUB_OFFLINE=1; cd $REPO;"

echo "launching 2 shards on job $JOBID (cap=$CAP) ..."
srun --jobid="$JOBID" --overlap bash -c "$ENV $PY -u src/inference.py \
  --config configs/config.psc.emnlp.ami_dur.yaml --checkpoint $D1/best.pt \
  --test_dir $TEST --start 0 --end 1827 --top_k 1 --max_new_tokens $CAP" > "$D1/inference.log" 2>&1 &
P1=$!
srun --jobid="$JOBID" --overlap bash -c "$ENV $PY -u src/inference.py \
  --config configs/config.psc.emnlp.ami_dur.yaml --checkpoint $D2/best.pt \
  --test_dir $TEST --start 1827 --end 3654 --top_k 1 --max_new_tokens $CAP" > "$D2/inference.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "both shards finished; merging ..."
$PY "$REPO/scripts/merge_shard_results.py" \
  --results "$D1/inference_results.json" "$D2/inference_results.json" \
  --output_dir "$D1" --test_dir "$TEST"
echo "== per-feature =="
$PY "$REPO/scripts/per_feature_sfs.py" --inference_results "$D1/inference_results.json" || true
echo "DONE -> $D1/inference_summary.json"
