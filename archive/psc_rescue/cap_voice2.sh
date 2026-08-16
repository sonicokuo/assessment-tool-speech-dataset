#!/bin/bash
# Resume the hnr/jitter/shimmer saliency capture on the user's NEW allocation.
# Does NOT scancel: the node is user-owned, borrowed via --overlap only.
SH=/ocean/projects/cis260125p/shared
JOB=$1
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
grep -q "attr.sum(dim=-1)" scripts/capture_saliency.py || { echo "[fatal] D1 fix absent"; exit 1; }
echo "GUARD_PASSED"
for i in $(seq 1 960); do
  st=$(squeue -j $JOB -h -o "%T" 2>/dev/null | tr -d " ")
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "ALLOCATION_LOST"; exit 1; }
  sleep 30
done
echo "NODE_READY"
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py \
  --checkpoint $CK --test_dir $SH/data/processed_corrected/test \
  --features hnr,jitter,shimmer --method gradxinput --out $SH/v4_gxi_voice.npz || exit 1
echo "VOICE_SALIENCY_DONE"
