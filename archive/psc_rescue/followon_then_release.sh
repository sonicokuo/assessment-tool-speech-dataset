#!/bin/bash
# Waits for the first two analyses, runs the D1 re-capture on the SAME allocation,
# then ALWAYS releases the node. The node must never sit idle holding a GPU.
SH=/ocean/projects/cis260125p/shared
JOB=$1
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
release() { echo "RELEASING NODE $JOB"; scancel $JOB 2>/dev/null; echo "NODE_RELEASED"; }
trap release EXIT                      # release on ANY exit path, including error

for i in $(seq 1 960); do
  grep -q "BOTH_ANALYSES_DONE" $SH/logs/on_node.log 2>/dev/null && break
  grep -q "ALLOCATION_LOST" $SH/logs/on_node.log 2>/dev/null && { echo "upstream lost"; exit 1; }
  [ -z "$(squeue -j $JOB -h -o %T 2>/dev/null)" ] && { echo "job gone"; exit 1; }
  sleep 30
done

cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
echo "=== [3/3] D1 re-capture: SIGNED reduction for ig / gradxinput ==="
P="setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test"
$P --features f0_sd,f0_mean,overlap_ratio --method gradxinput --out $SH/v3_gxi_signed.npz
$P --features speaking_rate,pause_count,pause_rate --method gradxinput --out $SH/v3_gxi_signed_evt.npz
$P --features f0_sd,f0_mean --method ig --ig_steps 8 --limit 1500 --out $SH/v3_ig_signed.npz
echo "D1_RECAPTURE_DONE"
