#!/bin/bash
# Re-capture with the D1 SIGNED reduction. GUARD FIRST: refuse to run unless the fix is
# actually present in the file this job will execute. The previous attempt silently ran the
# OLD code (the local fix was never scp'd) and produced "signed" outputs identical to the
# unsigned ones to four decimals.
SH=/ocean/projects/cis260125p/shared
JOB=$1
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
release() { echo "RELEASING NODE $JOB"; scancel $JOB 2>/dev/null; echo "NODE_RELEASED"; }
trap release EXIT

if [ "$(grep -c 'attr.sum(dim=-1)' $SH/repo_verify/scripts/capture_saliency.py)" -lt 1 ]; then
  echo "[fatal] D1 fix ABSENT from capture_saliency.py on PSC - refusing to run"; exit 1
fi
echo "GUARD_PASSED signed reduction present"

for i in $(seq 1 960); do
  st=$(squeue -j $JOB -h -o "%T" 2>/dev/null | tr -d " ")
  [ "$st" = "RUNNING" ] && break
  [ -z "$st" ] && { echo "ALLOCATION_LOST"; exit 1; }
  sleep 30
done
echo "NODE_READY"
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
P="setsid srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test"
$P --features f0_sd,f0_mean,overlap_ratio --method gradxinput --out $SH/v4_gxi_signed.npz
$P --features f0_sd,f0_mean --method ig --ig_steps 8 --limit 1500 --out $SH/v4_ig_signed.npz
echo "V4_SIGNED_DONE"
