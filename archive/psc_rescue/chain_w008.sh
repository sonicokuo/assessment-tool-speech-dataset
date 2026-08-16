#!/bin/bash
# Capture the missing voice saliency, then score all clean features. Borrows the user's
# allocation via --overlap; never scancels it.
SH=/ocean/projects/cis260125p/shared
JOB=43533677
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
grep -q "attr.sum(dim=-1)" scripts/capture_saliency.py || { echo "[fatal] D1 fix absent"; exit 1; }
echo "GUARD_PASSED"
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py \
  --checkpoint $CK --test_dir $SH/data/processed_corrected/test \
  --features hnr,jitter,shimmer --method gradxinput --out $SH/v4_gxi_voice.npz || exit 1
echo "=== SCORING the clean panel ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/score_voice.py
echo "CHAIN_W008_DONE"
