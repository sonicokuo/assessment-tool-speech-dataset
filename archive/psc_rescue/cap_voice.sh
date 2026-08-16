#!/bin/bash
# Saliency for the newly-clean features. hnr/jitter/shimmer were never captured.
# SIGNED reduction (D1) because hnr and shimmer have deviation (sign-varying) references.
SH=/ocean/projects/cis260125p/shared
JOB=43530605
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
grep -q "attr.sum(dim=-1)" scripts/capture_saliency.py || { echo "[fatal] D1 fix absent"; exit 1; }
echo "GUARD_PASSED"
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py \
  --checkpoint $CK --test_dir $SH/data/processed_corrected/test \
  --features hnr,jitter,shimmer --method gradxinput --out $SH/v4_gxi_voice.npz || exit 1
echo "VOICE_SALIENCY_DONE"
