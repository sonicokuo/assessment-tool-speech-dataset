#!/bin/bash
# (a) SIGNED event capture -- v3_gxi_signed_evt.npz was produced by the OLD unsigned code,
#     so pause_count_boundary could never be scored.
# (b) IG for hnr/shimmer -- signed IG beat grad*input on f0_sd (0.1159 -> 0.1285), so it
#     may lift hnr too. Same candidate on every feature keeps the panel comparable.
SH=/ocean/projects/cis260125p/shared
JOB=43533677
CK=$SH/checkpoints/full/bsigma_attnconcat_seed73/best.pt
cd $SH/repo_verify
export HF_HOME=$SH/hf_cache HF_HUB_OFFLINE=1
grep -q "attr.sum(dim=-1)" scripts/capture_saliency.py || { echo "[fatal] D1 fix absent"; exit 1; }
echo "GUARD_PASSED"
P="srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u scripts/capture_saliency.py --checkpoint $CK --test_dir $SH/data/processed_corrected/test"
$P --features speaking_rate,pause_count,pause_rate --method gradxinput --out $SH/v4_gxi_signed_evt.npz || exit 1
$P --features hnr,shimmer --method ig --ig_steps 8 --out $SH/v4_ig_voice.npz || exit 1
echo "=== rescoring with IG + boundary pause ==="
srun --jobid=$JOB --overlap $SH/envs/project/bin/python -u $SH/score_voice.py
echo "CHAIN2_DONE"
