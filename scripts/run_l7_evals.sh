#!/bin/bash
# Test eval for the AUDIO-ONLY + LAYER-7 arms (seeds 73, 42).
#
# WHY THIS IS THE PACING ITEM
# Both retrains finished 2026-08-08 (4/4 epochs) but NEITHER was ever test-evaluated, so
# every headline in the paper still comes from the ORACLE-OVERLAP-CONDITIONED arm that the
# 2026-08-07 decision retired. Val says the retrain did not cost accuracy (0.7451/0.7622 free
# decode vs the record arm's 0.7194/0.7230), but val is FREE DECODE and the headline is
# VERIFIED-SLOT -- different readouts, not two splits. Only this eval settles it.
#
# ONE owning chain per allocation. The two-chains-racing-one-node bug (both carrying
# `trap release EXIT`, first to finish killed the node) cost a full run. There is deliberately
# NO trap here: this is the USER'S interactive node and must never be auto-released.
set -u
SH=/ocean/projects/cis260125p/shared
RV=$SH/repo_verify
PY=$SH/envs/project/bin/python
export HF_HOME="$SH/hf_cache" HF_HUB_OFFLINE=1
export WANDB_DATA_DIR="$SH/wandb_data" WANDB_CACHE_DIR="$SH/wandb_cache"
cd "$RV" || exit 1
GT=$SH/data/descriptions_corrected_fw2.json

for V in l7audioonly_attnconcat_seed73 l7audioonly_attnconcat_seed42; do
  SEED=${V##*seed}
  CFG=$RV/configs/config.l7audioonly.s${SEED}.fw2.yaml
  CK=$SH/checkpoints/full/$V/best.pt
  echo "=== MARK:START $V $(date) cfg=$CFG ==="
  if [ ! -f "$CFG" ]; then echo "=== MARK:FAIL missing cfg $CFG ==="; continue; fi
  if [ ! -f "$CK" ];  then echo "=== MARK:FAIL missing ckpt $CK ==="; continue; fi
  # test_dir is LAYER-7 features: these arms were trained on processed_layer7, and feeding
  # layer-24 .pt files would silently score a distribution the model never saw.
  "$PY" -u src/inference.py --config "$CFG" --checkpoint "$CK" \
        --test_dir "$SH/data/processed_layer7/test" --top_k 1
  echo "=== MARK:EVALDONE $V $(date) ==="
  "$PY" "$RV/scripts/score_matched_test.py" "$GT" \
        "$SH/checkpoints/full/$V/inference_results.json"
  echo "=== MARK:SCORED $V $(date) ==="
done
echo "=== MARK:ALLDONE $(date) ==="
