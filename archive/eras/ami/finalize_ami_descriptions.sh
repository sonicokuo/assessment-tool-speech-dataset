#!/usr/bin/env bash
# After feature extraction completes, finalize the AMI test-set GT:
#   1) oracle overlap GT (*_vad columns) from AMI manual annotations
#   2) header-only train-100/dev stubs so build_descriptions --all runs test-only
#   3) descriptions in BOTH formats:
#        - nodur  (descriptions_ami.json)     → for the v9 adapter sweep
#        - withdur(descriptions_ami_dur.json) → for the v9_lora_8b_dur headline
# CPU only, ~seconds. Idempotent.
set -euo pipefail
SHARED=/ocean/projects/cis260125p/shared
REPO=$SHARED/assessment-tool-speech-dataset
PY=$SHARED/envs/project/bin/python
FEAT=$SHARED/data/features_ami

echo "== [1/3] oracle overlap GT =="
$PY "$REPO/scripts/compute_ami_overlap_gt.py" \
  --features_csv "$FEAT/test.csv" \
  --manifest     "$SHARED/data/ami_sdm/manifest_test.csv" \
  --segments_csv "$SHARED/data/ami_sdm/segments_test.csv"

echo "== [2/3] build_descriptions stubs =="
hdr=$(head -1 "$FEAT/test.csv")
echo "$hdr" > "$FEAT/train-100.csv"
echo "$hdr" > "$FEAT/dev.csv"

echo "== [3/3] descriptions (nodur + withdur) =="
$PY "$REPO/scripts/build_descriptions_deterministic.py" --all --untagged \
  --no-overlap-segments --no-duration \
  --features-dir "$FEAT" --output "$SHARED/data/descriptions_ami.json"
$PY "$REPO/scripts/build_descriptions_deterministic.py" --all --untagged \
  --no-overlap-segments \
  --features-dir "$FEAT" --output "$SHARED/data/descriptions_ami_dur.json"

echo "== DONE =="
echo "  nodur  : $SHARED/data/descriptions_ami.json"
echo "  withdur: $SHARED/data/descriptions_ami_dur.json"
