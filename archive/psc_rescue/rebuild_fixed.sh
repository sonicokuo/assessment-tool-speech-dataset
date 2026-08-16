#!/bin/bash
# Rebuild canonical descriptions with the 2026-07-28 fixes (hnr/shimmer column names +
# clean-by-construction f0 fallback). Replicates the original _fw invocation exactly:
# no --clean_features, default --overlap_threshold 0.5. Writes to NEW filenames.
set -e
SH=/ocean/projects/cis260125p/shared
RV=$SH/repo_verify
PY=$SH/envs/project/bin/python
cd $RV
for split in train-100:train dev:dev test:test; do
  csv="${split%%:*}"; f0="${split##*:}"
  echo "=== $csv ==="
  $PY -u scripts/build_canonical_descriptions.py \
    --features_csv $SH/data/features_corrected_merged/${csv}.csv \
    --clean_f0     $SH/data/clean_f0_${f0}.json \
    --output       $SH/data/descriptions_fixed_${f0}.json
done
echo "=== MERGE ==="
$PY - <<'PY'
import json
SH="/ocean/projects/cis260125p/shared"
out={}
for s in ("train","dev","test"):
    d=json.load(open(f"{SH}/data/descriptions_fixed_{s}.json"))
    dup=set(out)&set(d)
    if dup: raise SystemExit(f"FATAL: {len(dup)} duplicate keys when merging {s}")
    out.update(d); print(f"  {s}: {len(d)}")
json.dump(out, open(f"{SH}/data/descriptions_corrected_fw2.json","w"))
print("merged total:", len(out))
old=json.load(open(f"{SH}/data/descriptions_corrected_fw.json"))
print("old file total:", len(old), "| key sets identical:", set(old)==set(out))
PY
