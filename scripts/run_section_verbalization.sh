#!/usr/bin/env bash
# Section-tagged verbalization launcher for the 3-way distributed run.
#
# Usage:
#   bash scripts/run_section_verbalization.sh <PART>
# where <PART> is 1, 2, or 3 — one per teammate.
#
# Each PART covers a disjoint third of train-100 + dev + test:
#   PART 1: train-100[0:4634]    dev[0:1000]      test[0:1000]
#   PART 2: train-100[4634:9268] dev[1000:2000]   test[1000:2000]
#   PART 3: train-100[9268:...]  dev[2000:3000]   test[2000:3000]
#
# Launches 4 background python processes per part: 2 train sub-shards
# (2317 clips each) + 1 dev shard + 1 test shard. Output filenames are
# part-namespaced so the three teammates never collide in $SHARED/data/verbalized_v2.
#
# PREREQUISITES (run the one-time setup block first, then warm up Ollama):
#   1. conda env active, $SHARED exported, cwd = repo root, git pulled
#   2. Ollama running on this node
#   3. gemma4:e2b WARMED UP — the model is ~7 GB and loads slowly from
#      shared storage; cold first requests time out. Warm it with:
#        until curl -s http://localhost:11434/api/generate -d \
#          '{"model":"gemma4:e2b","prompt":"hi","stream":false,"keep_alive":"6h"}' \
#          2>/dev/null | grep -q '"response"'; do echo "  loading..."; sleep 15; done
set -euo pipefail

PART="${1:?usage: bash scripts/run_section_verbalization.sh <PART 1|2|3>}"
case "$PART" in
  1|2|3) : ;;
  *) echo "ERROR: PART must be 1, 2, or 3 (got: $PART)" >&2; exit 1 ;;
esac
: "${SHARED:?ERROR: \$SHARED not set — run the one-time setup block first}"

p=$(( PART - 1 ))
TRAIN_CHUNK=4634
DEVTEST_CHUNK=1000
HALF=$(( TRAIN_CHUNK / 2 ))

train_off=$(( p * TRAIN_CHUNK ))
devtest_off=$(( p * DEVTEST_CHUNK ))

OUT_DIR="$SHARED/data/verbalized_v2"
mkdir -p "$OUT_DIR"

FEATURES_DIR="$SHARED/data/features_pyannote"

echo "=== Part $PART verbalization launch ==="
echo "  train-100: offset $train_off, 2 sub-shards of $HALF clips"
echo "  dev:       offset $devtest_off, limit $DEVTEST_CHUNK"
echo "  test:      offset $devtest_off, limit $DEVTEST_CHUNK"
echo "  output:    $OUT_DIR/*.part${PART}.*.csv"
echo

# Sanity-check Ollama is up and the model is warm before launching — a cold
# model load can take minutes and would time out every shard's first request.
if ! curl -s -m 10 http://localhost:11434/api/generate -d \
      '{"model":"gemma4:e2b","prompt":"hi","stream":false,"keep_alive":"6h"}' \
      2>/dev/null | grep -q '"response"'; then
  echo "ERROR: gemma4:e2b is not responding (still loading, or Ollama down)." >&2
  echo "       Warm it up first — see PREREQUISITES in this script's header." >&2
  exit 1
fi
echo "  Ollama check: gemma4:e2b is warm and responding"
echo

# ── train-100: 2 sub-shards for speed ──────────────────────────────────────
for sub in 0 1; do
  off=$(( train_off + sub * HALF ))
  log="/tmp/verb-train-p${PART}s${sub}-${USER}.log"
  nohup python -u scripts/feature_verbalization.py --section-tagged --resume --retry-errors \
    --input  "$FEATURES_DIR/train-100.csv" \
    --output "$OUT_DIR/train-100.part${PART}.sub${sub}.csv" \
    --offset "$off" --limit "$HALF" \
    > "$log" 2>&1 &
  echo "  launched train sub$sub  (offset $off, limit $HALF)  log=$log"
done

# ── dev ──────────────────────────────────────────────────────────────────
dev_log="/tmp/verb-dev-p${PART}-${USER}.log"
nohup python -u scripts/feature_verbalization.py --section-tagged --resume --retry-errors \
  --input  "$FEATURES_DIR/dev.csv" \
  --output "$OUT_DIR/dev.part${PART}.csv" \
  --offset "$devtest_off" --limit "$DEVTEST_CHUNK" \
  > "$dev_log" 2>&1 &
echo "  launched dev            (offset $devtest_off, limit $DEVTEST_CHUNK)  log=$dev_log"

# ── test ─────────────────────────────────────────────────────────────────
test_log="/tmp/verb-test-p${PART}-${USER}.log"
nohup python -u scripts/feature_verbalization.py --section-tagged --resume --retry-errors \
  --input  "$FEATURES_DIR/test.csv" \
  --output "$OUT_DIR/test.part${PART}.csv" \
  --offset "$devtest_off" --limit "$DEVTEST_CHUNK" \
  > "$test_log" 2>&1 &
echo "  launched test           (offset $devtest_off, limit $DEVTEST_CHUNK)  log=$test_log"

echo
echo "Part $PART launched — 4 background processes."
echo "  progress: for f in $OUT_DIR/*part${PART}*.csv; do echo \"\$f \$(( \$(wc -l < \"\$f\") - 1 )) rows\"; done"
echo "  procs:    pgrep -af 'feature_verbalization.*part${PART}'"
