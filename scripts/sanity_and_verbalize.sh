#!/usr/bin/env bash
# Full sanity-check + verbalization launch for one teammate's PART.
#
# Usage:  bash scripts/sanity_and_verbalize.sh <PART 1|2|3>
#
# Assumes the one-time setup is already done in this shell:
#   conda activate /ocean/projects/cis260125p/shared/envs/project
#   export PYTHONNOUSERSITE=1
#   export SHARED=/ocean/projects/cis260125p/shared
#   cd $SHARED/assessment-tool-speech-dataset
#   git pull origin main
#
# This script then:
#   1. Kills any stale ollama processes — serve AND runner orphans (the
#      narrow `pkill -f 'ollama serve'` leaves runner subprocesses alive,
#      which wedge the GPU; this uses `pkill -9 -f ollama`).
#   2. Starts a fresh `ollama serve` (OLLAMA_NUM_PARALLEL=2).
#   3. Warm-up sanity check: loads gemma4:e2b, watches VRAM climb, 180s timeout.
#      PASS only if a real response comes back AND VRAM moved.
#   4. If sanity PASSES → launches the 4-process verbalization for this PART.
#      If sanity FAILS → exits with a diagnosis (bad node vs just-slow), so
#      you never launch shards into guaranteed timeouts.
set -uo pipefail

PART="${1:?usage: bash scripts/sanity_and_verbalize.sh <PART 1|2|3>}"
case "$PART" in 1|2|3) : ;; *) echo "ERROR: PART must be 1/2/3 (got $PART)" >&2; exit 1 ;; esac
: "${SHARED:?ERROR: \$SHARED not set — run the one-time setup block first}"

OLLAMA_LOG="/tmp/ollama-${USER}.log"

echo "==================================================================="
echo " Sanity + verbalization launch  —  Part $PART"
echo "==================================================================="

# ── Step 1: kill stale ollama (serve + runner orphans) ──────────────────
echo
echo "[1/4] Killing stale ollama processes (serve + runner orphans)..."
pkill -9 -f ollama 2>/dev/null || true
sleep 3
if pgrep -f ollama > /dev/null; then
  echo "  still alive after first kill — retrying:"
  pgrep -af ollama
  pkill -9 -f ollama 2>/dev/null || true
  sleep 3
fi
n_ollama=$(pgrep -fc ollama 2>/dev/null || echo 0)
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "  ollama processes remaining: ${n_ollama}  (want 0)"
echo "  GPU memory in use: ${gpu_used} MiB  (want near 0)"
if [ "${gpu_used}" -gt 2000 ] 2>/dev/null; then
  echo "  WARNING: GPU still has >2 GiB used by someone. On -p GPU-shared that"
  echo "           may be another user. The load may contend; proceed with care."
fi

# ── Step 2: start fresh ollama serve ────────────────────────────────────
echo
echo "[2/4] Starting fresh ollama serve..."
export PATH="$SHARED/ollama/bin:$PATH"
export LD_LIBRARY_PATH="$SHARED/ollama/lib:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="$SHARED/ollama_models"
export OLLAMA_NUM_PARALLEL=2
nohup ollama serve > "$OLLAMA_LOG" 2>&1 &
sleep 5
if ! curl -s -m 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  ERROR: ollama serve did not come up. Last log lines:" >&2
  tail -15 "$OLLAMA_LOG" >&2
  exit 1
fi
echo "  ollama serve is up (api/tags responds)"

# ── Step 3: warm-up sanity check with VRAM monitoring + timeout ─────────
echo
echo "[3/4] Warm-up: loading gemma4:e2b (180s timeout). VRAM should climb"
echo "      from ~0 toward ~9000 MiB — if it stays flat, the runner can't"
echo "      reach the GPU and this node is bad."
( for _ in $(seq 1 45); do
    printf "    %s  VRAM=%s MiB\n" "$(date +%H:%M:%S)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    sleep 4
  done ) &
WATCHER=$!

resp=$(curl -s --max-time 180 http://localhost:11434/api/generate \
  -d '{"model":"gemma4:e2b","prompt":"hi","stream":false,"keep_alive":"6h"}' 2>/dev/null || true)
kill "$WATCHER" 2>/dev/null || true
wait "$WATCHER" 2>/dev/null || true

gpu_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if echo "$resp" | grep -q '"response"'; then
  echo "  SANITY PASS — gemma4:e2b loaded and responded. GPU now: ${gpu_after} MiB"
else
  echo >&2
  echo "  SANITY FAIL — gemma4:e2b did not respond within 180s." >&2
  echo "  response: ${resp:0:200}" >&2
  echo "  GPU memory now: ${gpu_after} MiB" >&2
  echo >&2
  if [ "${gpu_after}" -lt 1000 ] 2>/dev/null; then
    echo "  DIAGNOSIS: VRAM never climbed — the llama runner can't init CUDA on" >&2
    echo "  this node. Restarting ollama will NOT help. Run 'exit' to drop this" >&2
    echo "  interact session and reallocate a fresh node, then re-run this script." >&2
  else
    echo "  DIAGNOSIS: VRAM climbed but the load didn't finish in 180s — likely" >&2
    echo "  just slow. The model file is now warm; re-run this script and it" >&2
    echo "  should load faster the second time." >&2
  fi
  echo "  Last ollama log lines:" >&2
  tail -15 "$OLLAMA_LOG" >&2
  exit 1
fi

# ── Step 4: launch verbalization for this PART ─────────────────────────
echo
echo "[4/4] Launching verbalization for Part $PART..."
bash scripts/run_section_verbalization.sh "$PART"

echo
echo "==================================================================="
echo " Done. Verbalization running in the background for Part $PART."
echo
echo " Progress:"
echo "   for f in \$SHARED/data/verbalized_v2/*part${PART}*.csv; do \\"
echo "     echo \"\$(basename \$f) \$(( \$(wc -l < \"\$f\") - 1 )) rows\"; done"
echo
echo " Processes:"
echo "   pgrep -af 'feature_verbalization.*part${PART}'"
echo "==================================================================="
