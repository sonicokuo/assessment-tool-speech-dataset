#!/bin/bash
# Show verbalization progress for one PART. Run from the repo root in your
# interact shell on the compute node. Reads the output CSVs on $SHARED
# (persistent — survives /tmp wipes and job expirations), so the progress
# section works from any node. The worker and Ollama checks only see the
# host this script runs on.
#
# Usage:  bash scripts/check_progress.sh [PART]    (default PART=1)
PART="${1:-1}"
OUT_DIR=/ocean/projects/cis260125p/shared/data/verbalized_v2

echo "== where =="
echo "  host: $(hostname -s)"
echo "  job:  ${SLURM_JOB_ID:-(none — looks like a login node)}"
echo

echo "== workers for part ${PART} =="
hits=$(pgrep -af 'feature_verbalization\.py --section' | grep -F "part${PART}.")
if [ -n "$hits" ]; then
  echo "$hits" | sed 's/^/  /'
  echo "  total: $(echo "$hits" | wc -l) worker(s) — expect 4 when actively running"
else
  echo "  none — verbalization is not running on this host"
fi
echo

echo "== progress (CSVs on \$SHARED) =="
shopt -s nullglob
files=( "$OUT_DIR"/*part${PART}*.csv )
if [ "${#files[@]}" -eq 0 ]; then
  echo "  no output files for part ${PART} yet"
else
  total_good=0
  total_target=0
  for f in "${files[@]}"; do
    tot=$(( $(wc -l < "$f") - 1 ))
    err=$(grep -c '\[ERROR\]' "$f")
    good=$(( tot - err ))
    case "$f" in *train*) tgt=2317 ;; *) tgt=1000 ;; esac
    total_good=$(( total_good + good ))
    total_target=$(( total_target + tgt ))
    status=""
    [ "$good" -ge "$tgt" ] && status="  <- DONE"
    printf "  %-30s %5d / %-5d good (%3d%%)%s\n" \
      "$(basename "$f")" "$good" "$tgt" "$(( good * 100 / tgt ))" "$status"
  done
  printf "  %-30s %5d / %-5d good (%3d%%)\n" \
    "TOTAL" "$total_good" "$total_target" "$(( total_good * 100 / total_target ))"
fi
echo

echo "== ollama on this host =="
if curl -s -m 3 localhost:11434/api/tags >/dev/null 2>&1; then
  echo "  up"
  loaded=$(curl -s -m 3 localhost:11434/api/ps 2>/dev/null)
  echo "  loaded models: $loaded"
else
  echo "  DOWN"
fi
