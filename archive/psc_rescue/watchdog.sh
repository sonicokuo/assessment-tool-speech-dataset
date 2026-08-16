#!/bin/bash
# Wake the operator the moment COMPUTE GOES TO WASTE, not on a fixed schedule.
# Trips on: an allocation whose GPU is idle (job finished / walled / died silently),
# a traceback in any active log, or the job list shrinking. Otherwise stays quiet.
SH=/ocean/projects/cis260125p/shared
GRACE=3          # consecutive idle polls before reporting (rides out load/eval gaps)
declare -A idle
for i in $(seq 1 96); do            # 96 * 5min = 8h, one full allocation
  sleep 300
  JOBS=$(squeue -u slin32 -h -o '%i %N' 2>/dev/null)
  [ -z "$JOBS" ] && { echo "ALL_JOBS_GONE at poll $i ($(date '+%H:%M'))"; exit 0; }
  while read -r J N; do
    [ -z "$J" ] && continue
    M=$(srun --jobid=$J --overlap nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | grep -oE '^[0-9]+')
    if [ "${M:-0}" -lt 2000 ]; then
      idle[$J]=$(( ${idle[$J]:-0} + 1 ))
      if [ "${idle[$J]}" -ge "$GRACE" ]; then
        echo "IDLE_GPU job=$J node=$N mem=${M}MiB for $((GRACE*5))min at $(date '+%H:%M') -- RECLAIM"
        squeue -u slin32 -o '%i %T %N %L'
        exit 0
      fi
    else
      idle[$J]=0
    fi
  done <<< "$JOBS"
  # any fresh traceback in logs touched in the last 10 min
  ERR=$(find $SH/logs -mmin -10 \( -name 'train_*.log' -o -name 'w00*.log' \) 2>/dev/null | while read -r f; do
          grep -qE 'Traceback|CUDA out of memory' "$f" 2>/dev/null && echo "$f"; done)
  if [ -n "$ERR" ]; then echo "ERROR_IN_LOG at $(date '+%H:%M'): $ERR"; exit 1; fi
done
echo "WATCHDOG_8H_ELAPSED all jobs still busy"
