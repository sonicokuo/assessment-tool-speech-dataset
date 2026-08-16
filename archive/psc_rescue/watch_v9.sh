PY=/ocean/projects/cis260125p/shared/envs/project/bin/python
RF=/ocean/projects/cis260125p/shared/checkpoints/v9_rescore_cleanf0/inference_results.json
LOG=/ocean/projects/cis260125p/shared/logs/v9_infer_full.log
while true; do
  RUNNING=0; pgrep -f config.v9_rescore.yaml >/dev/null 2>&1 && RUNNING=1
  N=0; [ -f "$RF" ] && N=$($PY -c "import json;print(len(json.load(open('$RF'))))" 2>/dev/null || echo 0)
  if [ "$N" = "3000" ]; then echo "V9_COMPLETE n=3000"; break; fi
  if [ "$RUNNING" = 0 ]; then
    if grep -qiE "Traceback|Error|CUDA out of memory|Killed" "$LOG" 2>/dev/null; then
      echo "V9_PROC_DIED_WITH_ERROR n=$N (see $LOG)"; else echo "V9_PROC_EXITED n=$N (no 3000, no obvious error)"; fi
    break
  fi
  sleep 180
done
