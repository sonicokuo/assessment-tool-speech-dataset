ROOT=/ocean/projects/cis260125p/shared/checkpoints
for r in "$@"; do
  echo "===== $r ====="
  s="$ROOT/$r/inference_summary.json"
  if [ -f "$s" ]; then
    echo "[summary present] size=$(stat -c%s "$s")"
    cat "$s"
    echo ""
  else
    echo "[NO inference_summary.json]"
  fi
  ir="$ROOT/$r/inference_results.json"
  if [ -f "$ir" ]; then echo "[inference_results.json present size=$(stat -c%s "$ir")]"; else echo "[NO inference_results.json]"; fi
  echo ""
done
