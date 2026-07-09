#!/usr/bin/env bash
# Live dashboard for the run_everything.sh batch.
# Use a refreshing view:   watch -n 30 bash monitor.sh
# Or one-shot:             bash monitor.sh
cd "$(dirname "$0")"

echo "======== TwoTower run monitor  $(date '+%Y-%m-%d %H:%M:%S') ========"
echo ""
echo "-- progress (START / DONE / FAILED) --"
grep -E "START|DONE|FAILED|ALL DONE" results/run.log 2>/dev/null | tail -22 || echo "  (no results/run.log yet)"
echo ""
echo "-- is the batch still running? --"
if pgrep -f run_everything.sh >/dev/null; then echo "  YES (run_everything.sh alive)"; else echo "  NO (finished or not started)"; fi
echo ""
echo "-- outputs produced --"
ls -1sh results/*.jsonl results/*.npz results/*.pkl 2>/dev/null || echo "  (none yet)"
echo ""
echo "-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo ""
echo "-- live tail: most recent experiment log --"
newest=$(ls -t results/logs/*.log 2>/dev/null | head -1)
if [ -n "$newest" ]; then echo "  [$(basename "$newest")]"; tail -8 "$newest" | sed 's/^/  /'; else echo "  (no experiment logs yet)"; fi
