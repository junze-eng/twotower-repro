#!/usr/bin/env bash
# Main test batch — the remaining pod runs, in one sequential pass (one model at a time).
# Avoids hand-pasting long commands (which kept getting line-split).
#
#   cd /workspace/twotower-repro && git pull && bash run_batch.sh
#
# Survive SSH disconnect (recommended):
#   nohup bash run_batch.sh > results/logs/batch.log 2>&1 &
#   tail -f results/logs/batch.log        # watch progress; Ctrl-C only stops the tail, not the run
#
# Monitor GPU from another terminal:   watch -n 2 nvidia-smi
set -uo pipefail                          # NOT -e: a failed step is logged, the batch continues
cd "$(dirname "$0")"
source setup/env.sh
export HF_HUB_OFFLINE=1                    # weights+code are local; never re-pull
mkdir -p results/logs
PY=python

run () {
  echo "===== [$(date +%H:%M:%S)] START  $*"
  if "$@"; then echo "----- [$(date +%H:%M:%S)] DONE   $1 ..."
  else echo "!!!!! [$(date +%H:%M:%S)] FAILED $* (continuing)"; fi
}

# ① block-64 collapse triangle
run $PY src/exp0_capture.py --block-size 64 --max-new 128 --steps 16 --gamma 0.8 --out results/trace_tri_b64.npz
# ② AR baseline (own speedup / quality retention)
run $PY src/run_all.py --exp ar --prompts data/gsm8k_mini.jsonl --out results/ar.jsonl --limit 15
# ③ HumanEval code-side collapse + AR code baseline
run $PY src/run_all.py --exp e3 --prompts data/humaneval_mini.jsonl --out results/he_collapse.jsonl --limit 10
run $PY src/run_all.py --exp ar --prompts data/humaneval_mini.jsonl --out results/he_ar.jsonl --limit 10
# ④ long-context needle (32K may OOM -> auto-recorded, not fatal)
run $PY src/ruler_lite.py --lengths 2048 8192 16384 32768 --out results/ruler.jsonl --ar
# ⑤ top-k MoE ablation (optional footnote)
run $PY src/ablation_topk.py --prompts data/gsm8k_mini.jsonl --out results/abl_topk.pkl --limit 10
# ⑥ before/after bug demo: BUGGY word-salad (NOFIX) vs the fixed b16 you already captured
run env TWOTOWER_NOFIX=1 $PY src/exp0_capture.py --block-size 16 --max-new 64 --steps 16 --gamma 0.8 --out results/trace_buggy_demo.npz

echo "===== ALL DONE ====="
ls -la results/*.jsonl results/*.npz results/*.pkl 2>/dev/null
