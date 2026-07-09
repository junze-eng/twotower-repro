#!/usr/bin/env bash
# Run ALL experiments unattended (one full day is fine). Each step is ISOLATED: a failure is
# logged and the run continues, so one broken experiment can't abort the overnight batch.
#
# Launch (survives SSH disconnect; keep the pod RUNNING until done — stopping the pod kills it):
#   cd /workspace/twotower-repro
#   export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
#   nohup bash run_everything.sh > results/run.log 2>&1 &
#   tail -f results/run.log
#
# Per-experiment logs land in results/logs/<name>.log ; all outputs in results/.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p results/logs

PY=/usr/local/bin/python
PROMPTS=data/gsm8k_mini.jsonl
SPEED=data/speed_prompts.jsonl
T0=$(date +%s)

run () {                       # run <name> <args...>
  local name="$1"; shift
  local s=$(date +%s)
  echo "===== [$(date +%H:%M:%S)] START $name  |  $*"
  if env "$@" >"results/logs/$name.log" 2>&1; then
    echo "----- [$(date +%H:%M:%S)] DONE  $name  ($(($(date +%s)-s))s)"
  else
    echo "!!!!! [$(date +%H:%M:%S)] FAILED $name ($(($(date +%s)-s))s) -> results/logs/$name.log"
  fi
}

# 0) prompts (offline-safe embedded set if HF datasets unavailable)
$PY src/prep_data.py --which gsm8k --n 15 --out "$PROMPTS" || true

echo "########## CHEAP / HIGH-VALUE FIRST ##########"
# before/after: capture the BUGGY trace too (for the pre's before-vs-after visual)
run exp0_buggy_trace  TWOTOWER_NOFIX=1 $PY src/exp0_capture.py --out results/trace_buggy.npz
run exp0_capture      $PY src/exp0_capture.py --out results/trace.npz
run abl_remask        $PY src/ablation_remask.py    --prompts "$PROMPTS" --out results/abl_remask.jsonl   --limit 10
run collapse_e3       $PY src/run_all.py --exp e3    --prompts "$PROMPTS" --out results/e3.jsonl          --limit 10
run abl_denoiser      $PY src/ablation_denoiser.py  --prompts "$PROMPTS" --out results/abl_denoiser.jsonl --limit 10
run abl_topk          $PY src/ablation_topk.py      --prompts "$PROMPTS" --out results/abl_topk.pkl       --limit 10
run main_run_ABD      $PY src/main_run.py           --prompts "$PROMPTS" --out results/trace_main.pkl     --limit 3
run main_run_moe_E    $PY src/main_run.py --moe-hook --prompts "$PROMPTS" --out results/trace_moe.pkl     --limit 3

echo "########## EXPENSIVE (overnight) ##########"
run ar_baseline       $PY src/run_all.py --exp ar    --prompts "$PROMPTS" --out results/ar.jsonl          --limit 15
run pareto_e2         $PY src/run_all.py --exp e2    --prompts "$PROMPTS" --out results/e2.jsonl          --limit 15
run speed_e1          $PY src/run_all.py --exp e1    --prompts "$SPEED"   --out results/e1.jsonl          --limit 10

echo ""
echo "########## ALL DONE in $(( ($(date +%s)-T0)/60 )) min ##########"
echo "results:"; ls -la results/
echo "logs:";    ls -la results/logs/
echo "Download results/ to your Mac, then score+plot offline. Stop the pod to save money."
