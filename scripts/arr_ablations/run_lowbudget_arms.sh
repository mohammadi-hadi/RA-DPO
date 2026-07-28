#!/usr/bin/env bash
#
# Low-budget selection arms: does R_train's ranking carry any signal?
#
# Every matched control so far ran at a budget where the task is at or near
# its full-data ceiling, so it could not separate two rules. This runs the
# 554-pair budget, the only one with real headroom, on both open-weight
# backbones:
#
#   smart10      (already trained) unanimous, LOWEST model uncertainty
#   flip10       unanimous, HIGHEST model uncertainty
#   randunan10   unanimous, random
#   random10     random from the whole pool
#
#   randunan10 vs random10  -> agreement filter
#   smart10/flip10 vs randunan10 -> uncertainty ranking, either direction
#
#   nohup bash scripts/arr_ablations/run_lowbudget_arms.sh > /dev/null 2>&1 &
#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/results/local_pipeline/stratified_arms.log"
ARMS=(strat10_dpo noworst10_dpo)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "================ low-budget arms start ================"
for cfg in configs/local_pipeline.yaml configs/local_pipeline_qwen.yaml; do
  export LOCAL_PIPELINE_CONFIG="$cfg"
  bb=$("$PY" -c "import sys;sys.path.insert(0,'.');from scripts.local_pipeline import shortname;print(shortname())")
  say "=== backbone $bb ==="
  for v in "${ARMS[@]}"; do
    pi="results/local_pipeline/per_instance/${bb}_${v}_local.json"
    if [ -f "$ROOT/$pi" ]; then say "  SKIP $v (per-instance file exists)"; continue; fi

    adapter="results/local_pipeline/training/${bb}/${v}/adapter_model.safetensors"
    if [ -f "$ROOT/$adapter" ]; then
      say "  SKIP train $v (adapter complete)"
    else
      say "  train $v"
      t0=$SECONDS
      "$PY" scripts/local_pipeline/stage_05_dpo_variants.py --only "$v" >>"$LOG" 2>&1 \
        && say "    trained in $((SECONDS-t0))s" || { say "    TRAIN FAILED"; continue; }
    fi

    say "  eval $v"
    "$PY" scripts/local_pipeline/stage_06_eval_oof.py --variants "$v" >>"$LOG" 2>&1
    # Verify the artefact, not the exit code: a previous runner reported OK
    # right after eval had crashed.
    if [ -f "$ROOT/$pi" ]; then say "    eval OK"; else say "    EVAL FAILED (no $pi)"; fi
  done
  "$PY" scripts/local_pipeline/stage_07_unified_tables.py >>"$LOG" 2>&1 && say "  tables refreshed"
done
unset LOCAL_PIPELINE_CONFIG

say ">>> statistics"
for bb in llama32_3b qwen25_3b; do
  "$PY" scripts/arr_ablations/reliability_stats.py \
    --dir results/local_pipeline/per_instance --glob "${bb}_*_local.json" \
    --out "results/local_pipeline/unified/${bb}/stats.json" >>"$LOG" 2>&1
done

say "================ low-budget arms done ================"
