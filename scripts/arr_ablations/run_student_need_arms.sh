#!/usr/bin/env bash
#
# Student-need selection arms: reliable label x base-model-wrong ranking.
#
# Per backbone: (1) base-model inference over the 5,536-pair train pool,
# (2) build hard10/hardctl10/hardall10 (554 pairs each), (3) train + eval.
# Arms are per-backbone because the base model's errors differ per student.
#
# EXIST is the development bed (exploratory, everything reported); the
# surviving rule confirms once on EDOS.
#
#   nohup bash scripts/arr_ablations/run_student_need_arms.sh > /dev/null 2>&1 &
#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/results/local_pipeline/student_need_arms.log"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "================ student-need arms start ================"
for cfg in configs/local_pipeline.yaml configs/local_pipeline_qwen.yaml; do
  export LOCAL_PIPELINE_CONFIG="$cfg"
  bb=$("$PY" -c "import sys;sys.path.insert(0,'.');from scripts.local_pipeline import shortname;print(shortname())")
  case "$bb" in
    llama32_3b) sfx=llama ;;
    qwen25_3b)  sfx=qwen ;;
    *) say "unknown backbone $bb"; continue ;;
  esac
  ARMS=("hard10_${sfx}_dpo" "hardctl10_${sfx}_dpo" "hardall10_${sfx}_dpo")
  say "=== backbone $bb ==="

  pred="results/local_pipeline/train_pool_base/${bb}_train_base.json"
  if [ -f "$ROOT/$pred" ]; then
    say "  SKIP train-pool inference (exists)"
  else
    say "  train-pool inference"
    t0=$SECONDS
    "$PY" scripts/arr_ablations/train_pool_base_inference.py >>"$LOG" 2>&1 \
      && say "    inference done in $((SECONDS-t0))s" \
      || { say "    INFERENCE FAILED"; continue; }
    [ -f "$ROOT/$pred" ] || { say "    INFERENCE FAILED (no $pred)"; continue; }
  fi

  say "  build arms"
  "$PY" scripts/arr_ablations/build_student_need_arms.py --backbone "$bb" >>"$LOG" 2>&1 \
    || { say "    BUILD FAILED"; continue; }

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
    # Verify the artefact, not the exit code.
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

say "================ student-need arms done ================"
