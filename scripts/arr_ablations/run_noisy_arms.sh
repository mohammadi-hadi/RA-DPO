#!/usr/bin/env bash
#
# Single-annotator label-noise arms: the regime where selection should matter.
#
# The pool's majority-vote labels are already denoised, which is why every
# matched control came out null. Under a single-annotator protocol ~17-18% of
# a random draw's pairs carry a flipped label, while the top-554 reliability
# selection is 100% unanimous and therefore invariant. Two new arms per
# backbone (smart10 needs no retraining):
#
#   noisy1_random10   same items as random10, single-annotator labels
#   noisy1_strat10    same items as strat10,  single-annotator labels
#
# Pre-registered primary contrast: smart10 vs noisy1_random10 (McNemar).
#
#   nohup bash scripts/arr_ablations/run_noisy_arms.sh > /dev/null 2>&1 &
#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/results/local_pipeline/noisy_arms.log"
ARMS=(noisy1_random10_dpo noisy1_strat10_dpo)

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "================ noisy-label arms start ================"
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

say "================ noisy-label arms done ================"
