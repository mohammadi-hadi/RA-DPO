#!/usr/bin/env bash
#
# EDOS track expansion: random-subset controls on Llama + the full Qwen
# backbone, giving the {EXIST, EDOS} x {Llama, Qwen} grid.
#
# Sequential on purpose — the runs share one MPS device, so overlapping them
# would thrash unified memory rather than finish sooner.
#
# Resumable: a stage is skipped when its adapter dir (train) or per-instance
# file (infer) already exists. Safe to re-run after an interruption.
#
#   bash scripts/arr_ablations/run_edos_expansion.sh
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/venv/bin/python"
DRIVER="$ROOT/scripts/arr_ablations/edos_pipeline.py"
LOG="$ROOT/results/edos_pipeline/expansion_run.log"
mkdir -p "$(dirname "$LOG")"

CFG_LLAMA="configs/edos_pipeline.yaml"
CFG_QWEN="configs/edos_pipeline_qwen.yaml"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# train <config> <models_root> <variant>
train() {
    local cfg="$1" mroot="$2" variant="$3"
    # The completion marker is adapter_model.safetensors, written only at the
    # end of training. Testing dir non-emptiness instead would treat an
    # interrupted run's checkpoint-500 as a finished adapter and skip
    # retraining -- std_dpo (875 steps) and ra_dpo (1554) both write one.
    if [ -f "$ROOT/$mroot/$variant/adapter_model.safetensors" ]; then
        say "SKIP train $variant ($cfg) — adapter complete"
        return 0
    fi
    if [ -d "$ROOT/$mroot/$variant" ]; then
        say "PARTIAL train $variant — incomplete adapter, retraining from scratch"
        rm -rf "${ROOT:?}/$mroot/$variant"
    fi
    say "START train $variant ($cfg)"
    local t0=$SECONDS
    if "$PY" "$DRIVER" --config "$cfg" train --variant "$variant" >>"$LOG" 2>&1; then
        say "DONE  train $variant — $((SECONDS - t0))s"
    else
        say "FAIL  train $variant — see $LOG"
        return 1
    fi
}

# infer <config> <shortname> <variant>
infer() {
    local cfg="$1" short="$2" variant="$3"
    local out="$ROOT/results/edos_pipeline/per_instance/${short}_${variant}_test_edos.json"
    if [ -f "$out" ]; then
        say "SKIP infer $variant ($short) — per-instance file exists"
        return 0
    fi
    say "START infer $variant ($short)"
    local t0=$SECONDS
    if "$PY" "$DRIVER" --config "$cfg" infer --variant "$variant" --split test >>"$LOG" 2>&1; then
        say "DONE  infer $variant — $((SECONDS - t0))s"
    else
        say "FAIL  infer $variant — see $LOG"
        return 1
    fi
}

say "================ EDOS expansion start ================"

# ---- 1. Llama: the two random controls -------------------------------------
# random30 first: it is the size-matched control against Smart-30 and the
# single most informative run in this batch.
for v in random30_dpo random50_dpo; do
    train "$CFG_LLAMA" "models/edos_pipeline" "$v" || exit 1
    infer "$CFG_LLAMA" "llama32_3b" "$v"          || exit 1
done

say ">>> Llama controls complete — refreshing Llama unified tables"
"$PY" "$DRIVER" --config "$CFG_LLAMA" eval-oof >>"$LOG" 2>&1 \
    && say "DONE  eval-oof (llama32_3b)" \
    || say "FAIL  eval-oof (llama32_3b)"

# ---- 2. Qwen: full variant set ---------------------------------------------
say ">>> Qwen backbone"
infer "$CFG_QWEN" "qwen25_3b" "base" || exit 1
for v in sft smart30_dpo random30_dpo std_dpo random50_dpo ra_dpo; do
    train "$CFG_QWEN" "models/edos_pipeline_qwen" "$v" || exit 1
    infer "$CFG_QWEN" "qwen25_3b" "$v"                 || exit 1
done

say ">>> Qwen complete — unified tables"
"$PY" "$DRIVER" --config "$CFG_QWEN" eval-oof >>"$LOG" 2>&1 \
    && say "DONE  eval-oof (qwen25_3b)" \
    || say "FAIL  eval-oof (qwen25_3b)"

say "================ EDOS expansion done ================"
