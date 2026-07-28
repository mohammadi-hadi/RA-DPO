#!/usr/bin/env bash
#
# EXIST Random-30%: the size-matched control for Smart-30% (1,661 pairs).
#
# EXIST ships Random-50% (2,768), matched to Smart-50% -- not to the headline
# Smart-30%. The EDOS track now has a matched control and reports no
# significant gap (Smart-30 vs Random-30, McNemar p = 0.22). This runs the
# EXIST counterpart on both local backbones so the same question can be asked
# where the agreement signal has four levels instead of two.
#
#   gap on EXIST but not EDOS -> selection needs a granular agreement signal
#   no gap on either          -> the training-side result is data efficiency
#
# Waits for run_edos_expansion.sh to finish, runs finalize_edos.sh, then
# trains. Launch it any time; it blocks until the GPU is free.
#
#   nohup bash scripts/arr_ablations/run_exist_random30.sh >/dev/null 2>&1 &
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/venv/bin/python"
LOG="$ROOT/results/local_pipeline/exist_random30.log"
mkdir -p "$(dirname "$LOG")"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ---- 0. Wait for the EDOS run to release the GPU ---------------------------
say "waiting for run_edos_expansion.sh to finish"
while pgrep -f "run_edos_expansion.sh" >/dev/null; do sleep 120; done
say "EDOS run finished"

say "running finalize_edos.sh"
bash scripts/arr_ablations/finalize_edos.sh >>"$LOG" 2>&1 \
    && say "finalize_edos OK" || say "finalize_edos FAILED (continuing)"

# ---- 1. Train + eval random30_dpo on each EXIST backbone -------------------
for cfg in configs/local_pipeline.yaml configs/local_pipeline_qwen.yaml; do
    export LOCAL_PIPELINE_CONFIG="$cfg"
    bb=$("$PY" -c "
import sys; sys.path.insert(0,'.')
from scripts.local_pipeline import shortname; print(shortname())")
    say "=== $bb ($cfg) ==="

    pi="results/local_pipeline/per_instance/${bb}_random30_dpo_local.json"
    if [ -f "$ROOT/$pi" ]; then
        say "  SKIP — $pi exists"
        continue
    fi

    say "  training random30_dpo (1,661 pairs)"
    t0=$SECONDS
    "$PY" scripts/local_pipeline/stage_05_dpo_variants.py --only random30_dpo \
        >>"$LOG" 2>&1 \
        && say "  train OK ($((SECONDS-t0))s)" \
        || { say "  train FAILED"; continue; }

    say "  eval + OOF R(x)"
    "$PY" scripts/local_pipeline/stage_06_eval_oof.py --variants random30_dpo \
        >>"$LOG" 2>&1 \
        && say "  eval OK" || say "  eval FAILED"

    say "  unified tables"
    "$PY" scripts/local_pipeline/stage_07_unified_tables.py >>"$LOG" 2>&1 \
        && say "  tables OK" || say "  tables FAILED"

    say "  validator"
    "$PY" scripts/local_pipeline/validate.py >>"$LOG" 2>&1
    say "  validator exit=$?"
done
unset LOCAL_PIPELINE_CONFIG

# ---- 2. The comparison this was run to make --------------------------------
say "=== Smart-30 vs Random-30 on EXIST ==="
for bb in llama32_3b qwen25_3b; do
    "$PY" scripts/arr_ablations/reliability_stats.py \
        --dir results/local_pipeline/per_instance \
        --glob "${bb}_*_local.json" \
        --out "results/local_pipeline/unified/${bb}/stats.json" >>"$LOG" 2>&1
done

"$PY" - <<'PYEOF'
import json, pathlib
for bb in ("llama32_3b", "qwen25_3b"):
    p = pathlib.Path(f"results/local_pipeline/unified/{bb}/stats.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    b = d["bootstrap_f1"]
    ks = {"smart30": f"{bb}_smart30_dpo", "random30": f"{bb}_random30_dpo"}
    if not all(k in b for k in ks.values()):
        print(f"{bb}: random30 not present yet"); continue
    print(f"\n{bb} (EXIST, n={d['meta']['n_test']}):")
    for name, k in ks.items():
        v = b[k]
        print(f"  {name:<9} F1 {v['f1_mean']:.4f} "
              f"[{v['f1_ci_low']:.4f}, {v['f1_ci_high']:.4f}]")
    for rec in d["mcnemar"]:
        if {rec["model_a"], rec["model_b"]} == set(ks.values()):
            sig = "SIGNIFICANT" if rec["significant_at_05"] else "not significant"
            print(f"  McNemar p = {rec['p_value']:.4f}  -> {sig}")
print("\nEDOS reference: Smart-30 0.6562 vs Random-30 0.6482, p = 0.2191 "
      "(not significant)")
PYEOF

say "=== done ==="
