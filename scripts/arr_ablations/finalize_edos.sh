#!/usr/bin/env bash
#
# Post-training pass for the EDOS track. Run AFTER run_edos_expansion.sh has
# printed "EDOS expansion done".
#
#   bash scripts/arr_ablations/finalize_edos.sh
#
# Produces, for each backbone that has per-instance files:
#   - bootstrap F1 CIs + pairwise McNemar        unified/<bb>/stats.json
#   - oracle / deployable / no-agreement R(x)    unified/<bb>/coverage_*.csv
#   - the LaTeX results table                    arr_revision/*/tables/
#
# The agreement regressor is trained once: agreement is a property of the
# text and the corpus, not of the classifier backbone, so both backbones
# consume the same predicted-agreement array (the EXIST track shares one
# score set across backbones the same way).
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/venv/bin/python"
DRIVER="$ROOT/scripts/arr_ablations/edos_pipeline.py"
PRED_NPY="results/edos_pipeline/predicted_agreement/pred_agreement_twitter-xlmr-base.npy"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 1. Agreement regressor -> unlocks the deployable regime ---------------
if [ -f "$ROOT/$PRED_NPY" ]; then
    say "SKIP agreement regressor — $PRED_NPY exists"
else
    say "training EDOS agreement regressor (~10 min)"
    "$PY" scripts/arr_ablations/train_edos_agreement_predictor.py || {
        say "FAIL regressor — deployable regime will be omitted"
    }
fi

# bash 3.2 (what macOS ships, and what `env bash` resolves to here) errors on
# "${arr[@]}" for an empty array under `set -u`, so every expansion below uses
# the ${arr[@]+"${arr[@]}"} guard. Without it this script would die on exactly
# the no-regressor fallback path.
PRED_ARG=()
[ -f "$ROOT/$PRED_NPY" ] && PRED_ARG=(--pred-agreement "$PRED_NPY")
if [ ${#PRED_ARG[@]} -gt 0 ]; then REGIMES="oracle / no-agreement / deployable"
else REGIMES="oracle / no-agreement"; fi

# ---- 2. Per-backbone: stats, regimes, table -------------------------------
for bb in llama32_3b qwen25_3b; do
    n=$(ls results/edos_pipeline/per_instance/${bb}_*_test_edos.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" = "0" ]; then
        say "SKIP $bb — no per-instance files"
        continue
    fi
    say "=== $bb ($n variants) ==="

    cfg="configs/edos_pipeline.yaml"
    [ "$bb" = "qwen25_3b" ] && cfg="configs/edos_pipeline_qwen.yaml"

    say "  R(x) regimes ($REGIMES)"
    "$PY" "$DRIVER" --config "$cfg" eval-oof \
        ${PRED_ARG[@]+"${PRED_ARG[@]}"} >/dev/null \
        && say "  eval-oof OK" || say "  eval-oof FAILED"

    say "  bootstrap CIs + McNemar"
    "$PY" scripts/arr_ablations/reliability_stats.py \
        --dir results/edos_pipeline/per_instance \
        --glob "${bb}_*_test_edos.json" \
        --out "results/edos_pipeline/unified/${bb}/stats.json" >/dev/null \
        && say "  stats OK" || say "  stats FAILED"

    say "  LaTeX table"
    "$PY" scripts/arr_ablations/build_edos_table.py --backbone "$bb" \
        && say "  table OK" || say "  table FAILED"
done

# ---- 3. Summary ------------------------------------------------------------
say "=== headline numbers ==="
"$PY" - <<'PYEOF'
import json, pathlib
root = pathlib.Path(".")
for bb in ("llama32_3b", "qwen25_3b"):
    p = root / "results/edos_pipeline/unified" / bb / "summary.json"
    if not p.exists():
        continue
    d = json.load(open(p))["models"]
    print(f"\n{bb}:")
    print(f"  {'variant':<14}{'pairs':>7}{'F1':>8}{'acc@100':>9}{'acc@50':>8}"
          f"{'no-agr':>8}{'deploy':>8}")
    for k, m in d.items():
        dep = m.get("deployable", {}).get("acc_at_coverage", {}).get("acc@50%")
        na = m.get("no_agreement", {}).get("acc_at_coverage", {}).get("acc@50%")
        print(f"  {k:<14}{str(m.get('pairs') or '—'):>7}{m['f1_macro']:>8.4f}"
              f"{m['acc_at_coverage']['acc@100%']:>9.4f}"
              f"{m['acc_at_coverage']['acc@50%']:>8.4f}"
              f"{na if na is None else format(na, '.4f'):>8}"
              f"{dep if dep is None else format(dep, '.4f'):>8}")

# The decisive comparison: reliability selection vs random at equal budget.
p = root / "results/edos_pipeline/unified/llama32_3b/summary.json"
if p.exists():
    d = json.load(open(p))["models"]
    if "smart30_dpo" in d and "random30_dpo" in d:
        s, r = d["smart30_dpo"], d["random30_dpo"]
        print(f"\nSmart-30 vs Random-30 (both 4,200 pairs) — does the "
              f"reliability ranking beat chance at equal budget?")
        print(f"  F1     {s['f1_macro']:.4f} vs {r['f1_macro']:.4f}  "
              f"(delta {s['f1_macro'] - r['f1_macro']:+.4f})")
        print(f"  acc@50 {s['acc_at_coverage']['acc@50%']:.4f} vs "
              f"{r['acc_at_coverage']['acc@50%']:.4f}")
        st = root / "results/edos_pipeline/unified/llama32_3b/stats.json"
        if st.exists():
            for rec in json.load(open(st))["mcnemar"]:
                if {rec["model_a"], rec["model_b"]} == {
                        "llama32_3b_smart30_dpo", "llama32_3b_random30_dpo"}:
                    print(f"  McNemar p = {rec['p_value']:.4f}")
PYEOF

say "=== finalize done ==="
echo
echo "Still open (EXIST-hardcoded, need the same constant edits):"
echo "  scripts/analysis/shared_weights_baseline.py   (IN_DIR, n_per=692)"
echo "  scripts/arr_ablations/tau_sensitivity.py      (MODELS abs paths)"
