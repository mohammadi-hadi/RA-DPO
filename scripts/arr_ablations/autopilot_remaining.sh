#!/bin/bash
# Self-driving completion of the remaining hosted-track work. Runs detached
# (nohup) so it survives any orchestrator/session interruption:
#   1. Polls the three gpt-4o ablation fine-tune jobs every 10 minutes.
#   2. When a job succeeds, evaluates its model on the 692-post test set
#      (writes results/final_reliability_3factor/gpt-4o_<variant>.json).
#   3. When all three are evaluated, regenerates the training-ablation table
#      and copies it into the paper workspace.
# Progress: results/arr_ablations/autopilot.log ; done marker:
# results/arr_ablations/autopilot.done
# NOTE: macOS bash 3.2 compatible (no associative arrays).
set -u
cd "$(dirname "$0")/../.."
PY=venv/bin/python
LOG=results/arr_ablations/autopilot.log
set -a; source .env; set +a

job_id() {
  case "$1" in
    uncert30) echo ftjob-3NGek8JfDre9BG2AWsjm3A0Z ;;
    agree30)  echo ftjob-4lyvQqzd21hNuC8RMwz62WoG ;;
    conf30)   echo ftjob-88HHnbQu8IyENndiz07ZphaZ ;;
  esac
}

status_of() {
  curl -s --max-time 40 "https://api.openai.com/v1/fine_tuning/jobs/$1" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    | $PY -c "import json,sys
try: print(json.loads(sys.stdin.read()).get('status',''))
except Exception: print('')"
}

evaluated() { [ -f "results/final_reliability_3factor/gpt-4o_$1.json" ]; }

echo "$(date '+%F %T') autopilot started (pid $$)" >> "$LOG"
while true; do
  for v in uncert30 agree30 conf30; do
    if evaluated "$v" || [ -f "results/arr_ablations/${v}.failed" ]; then continue; fi
    s=$(status_of "$(job_id "$v")")
    echo "$(date '+%F %T') $v: ${s:-poll-failed}" >> "$LOG"
    case "$s" in
      succeeded)
        echo "$(date '+%F %T') evaluating $v ..." >> "$LOG"
        if $PY scripts/arr_ablations/submit_ablation_finetunes.py eval --variant "$v" >> "$LOG" 2>&1; then
          echo "$(date '+%F %T') $v evaluated OK" >> "$LOG"
        else
          echo "$(date '+%F %T') $v EVAL FAILED (will retry next cycle)" >> "$LOG"
        fi
        ;;
      failed|cancelled)
        echo "$(date '+%F %T') $v terminal-failed; skipping" >> "$LOG"
        touch "results/arr_ablations/${v}.failed"
        ;;
    esac
  done
  ok=1
  for v in uncert30 agree30 conf30; do
    if ! evaluated "$v" && [ ! -f "results/arr_ablations/${v}.failed" ]; then ok=0; fi
  done
  if [ "$ok" -eq 1 ]; then
    $PY scripts/arr_ablations/build_training_ablation_table.py >> "$LOG" 2>&1
    cp arr_revision/experiments/tables/tab_training_ablation.tex arr_revision/paper/tables/ 2>>"$LOG"
    echo "$(date '+%F %T') ALL DONE" >> "$LOG"
    touch results/arr_ablations/autopilot.done
    exit 0
  fi
  sleep 600
done
