"""Check the three OpenAI fine-tune jobs and, for each that is succeeded,
run the evaluation pipeline on the 692-sample EN+ES structured-prompt test
set. Writes final_reliability_3factor/{gpt-4o_SFT,gpt-4o_Smart-10pct_DPO,
gpt-4o_Ambiguous_only_DPO}.json and regenerates the unified tables.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI

JOBS_FILE = ROOT / "results" / "unified_gpt4o" / "new_jobs.json"
SFT_FILE = ROOT / "results" / "unified_gpt4o" / "sft_job_id.txt"

MAPPING = {
    "sft":     ("gpt-4o (SFT)",             5535, "gpt-4o_SFT.json"),
    "smart10": ("gpt-4o (Smart-10% DPO)",   554,  "gpt-4o_Smart-10pct_DPO.json"),
    "ambig":   ("gpt-4o (Ambiguous-only DPO)", 665, "gpt-4o_Ambiguous_only_DPO.json"),
}


def status_map():
    client = OpenAI()
    jobs = {"sft": SFT_FILE.read_text().strip()}
    jobs.update(json.load(open(JOBS_FILE))["job_ids"])
    rows = {}
    for tag, jid in jobs.items():
        j = client.fine_tuning.jobs.retrieve(jid)
        rows[tag] = {"id": jid, "status": j.status, "model": j.fine_tuned_model,
                     "tokens": j.trained_tokens}
    return rows


def ensure_evaluated(tag, status):
    from scripts.evaluate_sft_4o import run  # evaluator is generic
    label, pairs, out_fname = MAPPING[tag]
    out_path = ROOT / "results" / "final_reliability_3factor" / out_fname
    if out_path.exists():
        print(f"[{tag}] already evaluated → {out_path.name}")
        return True
    if status["status"] != "succeeded" or not status["model"]:
        print(f"[{tag}] not ready ({status['status']}) — skip")
        return False
    print(f"[{tag}] evaluating {status['model']} ...")
    # Patch evaluator's OUT path by moving the file after run(); the eval writes gpt-4o_SFT.json by default.
    from scripts import evaluate_sft_4o as ev
    orig = ev.OUT_DIR / "gpt-4o_SFT.json"
    if orig.exists():
        orig.unlink()
    ev.run(status["model"], label, training_pairs=pairs)
    assert orig.exists(), "evaluator did not write gpt-4o_SFT.json"
    orig.rename(out_path)
    print(f"[{tag}] wrote {out_path}")
    return True


def main():
    rows = status_map()
    for tag, r in rows.items():
        print(f"{tag:>10}  status={r['status']:>15}  tokens={r['tokens']}  model={r['model']}")
    print()
    evaluated_any = False
    for tag, r in rows.items():
        if ensure_evaluated(tag, r):
            evaluated_any = True

    # If any succeeded / evaluated, regenerate unified tables
    if evaluated_any:
        print("\nRegenerating unified tables...")
        import subprocess
        subprocess.run(["python", str(ROOT / "scripts" / "build_unified_gpt4o_tables.py")], check=True)
        subprocess.run(["python", str(ROOT / "scripts" / "analysis" / "smart_curve_stats.py")], check=True)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment")
    main()
