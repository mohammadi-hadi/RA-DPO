"""Stage 07 — build unified tables from all evaluated variants.

Parallels results/unified_gpt4o/ for the local pipeline. Writes:
    results/local_pipeline/unified/fine_tuning.csv
    results/local_pipeline/unified/coverage_accuracy.csv
    results/local_pipeline/unified/weights.csv
    results/local_pipeline/unified/summary.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import (
    per_instance_dir, unified_dir, shortname, write_checkpoint,
)


def main():
    sn = shortname()
    rows_ft, rows_cov, rows_w = [], [], []
    summary = {}
    for path in sorted(per_instance_dir().glob(f"{sn}_*_local.json")):
        d = json.load(open(path))
        variant = path.stem.replace(f"{sn}_", "").replace("_local", "")
        sm = d["standard_metrics"]
        row_ft = {"method": d["model"], "variant": variant,
                  "pairs": d.get("training_pairs") or "—",
                  "f1_macro": round(sm["f1_macro"], 4),
                  "accuracy": round(sm["accuracy"], 4)}
        rows_ft.append(row_ft)
        if "accuracy_at_coverage" in d:
            row = {"variant": variant}
            row.update({k: round(v, 4) for k, v in d["accuracy_at_coverage"].items()})
            rows_cov.append(row)
        if "optimized_weights" in d:
            w = d["optimized_weights"]
            rows_w.append({"variant": variant,
                           "alpha": round(w["alpha"], 3),
                           "beta": round(w["beta"], 3),
                           "gamma": round(w["gamma"], 3)})
        summary[variant] = {"f1": sm["f1_macro"], "accuracy": sm["accuracy"]}

    uni = unified_dir()
    pd.DataFrame(rows_ft).to_csv(uni / "fine_tuning.csv", index=False)
    if rows_cov:
        pd.DataFrame(rows_cov).to_csv(uni / "coverage_accuracy.csv", index=False)
    if rows_w:
        pd.DataFrame(rows_w).to_csv(uni / "weights.csv", index=False)
    json.dump(summary, open(uni / "summary.json", "w"), indent=2)
    print("=== local pipeline unified tables ===")
    print(pd.DataFrame(rows_ft).to_string(index=False))
    if rows_cov:
        print(pd.DataFrame(rows_cov).to_string(index=False))
    if rows_w:
        print(pd.DataFrame(rows_w).to_string(index=False))
    write_checkpoint("stage_07_unified_tables", "ok")


if __name__ == "__main__":
    main()
