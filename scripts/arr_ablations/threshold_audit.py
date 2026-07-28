"""Decision-threshold audit: which fine-tuning gains exceed recalibration?

For every track and variant, compare the default-threshold F1-macro with the
best F1 over a swept decision threshold (oracle, applied to the stored test
p_YES). A fine-tuned model whose DEFAULT F1 exceeds the BASE model's
oracle-tuned ceiling has learned something no threshold move can supply; a
tuned-vs-tuned comparison is the fair ranking between variants.

Deployable (val-tuned) numbers for the two EXIST open-weight bases come from
scripts/arr_ablations/val_threshold_baseline.py.

    venv/bin/python scripts/arr_ablations/threshold_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "arr_ablations" / "threshold_audit.json"

TRACKS = {
    "exist_gpt4o": ("results/final_reliability_3factor/gpt-4o_{v}.json",
                    ["base", "SFT", "Standard_DPO", "Smart-30pct_DPO", "RA-DPO"]),
    "exist_llama": ("results/local_pipeline/per_instance/llama32_3b_{v}_local.json",
                    ["base", "sft", "std_dpo", "ra_dpo", "random10_dpo",
                     "hard10_llama_dpo", "hardctl10_llama_dpo"]),
    "exist_qwen": ("results/local_pipeline/per_instance/qwen25_3b_{v}_local.json",
                   ["base", "sft", "std_dpo", "ra_dpo", "random10_dpo",
                    "hard10_qwen_dpo", "hardctl10_qwen_dpo"]),
    "edos_llama": ("results/edos_pipeline/per_instance/llama32_3b_{v}_test_edos.json",
                   ["base", "sft", "std_dpo", "ra_dpo"]),
    "edos_qwen": ("results/edos_pipeline/per_instance/qwen25_3b_{v}_test_edos.json",
                  ["base", "sft", "std_dpo", "ra_dpo"]),
}
THRESHOLDS = np.linspace(0.02, 0.98, 193)


def audit(path):
    d = json.load(open(ROOT / path))
    pi = d["per_instance"]
    pred, corr = pi["predictions"], pi["correct"]
    gold = np.array([p if c else ("NO" if p == "YES" else "YES")
                     for p, c in zip(pred, corr)])
    conf = np.array(pi["confidences"])
    p_yes = np.where(np.array(pred) == "YES", conf, 1 - conf)
    tuned = max(f1_score(gold, np.where(p_yes >= t, "YES", "NO"),
                         average="macro") for t in THRESHOLDS)
    return {"f1_default": round(float(f1_score(gold, pred, average="macro")), 4),
            "f1_oracle_tuned": round(float(tuned), 4)}


def main():
    res = {}
    for track, (pat, variants) in TRACKS.items():
        res[track] = {}
        for v in variants:
            try:
                res[track][v] = audit(pat.format(v=v))
            except FileNotFoundError:
                res[track][v] = None
        base_ceiling = res[track]["base"]["f1_oracle_tuned"]
        print(f"\n=== {track} (base oracle ceiling {base_ceiling}) ===")
        for v, r in res[track].items():
            if r is None:
                print(f"  {v}: missing"); continue
            beyond = r["f1_default"] - base_ceiling
            print(f"  {v:<18} default {r['f1_default']:.4f}  "
                  f"tuned {r['f1_oracle_tuned']:.4f}  "
                  f"default-minus-base-ceiling {beyond:+.4f}")

    for bb in ("llama32_3b", "qwen25_3b"):
        p = ROOT / "results" / "local_pipeline" / "train_pool_base" / f"{bb}_val_threshold.json"
        if p.exists():
            res[f"val_tuned_{bb}"] = json.load(open(p))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
