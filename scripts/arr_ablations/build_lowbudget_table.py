"""Emit the low-budget selection table (tab:lowbudget) from the unified CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [ROOT / "arr_revision" / "experiments" / "tables",
            ROOT / "arr_revision" / "paper" / "tables"]

ARMS = [
    # distribution matched to the pool
    ("random10_dpo",   "Random, whole pool"),
    ("strat10_dpo",    "Reliability, stratified"),
    ("noworst10_dpo",  "Drop contested stratum"),
    # distribution narrowed to the unanimous tail
    ("randunan10_dpo", "Random, unanimous only"),
    ("smart10_dpo",    "Smart-10\\% (shipped rule)"),
    ("flip10_dpo",     "Highest uncertainty"),
    # ranked by base-model error (per-backbone variant names)
    ({"llama32_3b": "hard10_llama_dpo", "qwen25_3b": "hard10_qwen_dpo"},
     "Wrong-first, agreement gated"),
    ({"llama32_3b": "hardctl10_llama_dpo", "qwen25_3b": "hardctl10_qwen_dpo"},
     "Label-matched random, gated"),
    ({"llama32_3b": "hardall10_llama_dpo", "qwen25_3b": "hardall10_qwen_dpo"},
     "Wrong-first, ungated"),
]
SPLIT_AFTER_IDX = {2, 5}        # rules between the three blocks
BACKBONES = [("llama32_3b", "Llama"), ("qwen25_3b", "Qwen")]


def main():
    data = {}
    for bb, _ in BACKBONES:
        base = ROOT / "results" / "local_pipeline" / "unified" / bb
        # Full precision from the per-instance files: the 4-decimal unified
        # CSVs double-round (0.54253 -> 0.5425 -> 0.542 instead of 0.543).
        f1, cov = {}, {}
        pi_dir = ROOT / "results" / "local_pipeline" / "per_instance"
        for p in pi_dir.glob(f"{bb}_*_local.json"):
            variant = p.name[len(bb) + 1:-len("_local.json")]
            d = json.load(open(p))
            f1[variant] = float(d["standard_metrics"]["f1_macro"])
            acc = d.get("accuracy_at_coverage") or {}
            if "acc@50%" in acc:
                cov[variant] = float(acc["acc@50%"])
        st = json.load(open(base / "stats.json"))
        ci = {k[len(bb) + 1:]: v for k, v in st["bootstrap_f1"].items()
              if k.startswith(bb + "_")}
        data[bb] = (f1, cov, ci)

    lines = [
        "\\begin{table}[t]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}", "",
        "\\caption{Selection at a matched 554-pair budget: distribution-matched, "
        "unanimous-tail, and error-ranked blocks; last row: tuned base "
        "threshold, no training. F1 with 95\\% CIs.}",
        "\\label{tab:lowbudget}", "",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "cc" * len(BACKBONES) + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in BACKBONES) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(BACKBONES))),
        "Selection & " + " & ".join("F1 & @50" for _ in BACKBONES) + " \\\\",
        "\\midrule",
    ]
    for i, (key, label) in enumerate(ARMS):
        cells = []
        for bb, _ in BACKBONES:
            f1, cov, ci = data[bb]
            k = key[bb] if isinstance(key, dict) else key
            if k not in f1:
                cells += ["--", "--"]; continue
            c = ci.get(k)
            s = (f"{f1[k]:.3f} [{c['f1_ci_low']:.3f}, {c['f1_ci_high']:.3f}]"
                 if c else f"{f1[k]:.3f}")
            cells += [s, f"{cov.get(k, float('nan')):.3f}"]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if i in SPLIT_AFTER_IDX:
            lines.append("\\midrule")
    lines += ["\\midrule"]
    cells = []
    for bb, _ in BACKBONES:
        f1, cov, _ = data[bb]
        cells += [f"{f1['std_dpo']:.3f}", f"{cov.get('std_dpo', float('nan')):.3f}"]
    lines.append("Full data (5,536) & " + " & ".join(cells) + " \\\\")
    cells = []
    for bb, _ in BACKBONES:
        vt = ROOT / "results" / "local_pipeline" / "train_pool_base" / f"{bb}_val_threshold.json"
        if vt.exists():
            cells += [f"{json.load(open(vt))['f1_test_val_tuned']:.3f}", "--"]
        else:
            cells += ["--", "--"]
    lines.append("Tuned threshold, no training & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]

    text = "\n".join(lines)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / "tab_lowbudget.tex").write_text(text)
        print("wrote", (d / "tab_lowbudget.tex").relative_to(ROOT))


if __name__ == "__main__":
    main()
