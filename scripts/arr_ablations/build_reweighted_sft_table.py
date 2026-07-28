"""Build the reweighted-SFT baseline table (tab_reweighted_sft).

Compares plain SFT, agreement-weighted SFT (wsft), soft-label SFT
(annotator-distribution targets), Standard DPO, and RA-DPO on both local
backbones — the weighted/soft-label baselines a reviewer requested.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_TEX = ROOT / "arr_revision" / "experiments" / "tables" / "tab_reweighted_sft.tex"

ROWS = [
    ("sft",           "SFT (plain)"),
    ("wsft",          "SFT, agreement-weighted loss"),
    ("softlabel_sft", "SFT, soft labels"),
    ("std_dpo",       "Standard DPO"),
    ("ra_dpo",        "RA-DPO"),
]
BACKBONES = [("llama32_3b", "Llama-3.2-3B"), ("qwen25_3b", "Qwen2.5-3B")]


def read(sn: str, v: str):
    p = ROOT / "results" / "local_pipeline" / "per_instance" / f"{sn}_{v}_local.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["standard_metrics"]["f1_macro"], (d.get("accuracy_at_coverage") or {}).get("acc@50%")


def fmt(c):
    return f"{c:.3f}" if c is not None else "--"


def main():
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "",
        "\\caption{Reweighted-SFT baselines on the open-weight backbones:",
        "per-example loss weighted by annotator agreement (wsft) and",
        "soft-label training toward the annotator distribution. The hosted",
        "fine-tuning API supports neither, so these run locally. The plain-SFT",
        "Llama cell is the chat-template artifact discussed in Section~4.}",
        "\\label{tab:reweighted-sft}",
        "",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Llama-3.2-3B} & \\multicolumn{2}{c}{Qwen2.5-3B} \\\\",
        "Method & F1 & @50 & F1 & @50 \\\\",
        "\\midrule",
    ]
    for v, label in ROWS:
        cells = []
        for sn, _ in BACKBONES:
            r = read(sn, v)
            cells.extend([None, None] if r is None else [r[0], r[1]])
        lines.append(label + " & " + " & ".join(fmt(c) for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX}")
    for v, label in ROWS:
        vals = []
        for sn, _ in BACKBONES:
            r = read(sn, v)
            vals.append("--" if r is None else f"{r[0]:.3f}/{r[1]:.3f}")
        print(f"  {label:34s} {vals[0]:15s} {vals[1]}")


if __name__ == "__main__":
    main()
