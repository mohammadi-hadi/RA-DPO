"""Build the EDOS results table (tab_edos) from the unified summary.

Reads results/edos_pipeline/unified/<backbone>/{summary,stats}.json and emits
LaTeX to both table locations the revision keeps in sync:
  arr_revision/experiments/tables/  (canonical, script-generated)
  arr_revision/paper/tables/        (what main.tex \\input's)

Usage:
    python scripts/arr_ablations/build_edos_table.py                 # llama
    python scripts/arr_ablations/build_edos_table.py --backbone qwen25_3b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    ROOT / "arr_revision" / "experiments" / "tables",
    ROOT / "arr_revision" / "paper" / "tables",
]

LABELS = {
    "base": "Base (prompted)",
    "sft": "SFT",
    "std_dpo": "Standard DPO",
    "smart30_dpo": "Smart-30\\%",
    "random30_dpo": "Random-30\\%",
    "random50_dpo": "Random-50\\%",
    "ra_dpo": "RA-DPO",
}

BACKBONE_NAMES = {
    "llama32_3b": "Llama-3.2-3B-Instruct",
    "qwen25_3b": "Qwen2.5-3B-Instruct",
}

CAPTION = ("\\caption{{EDOS results on {backbone} ($n = 4{{,}}000$). "
           "``Pred.'' predicts agreement from text, ``no-agr.'' drops the "
           "agreement term; "
           "$\\alpha/\\beta/\\gamma$ are the learned weights.}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="llama32_3b",
                    choices=sorted(BACKBONE_NAMES))
    args = ap.parse_args()

    base = ROOT / "results" / "edos_pipeline" / "unified" / args.backbone
    summary = json.load(open(base / "summary.json"))
    models = summary["models"]

    stats_path = base / "stats.json"
    cis = {}
    if stats_path.exists():
        raw = json.load(open(stats_path))["bootstrap_f1"]
        # stats.json keys are per-instance filenames stripped of suffixes,
        # e.g. "llama32_3b_ra_dpo" -> variant "ra_dpo".
        prefix = f"{args.backbone}_"
        cis = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    boot = json.load(open(stats_path))["meta"]["bootstrap_n"] if cis else 0

    has_na = any("no_agreement" in m for m in models.values())
    has_dep = any("deployable" in m for m in models.values())
    cols = "lrccc" + ("c" if has_dep else "") + ("c" if has_na else "") + "c"
    header = ["Variant", "Pairs", "F1 (95\\% CI)", "Acc@100", "Acc@50"]
    if has_dep:
        header.append("Acc@50 (Pred.)")
    if has_na:
        header.append("Acc@50 (no-agr.)")
    header.append("$\\alpha/\\beta/\\gamma$")

    # Full-width float: eight columns squeezed into one column via resizebox
    # rendered at an unreadable size. 3pt keeps it inside \textwidth.
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "",
        CAPTION.format(backbone=BACKBONE_NAMES[args.backbone], boot=boot),
        "\\label{tab:edos}" if args.backbone == "llama32_3b"
        else f"\\label{{tab:edos_{args.backbone}}}",
        "",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]

    for key, label in LABELS.items():
        m = models.get(key)
        if m is None:
            continue
        w = m["weights_oof"]
        pairs = f"{m['pairs']:,}" if m.get("pairs") else "--"
        ci = cis.get(key)
        f1 = (f"{m['f1_macro']:.3f} [{ci['f1_ci_low']:.3f}, {ci['f1_ci_high']:.3f}]"
              if ci else f"{m['f1_macro']:.3f}")
        row = [label, pairs, f1,
               f"{m['acc_at_coverage']['acc@100%']:.3f}",
               f"{m['acc_at_coverage']['acc@50%']:.3f}"]
        if has_dep:
            dp = m.get("deployable", {}).get("acc_at_coverage", {})
            row.append(f"{dp['acc@50%']:.3f}" if dp else "--")
        if has_na:
            na = m.get("no_agreement", {}).get("acc_at_coverage", {})
            row.append(f"{na['acc@50%']:.3f}" if na else "--")
        row.append(f"{w['alpha']:.2f}/{w['beta']:.2f}/{w['gamma']:.2f}")
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    text = "\n".join(lines)

    name = ("tab_edos.tex" if args.backbone == "llama32_3b"
            else f"tab_edos_{args.backbone}.tex")
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
        print("wrote", (d / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
