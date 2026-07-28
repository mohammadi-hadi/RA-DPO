"""Build the training-side component-ablation table (tab_training_ablation).

Collects F1-macro and acc@50% for the top-30% single-component DPO
selections on every backbone with saved per-instance results:
  - local backbones: results/local_pipeline/per_instance/<sn>_<variant>_local.json
  - hosted gpt-4o:   results/final_reliability_3factor/gpt-4o_<variant>.json

Emits arr_revision/experiments/tables/tab_training_ablation.tex and a CSV
next to it. Rerun whenever a new backbone or variant finishes; the table
grows accordingly.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_TEX = ROOT / "arr_revision" / "experiments" / "tables" / "tab_training_ablation.tex"
OUT_CSV = ROOT / "results" / "arr_ablations" / "training_ablation.csv"

VARIANTS = [
    ("smart30_dpo",     "Smart-30\\% ($R_{\\mathrm{train}}$)"),
    ("agree30_dpo",     "Agreement-only"),
    ("agree30_tb2_dpo", "Agreement-only (alt tie-break)"),
    ("uncert30_dpo",    "Uncertainty-only"),
    ("conf30_dpo",      "Confidence-only"),
]
LOCAL_BACKBONES = [("llama32_3b", "Llama-3.2-3B"), ("qwen25_3b", "Qwen2.5-3B")]


def read_local(shortname: str, variant: str):
    p = ROOT / "results" / "local_pipeline" / "per_instance" / f"{shortname}_{variant}_local.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d["standard_metrics"]["f1_macro"], (d.get("accuracy_at_coverage") or {}).get("acc@50%")


def read_gpt4o(variant: str):
    key = variant.replace("_dpo", "")
    # Pre-existing paper results use a different file-name convention.
    aliases = {"smart30": "Smart-30pct_DPO"}
    p = ROOT / "results" / "final_reliability_3factor" / f"gpt-4o_{key}.json"
    if not p.exists() and key in aliases:
        p = ROOT / "results" / "final_reliability_3factor" / f"gpt-4o_{aliases[key]}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    f1 = d["standard_metrics"]["f1_macro"]
    # Leakage-free out-of-fold coverage: use the stored OOF block when
    # present, otherwise recompute it from the per-instance arrays with the
    # same recipe (5-fold stratified, seed 42) so every row is comparable.
    oof = d.get("oof") or {}
    cov = oof.get("accuracy_at_coverage_oof")
    if not cov:
        cov = {"acc@50%": _oof_acc50(d["per_instance"])}
    return f1, cov.get("acc@50%")


def _oof_acc50(pi: dict) -> float:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    conf = np.asarray(pi["confidences"], float)
    agree = np.asarray(pi["agreements"], float)
    sig = np.asarray(pi["sigmoid_scores"], float)
    correct = np.asarray(pi["correct"], int)
    X = np.column_stack([conf, agree, 1 - sig])
    r = np.zeros(len(correct))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, correct):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), correct[tr])
        w = np.abs(lr.coef_[0]); w = w / w.sum()
        r[te] = X[te] @ w
    k = max(1, int(round(len(r) * 0.5)))
    return float(correct[np.argsort(-r)[:k]].mean())


def fmt(cell):
    return f"{cell:.3f}" if cell is not None else "--"


def main():
    backbones = []
    if any(read_gpt4o(v) for v, _ in VARIANTS):
        backbones.append(("gpt4o", "gpt-4o", read_gpt4o))
    for sn, label in LOCAL_BACKBONES:
        if any(read_local(sn, v) for v, _ in VARIANTS):
            backbones.append((sn, label, lambda v, sn=sn: read_local(sn, v)))

    rows = []
    for variant, label in VARIANTS:
        cells = []
        for _, _, reader in backbones:
            r = reader(variant)
            cells.extend([None, None] if r is None else [r[0], r[1]])
        if any(c is not None for c in cells):
            rows.append((label, cells))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        header = ["selection"]
        for _, label, _ in backbones:
            header += [f"{label} F1", f"{label} acc@50"]
        w.writerow(header)
        for label, cells in rows:
            w.writerow([label] + [("" if c is None else f"{c:.4f}") for c in cells])

    col_spec = "l" + "cc" * len(backbones)
    head1 = " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in backbones) + " \\\\"
    head2 = "Selection (top 30\\%) & " + " & ".join(["F1 & @50"] * len(backbones)) + " \\\\"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "",
        "\\caption{Training-side component ablation: DPO trained on the top",
        "30\\% of preference pairs (1,661) ranked by a single component,",
        "against the $R_{\\mathrm{train}}$ ranking (Smart-30\\%). Agreement-only",
        "is reported with two tie-break orders because 1,805 unanimous pairs",
        "compete for 1,661 slots. On Llama, all pairwise differences among the",
        "30\\% selections are non-significant (McNemar $p \\geq 0.34$).}",
        "\\label{tab:training-ablation}",
        "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        head1,
        head2,
        "\\midrule",
    ]
    for label, cells in rows:
        lines.append(label + " & " + " & ".join(fmt(c) for c in cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT_TEX.write_text("\n".join(lines))
    print(f"wrote {OUT_TEX} ({len(rows)} rows x {len(backbones)} backbones)")
    for label, cells in rows:
        print(f"  {label:34s} " + " ".join(fmt(c) for c in cells))


if __name__ == "__main__":
    main()
