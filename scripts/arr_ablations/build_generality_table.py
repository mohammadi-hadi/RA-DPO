"""Build the cross-backbone, cross-dataset generality table (tab:local).

Replaces the EXIST-only local-corroboration table with the full grid:

    EXIST x {gpt-4o, Qwen, Llama}   and   EDOS x {Llama, Qwen}

Two metrics per cell (F1 and accuracy at 50% coverage) rather than three, so
the grid fits the existing table* without a new float; acc@100 for every cell
is in the appendix coverage tables.

Cells whose run has not finished are emitted as \\pending{} markers rather
than blanks, so a draft never silently reads as complete. Numbers are pulled
from the unified CSVs, never transcribed by hand.

    python scripts/arr_ablations/build_generality_table.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [ROOT / "arr_revision" / "experiments" / "tables",
            ROOT / "arr_revision" / "paper" / "tables"]

# The gpt-4o tables key rows by a display method name; the local-pipeline and
# EDOS tables key them by variant id. Both are needed per row.
# (display label, gpt-4o key, EXIST-local key, EDOS key)
ROWS = [
    ("SFT",          "SFT",                "sft",            "sft"),
    ("Smart-10\\%",  "Smart-10pct DPO",    "smart10_dpo",    None),
    ("Smart-30\\%",  "Smart-30pct DPO",    "smart30_dpo",    "smart30_dpo"),
    ("Random-30\\%", None,                 "random30_dpo",   "random30_dpo"),
    ("Smart-50\\%",  "Smart-50pct DPO",    "smart50_dpo",    None),
    ("Random-50\\%", "Random-50pct DPO",   "random50_dpo",   "random50_dpo"),
    ("Standard",     "Standard DPO",       "std_dpo",        "std_dpo"),
    ("Ambiguous",    "Ambiguous-only DPO", "ambiguous_dpo",  None),
    ("\\textbf{RA-DPO}", "RA-DPO",         "ra_dpo",         "ra_dpo"),
]

PENDING = "\\pending{}"


def norm(s: str) -> str:
    return (s.lower().replace("-", "").replace("_", "").replace("%", "pct")
            .replace(" ", "").replace("(", "").replace(")", ""))


def load_exist_gpt4o() -> dict:
    """results/unified_gpt4o: 'method' is 'gpt-4o (<variant>)'."""
    f1, cov = {}, {}
    for r in csv.DictReader(open(ROOT / "results/unified_gpt4o/fine_tuning.csv")):
        key = norm(r["method"].replace("gpt-4o", "").strip())
        f1[key] = float(r["f1_macro"])
    for r in csv.DictReader(
            open(ROOT / "results/unified_gpt4o/coverage_accuracy.csv")):
        # this file labels the column 'model', not 'method'
        label = r.get("model") or r.get("method") or r.get("variant")
        cov[norm(label.replace("gpt-4o", "").strip())] = float(r["acc@50%"])
    return {"f1": f1, "cov": cov}


def load_unified(base: Path) -> dict:
    """local_pipeline / edos_pipeline unified dirs: keyed by 'variant'."""
    f1, cov = {}, {}
    ft = base / "fine_tuning.csv"
    ca = base / "coverage_accuracy.csv"
    if not ft.exists():
        return {"f1": {}, "cov": {}}
    for r in csv.DictReader(open(ft)):
        f1[norm(r["variant"])] = float(r["f1_macro"])
    for r in csv.DictReader(open(ca)):
        cov[norm(r["variant"])] = float(r["acc@50%"])
    return {"f1": f1, "cov": cov}


def cell(src: dict, key) -> tuple[str, str]:
    if key is None:
        return "--", "--"
    k = norm(key)
    if k not in src["f1"]:
        return PENDING, PENDING
    return f"{src['f1'][k]:.3f}", f"{src['cov'].get(k, float('nan')):.3f}"


def main():
    tracks = [
        ("EXIST", "OpenAI", load_exist_gpt4o(), 1),
        ("EXIST", "Qwen",
         load_unified(ROOT / "results/local_pipeline/unified/qwen25_3b"), 1),
        ("EXIST", "Llama",
         load_unified(ROOT / "results/local_pipeline/unified/llama32_3b"), 1),
        ("EDOS", "Llama",
         load_unified(ROOT / "results/edos_pipeline/unified/llama32_3b"), 2),
        ("EDOS", "Qwen",
         load_unified(ROOT / "results/edos_pipeline/unified/qwen25_3b"), 2),
    ]

    n = len(tracks)
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}", "",
        "\\caption{F1 and accuracy at 50\\% coverage across three backbones "
        "and both corpora (true-agreement setting). Bold = column maximum. "
        "\\pendingnote}",
        "\\label{tab:local}", "",
        "\\begin{tabular}{l" + "cc" * n + "}", "\\toprule",
    ]

    # Two header rows: corpus spans, then backbone spans.
    hdr1, hdr2, rules = [""], [""], []
    col = 2
    for corpus, bb, _, _ in tracks:
        hdr1.append(f"\\multicolumn{{2}}{{c}}{{{bb}}}")
        hdr2.append("F1 & @50")
        rules.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
        col += 2
    corpus_hdr = ["Variant"]
    i = 0
    while i < len(tracks):
        corpus = tracks[i][0]
        span = sum(1 for t in tracks if t[0] == corpus)
        corpus_hdr.append(
            f"\\multicolumn{{{span * 2}}}{{c}}{{\\textbf{{{corpus}}}}}")
        i += span
    lines.append(" & ".join(corpus_hdr) + " \\\\")
    lines.append(" & ".join(hdr1) + " \\\\")
    lines.append("".join(rules))
    lines.append(" & ".join(hdr2) + " \\\\")
    lines.append("\\midrule")

    body = []
    for label, gk, lk, dk in ROWS:
        cells = []
        for corpus, bb, src, _ in tracks:
            if corpus == "EDOS":
                key = dk
            elif bb == "OpenAI":
                key = gk
            else:
                key = lk
            f1, c50 = cell(src, key)
            cells += [f1, c50]
        body.append((label, cells))

    # Bold the column maximum among numeric entries.
    for j in range(len(tracks) * 2):
        vals = [(i, c[1][j]) for i, c in enumerate(body)
                if c[1][j] not in ("--", PENDING)]
        if vals:
            bi = max(vals, key=lambda t: float(t[1]))[0]
            body[bi][1][j] = f"\\textbf{{{body[bi][1][j]}}}"

    for label, cells in body:
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    text = "\n".join(lines)

    n_pending = text.count(PENDING)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / "tab_local_corroboration.tex").write_text(text)
        print("wrote", (d / "tab_local_corroboration.tex").relative_to(ROOT))
    print(f"pending cells: {n_pending // 2} (runs still in flight)")


if __name__ == "__main__":
    main()
