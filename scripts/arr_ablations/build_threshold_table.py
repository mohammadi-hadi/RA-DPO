"""Emit the decision-threshold audit table (tab:threshold) from the audit JSON.

Every threshold number quoted in the paper needs a float to point at. Source:
results/arr_ablations/threshold_audit.json (scripts/arr_ablations/threshold_audit.py),
plus the validation-tuned base rows from the two EXIST open-weight backbones.

    venv/bin/python scripts/arr_ablations/build_threshold_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "arr_ablations" / "threshold_audit.json"
OUT_DIRS = [ROOT / "arr_revision" / "experiments" / "tables",
            ROOT / "arr_revision" / "paper" / "tables"]

TRACKS = [("exist_gpt4o", "EXIST-OpenAI"), ("exist_llama", "EXIST-Llama"),
          ("exist_qwen", "EXIST-Qwen"), ("edos_llama", "EDOS-Llama"),
          ("edos_qwen", "EDOS-Qwen")]
# display label -> per-track variant keys (names differ across tracks)
ROWS = [("Base (prompted)", {"exist_gpt4o": "base", "exist_llama": "base",
                             "exist_qwen": "base", "edos_llama": "base",
                             "edos_qwen": "base"}),
        ("SFT", {"exist_gpt4o": "SFT", "exist_llama": "sft",
                 "exist_qwen": "sft", "edos_llama": "sft", "edos_qwen": "sft"}),
        ("Standard DPO", {"exist_gpt4o": "Standard_DPO", "exist_llama": "std_dpo",
                          "exist_qwen": "std_dpo", "edos_llama": "std_dpo",
                          "edos_qwen": "std_dpo"}),
        ("RA-DPO", {"exist_gpt4o": "RA-DPO", "exist_llama": "ra_dpo",
                    "exist_qwen": "ra_dpo", "edos_llama": "ra_dpo",
                    "edos_qwen": "ra_dpo"})]
VAL_TUNED = {"exist_llama": "val_tuned_llama32_3b",
             "exist_qwen": "val_tuned_qwen25_3b"}


def main():
    d = json.load(open(SRC))
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}", "",
        "\\caption{Decision-threshold audit: F1 at the default threshold and at "
        "the best swept threshold. Last row tunes the base threshold on validation.}",
        "\\label{tab:threshold}", "",
        "\\begin{tabular}{l" + "cc" * len(TRACKS) + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in TRACKS) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(TRACKS))),
        "Model & " + " & ".join("default & tuned" for _ in TRACKS) + " \\\\",
        "\\midrule",
    ]
    for label, keys in ROWS:
        cells = []
        for tk, _ in TRACKS:
            r = d.get(tk, {}).get(keys.get(tk))
            cells += ([f"{r['f1_default']:.3f}", f"{r['f1_oracle_tuned']:.3f}"]
                      if r else ["--", "--"])
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    cells = []
    for tk, _ in TRACKS:
        v = d.get(VAL_TUNED.get(tk, ""), None)
        cells += ["--", f"{v['f1_test_val_tuned']:.3f}"] if v else ["--", "--"]
    lines.append("Base, tuned on validation & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    text = "\n".join(lines)
    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        (od / "tab_threshold.tex").write_text(text)
        print("wrote", (od / "tab_threshold.tex").relative_to(ROOT))


if __name__ == "__main__":
    main()
