#!/usr/bin/env python3
"""Leave-one-out component-importance figure for R(x) at inference.

For each gpt-4o variant we drop one component from the learned three-factor
R(x) and measure how much expected accuracy at 50% coverage falls. The gap
is the leave-one-out (LOO) importance of that component. Values are read from
the inference ablation summary so the figure cannot drift from Table 9:

    drop(agree)   = acc50(full) - acc50(conf + 1-sig)
    drop(conf)    = acc50(full) - acc50(agree + 1-sig)
    drop(tok-unc) = acc50(full) - acc50(conf + agree)

Outputs arr_revision/experiments/figures/fig_loo_importance.{pdf,png} and a
copy of the PDF under arr_revision/paper/figures/ for the paper build.
Run: venv/bin/python scripts/arr_ablations/fig_loo_importance.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_assets"))
from data import COLORS, apply_style  # noqa: E402  (paper figure style)

import matplotlib.pyplot as plt  # noqa: E402

SUMMARY = ROOT / "results" / "arr_ablations" / "rx_component_ablation" / "summary.csv"
FIG_DIR = ROOT / "arr_revision" / "experiments" / "figures"
PAPER_FIG_DIR = ROOT / "arr_revision" / "paper" / "figures"

# gpt-4o variants in display order, mapped to their summary.csv row prefix.
MODELS = [
    ("base", "gpt-4o_base"),
    ("SFT", "gpt-4o_SFT"),
    ("Standard DPO", "gpt-4o_Standard_DPO"),
    ("RA-DPO", "gpt-4o_RA-DPO"),
]

# component label -> (two-feature rule that drops it, bar color)
COMPONENTS = [
    ("Agreement", "oof_conf_sig", COLORS["ra_dpo"]),
    ("Confidence", "oof_agree_sig", COLORS["standard"]),
    ("Token-uncertainty", "oof_conf_agree", COLORS["smart30"]),
]


def load_acc50() -> dict[str, dict[str, float]]:
    """model_prefix -> {rule -> acc@50}."""
    out: dict[str, dict[str, float]] = {}
    with open(SUMMARY, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["model"], {})[row["rule"]] = float(row["acc@50"])
    return out


def main() -> int:
    acc = load_acc50()

    # deltas[component][model_idx] = LOO drop in pp
    deltas = {c[0]: [] for c in COMPONENTS}
    for _, prefix in MODELS:
        full = acc[prefix]["full_rx"]
        for label, drop_rule, _ in COMPONENTS:
            deltas[label].append((full - acc[prefix][drop_rule]) * 100.0)

    # Self-check against the number quoted in the paper/response.
    ra_agree = deltas["Agreement"][MODELS.index(("RA-DPO", "gpt-4o_RA-DPO"))]
    ok = abs(ra_agree - 10.9) < 0.15
    print(f"SELF-CHECK RA-DPO drop(agree) = {ra_agree:.2f}pp (expect ~10.9): "
          f"{'PASSED' if ok else 'FAILED'}")

    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(MODELS))
    width = 0.26
    for i, (label, _, color) in enumerate(COMPONENTS):
        offset = (i - 1) * width
        vals = deltas[label]
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(b.get_x() + b.get_width() / 2,
                                        v + (0.2 if v >= 0 else -0.2)),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7)

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MODELS])
    ax.set_ylabel("LOO drop in accuracy@50 (pp)")
    ax.set_title("Component importance in $R(x)$ at inference (gpt-4o)")
    ax.set_ylim(-3.5, 14.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.95,
              columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf = FIG_DIR / "fig_loo_importance.pdf"
    fig.savefig(save_pdf)
    fig.savefig(save_pdf.with_suffix(".png"), dpi=150)
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(save_pdf, PAPER_FIG_DIR / "fig_loo_importance.pdf")
    print(f"wrote {save_pdf} and copied to {PAPER_FIG_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
