"""Decompose R_train at the 554-pair budget: does either half carry signal?

R_train(x) = 0.8 * agreement + 0.2 * (1 - sigmoid_score) makes two separable
claims. At a budget below the full-data ceiling, four arms of identical size
separate them:

    random10     random from the whole pool          (no selection at all)
    randunan10   random from the unanimous pool       (agreement filter only)
    smart10      unanimous, lowest model uncertainty  (the shipped rule)
    flip10       unanimous, highest model uncertainty (opposite direction)

    randunan10 - random10   = the agreement filter
    smart10    - randunan10 = the uncertainty ranking, as shipped
    flip10     - randunan10 = the uncertainty ranking, reversed
    smart10    - flip10     = whether the direction matters at all

Every comparison is size-matched at 554 pairs, so any difference is
attributable to which pairs were chosen rather than how many.

    python scripts/arr_ablations/analyze_lowbudget_arms.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "random10_dpo": "random from pool      (no selection)",
    "randunan10_dpo": "random from unanimous (agreement filter)",
    "smart10_dpo": "unanimous, low unc.   (shipped rule)",
    "flip10_dpo": "unanimous, high unc.  (reversed)",
    "strat10_dpo": "stratified reliability (dist. matched)",
    "noworst10_dpo": "drop contested stratum",
    "noisy1_random10_dpo": "random items, 1-annot labels (17% flip)",
    "noisy1_strat10_dpo": "strat items, 1-annot labels  (18% flip)",
}
CONTRASTS = [
    ("randunan10_dpo", "random10_dpo", "agreement filter"),
    ("smart10_dpo", "randunan10_dpo", "uncertainty ranking (shipped)"),
    ("flip10_dpo", "randunan10_dpo", "uncertainty ranking (reversed)"),
    ("smart10_dpo", "flip10_dpo", "direction matters?"),
    ("strat10_dpo", "random10_dpo", "stratified vs random (clean)"),
    # Single-annotator protocol. smart10 is protocol-invariant (100%
    # unanimous), so it doubles as the selection arm under noise.
    ("smart10_dpo", "noisy1_random10_dpo", "PRIMARY: selection vs random, noisy"),
    ("noisy1_random10_dpo", "random10_dpo", "noise damage (same items)"),
    ("noisy1_strat10_dpo", "noisy1_random10_dpo", "stratified vs random, noisy"),
    ("smart10_dpo", "noisy1_strat10_dpo", "clean tail vs matched dist, noisy"),
]
# Student-need arms are per-backbone ({sfx} = llama / qwen); resolved in main.
ARMS_TPL = {
    "hard10_{sfx}_dpo": "gated wrong-first     (student need)",
    "hardctl10_{sfx}_dpo": "label-matched random  (gated pool)",
    "hardall10_{sfx}_dpo": "ungated wrong-first",
}
CONTRASTS_TPL = [
    ("hard10_{sfx}_dpo", "random10_dpo", "NEW RULE: student-need vs random"),
    ("hard10_{sfx}_dpo", "hardctl10_{sfx}_dpo", "informativeness beyond label mix"),
    ("hardall10_{sfx}_dpo", "hard10_{sfx}_dpo", "agreement gate effect"),
    ("hard10_{sfx}_dpo", "strat10_dpo", "vs stratified reliability"),
    ("hard10_{sfx}_dpo", "smart10_dpo", "vs shipped rule"),
]
SFX = {"llama32_3b": "llama", "qwen25_3b": "qwen"}


def main():
    for bb in ("llama32_3b", "qwen25_3b"):
        base = ROOT / "results" / "local_pipeline" / "unified" / bb
        ft, st = base / "fine_tuning.csv", base / "stats.json"
        if not ft.exists():
            print(f"\n{bb}: no results"); continue
        f1 = {r["variant"]: float(r["f1_macro"]) for r in csv.DictReader(open(ft))}
        cov = {}
        cp = base / "coverage_accuracy.csv"
        if cp.exists():
            rows = list(csv.DictReader(open(cp)))
            k = [c for c in rows[0] if "50" in c][0]
            cov = {r["variant"]: float(r[k]) for r in rows}
        stats = json.load(open(st)) if st.exists() else {"bootstrap_f1": {}, "mcnemar": []}
        ci = {k[len(bb) + 1:]: v for k, v in stats["bootstrap_f1"].items()
              if k.startswith(bb + "_")}
        mc = {frozenset((a["model_a"][len(bb) + 1:], a["model_b"][len(bb) + 1:])):
              a["p_value"] for a in stats["mcnemar"]}

        sfx = SFX[bb]
        arms = {**ARMS,
                **{k.format(sfx=sfx): v for k, v in ARMS_TPL.items()}}
        contrasts = CONTRASTS + [(a.format(sfx=sfx), b.format(sfx=sfx), lab)
                                 for a, b, lab in CONTRASTS_TPL]

        ceiling = f1.get("std_dpo")
        print(f"\n{'='*72}\n{bb}   (full-data ceiling F1 = {ceiling:.4f})\n{'='*72}")
        print(f"  {'arm':<38}{'F1':>8}{'95% CI':>18}{'acc@50':>9}")
        for v, desc in arms.items():
            if v not in f1:
                print(f"  {desc:<38}{'--':>8}"); continue
            c = ci.get(v)
            cis = f"[{c['f1_ci_low']:.3f}, {c['f1_ci_high']:.3f}]" if c else "--"
            print(f"  {desc:<38}{f1[v]:>8.4f}{cis:>18}{cov.get(v, float('nan')):>9.4f}")

        print(f"\n  {'contrast':<34}{'delta F1':>10}{'McNemar p':>12}  verdict")
        for a, b, label in contrasts:
            if a not in f1 or b not in f1:
                continue
            d = f1[a] - f1[b]
            p = mc.get(frozenset((a, b)))
            ps = f"{p:.4f}" if p is not None else "--"
            verdict = ("carries signal" if p is not None and p < 0.05
                       else "no effect")
            print(f"  {label:<34}{d:>+10.4f}{ps:>12}  {verdict}")


if __name__ == "__main__":
    main()
