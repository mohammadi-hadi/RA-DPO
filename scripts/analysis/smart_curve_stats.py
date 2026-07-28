"""Bootstrap + McNemar analysis of the Smart-%/Std/Random DPO curve.

Goal: tell whether Smart-50% is really worse than Smart-30% and Standard DPO,
or whether the gap (0.8192 vs 0.8213 vs 0.8212) is within noise.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "final_reliability_3factor"

MODELS = {
    "base":           ("gpt-4o_base.json",            0),
    "Random-50%":     ("gpt-4o_Random-50pct_DPO.json",2768),
    "Smart-50%":      ("gpt-4o_Smart-50pct_DPO.json", 2768),
    "Smart-30%":      ("gpt-4o_Smart-30pct_DPO.json", 1661),
    "Standard (100%)":("gpt-4o_Standard_DPO.json",    5536),
    "RA-DPO":         ("gpt-4o_RA-DPO.json",          8984),
}

LABELS = ["NO", "YES"]


def load(label):
    fname, pairs = MODELS[label]
    d = json.load(open(IN_DIR / fname))
    pi = d["per_instance"]
    preds = np.asarray(pi["predictions"])
    correct = np.asarray(pi["correct"], dtype=int)
    # True labels: derive from predictions + correct flag
    true = np.where(correct == 1, preds, np.where(preds == "YES", "NO", "YES"))
    return preds, true, correct, pairs


def f1m(true, pred):
    return f1_score(true, pred, average="macro", labels=LABELS)


def bootstrap_f1_ci(true, pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores.append(f1m(true[idx], pred[idx]))
    arr = np.array(scores)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(arr.std())


def mcnemar(correct_a, correct_b):
    """McNemar's test for paired binary outcomes."""
    from scipy.stats import binomtest
    a_only = int(((correct_a == 1) & (correct_b == 0)).sum())
    b_only = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    p = binomtest(a_only, n, p=0.5).pvalue
    return p, a_only, b_only


# --- Load all models ---
data = {k: load(k) for k in MODELS}

# --- Bootstrap CIs ---
print("=== Bootstrap F1-macro 95% CI (2000 resamples) ===")
print(f"{'Model':<18} {'pairs':>6}  {'F1':>6}  {'CI95':>20}  {'SE':>7}")
f1_rows = {}
for k, (pred, true, correct, pairs) in data.items():
    m, lo, hi, sd = bootstrap_f1_ci(true, pred)
    f1_rows[k] = (m, lo, hi, sd)
    print(f"{k:<18} {pairs:>6}  {m:.4f}  [{lo:.4f}, {hi:.4f}]  {sd:.4f}")

# --- Pairwise McNemar (paired correctness on the same 692 test posts) ---
print("\n=== McNemar paired tests (p-value for 'two models differ') ===")
pairs = [
    ("Standard (100%)", "Smart-30%"),
    ("Standard (100%)", "Smart-50%"),
    ("Smart-30%", "Smart-50%"),
    ("Smart-30%", "Random-50%"),
    ("Smart-50%", "Random-50%"),
    ("RA-DPO", "Standard (100%)"),
    ("RA-DPO", "Smart-30%"),
]
for a, b in pairs:
    p, a_only, b_only = mcnemar(data[a][2], data[b][2])
    # flag signficant
    flag = "  *" if p < 0.05 else ""
    print(f"  {a:>18} vs {b:<18}  p={p:.4f}  only_{a[:6]}_right={a_only}  only_{b[:6]}_right={b_only}{flag}")

# --- Direct disagreement count between 30% / 50% / 100% ---
print("\n=== Disagreement map (# posts where these 2 disagree) ===")
for a, b in [("Standard (100%)", "Smart-30%"), ("Standard (100%)", "Smart-50%"), ("Smart-30%", "Smart-50%")]:
    pa, pb = data[a][0], data[b][0]
    disagree = int((pa != pb).sum())
    print(f"  {a:>18} vs {b:<18}: disagree on {disagree}/692 = {disagree/692*100:.1f}%")

# --- Save JSON summary ---
out = {k: {"pairs": data[k][3], "f1_mean": m, "f1_ci_low": lo, "f1_ci_high": hi, "f1_se": sd}
       for k, (m, lo, hi, sd) in f1_rows.items()}
with open(ROOT / "results" / "unified_gpt4o" / "smart_curve_stats.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/unified_gpt4o/smart_curve_stats.json")
