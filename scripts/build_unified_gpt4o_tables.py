"""
Build the unified gpt-4o / structured-prompt / OOF R(x) tables.

Single base model: gpt-4o.
Single prompt strategy: structured (best-on-average; used in Table 4 already).
Leakage-free R(x): 5-fold stratified CV.

Outputs:
  results/unified_gpt4o/fine_tuning.csv
  results/unified_gpt4o/coverage_accuracy.csv
  results/unified_gpt4o/weights.csv
  results/unified_gpt4o/summary.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "final_reliability_3factor"
OUT_DIR = ROOT / "results" / "unified_gpt4o"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# gpt-4o track only (structured prompt, EN+ES, 692 samples)
MODELS = [
    ("gpt-4o (base)",                "gpt-4o_base.json",                None),
    ("gpt-4o (SFT)",                 "gpt-4o_SFT.json",                 5535),
    ("gpt-4o (Smart-10% DPO)",       "gpt-4o_Smart-10pct_DPO.json",     535),
    ("gpt-4o (Smart-30% DPO)",       "gpt-4o_Smart-30pct_DPO.json",     1661),
    ("gpt-4o (Smart-50% DPO)",       "gpt-4o_Smart-50pct_DPO.json",     2768),
    ("gpt-4o (Random-50% DPO)",      "gpt-4o_Random-50pct_DPO.json",    2768),
    ("gpt-4o (Ambiguous-only DPO)",  "gpt-4o_Ambiguous_only_DPO.json",  665),
    ("gpt-4o (Standard DPO)",        "gpt-4o_Standard_DPO.json",        5536),
    ("gpt-4o (RA-DPO)",              "gpt-4o_RA-DPO.json",              8984),
]
COVERAGE = [1.00, 0.90, 0.80, 0.60, 0.50]
N_FOLDS = 5
SEED = 42


def fit_weights(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)
    a = np.abs(lr.coef_[0])
    s = a.sum()
    return (a / s) if s > 0 else np.array([1/3, 1/3, 1/3])


def oof_rx(conf, agree, sig, y):
    X = np.column_stack([conf, agree, 1 - sig])
    r = np.zeros(len(y))
    ws = []
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        w = fit_weights(X[tr], y[tr])
        ws.append(w)
        r[te] = w[0]*conf[te] + w[1]*agree[te] + w[2]*(1-sig[te])
    return r, np.mean(ws, axis=0)


def acc_top(r, correct, cov):
    k = max(1, int(round(len(r) * cov)))
    return float(correct[np.argsort(-r)[:k]].mean())


rows_ft = []
rows_cov = []
rows_w = []
summary = {"meta": {"base_model": "gpt-4o", "prompt_strategy": "structured",
                    "languages": "EN+ES", "n_test": 692, "n_folds": N_FOLDS},
           "models": {}}

for label, fname, pairs in MODELS:
    d = json.load(open(IN_DIR / fname))
    pi = d["per_instance"]
    conf = np.asarray(pi["confidences"], dtype=float)
    agree = np.asarray(pi["agreements"], dtype=float)
    sig = np.asarray(pi["sigmoid_scores"], dtype=float)
    correct = np.asarray(pi["correct"], dtype=int)
    sm = d["standard_metrics"]

    r_oof, w_mean = oof_rx(conf, agree, sig, correct)

    rows_ft.append({
        "method": label,
        "pairs": pairs if pairs is not None else "—",
        "f1_macro": round(sm["f1_macro"], 4),
        "accuracy": round(sm["accuracy"], 4),
        "precision": round(sm["precision_macro"], 4),
        "recall": round(sm["recall_macro"], 4),
    })

    row_cov = {"model": label}
    for c in COVERAGE:
        row_cov[f"acc@{int(c*100)}%"] = round(acc_top(r_oof, correct, c), 4)
    rows_cov.append(row_cov)

    rows_w.append({
        "model": label,
        "alpha (conf)": round(float(w_mean[0]), 3),
        "beta (agree)": round(float(w_mean[1]), 3),
        "gamma (tokens)": round(float(w_mean[2]), 3),
    })

    summary["models"][label] = {
        "pairs": pairs,
        "f1_macro": sm["f1_macro"],
        "accuracy": sm["accuracy"],
        "weights_oof": {"alpha": float(w_mean[0]), "beta": float(w_mean[1]), "gamma": float(w_mean[2])},
        "acc_at_coverage": {f"{int(c*100)}%": acc_top(r_oof, correct, c) for c in COVERAGE},
    }


df_ft = pd.DataFrame(rows_ft).sort_values("f1_macro")
df_cov = pd.DataFrame(rows_cov)
df_w = pd.DataFrame(rows_w)

df_ft.to_csv(OUT_DIR / "fine_tuning.csv", index=False)
df_cov.to_csv(OUT_DIR / "coverage_accuracy.csv", index=False)
df_w.to_csv(OUT_DIR / "weights.csv", index=False)
json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

# Pretty print
print("\n=== FINE-TUNING (gpt-4o, structured prompt, EN+ES, 692 samples) ===")
print(df_ft.to_string(index=False))
print("\n=== COVERAGE-ACCURACY (leakage-free 5-fold R(x)) ===")
print(df_cov.to_string(index=False))
print("\n=== LEARNED WEIGHTS (OOF mean) ===")
print(df_w.to_string(index=False))
print(f"\nSaved → {OUT_DIR}")
