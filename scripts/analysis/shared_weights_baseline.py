"""Shared-weights R(x) baseline.

Instead of giving each model its own α, β, γ, fit ONE universal set on the
pooled (conf, agree, 1-sig) → correct data across all gpt-4o variants, then
apply those same weights to every model.

Purpose: show whether RA-DPO's R(x) advantage comes from (a) having a better
confidence signal, or (b) getting more favorable weights.

Three settings:
  - Per-model OOF weights (current paper method)
  - Shared weights (pooled 5-fold OOF: fit on pooled train folds, apply to all)
  - Uniform weights (α=β=γ=1/3) — crude baseline
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

MODELS = [
    ("gpt-4o (base)",           "gpt-4o_base.json"),
    ("gpt-4o (Random-50% DPO)", "gpt-4o_Random-50pct_DPO.json"),
    ("gpt-4o (Standard DPO)",   "gpt-4o_Standard_DPO.json"),
    ("gpt-4o (Smart-50% DPO)",  "gpt-4o_Smart-50pct_DPO.json"),
    ("gpt-4o (Smart-30% DPO)",  "gpt-4o_Smart-30pct_DPO.json"),
    ("gpt-4o (RA-DPO)",         "gpt-4o_RA-DPO.json"),
]
COV = [1.00, 0.90, 0.80, 0.60, 0.50]
SEED = 42


def fit_w(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)
    a = np.abs(lr.coef_[0]); s = a.sum()
    return (a / s) if s > 0 else np.ones(X.shape[1]) / X.shape[1]


def acc_top(r, correct, cov):
    k = max(1, int(round(len(r) * cov)))
    return float(correct[np.argsort(-r)[:k]].mean())


def main():
    # Load all models' per-instance arrays
    all_data = {}
    for label, fname in MODELS:
        d = json.load(open(IN_DIR / fname))
        pi = d["per_instance"]
        all_data[label] = {
            "conf":  np.asarray(pi["confidences"], dtype=float),
            "agree": np.asarray(pi["agreements"], dtype=float),
            "sig":   np.asarray(pi["sigmoid_scores"], dtype=float),
            "correct": np.asarray(pi["correct"], dtype=int),
        }

    n_models = len(all_data)
    n_per = 692
    # Pool features + model identifier + correct
    X_pool = []
    y_pool = []
    model_idx = []
    for mi, (label, a) in enumerate(all_data.items()):
        X = np.column_stack([a["conf"], a["agree"], 1 - a["sig"]])
        X_pool.append(X)
        y_pool.append(a["correct"])
        model_idx.append(np.full(n_per, mi))
    X_pool = np.vstack(X_pool)
    y_pool = np.concatenate(y_pool)
    model_idx = np.concatenate(model_idx)

    print(f"Pooled data: X={X_pool.shape}, y={y_pool.shape}")

    # --- Variant A: shared OOF weights via group-aware 5-fold CV ---
    # For fairness, we split by (model, fold) so every instance gets OOF weights
    # fit on ALL models' other folds.
    r_shared = np.zeros(n_models * n_per)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_weights = []
    for tr, te in skf.split(X_pool, y_pool):
        w = fit_w(X_pool[tr], y_pool[tr])
        fold_weights.append(w)
        r_shared[te] = X_pool[te] @ w
    shared_mean_w = np.mean(fold_weights, axis=0)

    # --- Variant B: uniform weights ---
    uniform_w = np.array([1/3, 1/3, 1/3])
    r_uniform = X_pool @ uniform_w

    # --- Variant C: per-model OOF (reference: matches our unified tables) ---
    r_per = np.zeros(n_models * n_per)
    per_model_weights = {}
    for mi, (label, a) in enumerate(all_data.items()):
        X = np.column_stack([a["conf"], a["agree"], 1 - a["sig"]])
        ws = []
        for tr, te in skf.split(X, a["correct"]):
            w = fit_w(X[tr], a["correct"][tr])
            ws.append(w)
            start = mi * n_per + te
            r_per[start] = X[te] @ w
        per_model_weights[label] = np.mean(ws, axis=0)

    # Compile results
    rows = []
    for mi, (label, a) in enumerate(all_data.items()):
        mask = model_idx == mi
        correct = a["correct"]
        row = {"model": label}
        for tag, rs in [("per", r_per[mask]), ("shared", r_shared[mask]), ("uniform", r_uniform[mask])]:
            for c in COV:
                row[f"{tag}_acc@{int(c*100)}%"] = round(acc_top(rs, correct, c), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "shared_weights_comparison.csv", index=False)

    # Print summary
    print(f"\n=== Shared-weights (OOF, pooled across all 6 models) ===")
    print(f"  α = {shared_mean_w[0]:.3f}   β = {shared_mean_w[1]:.3f}   γ = {shared_mean_w[2]:.3f}")

    print(f"\n=== Coverage-accuracy: per-model / shared / uniform ===")
    print(f"{'Model':<25}  {'@100%':>22}  {'@60%':>22}  {'@50%':>22}")
    print(f"{'':<25}  {'per | shared | uniform':>22}  {'per | shared | uniform':>22}  {'per | shared | uniform':>22}")
    for _, r in df.iterrows():
        cells = []
        for c in [100, 60, 50]:
            p = r[f"per_acc@{c}%"]; s = r[f"shared_acc@{c}%"]; u = r[f"uniform_acc@{c}%"]
            cells.append(f"{p:.3f} | {s:.3f} | {u:.3f}")
        print(f"{r['model']:<25}  {cells[0]:>22}  {cells[1]:>22}  {cells[2]:>22}")

    print(f"\n=== Per-model weights vs shared ===")
    print(f"{'Model':<25}  {'α':>12}  {'β':>12}  {'γ':>12}")
    for label, w in per_model_weights.items():
        print(f"{label:<25}  {w[0]:>12.3f}  {w[1]:>12.3f}  {w[2]:>12.3f}")
    print(f"{'SHARED (pooled OOF)':<25}  {shared_mean_w[0]:>12.3f}  {shared_mean_w[1]:>12.3f}  {shared_mean_w[2]:>12.3f}")

    # Save to JSON
    out = {
        "shared_weights_oof": {"alpha": float(shared_mean_w[0]), "beta": float(shared_mean_w[1]), "gamma": float(shared_mean_w[2])},
        "uniform_weights": [1/3, 1/3, 1/3],
        "per_model_weights_oof": {k: v.tolist() for k, v in per_model_weights.items()},
        "results": df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "shared_weights_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
