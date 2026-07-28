"""
Fix the data-leakage issue in R(x) weight optimization.

The report fits logistic regression for alpha/beta/gamma on the SAME test set
on which coverage-accuracy is reported. We replace this with 5-fold
cross-validation so each instance's R(x) is scored by weights learned on
OTHER instances. This gives an unbiased coverage-accuracy curve.

Outputs:
  results/leakage_fix/<model>.json    per-model OOF results
  results/leakage_fix/summary.json    comparison (old vs OOF)
  results/leakage_fix/comparison.csv  human-readable table
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
OUT_DIR = ROOT / "results" / "leakage_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "gpt-4o (base)": "gpt-4o_base.json",
    "gpt-4o-mini (base)": "gpt-4o-mini_base.json",
    "gpt-4o-mini (SFT)": "gpt-4o-mini_SFT.json",
    "gpt-4o (Std DPO)": "gpt-4o_Standard_DPO.json",
    "gpt-4o (RA-DPO)": "gpt-4o_RA-DPO.json",
    "gpt-4o (Smart-30%)": "gpt-4o_Smart-30pct_DPO.json",
    "gpt-4o (Smart-50%)": "gpt-4o_Smart-50pct_DPO.json",
    "gpt-4o (Random-50%)": "gpt-4o_Random-50pct_DPO.json",
}

COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.60, 0.50]
N_FOLDS = 5
SEED = 42


def build_features(conf: np.ndarray, agree: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return np.column_stack([conf, agree, 1.0 - sig])


def fit_normalized_weights(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, LogisticRegression, StandardScaler]:
    """Fit LR, return normalized [alpha, beta, gamma]. Matches the report
    method: take abs of coefs, divide by sum. If all coefs are negative /
    non-informative, fall back to equal weights."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(Xs, y)
    raw = lr.coef_[0]
    abs_coefs = np.abs(raw)
    total = abs_coefs.sum()
    w = abs_coefs / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
    return w, lr, scaler


def score_rx(weights: np.ndarray, conf: np.ndarray, agree: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return weights[0] * conf + weights[1] * agree + weights[2] * (1.0 - sig)


def acc_at_coverage(r_scores: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy when we answer the top `coverage` fraction by R(x)."""
    n = len(r_scores)
    k = max(1, int(round(n * coverage)))
    # top-k by R(x) descending
    idx = np.argsort(-r_scores)[:k]
    return float(correct[idx].mean())


def evaluate_model(model_label: str, path: Path) -> dict:
    d = json.load(open(path))
    pi = d["per_instance"]
    conf = np.asarray(pi["confidences"], dtype=float)
    agree = np.asarray(pi["agreements"], dtype=float)
    sig = np.asarray(pi["sigmoid_scores"], dtype=float)
    correct = np.asarray(pi["correct"], dtype=int)

    # --- OOF R(x) via 5-fold CV ---
    X = build_features(conf, agree, sig)
    y = correct
    oof_r = np.zeros(len(y))
    fold_weights = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in skf.split(X, y):
        w, _, _ = fit_normalized_weights(X[tr_idx], y[tr_idx])
        fold_weights.append(w)
        oof_r[te_idx] = score_rx(w, conf[te_idx], agree[te_idx], sig[te_idx])

    mean_w = np.mean(fold_weights, axis=0)

    # --- Full-data (leaky) R(x) — reproduces the report's setup ---
    w_full, _, _ = fit_normalized_weights(X, y)
    leaky_r = score_rx(w_full, conf, agree, sig)

    # Coverage-accuracy for both
    results = {
        "model": model_label,
        "n": int(len(y)),
        "base_accuracy": float(correct.mean()),  # acc @ 100%
        "weights_leaky": {
            "alpha": float(w_full[0]),
            "beta": float(w_full[1]),
            "gamma": float(w_full[2]),
        },
        "weights_oof_mean": {
            "alpha": float(mean_w[0]),
            "beta": float(mean_w[1]),
            "gamma": float(mean_w[2]),
        },
        "acc_at_coverage_leaky": {},
        "acc_at_coverage_oof": {},
    }
    for cov in COVERAGE_LEVELS:
        results["acc_at_coverage_leaky"][f"{int(cov*100)}%"] = acc_at_coverage(leaky_r, correct, cov)
        results["acc_at_coverage_oof"][f"{int(cov*100)}%"] = acc_at_coverage(oof_r, correct, cov)

    return results


def main() -> None:
    print(f"Running 5-fold cross-validated R(x) on {len(MODEL_FILES)} models...")
    all_results = {}
    rows = []
    for label, fname in MODEL_FILES.items():
        path = IN_DIR / fname
        if not path.exists():
            print(f"  SKIP {label}: {path} missing")
            continue
        r = evaluate_model(label, path)
        all_results[label] = r
        with open(OUT_DIR / fname, "w") as f:
            json.dump(r, f, indent=2)

        leaky = r["acc_at_coverage_leaky"]
        oof = r["acc_at_coverage_oof"]
        rows.append({
            "model": label,
            "n": r["n"],
            "alpha_leaky": r["weights_leaky"]["alpha"],
            "beta_leaky": r["weights_leaky"]["beta"],
            "gamma_leaky": r["weights_leaky"]["gamma"],
            "alpha_oof": r["weights_oof_mean"]["alpha"],
            "beta_oof": r["weights_oof_mean"]["beta"],
            "gamma_oof": r["weights_oof_mean"]["gamma"],
            **{f"leaky@{k}": v for k, v in leaky.items()},
            **{f"oof@{k}": v for k, v in oof.items()},
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print human-readable table
    print("\n" + "=" * 110)
    print(f"{'Model':<24}  {'100%':>18}  {'90%':>18}  {'80%':>18}  {'60%':>18}  {'50%':>18}")
    print(f"{'':<24}  {'leaky -> OOF':>18}  {'leaky -> OOF':>18}  {'leaky -> OOF':>18}  {'leaky -> OOF':>18}  {'leaky -> OOF':>18}")
    print("-" * 130)
    for label, r in all_results.items():
        parts = [f"{label:<24}"]
        for cov in COVERAGE_LEVELS:
            k = f"{int(cov*100)}%"
            l = r["acc_at_coverage_leaky"][k]
            o = r["acc_at_coverage_oof"][k]
            diff = (o - l) * 100
            parts.append(f"{l:.3f}->{o:.3f} ({diff:+.1f}pp)".rjust(18))
        print("  ".join(parts))

    print("\nWeights (leaky vs OOF-mean):")
    print(f"{'Model':<24}  {'alpha':>16}  {'beta':>16}  {'gamma':>16}")
    for label, r in all_results.items():
        wl, wo = r["weights_leaky"], r["weights_oof_mean"]
        print(f"{label:<24}  {wl['alpha']:.3f}->{wo['alpha']:.3f}       {wl['beta']:.3f}->{wo['beta']:.3f}       {wl['gamma']:.3f}->{wo['gamma']:.3f}")

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
