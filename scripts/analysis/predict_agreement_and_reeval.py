"""
Address the agreement-leakage concern.

The true annotator-agreement score at test time comes from the EXIST dataset's
6-annotator labels. In real deployment, we wouldn't have that. We re-score the
test set using a text → agreement regression model that was trained ONLY on
the training split's tweets. We then recompute the coverage-accuracy curve with
this predicted agreement in place of the true one.

Also computes a "no-agreement" baseline (α + γ only) as an extreme case.

Outputs:
  results/unified_gpt4o/predicted_agreement/*.json  (per-model OOF results)
  results/unified_gpt4o/predicted_agreement/comparison.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.models.agreement_predictor import AgreementPredictor

IN_DIR = ROOT / "results" / "final_reliability_3factor"
OUT_DIR = ROOT / "results" / "unified_gpt4o" / "predicted_agreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR_DIR = ROOT / "results" / "agreement_predictor"

MODELS = [
    ("gpt-4o (base)",           "gpt-4o_base.json"),
    ("gpt-4o (Random-50% DPO)", "gpt-4o_Random-50pct_DPO.json"),
    ("gpt-4o (Standard DPO)",   "gpt-4o_Standard_DPO.json"),
    ("gpt-4o (Smart-50% DPO)",  "gpt-4o_Smart-50pct_DPO.json"),
    ("gpt-4o (Smart-30% DPO)",  "gpt-4o_Smart-30pct_DPO.json"),
    ("gpt-4o (RA-DPO)",         "gpt-4o_RA-DPO.json"),
]
COV = [1.00, 0.90, 0.80, 0.60, 0.50]
N_FOLDS = 5
SEED = 42


def predict_agreement_for_test():
    """Load cached predicted agreements from the freshly trained predictor."""
    pred_file = OUT_DIR / "pred_agreement.npy"
    true_file = OUT_DIR / "true_agreement.npy"
    if not pred_file.exists():
        raise RuntimeError(
            "pred_agreement.npy missing — run scripts/analysis/train_agreement_predictor_fresh.py first"
        )
    pred = np.load(pred_file)
    true = np.load(true_file) if true_file.exists() else None

    # Sanity: verify order matches saved per_instance arrays
    ref = json.load(open(IN_DIR / "gpt-4o_base.json"))
    true_saved = np.asarray(ref["per_instance"]["agreements"], dtype=float)
    if true is None:
        true = true_saved
    assert np.allclose(true_saved, true), "true-agreement order mismatch"
    assert len(pred) == len(true_saved) == 692
    return pred, true_saved


def fit_w(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)
    a = np.abs(lr.coef_[0])
    s = a.sum()
    return (a / s) if s > 0 else np.ones(X.shape[1]) / X.shape[1]


def acc_top(r, correct, cov):
    k = max(1, int(round(len(r) * cov)))
    return float(correct[np.argsort(-r)[:k]].mean())


def oof_curve(feat, correct):
    r = np.zeros(len(correct))
    ws = []
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(feat, correct):
        w = fit_w(feat[tr], correct[tr])
        ws.append(w)
        r[te] = feat[te] @ w
    return r, np.mean(ws, axis=0)


def main():
    print("Predicting agreement on test set via mBERT regressor...")
    pred_agree, true_agree = predict_agreement_for_test()
    print(f"  true agreement : mean={true_agree.mean():.3f} std={true_agree.std():.3f}")
    print(f"  pred agreement : mean={pred_agree.mean():.3f} std={pred_agree.std():.3f}")
    print(f"  Pearson r(true,pred) = {np.corrcoef(true_agree, pred_agree)[0,1]:.3f}")

    rows = []
    for label, fname in MODELS:
        d = json.load(open(IN_DIR / fname))
        pi = d["per_instance"]
        conf = np.asarray(pi["confidences"], dtype=float)
        sig = np.asarray(pi["sigmoid_scores"], dtype=float)
        correct = np.asarray(pi["correct"], dtype=int)

        row = {"model": label}

        # Variant A: TRUE agreement (report's baseline)
        X = np.column_stack([conf, true_agree, 1 - sig])
        r_true, w_true = oof_curve(X, correct)
        for c in COV:
            row[f"true_acc@{int(c*100)}%"] = round(acc_top(r_true, correct, c), 4)
        row["alpha_true"], row["beta_true"], row["gamma_true"] = [round(float(x), 3) for x in w_true]

        # Variant B: PREDICTED agreement (deployment-realistic)
        X = np.column_stack([conf, pred_agree, 1 - sig])
        r_pred, w_pred = oof_curve(X, correct)
        for c in COV:
            row[f"pred_acc@{int(c*100)}%"] = round(acc_top(r_pred, correct, c), 4)
        row["alpha_pred"], row["beta_pred"], row["gamma_pred"] = [round(float(x), 3) for x in w_pred]

        # Variant C: NO agreement feature (conf + 1-sig only)
        X = np.column_stack([conf, 1 - sig])
        r_noagr, w_noagr = oof_curve(X, correct)
        for c in COV:
            row[f"noagr_acc@{int(c*100)}%"] = round(acc_top(r_noagr, correct, c), 4)
        row["alpha_noagr"], row["gamma_noagr"] = [round(float(x), 3) for x in w_noagr]

        rows.append(row)

        # Per-model save
        out = {
            "model": label,
            "true": {"weights": list(w_true.tolist()),
                     "acc_at_coverage": {f"{int(c*100)}%": acc_top(r_true, correct, c) for c in COV}},
            "predicted": {"weights": list(w_pred.tolist()),
                          "acc_at_coverage": {f"{int(c*100)}%": acc_top(r_pred, correct, c) for c in COV}},
            "no_agreement": {"weights": list(w_noagr.tolist()),
                             "acc_at_coverage": {f"{int(c*100)}%": acc_top(r_noagr, correct, c) for c in COV}},
        }
        with open(OUT_DIR / fname, "w") as f:
            json.dump(out, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison.csv", index=False)

    # Pretty print
    print("\n=== Coverage-Accuracy (true vs predicted vs none) ===")
    print(f"{'Model':<26} | {'@100%':>22} | {'@60%':>22} | {'@50%':>22}")
    print(f"{'':<26} | {'true | pred | none':>22} | {'true | pred | none':>22} | {'true | pred | none':>22}")
    print("-" * 100)
    for _, r in df.iterrows():
        cells = []
        for c in [100, 60, 50]:
            cells.append(f"{r[f'true_acc@{c}%']:.3f} | {r[f'pred_acc@{c}%']:.3f} | {r[f'noagr_acc@{c}%']:.3f}")
        print(f"{r['model']:<26} | {cells[0]:>22} | {cells[1]:>22} | {cells[2]:>22}")

    print("\n=== Weight shifts (alpha, beta, gamma) ===")
    print(f"{'Model':<26}  true α/β/γ                 pred α/β/γ")
    for _, r in df.iterrows():
        print(f"{r['model']:<26}  "
              f"{r['alpha_true']:.2f}/{r['beta_true']:.2f}/{r['gamma_true']:.2f}           "
              f"{r['alpha_pred']:.2f}/{r['beta_pred']:.2f}/{r['gamma_pred']:.2f}")

    # Save predicted agreements for reuse
    np.save(OUT_DIR / "pred_agreement.npy", pred_agree)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
