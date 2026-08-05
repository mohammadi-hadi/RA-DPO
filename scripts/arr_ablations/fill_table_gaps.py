"""Fill the reporting gaps reviewers flagged for the RA-DPO paper.

Three independent analyses over the saved per-instance arrays (692-item
EXIST 2023 test split, fixed EXISTDataLoader order):

1. Predicted-agreement and no-agreement coverage rows for ALL nine gpt-4o
   variants (paper Table 4 only reported the predicted regime for Base,
   Smart-30, Standard and RA-DPO). Replicates
   scripts/analysis/predict_agreement_and_reeval.py exactly:
     - predicted agreement  = results/unified_gpt4o/predicted_agreement/pred_agreement.npy
     - no-agreement         = 5-fold OOF logistic on [conf, 1-sig] only
   Reports accuracy @100/@60/@50 per regime.

2. Bootstrap 95% CI for the Ambiguous-only DPO macro-F1 (Table 3 currently
   says n/a). Gold labels are reconstructed exactly as in
   scripts/analysis/smart_curve_stats.py: gold = prediction if correct else
   the flipped binary label. 10000 resamples.

3. Per-language quality of the deployable agreement regressor
   (twitter-xlmr-base): Pearson r, Spearman rho and MAE overall and per
   language (en/es), joining lang metadata from EXIST2023_training.json
   after a bit-level alignment check on the recomputed agreement scores.

Outputs:
  results/arr_ablations/table_gaps/table4_missing_rows.csv
  results/arr_ablations/table_gaps/ambiguous_ci.json
  results/arr_ablations/table_gaps/regressor_by_lang.json
  arr_revision/experiments/tables/tab_coverage_complete.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import agreement_score  # noqa: E402

IN_DIR = ROOT / "results" / "final_reliability_3factor"
PRED_AGREE_FILE = ROOT / "results" / "unified_gpt4o" / "predicted_agreement" / "pred_agreement.npy"
REGRESSOR_DIR = ROOT / "results" / "agreement_predictor_comparison" / "twitter-xlmr-base"
TRAINING_JSON = ROOT / "EXIST2023_training.json"

OUT_DIR = ROOT / "results" / "arr_ablations" / "table_gaps"
TEX_DIR = ROOT / "arr_revision" / "experiments" / "tables"

# All nine gpt-4o variants, in Table-4 order.
MODELS = [
    ("Base",         "gpt-4o_base.json"),
    ("SFT",          "gpt-4o_SFT.json"),
    ("Smart-10",     "gpt-4o_Smart-10pct_DPO.json"),
    ("Smart-30",     "gpt-4o_Smart-30pct_DPO.json"),
    ("Smart-50",     "gpt-4o_Smart-50pct_DPO.json"),
    ("Random-50",    "gpt-4o_Random-50pct_DPO.json"),
    ("Standard DPO", "gpt-4o_Standard_DPO.json"),
    ("Ambiguous",    "gpt-4o_Ambiguous_only_DPO.json"),
    ("RA-DPO",       "gpt-4o_RA-DPO.json"),
]
# Variants whose predicted/no-agreement rows already appear in paper Table 4.
ALREADY_IN_TABLE4 = {"Base", "Smart-30", "Standard DPO", "RA-DPO"}
COV = [1.00, 0.60, 0.50]
N_FOLDS = 5
SEED = 42
N_BOOT = 10000
LABELS = ["NO", "YES"]


# ---------------------------------------------------------------------------
# Shared OOF R(x) machinery (identical to predict_agreement_and_reeval.py)
# ---------------------------------------------------------------------------
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


def load_per_instance(fname):
    d = json.load(open(IN_DIR / fname))
    pi = d["per_instance"]
    return {
        "predictions": np.asarray(pi["predictions"]),
        "conf": np.asarray(pi["confidences"], dtype=float),
        "agree": np.asarray(pi["agreements"], dtype=float),
        "sig": np.asarray(pi["sigmoid_scores"], dtype=float),
        "correct": np.asarray(pi["correct"], dtype=int),
    }


# ---------------------------------------------------------------------------
# Part 1 — full coverage table (true / predicted / no-agreement)
# ---------------------------------------------------------------------------
def coverage_table():
    pred_agree = np.load(PRED_AGREE_FILE)
    assert len(pred_agree) == 692, "pred_agreement.npy must cover the 692-item test set"

    rows = []
    for label, fname in MODELS:
        pi = load_per_instance(fname)
        conf, sig, correct = pi["conf"], pi["sig"], pi["correct"]
        true_agree = pi["agree"]

        regimes = {
            "True":    np.column_stack([conf, true_agree, 1 - sig]),
            "Pred.":   np.column_stack([conf, pred_agree, 1 - sig]),
            "No-agr.": np.column_stack([conf, 1 - sig]),
        }
        for regime, X in regimes.items():
            r, w = oof_curve(X, correct)
            row = {
                "model": label,
                "regime": regime,
                "new_in_table4": (regime != "True") and (label not in ALREADY_IN_TABLE4),
            }
            for c in COV:
                row[f"acc@{int(c * 100)}"] = round(acc_top(r, correct, c), 4)
            row["weights"] = "/".join(f"{x:.3f}" for x in w)
            rows.append(row)
    return pd.DataFrame(rows)


def write_latex(df):
    """Full Table-4 replacement: 9 variants x 3 regimes x @100/@60/@50."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        "",
        r"\caption{Accuracy at matched coverage under three agreement settings"
        r" (True/Pred./No-agr.), reported for every gpt-4o variant."
        r" Reliability weights are refit per model and regime with 5-fold"
        r" out-of-fold logistic regression.}",
        r"\label{tab:coverage-complete}",
        "",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Setting & @100 & @60 & @50 \\",
        r"\midrule",
    ]
    for mi, (label, _) in enumerate(MODELS):
        sub = df[df["model"] == label]
        for ri, (_, r) in enumerate(sub.iterrows()):
            name = label if ri == 0 else ""
            cells = " & ".join(f"{r[f'acc@{c}']:.3f}" for c in [100, 60, 50])
            setting = r["regime"]
            if label == "RA-DPO" and setting == "Pred.":
                setting = r"\textbf{Pred.}"
                cells = " & ".join(rf"\textbf{{{r[f'acc@{c}']:.3f}}}" for c in [100, 60, 50])
            lines.append(f"{name:<12} & {setting:<9} & {cells} \\\\")
        if mi < len(MODELS) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 2 — Ambiguous-only bootstrap CI
# ---------------------------------------------------------------------------
def ambiguous_ci():
    pi = load_per_instance("gpt-4o_Ambiguous_only_DPO.json")
    preds, correct = pi["predictions"], pi["correct"]
    # Binary task: gold label is fully recoverable from prediction + correct.
    true = np.where(correct == 1, preds, np.where(preds == "YES", "NO", "YES"))

    f1 = f1_score(true, preds, average="macro", labels=LABELS)

    rng = np.random.default_rng(SEED)
    n = len(true)
    scores = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        scores[b] = f1_score(true[idx], preds[idx], average="macro", labels=LABELS)

    return {
        "model": "gpt-4o (Ambiguous-only DPO)",
        "n_test": int(n),
        "n_resamples": N_BOOT,
        "seed": SEED,
        "f1_macro": float(f1),
        "f1_boot_mean": float(scores.mean()),
        "f1_ci_low": float(np.quantile(scores, 0.025)),
        "f1_ci_high": float(np.quantile(scores, 0.975)),
        "f1_se": float(scores.std()),
    }


# ---------------------------------------------------------------------------
# Part 3 — per-language regressor quality
# ---------------------------------------------------------------------------
def regressor_by_lang():
    pred = np.load(REGRESSOR_DIR / "predictions_test.npy").astype(float)
    targ = np.load(REGRESSOR_DIR / "targets_test.npy").astype(float)
    order = json.load(open(REGRESSOR_DIR / "test_order.json"))
    ids = [r["id_or_idx"] for r in order["rows"]]
    assert len(ids) == len(pred) == len(targ) == 692

    # Alignment check: recompute agreement from the raw dataset for these ids
    # and bit-match against both the regressor targets and the saved arrays.
    raw = json.load(open(TRAINING_JSON))
    recomputed = np.array([agreement_score(raw[i]["labels_task1"]) for i in ids])
    langs = np.array([raw[i]["lang"] for i in ids])
    saved_agree = load_per_instance("gpt-4o_base.json")["agree"]
    assert np.array_equal(recomputed, targ), "test_order ids misaligned with targets_test.npy"
    assert np.array_equal(recomputed, saved_agree), "test_order ids misaligned with saved agreements"

    def stats(mask):
        p, t = pred[mask], targ[mask]
        r, r_p = pearsonr(p, t)
        rho, rho_p = spearmanr(p, t)
        return {
            "n": int(mask.sum()),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "mae": float(np.abs(p - t).mean()),
        }

    return {
        "regressor": "cardiffnlp/twitter-xlm-roberta-base",
        "overall": stats(np.ones(len(pred), dtype=bool)),
        "en": stats(langs == "en"),
        "es": stats(langs == "es"),
    }


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- Part 1 ---
    df = coverage_table()
    df.to_csv(OUT_DIR / "table4_missing_rows.csv", index=False)
    tex = write_latex(df)
    (TEX_DIR / "tab_coverage_complete.tex").write_text(tex)

    print("=== Coverage @100/@60/@50 per regime (new Table-4 rows marked *) ===")
    print(f"{'Model':<14} {'Regime':<8} {'@100':>6} {'@60':>6} {'@50':>6}   weights")
    for _, r in df.iterrows():
        star = "*" if r["new_in_table4"] else " "
        print(f"{r['model']:<14} {r['regime']:<8} {r['acc@100']:>6.3f} {r['acc@60']:>6.3f} "
              f"{r['acc@50']:>6.3f} {star} {r['weights']}")

    # --- Part 2 ---
    ci = ambiguous_ci()
    with open(OUT_DIR / "ambiguous_ci.json", "w") as f:
        json.dump(ci, f, indent=2)
    print("\n=== Ambiguous-only DPO bootstrap CI ===")
    print(f"F1-macro = {ci['f1_macro']:.4f}  "
          f"95% CI [{ci['f1_ci_low']:.4f}, {ci['f1_ci_high']:.4f}]  SE {ci['f1_se']:.4f}")

    # --- Part 3 ---
    reg = regressor_by_lang()
    with open(OUT_DIR / "regressor_by_lang.json", "w") as f:
        json.dump(reg, f, indent=2)
    print("\n=== Agreement regressor (twitter-xlmr-base) by language ===")
    print(f"{'Split':<8} {'n':>4} {'Pearson r':>10} {'Spearman':>10} {'MAE':>8}")
    for k in ["overall", "en", "es"]:
        s = reg[k]
        print(f"{k:<8} {s['n']:>4} {s['pearson_r']:>10.4f} {s['spearman_rho']:>10.4f} {s['mae']:>8.4f}")

    # --- SELF-CHECK ---
    def get(model, regime, col):
        return float(df[(df["model"] == model) & (df["regime"] == regime)][col].iloc[0])

    checks = {
        "RA-DPO pred@50 ~ 0.887": abs(get("RA-DPO", "Pred.", "acc@50") - 0.887) <= 0.005,
        "RA-DPO noagr@50 ~ 0.853": abs(get("RA-DPO", "No-agr.", "acc@50") - 0.853) <= 0.005,
        "SFT true@50 ~ 0.965": abs(get("SFT", "True", "acc@50") - 0.965) <= 0.005,
        "Ambiguous F1 ~ 0.653": abs(ci["f1_macro"] - 0.653) <= 0.005,
        "overall Pearson r ~ 0.3508": abs(reg["overall"]["pearson_r"] - 0.3508) <= 0.005,
    }
    print("\n=== SELF-CHECK ===")
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    if not ok:
        sys.exit(1)
    print("All self-checks passed.")


if __name__ == "__main__":
    main()
