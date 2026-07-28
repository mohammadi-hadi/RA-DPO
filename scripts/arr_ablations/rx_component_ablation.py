"""
Inference-time component ablation of the reliability score R(x).

Which component of R(x) = alpha*conf + beta*agree + gamma*(1 - sig) drives
selective prediction? For every saved model (gpt-4o track + local track) we
rank the 692 test items by each of seven scoring rules and report accuracy
at matched coverage plus AUARC:

  1. conf-only            confidence alone (= MaxProb baseline, no fitting)
  2. agree-only           true annotator agreement alone
  3. (1-sig)-only         inverted sigmoid token score alone
  4. conf+agree (OOF)     leave-one-out: logistic on 2 of 3 features,
  5. conf+(1-sig) (OOF)   5-fold stratified CV, same pattern as
  6. agree+(1-sig) (OOF)  stage_06_eval_oof.oof_rx()
  7. full R(x) (OOF)      all 3 features -- must reproduce paper numbers

Tie handling: agreements take only 4 values {0.5, 0.667, 0.833, 1.0}, so a
coverage cut usually lands inside a tie group. We compute EXPECTED accuracy
under uniform random tie-breaking analytically: all items strictly above the
cut value are taken fully; the boundary tie group contributes its mean
accuracy weighted by the fraction of the group needed to reach the target
coverage. The same tie-aware logic is applied to every scoring rule.

Alignment guard: before any computation we recompute agreement_score from
EXIST2023_training.json for the ids in
results/agreement_predictor_comparison/twitter-xlmr-base/test_order.json and
bit-match the result against the saved `agreements` array of every model
file. Any mismatch aborts the run.

Outputs:
  results/arr_ablations/rx_component_ablation/full_results.json
  results/arr_ablations/rx_component_ablation/summary.csv
  arr_revision/experiments/tables/tab_rx_ablation.tex
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

from src.data.data_loader import agreement_score  # noqa: E402

GPT_DIR = ROOT / "results" / "final_reliability_3factor"
LOCAL_DIR = ROOT / "results" / "local_pipeline" / "per_instance"
OUT_DIR = ROOT / "results" / "arr_ablations" / "rx_component_ablation"
TEX_DIR = ROOT / "arr_revision" / "experiments" / "tables"
TEST_ORDER = (
    ROOT / "results" / "agreement_predictor_comparison"
    / "twitter-xlmr-base" / "test_order.json"
)
TRAINING_JSON = ROOT / "EXIST2023_training.json"

REPORT_COVERAGES = [1.0, 0.9, 0.8, 0.6, 0.5]
AUARC_GRID = np.round(np.arange(0.05, 1.0 + 1e-9, 0.05), 2)  # 0.05 .. 1.00
N_FOLDS = 5
SEED = 42

# rule name -> feature column subset of [conf, agree, 1-sig]; None = raw score
RULES = {
    "conf_only": None,
    "agree_only": None,
    "one_minus_sig_only": None,
    "oof_conf_agree": [0, 1],
    "oof_conf_sig": [0, 2],
    "oof_agree_sig": [1, 2],
    "full_rx": [0, 1, 2],
}
FEATURE_NAMES = ["conf", "agree", "1-sig"]


# --------------------------------------------------------------------------
# Alignment guard
# --------------------------------------------------------------------------

def recompute_agreement_reference() -> np.ndarray:
    """Recompute agreement_score for the canonical test order from the raw
    EXIST 2023 training file. Must bit-match every saved agreements array."""
    order = json.load(open(TEST_ORDER))
    assert order["split"] == "test" and order["n"] == 692, "unexpected test_order"
    raw = json.load(open(TRAINING_JSON))
    id2labels = {str(k): v.get("labels_task1", []) for k, v in raw.items()}
    ref = np.array(
        [agreement_score(id2labels[str(r["id_or_idx"])]) for r in order["rows"]],
        dtype=float,
    )
    # cross-check against the agreement stored in test_order itself
    stored = np.array([r["agreement_true"] for r in order["rows"]], dtype=float)
    assert np.array_equal(ref, stored), "test_order agreement_true mismatch"
    return ref


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def oof_scores(X: np.ndarray, correct: np.ndarray, seed: int = SEED):
    """Out-of-fold linear reliability scores, identical pattern to
    scripts/local_pipeline/stage_06_eval_oof.oof_rx() but for any
    feature subset."""
    r = np.zeros(len(correct))
    ws = []
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, correct):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(
            sc.transform(X[tr]), correct[tr]
        )
        a = np.abs(lr.coef_[0])
        a = a / a.sum() if a.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])
        ws.append(a)
        r[te] = X[te] @ a
    return r, np.mean(ws, axis=0)


def expected_acc_at_coverage(r: np.ndarray, correct: np.ndarray, cov: float) -> float:
    """Expected accuracy of answering the top `cov` fraction ranked by r,
    under uniform random tie-breaking.

    Groups items by score value (descending). Full groups above the cut are
    taken entirely; the boundary tie group contributes fractionally: each of
    its items is included with equal probability, so its expected number of
    correct answers is (needed / group_size) * group_correct_sum.
    """
    n = len(r)
    k = cov * n  # real-valued target count
    if k <= 0:
        return float("nan")
    order = np.argsort(-r, kind="mergesort")
    r_sorted = r[order]
    c_sorted = correct[order].astype(float)
    # group boundaries: indices where score value changes
    vals, starts = np.unique(-r_sorted, return_index=True)  # ascending on -r
    starts = np.sort(starts)
    ends = np.append(starts[1:], n)
    expected_correct = 0.0
    taken = 0.0
    for s, e in zip(starts, ends):
        size = e - s
        group_correct = c_sorted[s:e].sum()
        if taken + size <= k:
            expected_correct += group_correct
            taken += size
        else:
            needed = k - taken
            expected_correct += (needed / size) * group_correct
            taken = k
            break
    return float(expected_correct / k)


def auarc(r: np.ndarray, correct: np.ndarray) -> float:
    """Area under the accuracy-coverage curve, trapezoid over AUARC_GRID."""
    accs = [expected_acc_at_coverage(r, correct, c) for c in AUARC_GRID]
    return float(np.trapezoid(accs, AUARC_GRID))


# --------------------------------------------------------------------------
# Per-model evaluation
# --------------------------------------------------------------------------

def evaluate_model(path: Path, ref_agree: np.ndarray) -> dict:
    d = json.load(open(path))
    pi = d["per_instance"]
    conf = np.asarray(pi["confidences"], dtype=float)
    agree = np.asarray(pi["agreements"], dtype=float)
    sig = np.asarray(pi["sigmoid_scores"], dtype=float)
    correct = np.asarray(pi["correct"], dtype=int)
    assert len(correct) == 692, f"{path.name}: n != 692"
    assert np.array_equal(agree, ref_agree), (
        f"{path.name}: agreements do not bit-match the recomputed reference -- "
        "instance order is broken, refusing to continue"
    )

    X = np.column_stack([conf, agree, 1.0 - sig])
    raw = {"conf_only": conf, "agree_only": agree, "one_minus_sig_only": 1.0 - sig}

    out = {}
    for rule, cols in RULES.items():
        if cols is None:
            r = raw[rule]
            weights = None
        else:
            r, w = oof_scores(X[:, cols], correct)
            weights = {FEATURE_NAMES[c]: float(wi) for c, wi in zip(cols, w)}
        entry = {
            "acc_at_coverage": {
                f"{cov:.2f}": expected_acc_at_coverage(r, correct, cov)
                for cov in REPORT_COVERAGES
            },
            "auarc": auarc(r, correct),
        }
        if weights is not None:
            entry["oof_mean_weights"] = weights
        out[rule] = entry
    return out


# --------------------------------------------------------------------------
# LaTeX table
# --------------------------------------------------------------------------

TEX_RULE_LABELS = {
    "conf_only": r"Conf-only (MaxProb)",
    "agree_only": r"Agree-only",
    "one_minus_sig_only": r"$(1-\mathrm{sig})$-only",
    "oof_conf_agree": r"Conf $+$ Agree",
    "oof_conf_sig": r"Conf $+$ $(1-\mathrm{sig})$",
    "oof_agree_sig": r"Agree $+$ $(1-\mathrm{sig})$",
    "full_rx": r"Full $R(x)$",
}
TEX_MODELS = [
    ("gpt-4o_base", "gpt-4o (base)"),
    ("gpt-4o_SFT", "gpt-4o (SFT)"),
    ("gpt-4o_Standard_DPO", "gpt-4o (Standard DPO)"),
    ("gpt-4o_RA-DPO", "gpt-4o (RA-DPO)"),
]


def write_latex(results: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        "",
        r"\caption{Inference-time ablation of the reliability score"
        r" $R(x)$. Test items are ranked by each scoring rule; we report"
        r" expected accuracy at matched coverage (uniform random"
        r" tie-breaking) and the area under the accuracy--coverage curve"
        r" (AUARC, trapezoid over coverage $0.05$--$1.0$). Two-feature and"
        r" full-$R(x)$ rules use out-of-fold weights (5-fold stratified"
        r" CV). Conf-only is the MaxProb baseline. Results for the 18"
        r" local-model runs (Llama-3.2-3B, Qwen2.5-3B) are provided in the"
        r" released JSON.}",
        r"\label{tab:rx_ablation}",
        "",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Scoring rule & @100 & @60 & @50 & AUARC \\",
    ]
    for key, label in TEX_MODELS:
        res = results[key]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % label)
        best50 = max(res[r]["acc_at_coverage"]["0.50"] for r in RULES)
        for rule in RULES:
            acc = res[rule]["acc_at_coverage"]
            a100, a60, a50 = acc["1.00"], acc["0.60"], acc["0.50"]
            au = res[rule]["auarc"]
            f50 = f"{a50:.3f}"
            if abs(a50 - best50) < 5e-4:
                f50 = r"\textbf{%s}" % f50
            lines.append(
                f"{TEX_RULE_LABELS[rule]} & {a100:.3f} & {a60:.3f} & {f50} & {au:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    ref_agree = recompute_agreement_reference()
    print(f"Alignment reference OK: 692 agreements recomputed from "
          f"{TRAINING_JSON.name}, values {sorted(set(np.round(ref_agree, 3)))}")

    model_files = {}
    for p in sorted(GPT_DIR.glob("*.json")):
        if p.name == "summary.json":
            continue
        model_files[p.stem] = p
    for p in sorted(LOCAL_DIR.glob("*.json")):
        model_files[p.stem] = p
    print(f"Evaluating {len(model_files)} models x {len(RULES)} scoring rules\n")

    results = {}
    rows = []
    for name, path in model_files.items():
        res = evaluate_model(path, ref_agree)
        results[name] = res
        for rule in RULES:
            acc = res[rule]["acc_at_coverage"]
            rows.append({
                "model": name,
                "rule": rule,
                **{f"acc@{int(float(c) * 100)}": v for c, v in acc.items()},
                "auarc": res[rule]["auarc"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    meta = {
        "n_test": 692,
        "coverages_reported": REPORT_COVERAGES,
        "auarc_grid": [float(c) for c in AUARC_GRID],
        "n_folds": N_FOLDS,
        "seed": SEED,
        "tie_breaking": "expected accuracy under uniform random tie-breaking",
    }
    with open(OUT_DIR / "full_results.json", "w") as f:
        json.dump({"meta": meta, "models": results}, f, indent=2)
    write_latex(results, TEX_DIR / "tab_rx_ablation.tex")

    # ---------------- summary table ----------------
    print(f"{'Model':<34} {'Rule':<20} {'@100':>7} {'@60':>7} {'@50':>7} {'AUARC':>7}")
    print("-" * 86)
    for name in model_files:
        for rule in RULES:
            acc = results[name][rule]["acc_at_coverage"]
            print(f"{name:<34} {rule:<20} {acc['1.00']:>7.3f} {acc['0.60']:>7.3f} "
                  f"{acc['0.50']:>7.3f} {results[name][rule]['auarc']:>7.3f}")
        print("-" * 86)

    # ---------------- self-check ----------------
    ra = results["gpt-4o_RA-DPO"]["full_rx"]
    a50 = ra["acc_at_coverage"]["0.50"]
    a100 = ra["acc_at_coverage"]["1.00"]
    w = ra["oof_mean_weights"]
    checks = [
        ("full R(x) acc@0.5 in 0.962 +/- 0.005", abs(a50 - 0.962) <= 0.005, a50),
        ("full R(x) acc@1.0 == 0.828", abs(a100 - 0.828) <= 0.0005, a100),
        ("OOF alpha ~ 0.15", abs(w["conf"] - 0.15) <= 0.03, w["conf"]),
        ("OOF beta ~ 0.83", abs(w["agree"] - 0.83) <= 0.03, w["agree"]),
        ("OOF gamma ~ 0.02", abs(w["1-sig"] - 0.02) <= 0.03, w["1-sig"]),
    ]
    print("\nSELF-CHECK (gpt-4o RA-DPO, full R(x), OOF):")
    ok = True
    for label, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}  (got {val:.4f})")
        ok &= passed
    print(f"\nSaved: {OUT_DIR / 'full_results.json'}")
    print(f"       {OUT_DIR / 'summary.csv'}")
    print(f"       {TEX_DIR / 'tab_rx_ablation.tex'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
