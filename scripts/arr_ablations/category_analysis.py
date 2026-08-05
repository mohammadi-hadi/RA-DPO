"""Sexism-category analysis of R(x)-gated abstention (ARR reviewer request).

Breaks the 692-item EXIST 2023 test split down by sexism type and shows,
for gpt-4o {base, RA-DPO} and llama32_3b ra_dpo, what the reliability gate
does to each category at 50% coverage:

  1. Align test-split ids (test_order.json) with EXIST2023_training.json.
     MANDATORY alignment proof: agreement_score recomputed from the raw
     annotations must bit-match the saved `agreements` array
     (max abs diff < 1e-9) before any metadata is joined.
  2. Per item derive:
       - task2 bucket: plurality label among annotators who gave a non-'-'
         task2 label (DIRECT / REPORTED / JUDGEMENTAL); empty or tied
         plurality -> 'NONE'. The rare 'UNKNOWN' vote (annotator marked
         sexist but could not judge intention; 2 test items affected) is
         treated as a non-vote like '-'.
       - task3 bucket: most frequent category across all annotators'
         (flattened) task3 lists, ignoring '-' and 'UNKNOWN' (4 test items
         carry UNKNOWN votes); ties -> alphabetical first; all '-' -> 'NONE'.
  3. Per bucket (task2, task3, language) and per model report:
       n, full-coverage accuracy, accuracy on the retained half at 50%
       coverage under the full OOF R(x), and the abstention rate
       (fraction of the bucket falling in the abstained half).

Outputs:
  results/arr_ablations/category_analysis/category_results.json
  results/arr_ablations/category_analysis/category_table.csv
  arr_revision/experiments/tables/tab_category_analysis.tex
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import agreement_score  # noqa: E402

TEST_ORDER = ROOT / "results" / "agreement_predictor_comparison" / "twitter-xlmr-base" / "test_order.json"
RAW_JSON = ROOT / "EXIST2023_training.json"
MODELS = {
    "gpt-4o_base": ROOT / "results" / "final_reliability_3factor" / "gpt-4o_base.json",
    "gpt-4o_RA-DPO": ROOT / "results" / "final_reliability_3factor" / "gpt-4o_RA-DPO.json",
    "llama32_3b_ra_dpo": ROOT / "results" / "local_pipeline" / "per_instance" / "llama32_3b_ra_dpo_local.json",
}
ALIGN_REF = ROOT / "results" / "final_reliability_3factor" / "gpt-4o_RA-DPO.json"

OUT_DIR = ROOT / "results" / "arr_ablations" / "category_analysis"
TEX_PATH = ROOT / "arr_revision" / "experiments" / "tables" / "tab_category_analysis.tex"
COVERAGE = 0.50


# ---------------------------------------------------------------- OOF R(x)
def oof_rx(conf, agree, sig, correct, seed=42):
    """5-fold OOF R(x) — identical to scripts/local_pipeline/stage_06_eval_oof.py."""
    X = np.column_stack([conf, agree, 1 - sig])
    r = np.zeros(len(correct))
    ws = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, correct):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), correct[tr])
        a = np.abs(lr.coef_[0]); a = a / a.sum()
        ws.append(a)
        r[te] = X[te] @ a
    return r, np.mean(ws, axis=0)


def retained_mask(r: np.ndarray, coverage: float) -> np.ndarray:
    """Top-k retained set by R(x); stable sort keeps ties deterministic."""
    k = max(1, int(round(len(r) * coverage)))
    mask = np.zeros(len(r), dtype=bool)
    mask[np.argsort(-r, kind="stable")[:k]] = True
    return mask


# ---------------------------------------------------------------- buckets
NON_VOTES = {"-", "UNKNOWN"}


def task2_bucket(labels: list[str]) -> str:
    """Plurality task2 label among non-'-' votes; empty or tie -> NONE."""
    votes = [l for l in labels if l not in NON_VOTES]
    if not votes:
        return "NONE"
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "NONE"  # no majority
    return counts[0][0]


def task3_bucket(label_lists: list[list[str]]) -> str:
    """Most frequent task3 category (flattened, '-' ignored); ties -> alphabetical."""
    flat = [l for lst in label_lists for l in lst if l not in NON_VOTES]
    if not flat:
        return "NONE"
    counts = Counter(flat)
    best = max(counts.values())
    return sorted(c for c, n in counts.items() if n == best)[0]


# ---------------------------------------------------------------- analysis
def bucket_stats(buckets: pd.Series, correct: np.ndarray, retained: np.ndarray) -> list[dict]:
    rows = []
    for b in sorted(buckets.unique()):
        idx = (buckets == b).to_numpy()
        n = int(idx.sum())
        kept = idx & retained
        rows.append({
            "bucket": b,
            "n": n,
            "acc_full": round(float(correct[idx].mean()), 4),
            "acc_at_50_retained": round(float(correct[kept].mean()), 4) if kept.any() else None,
            "n_retained": int(kept.sum()),
            "abstain_rate": round(float(1.0 - kept.sum() / n), 4),
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: alignment proof -----------------------------------------
    order = json.load(open(TEST_ORDER))
    assert order["split"] == "test" and order["n"] == 692
    ids = [row["id_or_idx"] for row in order["rows"]]
    raw = json.load(open(RAW_JSON))

    recomputed = np.array([agreement_score(raw[i]["labels_task1"]) for i in ids])
    ref = json.load(open(ALIGN_REF))
    saved = np.array(ref["per_instance"]["agreements"], dtype=float)
    max_diff = float(np.max(np.abs(recomputed - saved)))
    print(f"[alignment] recomputed vs saved agreements: max abs diff = {max_diff:.2e}")
    assert max_diff < 1e-9, f"ALIGNMENT FAILED: max abs diff {max_diff}"
    print("[alignment] PASSED (< 1e-9) — safe to join metadata\n")

    # ---- Step 2: metadata buckets ----------------------------------------
    meta = pd.DataFrame({
        "id": ids,
        "lang": [raw[i]["lang"].upper() for i in ids],
        "task2": [task2_bucket(raw[i]["labels_task2"]) for i in ids],
        "task3": [task3_bucket(raw[i]["labels_task3"]) for i in ids],
    })
    print("[buckets] task2:", dict(meta["task2"].value_counts()))
    print("[buckets] task3:", dict(meta["task3"].value_counts()))
    print("[buckets] lang :", dict(meta["lang"].value_counts()), "\n")

    # ---- Step 3: per-model per-bucket stats ------------------------------
    results = {"alignment_max_abs_diff": max_diff, "coverage": COVERAGE, "models": {}}
    csv_rows = []
    for label, path in MODELS.items():
        d = json.load(open(path))
        pi = d["per_instance"]
        conf = np.array(pi["confidences"], dtype=float)
        agree = np.array(pi["agreements"], dtype=float)
        sig = np.array(pi["sigmoid_scores"], dtype=float)
        correct = np.array(pi["correct"], dtype=int)
        assert len(correct) == 692
        # every track shares the same agreements array (invariant I-bitmatch)
        assert float(np.max(np.abs(agree - saved))) < 1e-9, f"{label}: agreements drift"

        r, w = oof_rx(conf, agree, sig, correct)
        kept = retained_mask(r, COVERAGE)
        overall_50 = float(correct[kept].mean())
        print(f"=== {label}  (acc@100={correct.mean():.4f}, acc@50={overall_50:.4f}, "
              f"OOF w={np.round(w, 3).tolist()})")

        model_out = {
            "acc_full": round(float(correct.mean()), 4),
            "acc_at_50": round(overall_50, 4),
            "oof_weights_mean": {"alpha": round(float(w[0]), 4),
                                 "beta": round(float(w[1]), 4),
                                 "gamma": round(float(w[2]), 4)},
        }
        for dim in ("task2", "task3", "lang"):
            rows = bucket_stats(meta[dim], correct, kept)
            model_out[dim] = rows
            hdr = f"{'bucket':<30}{'n':>5}{'acc@100':>9}{'acc@50':>9}{'abstain%':>10}"
            print(f"-- {dim}\n{hdr}")
            for rw in rows:
                a50 = f"{rw['acc_at_50_retained']:.3f}" if rw["acc_at_50_retained"] is not None else "  n/a"
                print(f"{rw['bucket']:<30}{rw['n']:>5}{rw['acc_full']:>9.3f}{a50:>9}"
                      f"{100 * rw['abstain_rate']:>9.1f}%")
                csv_rows.append({"model": label, "dimension": dim, **rw})
            print()
        results["models"][label] = model_out

    # ---- Step 4: outputs --------------------------------------------------
    with open(OUT_DIR / "category_results.json", "w") as f:
        json.dump(results, f, indent=2)
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUT_DIR / "category_table.csv", index=False)

    write_latex(results["models"]["gpt-4o_RA-DPO"]["task3"])
    print(f"[out] {OUT_DIR / 'category_results.json'}")
    print(f"[out] {OUT_DIR / 'category_table.csv'}")
    print(f"[out] {TEX_PATH}")


def write_latex(task3_rows: list[dict]) -> None:
    """Booktabs table: task3 categories x (n, acc@100, acc@50, abstain%) for RA-DPO."""
    pretty = {
        "NONE": "Non-sexist / no category",
        "IDEOLOGICAL-INEQUALITY": "Ideological inequality",
        "STEREOTYPING-DOMINANCE": "Stereotyping \\& dominance",
        "OBJECTIFICATION": "Objectification",
        "SEXUAL-VIOLENCE": "Sexual violence",
        "MISOGYNY-NON-SEXUAL-VIOLENCE": "Misogyny / non-sexual violence",
    }
    order = ["NONE", "IDEOLOGICAL-INEQUALITY", "MISOGYNY-NON-SEXUAL-VIOLENCE",
             "OBJECTIFICATION", "SEXUAL-VIOLENCE", "STEREOTYPING-DOMINANCE"]
    by_bucket = {r["bucket"]: r for r in task3_rows}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "",
        "\\caption{RA-DPO (gpt-4o) per sexism category (majority task-3 label):",
        "accuracy at full coverage, accuracy on the retained half at 50\\%",
        "coverage under $R(x)$, and the share of each category that is",
        "abstained on.}",
        "\\label{tab:category-analysis}",
        "",
        "\\begin{tabular}{lrccc}",
        "\\toprule",
        "Category & $n$ & Acc@100 & Acc@50 & Abstain\\% \\\\",
        "\\midrule",
    ]
    for b in order:
        if b not in by_bucket:
            continue
        r = by_bucket[b]
        a50 = f"{r['acc_at_50_retained']:.3f}" if r["acc_at_50_retained"] is not None else "--"
        lines.append(
            f"{pretty[b]} & {r['n']} & {r['acc_full']:.3f} & {a50} & "
            f"{100 * r['abstain_rate']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TEX_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
