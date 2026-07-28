"""Bootstrap F1 CIs + pairwise McNemar for any per-instance array directory.

Dataset-agnostic replacement for scripts/analysis/smart_curve_stats.py, which
is hardcoded to the EXIST gpt-4o track (IN_DIR, a literal MODELS dict, and a
692 test-set size) and — more importantly — *prints* its McNemar p-values
without persisting them, so those numbers survive only in a terminal
scrollback. This script writes everything it computes.

Works on any directory whose files carry the standard per_instance block
(predictions / correct, plus optional training_pairs), which both tracks do:

    results/final_reliability_3factor/          EXIST, gpt-4o
    results/edos_pipeline/per_instance/         EDOS, local backbones

Usage:
    # EDOS, one backbone
    python scripts/arr_ablations/reliability_stats.py \
        --dir results/edos_pipeline/per_instance \
        --glob 'llama32_3b_*_test_edos.json' \
        --out results/edos_pipeline/unified/llama32_3b/stats.json

    # EXIST gpt-4o (reproduces smart_curve_stats.py and adds McNemar to disk)
    python scripts/arr_ablations/reliability_stats.py \
        --dir results/final_reliability_3factor \
        --glob 'gpt-4o_*.json' \
        --out results/unified_gpt4o/reliability_stats.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["NO", "YES"]


def load_model(path: Path):
    """Return (label, predictions, true, correct, n_pairs) for one file."""
    d = json.load(open(path))
    pi = d["per_instance"]
    preds = np.asarray(pi["predictions"])
    correct = np.asarray(pi["correct"], dtype=int)

    # Gold is not stored as its own array in either track; recover it from
    # prediction XOR correctness. Binary task, so the flip is well-defined.
    if preds.dtype.kind in "US":
        flipped = np.where(preds == "YES", "NO", "YES")
    else:
        flipped = 1 - preds
    true = np.where(correct == 1, preds, flipped)

    pairs = d.get("training_pairs")
    label = re.sub(r"_(test_edos|local)$", "", path.stem)
    return label, preds, true, correct, pairs


def f1m(true, pred):
    labels = LABELS if true.dtype.kind in "US" else [0, 1]
    return f1_score(true, pred, average="macro", labels=labels)


def bootstrap_f1_ci(true, pred, n: int, seed: int):
    rng = np.random.default_rng(seed)
    n_samples = len(true)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores[i] = f1m(true[idx], pred[idx])
    return {
        "f1_mean": float(scores.mean()),
        "f1_ci_low": float(np.quantile(scores, 0.025)),
        "f1_ci_high": float(np.quantile(scores, 0.975)),
        "f1_se": float(scores.std()),
    }


def mcnemar(correct_a, correct_b):
    """Exact McNemar on paired correctness over the same test items."""
    from scipy.stats import binomtest
    a_only = int(((correct_a == 1) & (correct_b == 0)).sum())
    b_only = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, p=0.5).pvalue)
    return p, a_only, b_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="per-instance array directory")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = (ROOT / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matching {args.glob!r} in {in_dir}")

    models = {}
    for path in files:
        label, preds, true, correct, pairs = load_model(path)
        models[label] = {"preds": preds, "true": true,
                         "correct": correct, "pairs": pairs}

    sizes = {len(m["correct"]) for m in models.values()}
    if len(sizes) != 1:
        raise SystemExit(f"per-instance arrays differ in length: {sizes}")
    n_test = sizes.pop()

    out = {"meta": {"source_dir": str(in_dir.relative_to(ROOT)),
                    "glob": args.glob,
                    "n_test": n_test,
                    "bootstrap_n": args.bootstrap_n,
                    "seed": args.seed},
           "bootstrap_f1": {}, "mcnemar": [], "disagreement": []}

    print(f"=== Bootstrap F1-macro 95% CI ({args.bootstrap_n} resamples, "
          f"n={n_test}) ===")
    print(f"{'Model':<28} {'pairs':>7}  {'F1':>6}  {'CI95':>18}  {'SE':>6}")
    for label, m in models.items():
        ci = bootstrap_f1_ci(m["true"], m["preds"], args.bootstrap_n, args.seed)
        ci["pairs"] = m["pairs"]
        out["bootstrap_f1"][label] = ci
        pairs_s = m["pairs"] if m["pairs"] is not None else "—"
        print(f"{label:<28} {str(pairs_s):>7}  {ci['f1_mean']:.4f}  "
              f"[{ci['f1_ci_low']:.4f}, {ci['f1_ci_high']:.4f}]  {ci['f1_se']:.4f}")

    # Every pair, not a hand-picked list — the omitted comparison is always
    # the one a reader wants.
    print(f"\n=== McNemar paired tests (all {len(models)} choose 2) ===")
    for a, b in itertools.combinations(models, 2):
        p, a_only, b_only = mcnemar(models[a]["correct"], models[b]["correct"])
        rec = {"model_a": a, "model_b": b, "p_value": p,
               "only_a_correct": a_only, "only_b_correct": b_only,
               "significant_at_05": bool(p < 0.05)}
        out["mcnemar"].append(rec)
        disagree = int((models[a]["preds"] != models[b]["preds"]).sum())
        out["disagreement"].append({"model_a": a, "model_b": b,
                                    "n_disagree": disagree,
                                    "pct_disagree": round(
                                        disagree / n_test * 100, 2)})
        print(f"  {a:>26} vs {b:<26} p={p:.4f}  "
              f"a_only={a_only:<4} b_only={b_only:<4} "
              f"disagree={disagree}{'  *' if p < 0.05 else ''}")

    n_sig = sum(r["significant_at_05"] for r in out["mcnemar"])
    out["meta"]["n_significant_at_05"] = n_sig
    out["meta"]["n_comparisons"] = len(out["mcnemar"])
    print(f"\n{n_sig}/{len(out['mcnemar'])} comparisons significant at p<0.05")

    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
