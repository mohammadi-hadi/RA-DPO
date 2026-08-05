"""Training-selection overlap analysis for Smart-30% (ARR reviewer request).

Reviewers suspect the annotator-agreement component dominates the R(x)
ranking that defines the Smart-30% DPO subset. This script quantifies how
much of Smart-30% is *forced* by agreement alone:

  (a) Count unanimous (agreement = 1.0) train pairs and check that the
      1,661 Smart-30% pairs are fully contained in that set.
  (b) Tie-aware overlap bounds: because all unanimous pairs share the same
      agreement value, ANY "top-30% by agreement" selection is an arbitrary
      1,661-subset of the unanimous set. Minimum possible overlap with
      Smart-30% is (1661 - (1805 - 1661)) / 1661; maximum is 100%.
  (c) Uncertainty-only counterfactual: top-1,661 by (1 - sigmoid_score)
      with a deterministic tie-break by id. Overlap with Smart-30% and
      composition by agreement level {3/3, 4/6, 5/6, 6/6}.
  (d) Spearman correlation between the (1 - sigmoid) ranking and the
      agreement ranking over all 5,536 train pairs.
  (e) Composition table: uncertainty-only-30% vs Smart-30% vs the random
      expectation (train marginal).

NOTE / TODO: the confidence component of R(x) cannot be replayed yet --
the original per-train-instance confidences were not persisted. Once they
are regenerated, pass --confidences path/to/conf.json (a {id: conf} map)
and the confidence-only top-30% overlap + composition will be appended to
the same outputs.

Outputs:
  results/arr_ablations/selection_overlap/overlap.json
  results/arr_ablations/selection_overlap/composition.csv
  arr_revision/experiments/tables/tab_selection_overlap.tex
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, agreement_score

SMART30_PATH = ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl"
TOKEN_SCORES_PATH = ROOT / "results" / "token_scores" / "token_scores_cache.json"
OUT_DIR = ROOT / "results" / "arr_ablations" / "selection_overlap"
TEX_PATH = ROOT / "arr_revision" / "experiments" / "tables" / "tab_selection_overlap.tex"

PROMPT_PREFIX = "Post: "
PROMPT_SUFFIX = "\n\nClassification (YES or NO):"

AGREEMENT_LEVELS = ["3/3", "4/6", "5/6", "6/6"]


def agreement_level(a: float) -> str:
    """Map a 6-annotator agreement score to its vote-split label."""
    if a >= 1.0 - 1e-9:
        return "6/6"
    if a >= 5 / 6 - 1e-9:
        return "5/6"
    if a >= 4 / 6 - 1e-9:
        return "4/6"
    return "3/3"


def load_train_split() -> pd.DataFrame:
    """Load the train split exactly like scripts/build_smart10_and_ambiguous.py."""
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, _, _ = loader.create_train_val_test_split(df)
    return train_df.reset_index(drop=True)


def load_smart30_texts() -> list[str]:
    """Extract post texts from the Smart-30% DPO jsonl."""
    texts = []
    with open(SMART30_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)["input"]["messages"][1]["content"]
            assert msg.startswith(PROMPT_PREFIX) and msg.endswith(PROMPT_SUFFIX), (
                "Unexpected prompt format in smart30_dpo.jsonl"
            )
            texts.append(msg[len(PROMPT_PREFIX):-len(PROMPT_SUFFIX)])
    return texts


def composition(ids, agr) -> dict:
    """Counts and fractions per agreement level for a set of ids."""
    counts = {lvl: 0 for lvl in AGREEMENT_LEVELS}
    for i in ids:
        counts[agreement_level(agr[i])] += 1
    n = len(ids)
    return {
        "counts": counts,
        "fractions": {lvl: counts[lvl] / n for lvl in AGREEMENT_LEVELS},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--confidences", type=Path, default=None,
        help="TODO hook: JSON {id: conf} of regenerated train confidences; "
             "adds a confidence-only top-30% selection to the analysis.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train split + agreement + token sigmoid scores ----
    train_df = load_train_split()
    n_train = len(train_df)
    print(f"Train split: {n_train} tweets")
    agr = dict(zip(train_df["id"], train_df["agreement_score"]))
    train_ids = list(train_df["id"])

    cache = json.load(open(TOKEN_SCORES_PATH))
    missing = [i for i in train_ids if str(i) not in cache]
    assert not missing, f"{len(missing)} train ids missing from token score cache"
    sig = {i: cache[str(i)]["sigmoid_score"] for i in train_ids}

    # ---- 2. Smart-30% tweets → train ids by exact text ----
    smart_texts = load_smart30_texts()
    text2id = dict(zip(train_df["tweet"], train_df["id"]))
    assert len(text2id) == n_train, "Duplicate tweet texts in train split"
    matched = [t for t in smart_texts if t in text2id]
    match_rate = len(matched) / len(smart_texts)
    smart_ids = {text2id[t] for t in matched}
    n_smart = len(smart_ids)
    print(f"Smart-30%: {len(smart_texts)} pairs, {len(matched)} matched "
          f"({match_rate:.1%}), {n_smart} unique train ids")

    # ---- 3a. Unanimous containment ----
    unanimous_ids = {i for i in train_ids if agr[i] >= 1.0 - 1e-9}
    n_unanimous = len(unanimous_ids)
    n_outside = len(smart_ids - unanimous_ids)
    print(f"Unanimous (6/6) train pairs: {n_unanimous}")
    print(f"Smart-30% pairs outside the unanimous set: {n_outside}")

    # ---- 3b. Tie-aware overlap bounds for ANY top-30%-by-agreement ----
    # All unanimous pairs tie at agreement = 1.0, so any agreement-only
    # top-1661 is an arbitrary 1661-subset of the 1805 unanimous pairs.
    min_overlap = (n_smart - (n_unanimous - n_smart)) / n_smart
    max_overlap = 1.0
    # Expected overlap under a uniformly random tie-break (hypergeometric mean)
    exp_overlap_random_tiebreak = n_smart / n_unanimous
    print(f"Agreement-only top-30% overlap with Smart-30%: "
          f"min {min_overlap:.1%}, max {max_overlap:.0%} "
          f"(expected {exp_overlap_random_tiebreak:.1%} under random tie-break)")

    # ---- 3c. Uncertainty-only top-1661 by (1 - sigmoid), tie-break by id ----
    order = sorted(train_ids, key=lambda i: (-(1.0 - sig[i]), str(i)))
    unc_ids = set(order[:n_smart])
    unc_overlap = len(unc_ids & smart_ids) / n_smart
    unc_comp = composition(unc_ids, agr)
    print(f"Uncertainty-only top-30% overlap with Smart-30%: {unc_overlap:.1%}")

    # ---- 3d. Spearman: (1 - sigmoid) ranking vs agreement ranking ----
    u = np.array([1.0 - sig[i] for i in train_ids])
    a = np.array([agr[i] for i in train_ids])
    rho, pval = spearmanr(u, a)
    print(f"Spearman (1 - sigmoid) vs agreement over {n_train}: "
          f"rho = {rho:.3f} (p = {pval:.2e})")

    # ---- 3e. Composition table ----
    smart_comp = composition(smart_ids, agr)
    train_comp = composition(train_ids, agr)
    # Random expectation for a uniform 1661-subset = train marginal
    random_comp = {
        "counts": {lvl: train_comp["fractions"][lvl] * n_smart
                   for lvl in AGREEMENT_LEVELS},
        "fractions": dict(train_comp["fractions"]),
    }
    # Chance-level overlap of any random 1661-subset with Smart-30%
    chance_overlap = n_smart / n_train

    selections = {
        "Smart-30% (R(x) top-30%)": (smart_comp, 1.0),
        "Agreement-only top-30%": (smart_comp,  # forced into unanimous set
                                   None),      # overlap is a range, see json
        "Uncertainty-only top-30%": (unc_comp, unc_overlap),
        "Random 30% (expectation)": (random_comp, chance_overlap),
    }

    # ---- Optional confidence-only selection (TODO hook) ----
    conf_result = None
    if args.confidences is not None:
        conf = {k: float(v) for k, v in json.load(open(args.confidences)).items()}
        missing_c = [i for i in train_ids if str(i) not in conf and i not in conf]
        assert not missing_c, f"{len(missing_c)} train ids missing from {args.confidences}"
        get_c = lambda i: conf.get(i, conf.get(str(i)))
        order_c = sorted(train_ids, key=lambda i: (-get_c(i), str(i)))
        conf_ids = set(order_c[:n_smart])
        conf_overlap = len(conf_ids & smart_ids) / n_smart
        conf_comp = composition(conf_ids, agr)
        selections["Confidence-only top-30%"] = (conf_comp, conf_overlap)
        conf_result = {"overlap_with_smart30": conf_overlap,
                       "composition": conf_comp}
        print(f"Confidence-only top-30% overlap with Smart-30%: {conf_overlap:.1%}")

    # ---- SELF-CHECK ----
    ok = (n_unanimous == 1805) and (n_outside == 0) and (match_rate == 1.0)
    status = "PASSED" if ok else "FAILED"
    print(f"\nSELF-CHECK {status}: unanimous = {n_unanimous} (expect 1805), "
          f"smart30 outside unanimous = {n_outside} (expect 0), "
          f"text match rate = {match_rate:.1%} (expect 100%)")
    if not ok:
        sys.exit(1)

    # ---- Write overlap.json ----
    overlap = {
        "n_train": n_train,
        "n_smart30": n_smart,
        "smart30_text_match_rate": match_rate,
        "n_unanimous": n_unanimous,
        "smart30_outside_unanimous": n_outside,
        "smart30_contained_in_unanimous": n_outside == 0,
        "agreement_only_top30_overlap": {
            "min": min_overlap,
            "max": max_overlap,
            "expected_random_tiebreak": exp_overlap_random_tiebreak,
            "note": "All 1805 unanimous pairs tie at agreement=1.0; any "
                    "agreement-only top-1661 is an arbitrary subset of them.",
        },
        "uncertainty_only_top30": {
            "overlap_with_smart30": unc_overlap,
            "composition": unc_comp,
        },
        "chance_overlap_random_subset": chance_overlap,
        "spearman_1minus_sigmoid_vs_agreement": {"rho": rho, "p_value": pval},
        "train_marginal_composition": train_comp,
        "confidence_only_top30": conf_result,  # TODO: fill via --confidences
        "confidence_note": "Original train confidences were not persisted; "
                           "confidence-only overlap will be appended once "
                           "regenerated (pass --confidences).",
    }
    with open(OUT_DIR / "overlap.json", "w") as f:
        json.dump(overlap, f, indent=2)

    # ---- Write composition.csv ----
    rows = []
    for name, (comp, ov) in selections.items():
        row = {"selection": name, "n_pairs": n_smart,
               "overlap_with_smart30": (f"{min_overlap:.3f}-1.000"
                                        if ov is None else f"{ov:.3f}")}
        for lvl in AGREEMENT_LEVELS:
            row[f"count_{lvl}"] = round(comp["counts"][lvl], 1)
            row[f"pct_{lvl}"] = round(comp["fractions"][lvl] * 100, 1)
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_DIR / "composition.csv", index=False)
    print("\n=== Composition of 1,661-pair selections ===")
    print(comp_df.to_string(index=False))

    # ---- Write LaTeX table ----
    def pct(x):
        return f"{x * 100:.1f}\\%"

    tex_rows = []
    for name, (comp, ov) in selections.items():
        ov_str = (f"{min_overlap * 100:.1f}--100\\%" if ov is None
                  else pct(ov))
        cells = "  & ".join(pct(comp["fractions"][lvl]) for lvl in AGREEMENT_LEVELS)
        tex_rows.append(f"{name:<28} & {ov_str:>12} & {cells} \\\\")
    tex_rows[0] = tex_rows[0].replace("Smart-30% (R(x) top-30%)",
                                      "Smart-30\\% ($R(x)$ top-30\\%)")
    tex_rows[1] = tex_rows[1].replace("Agreement-only top-30%",
                                      "Agreement-only top-30\\%")
    tex_rows[2] = tex_rows[2].replace("Uncertainty-only top-30%",
                                      "Uncertainty-only top-30\\%")
    tex_rows[3] = tex_rows[3].replace("Random 30% (expectation)",
                                      "Random 30\\% (expectation)")

    tex = f"""\\begin{{table}}[h]
\\centering
\\fontsize{{10}}{{12}}\\selectfont
\\setlength{{\\tabcolsep}}{{3pt}}

\\caption{{Overlap of alternative 1,661-pair training selections with
Smart-30\\%. The train split has {n_unanimous:,} unanimous pairs, so any
agreement-only top-30\\% must overlap Smart-30\\% by at least
{min_overlap * 100:.1f}\\%. An uncertainty-only selection (rank by
$1-\\mathrm{{sig}}$) overlaps only {unc_overlap * 100:.1f}\\% --
chance level is {chance_overlap * 100:.1f}\\% -- and its composition
mirrors the train marginal (Spearman $\\rho$ between the two rankings:
${rho:.2f}$).}}
\\label{{tab:selection-overlap}}

\\begin{{tabular}}{{lccccc}}
\\toprule
Selection (1,661 pairs) & Overlap & 3/3 & 4/6 & 5/6 & 6/6 \\\\
\\midrule
{chr(10).join(tex_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(TEX_PATH, "w") as f:
        f.write(tex)

    print(f"\nSaved: {OUT_DIR / 'overlap.json'}")
    print(f"Saved: {OUT_DIR / 'composition.csv'}")
    print(f"Saved: {TEX_PATH}")


if __name__ == "__main__":
    main()
