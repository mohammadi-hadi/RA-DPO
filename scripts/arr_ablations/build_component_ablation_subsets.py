"""Build component-ablation DPO training subsets (training-side ablation).

R(x) mixes three signals: model confidence, annotator agreement, and token
uncertainty (sigmoid score). To isolate each component's contribution on the
TRAINING side, this script builds top-30% subsets (1661 of 5536 train pairs)
ranked by a single component:

  - agree30_dpo.jsonl      rank by agreement desc, tie-break id asc
  - agree30_tb2_dpo.jsonl  rank by agreement desc, tie-break id desc
                           (bounds the tie-break effect: >1661 unanimous rows)
  - uncert30_dpo.jsonl     rank by (1 - sigmoid_score) desc, tie-break id asc
                           (sigmoid_score from results/token_scores cache)
  - conf30_dpo.jsonl       only with --confidences <json>: {id: confidence}
                           rank by confidence desc, tie-break id asc

Output records are format-identical to results/smart_sampling/smart30_dpo.jsonl
(same JSON keys, same system prompt) so the same DPO fine-tuning jobs can
consume them. A manifest with line counts, agreement histograms and overlap
with the Smart-30% subset is written alongside.

Usage:
    python scripts/arr_ablations/build_component_ablation_subsets.py
    python scripts/arr_ablations/build_component_ablation_subsets.py \
        --confidences results/train_confidences.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score

OUT_DIR = ROOT / "results" / "smart_sampling"
TOKEN_SCORES = ROOT / "results" / "token_scores" / "token_scores_cache.json"
SMART30 = OUT_DIR / "smart30_dpo.jsonl"
FRACTION = 0.30

SYSTEM_PROMPT = (
    "You are an expert content moderator. Classify whether the social media "
    "post is sexist or not. Respond with ONLY YES or NO."
)


def make_pair(tweet: str, majority_label: str) -> dict:
    chosen = majority_label.upper()
    rejected = "NO" if chosen == "YES" else "YES"
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Post: {tweet}\n\nClassification (YES or NO):",
                },
            ]
        },
        "preferred_output": [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }


def load_train_df() -> pd.DataFrame:
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, _, _ = loader.create_train_val_test_split(df)
    return train_df.reset_index(drop=True)


def select_top(df: pd.DataFrame, score_col: str, n: int,
               id_ascending: bool) -> pd.DataFrame:
    """Top-n rows by score desc; deterministic tie-break on id (stable sort)."""
    ordered = df.sort_values("id", ascending=id_ascending, kind="mergesort")
    ordered = ordered.sort_values(score_col, ascending=False, kind="mergesort")
    return ordered.head(n)


def write_jsonl(subset: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        for _, row in subset.iterrows():
            f.write(json.dumps(make_pair(row["tweet"], row["majority_label"]),
                               ensure_ascii=False) + "\n")


def smart30_tweet_set() -> set:
    tweets = set()
    with open(SMART30) as f:
        for line in f:
            rec = json.loads(line)
            user_msg = rec["input"]["messages"][1]["content"]
            assert user_msg.startswith("Post: ")
            tweets.add(user_msg[len("Post: "):-len("\n\nClassification (YES or NO):")])
    return tweets


def describe(subset: pd.DataFrame, path: Path, smart30_tweets: set,
             params: dict) -> dict:
    n_lines = sum(1 for _ in open(path))
    hist = {f"{k:.4f}": int(v) for k, v in
            sorted(Counter(subset["agreement_score"]).items())}
    overlap = int(subset["tweet"].isin(smart30_tweets).sum())
    return {
        "file": path.name,
        "n_lines": n_lines,
        "agreement_histogram": hist,
        "overlap_with_smart30": {
            "count": overlap,
            "pct_of_file": round(100.0 * overlap / len(subset), 2),
        },
        "generation_params": params,
    }


def self_check(paths: list, n_expected: int, agree_paths: list,
               subsets: dict) -> None:
    # 1. line counts
    for p in paths:
        n = sum(1 for _ in open(p))
        assert n == n_expected, f"{p.name}: {n} lines, expected {n_expected}"
    # 2. agree30 subsets are all-unanimous
    for p in agree_paths:
        agr = subsets[p.name]["agreement_score"]
        assert (agr == 1.0).all(), f"{p.name}: non-unanimous rows selected"
    # 3. record format identical to smart30 (keys + system prompt of record 1)
    with open(SMART30) as f:
        ref = json.loads(f.readline())
    for p in paths:
        with open(p) as f:
            rec = json.loads(f.readline())
        assert set(rec) == set(ref), f"{p.name}: top-level keys differ"
        assert set(rec["input"]) == set(ref["input"])
        for got, want in zip(rec["input"]["messages"], ref["input"]["messages"]):
            assert set(got) == set(want) and got["role"] == want["role"]
        assert rec["input"]["messages"][0]["content"] == \
            ref["input"]["messages"][0]["content"], f"{p.name}: system prompt differs"
        for key in ("preferred_output", "non_preferred_output"):
            assert set(rec[key][0]) == set(ref[key][0])
    print("SELF-CHECK PASSED: line counts, agree30 unanimity, record format")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confidences", type=Path, default=None,
                    help="JSON {train_id: confidence}; if given, also build "
                         "conf30_dpo.jsonl (rank by confidence desc, id asc)")
    args = ap.parse_args()

    train_df = load_train_df()
    n_top = int(round(len(train_df) * FRACTION))
    print(f"Training set size: {len(train_df)} -> top-{FRACTION:.0%} = {n_top}")

    cache = json.load(open(TOKEN_SCORES))
    missing = set(train_df["id"]) - set(cache)
    assert not missing, f"{len(missing)} train ids missing from token cache"
    train_df["uncertainty"] = [1.0 - cache[i]["sigmoid_score"]
                               for i in train_df["id"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smart30_tweets = smart30_tweet_set()

    builds = [
        ("agree30_dpo.jsonl", "agreement_score", True,
         {"rank_by": "agreement desc", "tie_break": "id asc"}),
        ("agree30_tb2_dpo.jsonl", "agreement_score", False,
         {"rank_by": "agreement desc", "tie_break": "id desc"}),
        ("uncert30_dpo.jsonl", "uncertainty", True,
         {"rank_by": "(1 - sigmoid_score) desc", "tie_break": "id asc",
          "sigmoid_source": str(TOKEN_SCORES.relative_to(ROOT))}),
    ]
    if args.confidences is not None:
        conf = json.load(open(args.confidences))
        missing = set(train_df["id"]) - set(conf)
        assert not missing, f"{len(missing)} train ids missing from {args.confidences}"
        train_df["confidence"] = [float(conf[i]) for i in train_df["id"]]
        builds.append(("conf30_dpo.jsonl", "confidence", True,
                       {"rank_by": "confidence desc", "tie_break": "id asc",
                        "confidence_source": str(args.confidences)}))

    common = {"split": "EXISTDataLoader 80/10/10 stratified, seed 42",
              "n_train": len(train_df), "fraction": FRACTION, "n_selected": n_top}
    manifest, subsets, paths = [], {}, []
    for fname, col, id_asc, params in builds:
        subset = select_top(train_df, col, n_top, id_asc)
        path = OUT_DIR / fname
        write_jsonl(subset, path)
        subsets[fname] = subset
        paths.append(path)
        manifest.append(describe(subset, path, smart30_tweets, {**params, **common}))
        print(f"  {fname}: {len(subset)} pairs written")

    manifest_path = OUT_DIR / "component_ablation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {manifest_path}")

    agree_paths = [OUT_DIR / "agree30_dpo.jsonl", OUT_DIR / "agree30_tb2_dpo.jsonl"]
    self_check(paths, n_top, agree_paths, subsets)

    print(f"\n{'file':<24}{'overlap w/ smart30':>20}{'pct':>8}")
    for entry in manifest:
        ov = entry["overlap_with_smart30"]
        print(f"{entry['file']:<24}{ov['count']:>20}{ov['pct_of_file']:>7.1f}%")
    print("\nComposition by agreement level:")
    for entry in manifest:
        print(f"  {entry['file']}: {entry['agreement_histogram']}")


if __name__ == "__main__":
    main()
