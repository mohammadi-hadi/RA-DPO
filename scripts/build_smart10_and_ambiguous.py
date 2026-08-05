"""Build two new DPO training subsets:
  1. Smart-10%: top 554 pairs (all unanimous, further sorted by confidence+token).
  2. Ambiguous-only: pairs where annotator agreement <= 0.5 (3/3 splits).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score

OUT_DIR = ROOT / "results" / "smart_sampling"
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


def main():
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, _, _ = loader.create_train_val_test_split(df)
    print(f"Training set size: {len(train_df)}")

    train_df = train_df.reset_index(drop=True)
    counts = train_df["agreement_score"].value_counts().sort_index()
    print("Agreement distribution in training set:")
    for k, v in counts.items():
        print(f"  {k:.3f} -> {v}")

    # ---- Smart-10%: take top 554 from the existing Smart-30% file (already R(x)-sorted) ----
    smart30_path = OUT_DIR / "smart30_dpo.jsonl"
    smart10_path = OUT_DIR / "smart10_dpo.jsonl"
    with open(smart30_path) as f:
        smart30_lines = f.readlines()
    n10 = max(1, int(round(len(train_df) * 0.10)))
    smart10_lines = smart30_lines[:n10]
    with open(smart10_path, "w") as f:
        f.writelines(smart10_lines)
    print(f"\nSmart-10% → {n10} pairs written to {smart10_path}")

    # ---- Ambiguous-only: all pairs with agreement <= 0.5 (3/3 splits) ----
    ambig_df = train_df[train_df["agreement_score"] <= 0.5 + 1e-6].copy()
    print(f"\nAmbiguous-only candidates (agreement ≤ 0.5): {len(ambig_df)}")
    # Drop any pairs where majority_label is ambiguous (3/3 ties → arbitrary tie-break)
    print(f"  majority_label distribution: {ambig_df['majority_label'].value_counts().to_dict()}")

    ambig_path = OUT_DIR / "ambiguous_only_dpo.jsonl"
    with open(ambig_path, "w") as f:
        for _, row in ambig_df.iterrows():
            f.write(json.dumps(make_pair(row["tweet"], row["majority_label"]),
                               ensure_ascii=False) + "\n")
    print(f"Ambiguous-only → {len(ambig_df)} pairs written to {ambig_path}")


if __name__ == "__main__":
    main()
