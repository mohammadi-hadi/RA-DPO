"""Build the low-budget selection arms that decide whether R_train ranks at all.

Motivation. Every size-matched control so far was run at 30% or 50% of the
training pool, where the task is at or near its full-data ceiling (EXIST
gpt-4o: Smart-30% 0.8213 vs full-data 0.8212). A control at a saturated
budget cannot separate two selection rules no matter how different they are.
The 10% budget (554 pairs) is the only one with real headroom left
(+8.6pp on Llama, +4.5pp on Qwen, +2.1pp on gpt-4o) and has never had a
random counterpart.

R_train(x) = 0.8 * agreement + 0.2 * (1 - sigmoid_score) decomposes into two
claims, and this builds the arms that separate them at that budget:

  smart10      (exists)  unanimous pairs, ranked by LOWEST model uncertainty
  flip10       (new)     unanimous pairs, ranked by HIGHEST model uncertainty
  randunan10   (new)     unanimous pairs, drawn at random
  random10     (new)     drawn at random from the whole pool

  randunan10 vs random10   -> does the agreement filter carry signal?
  smart10 vs randunan10    -> does the uncertainty ranking beat chance?
  flip10  vs randunan10    -> does the opposite direction do better?
  smart10 vs flip10        -> does the direction matter at all?

The direction question is live because the current rule prefers pairs that
are clean for annotators AND easy for the model, which is the combination
that carries least gradient. Active learning would target clean-label,
model-uncertain instead.

Note the sigmoid score is near-degenerate on the unanimous block (std 0.013,
mean 0.489, capped at 0.500), so a null across all four arms is itself a
result: it would show the token-uncertainty term is noise rather than signal.

    python scripts/arr_ablations/build_lowbudget_selection_arms.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_component_ablation_subsets import (  # noqa: E402
    load_train_df, make_pair, TOKEN_SCORES, OUT_DIR,
)

SEED = 42
N_KEEP = 554          # matches the existing smart10_dpo budget exactly
UNANIMOUS = 0.999     # agreement_score threshold for 6/0


def write(df, name):
    path = OUT_DIR / name
    with open(path, "w") as f:
        for r in df.itertuples():
            f.write(json.dumps(make_pair(r.tweet, r.majority_label),
                               ensure_ascii=False) + "\n")
    print(f"  {name:<24} {len(df):>4} pairs -> {path.relative_to(ROOT)}")
    return path


def main():
    df = load_train_df()
    cache = json.load(open(TOKEN_SCORES))
    df["sig"] = [cache[i]["sigmoid_score"] for i in df["id"]]
    unan = df[df["agreement_score"] >= UNANIMOUS].reset_index(drop=True)
    print(f"train {len(df)} pairs, {len(unan)} unanimous; budget {N_KEEP}")
    print(f"sigmoid within unanimous: mean {unan['sig'].mean():.4f} "
          f"std {unan['sig'].std():.4f}")

    rng = np.random.default_rng(SEED)

    # Preserve original train row order in every file, as smart30/smart50 do.
    flip = unan.nlargest(N_KEEP, "sig").sort_index()
    runan = unan.iloc[np.sort(rng.choice(len(unan), N_KEEP, replace=False))]
    rall = df.iloc[np.sort(rng.choice(len(df), N_KEEP, replace=False))]

    write(flip, "flip10_dpo.jsonl")
    write(runan, "randunan10_dpo.jsonl")
    write(rall, "random10_dpo.jsonl")

    # Sanity: the two uncertainty directions must not overlap, and the random
    # arms must sit at their expected composition.
    smart_lo = set(unan.nsmallest(N_KEEP, "sig")["id"])
    print(f"\n  smart10 vs flip10 overlap : {len(smart_lo & set(flip['id']))}"
          f"/{N_KEEP}  (expect 0)")
    print(f"  random10 unanimous share  : "
          f"{(rall['agreement_score'] >= UNANIMOUS).mean():.3f}  "
          f"(pool rate {(df['agreement_score'] >= UNANIMOUS).mean():.3f})")
    print(f"  randunan10 unanimous share: "
          f"{(runan['agreement_score'] >= UNANIMOUS).mean():.3f}  (expect 1.000)")
    print(f"\n  add to configs/local_pipeline.yaml -> data.training_pairs:")
    for v in ("flip10_dpo", "randunan10_dpo", "random10_dpo"):
        print(f"    {v}: {N_KEEP}")


if __name__ == "__main__":
    main()
