"""Single-annotator label-noise arms: the regime where selection should matter.

Every matched control so far selects from a pool whose labels are 6-vote
majorities, i.e. already denoised, so the agreement term has no noise to
detect and random subsampling suffices. This builds the counterfactual the
method was designed for: each item carries ONE randomly drawn annotator's
label, the standard annotation budget in practice. Per stratum the drawn
label differs from the majority with probability 0 (6/0), 1/6 (5/1),
1/3 (4/2) and 1/2 (3/3), which puts roughly 19.5% flipped preference pairs
in the pool.

Arms, all 554 pairs, all reusing the ITEM sets of the existing clean arms so
label noise is the only difference:

  noisy1_random10   same items as random10_dpo, single-annotator labels
                    (inherits the pool flip rate, ~19-20%)
  noisy1_strat10    same items as strat10_dpo, single-annotator labels
                    (distribution-matched selection under noise)
  smart10 (exists)  the top-554 selection is 100% unanimous, so every
                    annotator agrees on every item and the single-annotator
                    file would be byte-identical to smart10_dpo.jsonl.
                    Asserted below; no new file, no new training.

Pre-registered primary contrast: smart10_dpo vs noisy1_random10_dpo, exact
McNemar, both backbones. Item seed and label seed fixed at 42, one draw.

    venv/bin/python scripts/arr_ablations/build_noisy_label_arms.py
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
from build_stratified_arms import largest_remainder  # noqa: E402

SEED_ITEMS = 42    # must replay build_lowbudget_selection_arms / stratified arms
SEED_LABELS = 42   # one draw, no re-rolls
N_KEEP, UNANIMOUS = 554, 0.999


def file_tweets(name):
    tweets = []
    with open(OUT_DIR / name) as f:
        for line in f:
            msg = json.loads(line)["input"]["messages"][1]["content"]
            assert msg.startswith("Post: ")
            tweets.append(msg[len("Post: "):-len("\n\nClassification (YES or NO):")])
    return tweets


def file_chosen(name):
    with open(OUT_DIR / name) as f:
        return [json.loads(line)["preferred_output"][0]["content"] for line in f]


def write(df, name):
    p = OUT_DIR / name
    with open(p, "w") as f:
        for r in df.itertuples():
            f.write(json.dumps(make_pair(r.tweet, r.noisy_label),
                               ensure_ascii=False) + "\n")
    print(f"  {name:<26} {len(df):>4} pairs -> {p.relative_to(ROOT)}")


def main():
    df = load_train_df()
    cache = json.load(open(TOKEN_SCORES))
    df["sig"] = [cache[i]["sigmoid_score"] for i in df["id"]]

    # --- single-annotator label draw over the whole pool, once ---------------
    rng_lab = np.random.default_rng(SEED_LABELS)
    ann = rng_lab.integers(0, 6, size=len(df))
    df["noisy_label"] = [labs[a] for labs, a in zip(df["labels_task1"], ann)]
    df["flipped"] = df["noisy_label"] != df["majority_label"]
    print(f"pool {len(df)}   flip rate {df['flipped'].mean():.3f} "
          f"({int(df['flipped'].sum())} items)")
    for a in sorted(df["agreement_score"].round(3).unique()):
        m = df["agreement_score"].round(3) == a
        print(f"  stratum {a:>5}: n={m.sum():>4}  flipped {df.loc[m, 'flipped'].mean():.3f}")

    # --- replay the random10 item draw (rng sequence from the original) ------
    unan_n = int((df["agreement_score"] >= UNANIMOUS).sum())
    rng = np.random.default_rng(SEED_ITEMS)
    rng.choice(unan_n, N_KEEP, replace=False)                      # runan draw
    rall = df.iloc[np.sort(rng.choice(len(df), N_KEEP, replace=False))]
    assert list(rall["tweet"]) == file_tweets("random10_dpo.jsonl"), \
        "random10 item replay does not match the shipped file"

    # --- recompute the strat10 item set (deterministic, no rng) --------------
    agr = df["agreement_score"].round(3)
    levels = sorted(agr.unique())
    counts = [int((agr == a).sum()) for a in levels]
    n_i = largest_remainder([c / len(df) for c in counts], N_KEEP)
    parts = [df[agr == a].nsmallest(n, "sig") for a, n in zip(levels, n_i)]
    strat = df.loc[np.sort(np.concatenate([p.index.values for p in parts]))]
    assert list(strat["tweet"]) == file_tweets("strat10_dpo.jsonl"), \
        "strat10 item recomputation does not match the shipped file"

    # --- smart10 invariance: unanimous items cannot flip ---------------------
    sm_tweets, sm_chosen = file_tweets("smart10_dpo.jsonl"), file_chosen("smart10_dpo.jsonl")
    sm = df.set_index("tweet").loc[sm_tweets]
    assert (sm["agreement_score"] >= UNANIMOUS).all(), "smart10 has non-unanimous items"
    assert list(sm["noisy_label"].str.upper()) == sm_chosen, \
        "smart10 labels change under the protocol (should be impossible)"
    print(f"\nsmart10 invariant: all {len(sm)} items unanimous, "
          f"single-annotator labels identical to the shipped file")

    print()
    write(rall, "noisy1_random10_dpo.jsonl")
    write(strat, "noisy1_strat10_dpo.jsonl")

    manifest = {
        "protocol": "one annotator label per item, drawn uniformly from the six",
        "seed_items": SEED_ITEMS, "seed_labels": SEED_LABELS, "n_pairs": N_KEEP,
        "pool_flip_rate": round(float(df["flipped"].mean()), 4),
        "arms": {}}
    for name, sub in (("noisy1_random10_dpo", rall), ("noisy1_strat10_dpo", strat)):
        manifest["arms"][name] = {
            "items_identical_to": name.replace("noisy1_", ""),
            "n_flipped": int(sub["flipped"].sum()),
            "flip_rate": round(float(sub["flipped"].mean()), 4),
            "flips_by_stratum": {
                str(a): int(sub.loc[sub["agreement_score"].round(3) == a, "flipped"].sum())
                for a in levels}}
    manifest["arms"]["smart10_dpo"] = {
        "note": "protocol-invariant: 100% unanimous, file unchanged, no retraining",
        "n_flipped": 0, "flip_rate": 0.0}
    mp = OUT_DIR / "noisy_arms_manifest.json"
    json.dump(manifest, open(mp, "w"), indent=2)
    print(f"\nmanifest -> {mp.relative_to(ROOT)}")
    for name, m in manifest["arms"].items():
        print(f"  {name:<22} flipped {m['n_flipped']:>3}  rate {m['flip_rate']}")


if __name__ == "__main__":
    main()
