"""Student-need selection arms: reliable label x pairs the base model gets wrong.

The factorial showed why the shipped rule fails: it selects unanimous pairs
that are also EASY for the model, and pairs the model already classifies
correctly contribute almost no DPO gradient. The revised rule keeps the
reliability signal where it is competent (the agreement gate certifies the
LABEL) and adds the axis the old rule lacked (the student's own error
certifies the pair is INFORMATIVE). This follows reducible-loss selection:
high student loss, low label noise.

Arms, 554 pairs each, per backbone (base predictions differ per student):

  hard10_<bb>     agreement >= 5/6 AND base model wrong, most confidently
                  wrong first; topped up with lowest-confidence correct pairs
                  from the same gated pool if the wrong pool is short.
  hardctl10_<bb>  control: random from the same gated pool with the SAME
                  YES/NO label counts as hard10. Separates "informative
                  pairs" from "class rebalancing" (base models over-predict
                  NO, so the wrong pool is YES-heavy).
  hardall10_<bb>  ungated: wrong-first over the whole pool. Separates the
                  agreement gate's contribution.

Labels are the clean 6-vote majorities (the noise regime is closed).
EXIST is the development bed; the surviving rule confirms once on EDOS.

    venv/bin/python scripts/arr_ablations/build_student_need_arms.py --backbone llama32_3b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_component_ablation_subsets import (  # noqa: E402
    load_train_df, make_pair, OUT_DIR,
)

PRED_DIR = ROOT / "results" / "local_pipeline" / "train_pool_base"
SUFFIX = {"llama32_3b": "llama", "qwen25_3b": "qwen"}
SEED, N_KEEP, GATE = 42, 554, 5.0 / 6.0 - 1e-9


def write(df, name):
    p = OUT_DIR / name
    with open(p, "w") as f:
        for r in df.itertuples():
            f.write(json.dumps(make_pair(r.tweet, r.majority_label),
                               ensure_ascii=False) + "\n")
    print(f"  {name:<24} {len(df):>4} pairs -> {p.relative_to(ROOT)}")


def compose(df):
    return {"n": len(df),
            "label_counts": df["majority_label"].value_counts().to_dict(),
            "strata": {str(a): int((df["agreement_score"].round(3) == a).sum())
                       for a in sorted(df["agreement_score"].round(3).unique())},
            "n_wrong_at_base": int(df["wrong"].sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=sorted(SUFFIX))
    args = ap.parse_args()
    sfx = SUFFIX[args.backbone]

    preds = json.load(open(PRED_DIR / f"{args.backbone}_train_base.json"))["items"]
    df = load_train_df()
    assert set(df["id"]) == set(preds), "train ids do not match prediction file"
    df["pred"] = [preds[i]["pred"] for i in df["id"]]
    df["conf"] = [preds[i]["conf"] for i in df["id"]]
    df["wrong"] = df["pred"] != df["majority_label"]

    gated = df[df["agreement_score"] >= GATE]
    gw = gated[gated["wrong"]]
    print(f"pool {len(df)}  wrong {int(df['wrong'].sum())} "
          f"({df['wrong'].mean():.1%})   gated(>=5/6) {len(gated)}  "
          f"gated-and-wrong {len(gw)}")

    # hard10: most confidently wrong first, top up with least-confident correct
    hard = gw.nlargest(min(N_KEEP, len(gw)), "conf")
    n_fill = N_KEEP - len(hard)
    if n_fill > 0:
        fill = gated[~gated["wrong"]].nsmallest(n_fill, "conf")
        hard = df.loc[np.sort(np.concatenate([hard.index, fill.index]))]
        print(f"  wrong pool short: filled {n_fill} with lowest-conf correct")
    else:
        hard = hard.sort_index()

    # hardctl10: label-matched random from the gated pool
    rng = np.random.default_rng(SEED)
    parts = []
    for lab, n in hard["majority_label"].value_counts().items():
        cand = gated[gated["majority_label"] == lab]
        parts.append(cand.iloc[np.sort(rng.choice(len(cand), n, replace=False))])
    ctl = df.loc[np.sort(np.concatenate([p.index.values for p in parts]))]

    # hardall10: ungated wrong-first
    aw = df[df["wrong"]]
    hall = aw.nlargest(min(N_KEEP, len(aw)), "conf")
    n_fill = N_KEEP - len(hall)
    if n_fill > 0:
        fill = df[~df["wrong"]].nsmallest(n_fill, "conf")
        hall = df.loc[np.sort(np.concatenate([hall.index, fill.index]))]
    else:
        hall = hall.sort_index()

    print()
    write(hard, f"hard10_{sfx}_dpo.jsonl")
    write(ctl, f"hardctl10_{sfx}_dpo.jsonl")
    write(hall, f"hardall10_{sfx}_dpo.jsonl")

    manifest_path = OUT_DIR / f"student_need_manifest_{sfx}.json"
    manifest = {"backbone": args.backbone, "seed": SEED, "n_pairs": N_KEEP,
                "gate": "agreement >= 5/6",
                "rank": "confidently-wrong desc, fill lowest-conf correct",
                "pool": {"n": len(df), "n_wrong": int(df["wrong"].sum()),
                         "n_gated": len(gated), "n_gated_wrong": len(gw)},
                "arms": {f"hard10_{sfx}_dpo": compose(hard),
                         f"hardctl10_{sfx}_dpo": compose(ctl),
                         f"hardall10_{sfx}_dpo": compose(hall)},
                "overlap_hard_vs_ctl": int(len(set(hard["id"]) & set(ctl["id"])))}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"\nmanifest -> {manifest_path.relative_to(ROOT)}")
    for name, m in manifest["arms"].items():
        print(f"  {name:<22} labels {m['label_counts']}  strata {m['strata']}  "
              f"wrong {m['n_wrong_at_base']}")


if __name__ == "__main__":
    main()
