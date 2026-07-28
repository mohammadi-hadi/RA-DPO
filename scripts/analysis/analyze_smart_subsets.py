"""Analyze composition of Smart-30%, Smart-50%, Random-50% training subsets
to explain why the F1 curve is flat.

Hypotheses to test:
  H1: Subsets have different agreement distributions (quality).
  H2: Subsets have different label / language balance.
  H3: Subsets have different tweet length profiles.
  H4: The subsets overlap enough that they're learning similar things.
  H5: Learning saturates — anything above ~1000 pairs is enough.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score

SUBSET_FILES = {
    "Smart-30%": ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl",
    "Smart-50%": ROOT / "results" / "smart_sampling" / "smart50_dpo.jsonl",
    "Random-50%": ROOT / "results" / "smart_sampling" / "random50_dpo.jsonl",
    "Standard (100%)": ROOT / "results" / "openai_dpo_train.jsonl",
}

OUT_DIR = ROOT / "results" / "unified_gpt4o" / "subset_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_tweet(user_content: str) -> str:
    """Pull the tweet text out of the user prompt."""
    m = re.search(r"Post:\s*(.*?)\s*\n\s*Classification", user_content, re.DOTALL)
    return m.group(1).strip() if m else user_content.strip()


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            inp = d["input"]
            msgs = inp["messages"] if isinstance(inp, dict) else inp
            user_msg = msgs[-1]["content"]
            tweet = extract_tweet(user_msg)
            pref = d["preferred_output"]
            if isinstance(pref, list):
                pref = pref[0]
            rej = d["non_preferred_output"]
            if isinstance(rej, list):
                rej = rej[0]
            pairs.append({
                "tweet": tweet,
                "chosen": pref["content"].strip().upper(),
                "rejected": rej["content"].strip().upper(),
            })
    return pairs


def build_train_lookup():
    """Build tweet → (agreement, lang, majority_label) lookup from TRAIN split."""
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, _, _ = loader.create_train_val_test_split(df)
    lookup = {}
    for _, row in train_df.iterrows():
        lookup[row["tweet"].strip()] = {
            "agreement": float(row["agreement_score"]),
            "lang": row["lang"],
            "majority_label": row["majority_label"],
            "length": len(row["tweet"]),
        }
    return lookup, len(train_df)


def main():
    print("Building training lookup...")
    lookup, n_train = build_train_lookup()
    print(f"  {n_train} unique training tweets")

    # Analyse each subset
    rows_summary = []
    subset_tweets = {}
    for name, path in SUBSET_FILES.items():
        pairs = load_pairs(path)
        matched = [lookup[p["tweet"]] for p in pairs if p["tweet"] in lookup]
        pct = len(matched) / len(pairs) * 100
        print(f"\n{name}: {len(pairs)} pairs, {len(matched)} matched ({pct:.1f}%)")

        agree = np.asarray([m["agreement"] for m in matched])
        langs = [m["lang"] for m in matched]
        labels = [m["majority_label"] for m in matched]
        lengths = np.asarray([m["length"] for m in matched])

        row = {
            "subset": name,
            "n_pairs": len(pairs),
            "agree_mean": float(agree.mean()),
            "agree_median": float(np.median(agree)),
            "agree_std": float(agree.std()),
            "agree_ge_5of6": float((agree >= 5 / 6).mean()),  # fraction with 5/6 agreement
            "agree_eq_6of6": float((agree >= 0.999).mean()),  # unanimous
            "pct_en": float(sum(1 for l in langs if l == "en") / len(langs)),
            "pct_yes": float(sum(1 for l in labels if l == "YES") / len(labels)),
            "tweet_len_median": float(np.median(lengths)),
            "tweet_len_p90": float(np.quantile(lengths, 0.9)),
        }
        rows_summary.append(row)
        subset_tweets[name] = set(p["tweet"] for p in pairs)

    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(OUT_DIR / "subset_composition.csv", index=False)

    print("\n=== Subset composition ===")
    print(df_summary.to_string(index=False))

    # Overlaps
    print("\n=== Overlap (fraction of Smaller subset contained in Larger) ===")
    overlaps = []
    names = list(subset_tweets.keys())
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j: continue
            a, b = names[i], names[j]
            inter = len(subset_tweets[a] & subset_tweets[b])
            frac_a = inter / len(subset_tweets[a])
            overlaps.append({"A": a, "B": b, "intersection": inter,
                             "frac_of_A_in_B": frac_a})
            print(f"  {a:>18} ∩ {b:<18}: {inter:>5} tweets  ({frac_a*100:.1f}% of {a})")
    pd.DataFrame(overlaps).to_csv(OUT_DIR / "overlaps.csv", index=False)

    # Agreement histogram per subset
    print("\n=== Agreement distribution (counts) ===")
    bins = [0.5, 0.67, 0.83, 1.0, 1.01]
    labels = ["3/6 (split)", "4/6", "5/6", "6/6 (unanimous)"]
    hist_rows = []
    for name, path in SUBSET_FILES.items():
        pairs = load_pairs(path)
        matched = [lookup[p["tweet"]] for p in pairs if p["tweet"] in lookup]
        agree = np.asarray([m["agreement"] for m in matched])
        counts, _ = np.histogram(agree, bins=bins)
        pct = counts / counts.sum() * 100
        hist_rows.append({"subset": name, **{lab: int(c) for lab, c in zip(labels, counts)}})
        print(f"  {name:<18}: " + "  ".join(f"{lab}={c} ({p:.0f}%)" for lab, c, p in zip(labels, counts, pct)))
    pd.DataFrame(hist_rows).to_csv(OUT_DIR / "agreement_histogram.csv", index=False)

    # ---- Simple learning-curve fit: F1 = a - b * exp(-c * N) ----
    from scipy.optimize import curve_fit
    f1_by_pairs = json.load(open(ROOT / "results" / "unified_gpt4o" / "smart_curve_stats.json"))
    # Use: (pairs, f1_mean, weight = 1/SE)
    points = [(v["pairs"], v["f1_mean"], v["f1_se"]) for v in f1_by_pairs.values() if v["pairs"] > 0]
    # Drop RA-DPO's oversampled 8984 (not the same regime)
    points = [p for p in points if p[0] <= 6000]
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    ws = 1.0 / np.array([p[2] for p in points], dtype=float)
    try:
        def model(n, a, b, c):
            return a - b * np.exp(-c * n)
        popt, _ = curve_fit(model, xs, ys, p0=[0.83, 0.2, 0.005], sigma=1 / ws, maxfev=5000)
        print(f"\n=== Learning-curve fit (F1 = a - b·exp(-c·N)) ===")
        print(f"  a (saturation)   = {popt[0]:.4f}")
        print(f"  b (initial drop) = {popt[1]:.4f}")
        print(f"  c (rate)         = {popt[2]:.6f}")
        n_to_99 = -np.log(0.01 / popt[1]) / popt[2]
        n_to_95 = -np.log(0.05 / popt[1]) / popt[2]
        print(f"  95% of max F1 reached at ~{int(n_to_95)} pairs")
        print(f"  99% of max F1 reached at ~{int(n_to_99)} pairs")
    except Exception as e:
        print(f"Learning-curve fit failed: {e}")

    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
