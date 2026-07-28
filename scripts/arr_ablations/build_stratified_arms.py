"""Stratified selection arms: does reliability help once the distribution is held fixed?

The low-budget factorial showed that selecting by reliability underperforms a
random draw, and the likely mechanism is distribution narrowing: Smart-k% is
100% unanimous, so it trains on the easy tail while the test set spans all
agreement levels. Both arms here keep the training distribution close to the
pool and vary only what is removed, which separates "reliability is useless"
from "reliability selection also happened to shift the distribution".

  strat10     all four agreement strata at their pool proportions, and within
              each stratum the lowest-uncertainty pairs. Distribution matched;
              the only thing that varies is the within-stratum ranking. Tests
              whether the token-uncertainty signal carries anything at all
              once agreement is controlled for.

  noworst10   drops only the fully contested 3/3 stratum (agreement 0.5, a
              coin-flip label) and samples the remaining three strata at their
              relative proportions. This is the minimal intervention that
              removes genuinely unreliable labels without narrowing to the
              easy tail. Prior evidence favours it: training ONLY on that
              stratum (Ambiguous-only) drops the hosted model to 0.653 from a
              prompted 0.724, so those pairs demonstrably hurt.

Both are 554 pairs, matching smart10/random10/randunan10/flip10 exactly, so
every comparison is size-matched.

    python scripts/arr_ablations/build_stratified_arms.py
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

SEED, N_KEEP, WORST = 42, 554, 0.5


def write(df, name):
    p = OUT_DIR / name
    with open(p, "w") as f:
        for r in df.itertuples():
            f.write(json.dumps(make_pair(r.tweet, r.majority_label),
                               ensure_ascii=False) + "\n")
    print(f"  {name:<22} {len(df):>4} pairs")
    return p


def largest_remainder(fracs, total):
    """Apportion `total` across strata so the parts sum exactly to total."""
    raw = [f * total for f in fracs]
    base = [int(np.floor(x)) for x in raw]
    for i in np.argsort([-(r - b) for r, b in zip(raw, base)])[:total - sum(base)]:
        base[i] += 1
    return base


def main():
    df = load_train_df()
    cache = json.load(open(TOKEN_SCORES))
    df["sig"] = [cache[i]["sigmoid_score"] for i in df["id"]]
    df["agr"] = df["agreement_score"].round(3)
    levels = sorted(df["agr"].unique())
    counts = [int((df["agr"] == a).sum()) for a in levels]
    print(f"pool {len(df)}   strata " +
          "  ".join(f"{a}:{c} ({c/len(df):.1%})" for a, c in zip(levels, counts)))

    rng = np.random.default_rng(SEED)

    # --- strat10: pool proportions, lowest uncertainty within each stratum ---
    n_i = largest_remainder([c / len(df) for c in counts], N_KEEP)
    parts = [df[df["agr"] == a].nsmallest(n, "sig") for a, n in zip(levels, n_i)]
    strat = df.loc[np.sort(np.concatenate([p.index.values for p in parts]))]
    print(f"\n  strat10    per-stratum {dict(zip(levels, n_i))}")
    write(strat, "strat10_dpo.jsonl")

    # --- noworst10: drop the 3/3 stratum, proportional over the rest --------
    keep = df[df["agr"] > WORST]
    lv = [a for a in levels if a > WORST]
    ct = [int((keep["agr"] == a).sum()) for a in lv]
    m_i = largest_remainder([c / len(keep) for c in ct], N_KEEP)
    parts = [keep[keep["agr"] == a].iloc[
                 np.sort(rng.choice((keep["agr"] == a).sum(), n, replace=False))]
             for a, n in zip(lv, m_i)]
    nw = df.loc[np.sort(np.concatenate([p.index.values for p in parts]))]
    print(f"  noworst10  per-stratum {dict(zip(lv, m_i))} (3/3 stratum dropped)")
    write(nw, "noworst10_dpo.jsonl")

    print("\n  composition check (share of each arm that is unanimous):")
    for nm, d in (("pool", df), ("strat10", strat), ("noworst10", nw)):
        print(f"    {nm:<10} {(d['agr'] >= 0.999).mean():.3f}   "
              f"contested(0.5) {(d['agr'] <= WORST).mean():.3f}")


if __name__ == "__main__":
    main()
