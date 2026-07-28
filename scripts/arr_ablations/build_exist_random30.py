"""Build the size-matched random control for EXIST's Smart-30% subset.

EXIST ships Random-50% (2,768 pairs), which is size-matched to Smart-50%, not
to the headline Smart-30% (1,661). Without a 1,661-pair random arm there is no
way to tell whether the reliability ranking selects better pairs or whether
1,661 pairs is simply enough. The EDOS track now has that control and reports
no significant gap (McNemar p = 0.22); this builds the EXIST counterpart so
the same question can be asked on the four-level agreement signal.

Records are sampled verbatim from results/openai_dpo_train.jsonl, so the JSONL
is byte-identical in format to every other training file by construction, and
row order follows the original train order exactly as smart30_dpo.jsonl does.

    python scripts/arr_ablations/build_exist_random30.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results" / "openai_dpo_train.jsonl"
SMART30 = ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl"
OUT = ROOT / "results" / "smart_sampling" / "random30_dpo.jsonl"
SEED = 42


def prompt_text(rec: dict) -> str:
    inp = rec["input"]
    msgs = inp["messages"] if isinstance(inp, dict) else inp
    return msgs[-1]["content"]


def main():
    lines = FULL.read_text().splitlines()
    n_total = len(lines)
    n_keep = sum(1 for _ in open(SMART30))
    print(f"full train pairs : {n_total}")
    print(f"smart30 pairs    : {n_keep}  (target size)")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
    subset = [lines[i] for i in idx]
    assert len(subset) == n_keep

    OUT.write_text("\n".join(subset) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(subset)} pairs, seed {SEED})")

    # Overlap with Smart-30 should sit at the chance rate; anything far from it
    # means the sample is not independent of the reliability ranking.
    s30 = {prompt_text(json.loads(l)) for l in open(SMART30)}
    r30 = [prompt_text(json.loads(l)) for l in subset]
    ov = len(s30 & set(r30)) / len(r30)
    chance = n_keep / n_total
    print(f"overlap with Smart-30: {ov:.3f}  (chance {chance:.3f})")
    assert abs(ov - chance) < 0.05, "sample is not independent of the ranking"

    # Agreement composition is the thing the ranking is supposed to change:
    # Smart-30 on EXIST is 100% unanimous, a random draw should not be.
    print("\nAdd to configs/local_pipeline.yaml -> data.training_pairs:")
    print(f"    random30_dpo: {n_keep}")


if __name__ == "__main__":
    main()
