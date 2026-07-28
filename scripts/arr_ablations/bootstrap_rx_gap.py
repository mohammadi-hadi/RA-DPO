#!/usr/bin/env python3
"""Paired bootstrap for the conf+agree vs full-R(x) gap at 50% coverage.

The reliability-ablation table notes that on Standard DPO the two-feature
confidence+agreement rule edges the full three-factor score at 50% coverage
(0.942 vs 0.922). This script substantiates the "within bootstrap confidence
intervals" caption claim: it resamples the 692 test items with replacement
(paired -- the same resample is scored by both rules) and reports the 95%
percentile interval of the accuracy@50 gap for every gpt-4o variant in the
table. OOF scores are computed once on the full test set (exactly as in the
table) and held fixed; the bootstrap varies only the evaluation sample.

Output: results/arr_ablations/rx_component_ablation/bootstrap_gap.json
Run: venv/bin/python scripts/arr_ablations/bootstrap_rx_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "arr_ablations"))
from rx_component_ablation import (  # noqa: E402
    GPT_DIR,
    OUT_DIR,
    expected_acc_at_coverage,
    oof_scores,
    recompute_agreement_reference,
)

MODELS = ["gpt-4o_base", "gpt-4o_SFT", "gpt-4o_Standard_DPO", "gpt-4o_RA-DPO"]
N_BOOT = 10_000
COV = 0.5
SEED = 42


def main() -> int:
    ref_agree = recompute_agreement_reference()
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "coverage": COV, "seed": SEED, "models": {}}
    ok = True

    for name in MODELS:
        pi = json.load(open(GPT_DIR / f"{name}.json"))["per_instance"]
        conf = np.asarray(pi["confidences"], dtype=float)
        agree = np.asarray(pi["agreements"], dtype=float)
        sig = np.asarray(pi["sigmoid_scores"], dtype=float)
        correct = np.asarray(pi["correct"], dtype=int)
        assert np.array_equal(agree, ref_agree), f"{name}: agreement mismatch"

        X = np.column_stack([conf, agree, 1.0 - sig])
        r_pair, _ = oof_scores(X[:, [0, 1]], correct)  # conf + agree
        r_full, _ = oof_scores(X, correct)             # full R(x)

        point_pair = expected_acc_at_coverage(r_pair, correct, COV)
        point_full = expected_acc_at_coverage(r_full, correct, COV)
        point_gap = point_pair - point_full

        n = len(correct)
        gaps = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            gaps[b] = expected_acc_at_coverage(
                r_pair[idx], correct[idx], COV
            ) - expected_acc_at_coverage(r_full[idx], correct[idx], COV)

        lo, hi = np.percentile(gaps, [2.5, 97.5])
        # two-sided sign test on the gap: is 0 inside the interval?
        p_two_sided = 2.0 * min((gaps <= 0).mean(), (gaps >= 0).mean())
        entry = {
            "acc50_conf_agree": round(point_pair, 4),
            "acc50_full_rx": round(point_full, 4),
            "gap_point_pp": round(point_gap * 100, 2),
            "gap_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "zero_inside_ci95": bool(lo <= 0.0 <= hi),
            "p_two_sided": round(min(p_two_sided, 1.0), 4),
        }
        out["models"][name] = entry
        print(f"{name}: gap {entry['gap_point_pp']:+.2f}pp, "
              f"95% CI [{entry['gap_ci95_pp'][0]:+.2f}, "
              f"{entry['gap_ci95_pp'][1]:+.2f}], p={entry['p_two_sided']:.3f}")

    # Self-check: Standard DPO point gap must reproduce the table (0.942-0.922).
    sd = out["models"]["gpt-4o_Standard_DPO"]
    if abs(sd["gap_point_pp"] - 2.02) > 0.15:
        print(f"SELF-CHECK FAILED: Standard DPO gap {sd['gap_point_pp']}pp "
              "(expect ~2.02)")
        ok = False
    else:
        print(f"SELF-CHECK PASSED: Standard DPO gap {sd['gap_point_pp']}pp")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "bootstrap_gap.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
