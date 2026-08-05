"""
Simple Statistical Tests for Research Analysis

Two tests:
1. Does higher annotator agreement make it easier for models? (chi-square + t-test)
2. Does text length affect prediction correctness? (t-test + correlation)
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple


def test_agreement_vs_correctness(
    agreement_scores: np.ndarray,
    is_correct: np.ndarray,
) -> Dict[str, Any]:
    """
    Test: instances with higher annotator agreement are easier for the model.

    Uses:
    - Independent t-test: compare agreement of correct vs incorrect predictions
    - Chi-square: bin agreement into high/low and test independence with correctness

    Returns dict with all test results.
    """
    agreement_scores = np.asarray(agreement_scores, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)

    correct_agree = agreement_scores[is_correct]
    incorrect_agree = agreement_scores[~is_correct]

    # T-test: is mean agreement different for correct vs incorrect?
    t_stat, t_p = stats.ttest_ind(correct_agree, incorrect_agree)

    # Chi-square: high agreement (>= 0.83) vs low (< 0.83) × correct/incorrect
    high = agreement_scores >= 0.83
    ct = np.array([
        [(high & is_correct).sum(), (high & ~is_correct).sum()],
        [(~high & is_correct).sum(), (~high & ~is_correct).sum()],
    ])
    chi2, chi_p, _, _ = stats.chi2_contingency(ct)

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        ((len(correct_agree) - 1) * correct_agree.std()**2 +
         (len(incorrect_agree) - 1) * incorrect_agree.std()**2) /
        (len(correct_agree) + len(incorrect_agree) - 2)
    ) if len(incorrect_agree) > 1 else 1.0
    cohens_d = (correct_agree.mean() - incorrect_agree.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        "t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "significance": _sig(t_p),
            "mean_correct": float(correct_agree.mean()),
            "mean_incorrect": float(incorrect_agree.mean()),
            "cohens_d": float(cohens_d),
        },
        "chi_square": {
            "chi2": float(chi2),
            "p_value": float(chi_p),
            "significance": _sig(chi_p),
            "contingency_table": ct.tolist(),
        },
        "n_correct": int(is_correct.sum()),
        "n_incorrect": int((~is_correct).sum()),
    }


def test_length_vs_correctness(
    text_lengths: np.ndarray,
    is_correct: np.ndarray,
) -> Dict[str, Any]:
    """
    Test: does text length affect whether the model predicts correctly?

    Uses:
    - Independent t-test: compare length of correct vs incorrect predictions
    - Spearman correlation: length vs correctness

    Returns dict with all test results.
    """
    text_lengths = np.asarray(text_lengths, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)

    correct_len = text_lengths[is_correct]
    incorrect_len = text_lengths[~is_correct]

    # T-test
    t_stat, t_p = stats.ttest_ind(correct_len, incorrect_len)

    # Spearman correlation
    rho, sp_p = stats.spearmanr(text_lengths, is_correct.astype(float))

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        ((len(correct_len) - 1) * correct_len.std()**2 +
         (len(incorrect_len) - 1) * incorrect_len.std()**2) /
        (len(correct_len) + len(incorrect_len) - 2)
    ) if len(incorrect_len) > 1 else 1.0
    cohens_d = (correct_len.mean() - incorrect_len.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        "t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "significance": _sig(t_p),
            "mean_correct": float(correct_len.mean()),
            "mean_incorrect": float(incorrect_len.mean()),
            "cohens_d": float(cohens_d),
        },
        "spearman": {
            "rho": float(rho),
            "p_value": float(sp_p),
            "significance": _sig(sp_p),
        },
        "n_correct": int(is_correct.sum()),
        "n_incorrect": int((~is_correct).sum()),
    }


def _sig(p: float) -> str:
    """Significance marker."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"
