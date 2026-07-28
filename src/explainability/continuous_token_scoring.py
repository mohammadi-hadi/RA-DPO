"""
Sigmoid-Based Continuous Token Scoring

Instead of the binary ConfPO approach (token below average → critical = 1, else 0),
this uses a sigmoid function to give continuous weights:

    w_i = sigmoid(k * (T - p_i))

Where:
    T = average token probability (same threshold as ConfPO)
    p_i = probability of token i (exp(logprob_i))
    k = steepness parameter (higher = closer to binary)

Effect: tokens slightly below threshold get moderate weight,
tokens far below get weight close to 1. This captures magnitude
of deviation, not just direction.

The weighted_critical_score replaces binary critical_fraction:
    score = sum(w_i for meaningful tokens) / n_meaningful

Literature context:
- ConfPO (2025): binary threshold at average logprob
- TIS-DPO (ICLR 2025): exponential token weighting w = k·exp(μ·r)
- Our approach: sigmoid interpolation between binary and continuous
"""

from typing import Dict, List, Any

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class SigmoidTokenScorer:
    """
    Sigmoid-based continuous token scoring.

    w_i = sigmoid(k * (T - p_i))

    Tokens with probability far below T get w_i ≈ 1 (very critical).
    Tokens with probability far above T get w_i ≈ 0 (not critical).
    Tokens near T get intermediate weights (the key advantage over binary).
    """

    def __init__(
        self,
        steepness: float = 10.0,
        min_token_length: int = 2,
        stopwords: Dict[str, List[str]] = None,
    ):
        """
        Args:
            steepness: k parameter. Higher = sharper transition (more binary-like).
                       Lower = smoother (more continuous). Default 10.0 works well.
            min_token_length: Minimum characters for a meaningful token.
            stopwords: Language-specific stopwords dict.
        """
        self.steepness = steepness

        from .token_importance import TokenImportanceExtractor
        _default = TokenImportanceExtractor(stopwords=stopwords)
        self.stopwords = _default.stopwords
        self.min_token_length = min_token_length

    def _is_meaningful(self, token: str, lang: str = "en") -> bool:
        """Check if token should be considered."""
        t = token.strip().lower()
        if len(t) < self.min_token_length:
            return False
        if t in self.stopwords.get(lang, []):
            return False
        if not any(c.isalpha() for c in t):
            return False
        return True

    def score_tokens(
        self,
        tokens: List[str],
        logprobs: np.ndarray,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Score tokens using sigmoid weighting.

        Args:
            tokens: Token strings
            logprobs: Per-token log-probabilities
            lang: Language code

        Returns:
            Dict with weights, sigmoid_critical_score, and comparison to binary
        """
        logprobs = np.asarray(logprobs, dtype=np.float64)

        # Convert to probabilities
        probs = np.exp(logprobs)

        # Threshold T = average probability (same as ConfPO)
        # Use log-sum-exp for numerical stability
        max_lp = logprobs.max()
        avg_prob = np.mean(np.exp(logprobs - max_lp)) * np.exp(max_lp)
        avg_logp = max_lp + np.log(np.mean(np.exp(logprobs - max_lp)))

        # Sigmoid weights: w_i = sigmoid(k * (T - p_i))
        # When p_i < T: (T - p_i) > 0 → w_i > 0.5 (critical)
        # When p_i > T: (T - p_i) < 0 → w_i < 0.5 (not critical)
        weights = sigmoid(self.steepness * (avg_prob - probs))

        # Build meaningful-token mask
        meaningful = np.array([self._is_meaningful(t, lang) for t in tokens])

        # Sigmoid critical score: mean weight over meaningful tokens
        if meaningful.sum() > 0:
            sigmoid_score = float(weights[meaningful].mean())
        else:
            sigmoid_score = 0.5

        # Binary critical fraction for comparison
        binary_mask = logprobs < avg_logp
        n_meaningful = meaningful.sum()
        binary_fraction = float((binary_mask & meaningful).sum() / n_meaningful) if n_meaningful > 0 else 0.0

        return {
            "weights": weights,
            "sigmoid_critical_score": sigmoid_score,
            "binary_critical_fraction": binary_fraction,
            "avg_prob": float(avg_prob),
            "avg_logp": float(avg_logp),
            "n_meaningful": int(n_meaningful),
            "meaningful_mask": meaningful,
        }
