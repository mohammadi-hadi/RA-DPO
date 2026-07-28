"""
Logistic Regression Weight Optimizer

Learns optimal weights (alpha, beta, gamma) for the reliability scoring
formula R(x) = alpha*conf + beta*agree + gamma*(1-critical_frac) using
logistic regression on validation data.

Instead of hand-tuning, the LR coefficients directly become the weights.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss


@dataclass
class OptimizedWeights:
    """Result of weight optimization."""
    confidence_weight: float   # alpha
    agreement_weight: float    # beta
    token_weight: float        # gamma
    intercept: float
    predict_threshold: float
    train_accuracy: float
    train_f1: float
    val_accuracy: Optional[float] = None
    val_f1: Optional[float] = None


class WeightOptimizer:
    """
    Learns reliability scoring weights via logistic regression.

    Features: [calibrated_confidence, agreement_score, 1 - critical_fraction]
    Target:   1 if model prediction is correct, 0 otherwise

    The fitted LR coefficients are normalized to sum to 1 and used as
    alpha, beta, gamma. The LR decision boundary gives the threshold.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Args:
            C: Inverse regularization strength for logistic regression.
            max_iter: Maximum iterations for solver convergence.
        """
        self.C = C
        self.max_iter = max_iter
        self.lr = None
        self.scaler = None
        self._fitted = False

    @staticmethod
    def _build_features(
        calibrated_confidences: np.ndarray,
        agreement_scores: np.ndarray,
        critical_fractions: np.ndarray,
    ) -> np.ndarray:
        """Stack the three signals into a feature matrix."""
        return np.column_stack([
            np.asarray(calibrated_confidences),
            np.asarray(agreement_scores),
            1.0 - np.asarray(critical_fractions),
        ])

    def fit(
        self,
        calibrated_confidences: np.ndarray,
        agreement_scores: np.ndarray,
        critical_fractions: np.ndarray,
        correct: np.ndarray,
    ) -> OptimizedWeights:
        """
        Fit logistic regression to learn optimal weights.

        Args:
            calibrated_confidences: Model confidence after calibration [0,1]
            agreement_scores:       Annotator agreement scores [0,1]
            critical_fractions:     Fraction of critical tokens [0,1]
            correct:                Boolean array — True if prediction was correct

        Returns:
            OptimizedWeights with learned alpha, beta, gamma and threshold.
        """
        X = self._build_features(
            calibrated_confidences, agreement_scores, critical_fractions
        )
        y = np.asarray(correct, dtype=int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.lr = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs"
        )
        self.lr.fit(X_scaled, y)
        self._fitted = True

        # Extract and normalize coefficients to get weights
        raw_coefs = self.lr.coef_[0]
        # Use absolute values — all three signals should contribute positively
        abs_coefs = np.abs(raw_coefs)
        total = abs_coefs.sum()
        if total > 0:
            weights = abs_coefs / total
        else:
            weights = np.array([1 / 3, 1 / 3, 1 / 3])

        # Compute threshold from decision boundary on unscaled features
        # R(x) = w·features, threshold = value where P(correct)=0.5
        # Approximate by finding the R(x) cutoff on training data
        r_scores = weights[0] * calibrated_confidences + \
                   weights[1] * agreement_scores + \
                   weights[2] * (1.0 - critical_fractions)
        threshold = self._find_threshold(r_scores, y)

        # Training metrics
        y_pred = self.lr.predict(X_scaled)
        train_acc = accuracy_score(y, y_pred)
        train_f1 = f1_score(y, y_pred, average="macro", zero_division=0.0)

        return OptimizedWeights(
            confidence_weight=float(weights[0]),
            agreement_weight=float(weights[1]),
            token_weight=float(weights[2]),
            intercept=float(self.lr.intercept_[0]),
            predict_threshold=float(threshold),
            train_accuracy=float(train_acc),
            train_f1=float(train_f1),
        )

    def evaluate(
        self,
        calibrated_confidences: np.ndarray,
        agreement_scores: np.ndarray,
        critical_fractions: np.ndarray,
        correct: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the fitted LR on held-out data."""
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate()")

        X = self._build_features(
            calibrated_confidences, agreement_scores, critical_fractions
        )
        y = np.asarray(correct, dtype=int)
        X_scaled = self.scaler.transform(X)

        y_pred = self.lr.predict(X_scaled)
        y_prob = self.lr.predict_proba(X_scaled)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0.0)),
            "log_loss": float(log_loss(y, y_prob)),
        }

    @staticmethod
    def _find_threshold(
        r_scores: np.ndarray,
        correct: np.ndarray,
        low: float = 0.3,
        high: float = 0.95,
        n_steps: int = 50,
        min_coverage: float = 0.3,
    ) -> float:
        """Sweep R(x) threshold to maximize F1(accuracy, coverage)."""
        best_f1 = 0.0
        best_tau = 0.5

        for tau in np.linspace(low, high, n_steps):
            mask = r_scores >= tau
            coverage = mask.mean()
            if coverage < min_coverage or mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            if acc + coverage > 0:
                f1 = 2 * acc * coverage / (acc + coverage)
            else:
                f1 = 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        return best_tau


def optimize_reliability_weights(
    calibrated_confidences: np.ndarray,
    agreement_scores: np.ndarray,
    critical_fractions: np.ndarray,
    correct: np.ndarray,
    val_calibrated_confidences: Optional[np.ndarray] = None,
    val_agreement_scores: Optional[np.ndarray] = None,
    val_critical_fractions: Optional[np.ndarray] = None,
    val_correct: Optional[np.ndarray] = None,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Convenience function: fit LR, return optimized weights and metrics.

    Args:
        calibrated_confidences: Training confidence scores
        agreement_scores:       Training agreement scores
        critical_fractions:     Training critical token fractions
        correct:                Training correctness labels
        val_*:                  Optional validation arrays
        C:                      LR regularization

    Returns:
        Dictionary with 'weights' (OptimizedWeights) and optional 'val_metrics'.
    """
    optimizer = WeightOptimizer(C=C)
    weights = optimizer.fit(
        calibrated_confidences, agreement_scores, critical_fractions, correct
    )

    result: Dict[str, Any] = {"weights": weights}

    if val_calibrated_confidences is not None:
        val_metrics = optimizer.evaluate(
            val_calibrated_confidences,
            val_agreement_scores,
            val_critical_fractions,
            val_correct,
        )
        weights.val_accuracy = val_metrics["accuracy"]
        weights.val_f1 = val_metrics["f1_macro"]
        result["val_metrics"] = val_metrics

    print(f"\nOptimized Reliability Weights:")
    print(f"  alpha (confidence): {weights.confidence_weight:.4f}")
    print(f"  beta  (agreement):  {weights.agreement_weight:.4f}")
    print(f"  gamma (token):      {weights.token_weight:.4f}")
    print(f"  threshold:          {weights.predict_threshold:.4f}")
    print(f"  train accuracy:     {weights.train_accuracy:.4f}")
    if weights.val_f1 is not None:
        print(f"  val F1:             {weights.val_f1:.4f}")

    return result
