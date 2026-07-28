"""
Confidence Calibration Utilities

Implements temperature scaling and Platt scaling for post-hoc
calibration of DPO model predictions (Cal-DPO inspired).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    method: str = "temperature"  # temperature, platt
    initial_temperature: float = 1.5
    max_iter: int = 100
    lr: float = 0.01
    n_bins: int = 15


class TemperatureScaler:
    """
    Temperature scaling for post-hoc calibration.

    Learns a single temperature parameter T such that
    calibrated_probs = softmax(logits / T) are well-calibrated.
    Optimized via L-BFGS-B on negative log-likelihood.
    """

    def __init__(self, initial_temperature: float = 1.5):
        self.temperature = initial_temperature
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature on validation logits and labels.

        Args:
            logits: Raw model logits, shape (N,) or (N, C)
            labels: Ground truth binary labels, shape (N,)
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        from scipy.optimize import minimize

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        # Handle 1D logits (binary case: logit for positive class)
        if logits.ndim == 1:
            logits_2d = np.stack([-logits, logits], axis=1)
        else:
            logits_2d = logits

        def nll_loss(t):
            t = max(t[0], 0.01)  # Prevent division by zero
            scaled = logits_2d / t
            # Numerically stable log-softmax
            max_vals = scaled.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(scaled - max_vals).sum(axis=1, keepdims=True)) + max_vals
            log_probs = scaled - log_sum_exp
            # NLL for true labels
            return -log_probs[np.arange(len(labels)), labels].mean()

        result = minimize(
            nll_loss,
            x0=[self.temperature],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0)],
            options={"maxiter": max_iter},
        )

        self.temperature = float(result.x[0])
        self._fitted = True
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw logits, shape (N,) or (N, C)

        Returns:
            Calibrated probabilities
        """
        logits = np.asarray(logits, dtype=np.float64)

        if logits.ndim == 1:
            logits_2d = np.stack([-logits, logits], axis=1)
        else:
            logits_2d = logits

        scaled = logits_2d / self.temperature
        # Stable softmax
        max_vals = scaled.max(axis=1, keepdims=True)
        exp_vals = np.exp(scaled - max_vals)
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)

        return probs

    def calibrate_confidence(self, confidence: np.ndarray) -> np.ndarray:
        """
        Calibrate confidence scores (already probabilities).

        Converts confidence to logit space, applies temperature, converts back.

        Args:
            confidence: Confidence scores in [0, 1], shape (N,)

        Returns:
            Calibrated confidence scores
        """
        confidence = np.asarray(confidence, dtype=np.float64)
        # Clamp to avoid log(0)
        eps = 1e-7
        confidence = np.clip(confidence, eps, 1.0 - eps)
        # Convert to logit
        logits = np.log(confidence / (1.0 - confidence))
        # Apply temperature scaling
        probs = self.calibrate(logits)
        # Return positive class probability
        return probs[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class PlattScaler:
    """
    Platt scaling: learns sigmoid(a * logit + b) for calibration.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """
        Fit Platt scaling parameters.

        Args:
            logits: Raw logits or confidence scores, shape (N,)
            labels: Binary labels, shape (N,)
            max_iter: Maximum iterations

        Returns:
            Tuple of (a, b) parameters
        """
        from scipy.optimize import minimize

        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        def nll_loss(params):
            a, b = params
            z = a * logits + b
            # Numerically stable binary cross-entropy
            loss = np.mean(np.maximum(z, 0) - z * labels + np.log(1 + np.exp(-np.abs(z))))
            return loss

        result = minimize(
            nll_loss,
            x0=[self.a, self.b],
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )

        self.a, self.b = float(result.x[0]), float(result.x[1])
        self._fitted = True
        return self.a, self.b

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.

        Args:
            logits: Raw logits, shape (N,)

        Returns:
            Calibrated probabilities for positive class
        """
        logits = np.asarray(logits, dtype=np.float64)
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def compute_calibration_metrics(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute calibration metrics: ECE, MCE, and reliability diagram data.

    Args:
        confidences: Predicted confidence scores, shape (N,)
        correct: Boolean array of whether predictions are correct, shape (N,)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE, MCE, and per-bin data
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()
        bin_counts.append(int(count))

        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            gap = abs(avg_acc - avg_conf)

            bin_centers.append(float((lo + hi) / 2))
            bin_accuracies.append(float(avg_acc))
            bin_confidences.append(float(avg_conf))

            ece += gap * count
            mce = max(mce, gap)
        else:
            bin_centers.append(float((lo + hi) / 2))
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    n_total = len(confidences)
    ece = ece / n_total if n_total > 0 else 0.0

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    logits = np.random.randn(100)
    labels = (logits + np.random.randn(100) * 0.5 > 0).astype(int)

    ts = TemperatureScaler()
    t = ts.fit(logits, labels)
    print(f"TemperatureScaler: T={t:.4f}")
    probs = ts.calibrate(logits)
    print(f"  Calibrated probs shape: {probs.shape}, mean: {probs[:, 1].mean():.4f}")

    ps = PlattScaler()
    a, b = ps.fit(logits, labels)
    print(f"PlattScaler: a={a:.4f}, b={b:.4f}")
    probs_platt = ps.calibrate(logits)
    print(f"  Calibrated probs mean: {probs_platt.mean():.4f}")

    # Test calibration metrics
    conf = np.random.uniform(0.5, 1.0, 100)
    correct = np.random.binomial(1, conf)
    metrics = compute_calibration_metrics(conf, correct)
    print(f"\nCalibration metrics: ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}")
