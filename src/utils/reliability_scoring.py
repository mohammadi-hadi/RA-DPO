"""
Reliability Scoring for Sexism Detection

Computes composite reliability scores combining calibrated confidence,
annotator agreement, and critical token analysis.
Implements the decision layer: PREDICT if R(x) >= threshold, else ABSTAIN.

Agreement can come from real annotator labels OR a trained AgreementPredictor.
Weights (alpha, beta, gamma) can be hand-tuned OR learned via WeightOptimizer.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReliabilityScoringConfig:
    """Configuration for reliability scoring."""
    confidence_weight: float = 0.5    # alpha
    agreement_weight: float = 0.3     # beta
    token_weight: float = 0.2         # gamma
    predict_threshold: float = 0.6


@dataclass
class ReliabilityResult:
    """Result for a single instance from the reliability pipeline."""
    text: str = ""
    prediction: str = ""
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    agreement_score: float = 0.0
    critical_fraction: float = 0.0
    reliability_score: float = 0.0
    decision: str = "ABSTAIN"  # PREDICT or ABSTAIN
    critical_tokens: List[str] = field(default_factory=list)
    token_weights: List[float] = field(default_factory=list)
    lang: str = "en"
    true_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200],  # Truncate for JSON
            "prediction": self.prediction,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "agreement_score": self.agreement_score,
            "critical_fraction": self.critical_fraction,
            "reliability_score": self.reliability_score,
            "decision": self.decision,
            "critical_tokens": self.critical_tokens[:20],
            "lang": self.lang,
            "true_label": self.true_label,
        }


class ReliabilityScorer:
    """
    Computes reliability scores and makes predict/abstain decisions.

    R(x) = alpha * calibrated_conf + beta * agreement + gamma * (1 - critical_fraction)
    Decision: PREDICT if R(x) >= threshold, else ABSTAIN
    """

    def __init__(self, config: Optional[ReliabilityScoringConfig] = None):
        self.config = config or ReliabilityScoringConfig()

    @classmethod
    def from_optimized_weights(cls, optimized_weights) -> "ReliabilityScorer":
        """
        Create a scorer using weights learned by WeightOptimizer.

        Args:
            optimized_weights: An OptimizedWeights instance from
                               ``utils.weight_optimizer.WeightOptimizer.fit()``.
        """
        config = ReliabilityScoringConfig(
            confidence_weight=optimized_weights.confidence_weight,
            agreement_weight=optimized_weights.agreement_weight,
            token_weight=optimized_weights.token_weight,
            predict_threshold=optimized_weights.predict_threshold,
        )
        return cls(config)

    def compute_score(
        self,
        calibrated_confidence: float,
        agreement_score: float,
        critical_fraction: float,
    ) -> float:
        """
        Compute reliability score R(x).

        Args:
            calibrated_confidence: Calibrated model confidence [0, 1]
            agreement_score: Annotator agreement score [0, 1]
            critical_fraction: Fraction of tokens that are critical [0, 1]

        Returns:
            Reliability score in [0, 1]
        """
        c = self.config
        score = (
            c.confidence_weight * calibrated_confidence
            + c.agreement_weight * agreement_score
            + c.token_weight * (1.0 - critical_fraction)
        )
        return float(np.clip(score, 0.0, 1.0))

    def make_decision(self, reliability_score: float) -> str:
        """Make predict/abstain decision based on reliability score."""
        return "PREDICT" if reliability_score >= self.config.predict_threshold else "ABSTAIN"

    def score_instance(
        self,
        calibrated_confidence: float,
        agreement_score: float,
        critical_fraction: float,
    ) -> Tuple[float, str]:
        """
        Score and decide for one instance.

        Returns:
            Tuple of (reliability_score, decision)
        """
        score = self.compute_score(
            calibrated_confidence, agreement_score, critical_fraction
        )
        decision = self.make_decision(score)
        return score, decision

    def score_batch(
        self,
        calibrated_confidences: np.ndarray,
        agreement_scores: np.ndarray,
        critical_fractions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score and decide for a batch of instances.

        Args:
            calibrated_confidences: Array of calibrated confidences
            agreement_scores: Array of agreement scores
            critical_fractions: Array of critical token fractions

        Returns:
            Tuple of (reliability_scores, decisions as bool array where True=PREDICT)
        """
        c = self.config
        scores = (
            c.confidence_weight * np.asarray(calibrated_confidences)
            + c.agreement_weight * np.asarray(agreement_scores)
            + c.token_weight * (1.0 - np.asarray(critical_fractions))
        )
        scores = np.clip(scores, 0.0, 1.0)
        decisions = scores >= c.predict_threshold
        return scores, decisions

    def find_optimal_threshold(
        self,
        reliability_scores: np.ndarray,
        correct: np.ndarray,
        threshold_range: Tuple[float, float] = (0.5, 0.95),
        n_steps: int = 19,
        min_coverage: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Find optimal threshold by sweeping and tracking coverage vs accuracy.

        Args:
            reliability_scores: Array of reliability scores
            correct: Boolean array of correct predictions
            threshold_range: Range of thresholds to try
            n_steps: Number of threshold steps
            min_coverage: Minimum acceptable coverage

        Returns:
            Dictionary with optimal threshold and sweep results
        """
        reliability_scores = np.asarray(reliability_scores)
        correct = np.asarray(correct, dtype=bool)

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        sweep_results = []
        best_f1 = 0.0
        best_threshold = threshold_range[0]

        for tau in thresholds:
            mask = reliability_scores >= tau
            coverage = mask.mean()

            if coverage < min_coverage or mask.sum() == 0:
                sweep_results.append({
                    "threshold": float(tau),
                    "coverage": float(coverage),
                    "accuracy": 0.0,
                    "f1": 0.0,
                })
                continue

            acc = correct[mask].mean()
            # Approximate F1 combining accuracy and coverage
            if acc + coverage > 0:
                f1 = 2 * acc * coverage / (acc + coverage)
            else:
                f1 = 0.0

            sweep_results.append({
                "threshold": float(tau),
                "coverage": float(coverage),
                "accuracy": float(acc),
                "f1": float(f1),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(tau)

        return {
            "optimal_threshold": best_threshold,
            "best_f1": float(best_f1),
            "sweep_results": sweep_results,
        }


def compute_abstention_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    decisions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for the predict/abstain framework.

    Args:
        predictions: Model predictions (e.g., 0 or 1)
        true_labels: Ground truth labels
        decisions: Boolean array where True = PREDICT, False = ABSTAIN

    Returns:
        Dictionary of metrics
    """
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    decisions = np.asarray(decisions, dtype=bool)

    n_total = len(predictions)
    n_predicted = decisions.sum()
    n_abstained = n_total - n_predicted

    coverage = n_predicted / n_total if n_total > 0 else 0.0
    abstention_rate = n_abstained / n_total if n_total > 0 else 0.0

    if n_predicted > 0:
        pred_correct = (predictions[decisions] == true_labels[decisions])
        accuracy_on_predicted = pred_correct.mean()

        # F1 on predicted instances
        from sklearn.metrics import f1_score
        f1_on_predicted = f1_score(
            true_labels[decisions], predictions[decisions],
            average="macro", zero_division=0.0,
        )
    else:
        accuracy_on_predicted = 0.0
        f1_on_predicted = 0.0

    # Overall accuracy (treating abstained as wrong)
    overall_correct = np.zeros(n_total, dtype=bool)
    if n_predicted > 0:
        overall_correct[decisions] = (predictions[decisions] == true_labels[decisions])
    overall_accuracy = overall_correct.mean()

    return {
        "coverage": float(coverage),
        "abstention_rate": float(abstention_rate),
        "n_predicted": int(n_predicted),
        "n_abstained": int(n_abstained),
        "accuracy_on_predicted": float(accuracy_on_predicted),
        "f1_on_predicted": float(f1_on_predicted),
        "overall_accuracy": float(overall_accuracy),
    }


def score_with_predicted_agreement(
    scorer: ReliabilityScorer,
    calibrated_confidences: np.ndarray,
    texts: List[str],
    critical_fractions: np.ndarray,
    agreement_predictor,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score a batch using agreement predicted from text instead of real labels.

    Args:
        scorer: Configured ReliabilityScorer instance.
        calibrated_confidences: Array of calibrated model confidences.
        texts: Raw tweet texts for the agreement predictor.
        critical_fractions: Array of critical token fractions.
        agreement_predictor: A trained ``AgreementPredictor`` instance.
        batch_size: Batch size for the agreement predictor.

    Returns:
        Tuple of (reliability_scores, predict_mask) where predict_mask
        is True for instances that should be predicted (not abstained).
    """
    predicted_agreement = agreement_predictor.predict(texts, batch_size=batch_size)
    return scorer.score_batch(
        calibrated_confidences,
        predicted_agreement,
        critical_fractions,
    )


if __name__ == "__main__":
    # Quick test
    scorer = ReliabilityScorer()
    score = scorer.compute_score(0.8, 0.9, 0.3)
    print(f"R(x) = {score:.4f}")
    decision = scorer.make_decision(score)
    print(f"Decision: {decision}")

    # Batch test
    np.random.seed(42)
    confs = np.random.uniform(0.5, 1.0, 50)
    agrees = np.random.uniform(0.5, 1.0, 50)
    crits = np.random.uniform(0.0, 0.5, 50)
    scores, decisions = scorer.score_batch(confs, agrees, crits)
    print(f"\nBatch: {decisions.sum()}/{len(decisions)} predicted")

    # Abstention metrics
    preds = np.random.randint(0, 2, 50)
    labels = np.random.randint(0, 2, 50)
    metrics = compute_abstention_metrics(preds, labels, decisions)
    print(f"Coverage: {metrics['coverage']:.2f}, Accuracy on predicted: {metrics['accuracy_on_predicted']:.2f}")
