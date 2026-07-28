"""Utility functions for sexism detection."""

from .metrics import compute_metrics, evaluate_predictions, generate_results_table
from .label_mapping import LabelMapper, map_llm_output
from .visualization import plot_results, plot_confusion_matrix, plot_token_importance
from .calibration import TemperatureScaler, PlattScaler, compute_calibration_metrics
from .reliability_scoring import ReliabilityScorer, ReliabilityResult, compute_abstention_metrics, score_with_predicted_agreement
from .weight_optimizer import WeightOptimizer, OptimizedWeights, optimize_reliability_weights
from .statistical_tests import test_agreement_vs_correctness, test_length_vs_correctness

__all__ = [
    "compute_metrics",
    "evaluate_predictions",
    "generate_results_table",
    "LabelMapper",
    "map_llm_output",
    "plot_results",
    "plot_confusion_matrix",
    "plot_token_importance",
    "TemperatureScaler",
    "PlattScaler",
    "compute_calibration_metrics",
    "ReliabilityScorer",
    "ReliabilityResult",
    "compute_abstention_metrics",
    "score_with_predicted_agreement",
    "WeightOptimizer",
    "OptimizedWeights",
    "optimize_reliability_weights",
    "test_agreement_vs_correctness",
    "test_length_vs_correctness",
]
