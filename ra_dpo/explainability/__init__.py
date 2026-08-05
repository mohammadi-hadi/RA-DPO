"""SHAP-based explainability for sexism detection models."""

from .shap_analysis import SHAPAnalyzer, compute_shap_values
from .token_importance import (
    TokenImportanceExtractor,
    extract_important_tokens,
    CriticalTokenDetector,
    PreferenceTokenHighlighter,
)
from .continuous_token_scoring import SigmoidTokenScorer

__all__ = [
    "SHAPAnalyzer",
    "compute_shap_values",
    "TokenImportanceExtractor",
    "extract_important_tokens",
    "CriticalTokenDetector",
    "PreferenceTokenHighlighter",
    "SigmoidTokenScorer",
]
