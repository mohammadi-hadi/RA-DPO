"""
Token Importance Extraction

Extracts and manages important tokens from SHAP analysis
for XAI-enhanced model training.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


class TokenImportanceExtractor:
    """
    Extracts and manages important tokens from explainability analysis.

    Supports filtering, thresholding, and language-specific token management.
    """

    def __init__(
        self,
        importance_threshold: float = 0.1,
        top_k: int = 50,
        min_token_length: int = 2,
        stopwords: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the token importance extractor.

        Args:
            importance_threshold: Minimum importance score
            top_k: Maximum number of tokens to extract
            min_token_length: Minimum token length
            stopwords: Dictionary of stopwords per language
        """
        self.importance_threshold = importance_threshold
        self.top_k = top_k
        self.min_token_length = min_token_length

        # Default stopwords
        self.stopwords = stopwords or {
            'en': ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                   'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                   'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                   'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                   'by', 'from', 'as', 'into', 'through', 'during', 'before',
                   'after', 'above', 'below', 'between', 'under', 'again',
                   'further', 'then', 'once', 'here', 'there', 'when', 'where',
                   'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                   'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                   'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
                   'or', 'because', 'until', 'while', 'this', 'that', 'these',
                   'those', 'what', 'which', 'who', 'whom', 'i', 'me', 'my',
                   'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                   'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                   'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                   'itself', 'they', 'them', 'their', 'theirs', 'themselves'],
            'es': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de',
                   'del', 'al', 'a', 'en', 'y', 'o', 'que', 'es', 'son', 'fue',
                   'fueron', 'ser', 'estar', 'ha', 'han', 'he', 'hemos', 'hay',
                   'por', 'para', 'con', 'sin', 'sobre', 'entre', 'pero', 'si',
                   'no', 'como', 'cuando', 'donde', 'quien', 'cual', 'cuyo',
                   'este', 'esta', 'estos', 'estas', 'ese', 'esa', 'esos',
                   'esas', 'aquel', 'aquella', 'yo', 'tu', 'el', 'ella',
                   'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te', 'le',
                   'nos', 'os', 'les', 'mi', 'mis', 'tu', 'tus', 'su', 'sus',
                   'muy', 'mas', 'ya', 'tambien', 'solo', 'todo', 'toda',
                   'todos', 'todas', 'otro', 'otra', 'otros', 'otras']
        }

        self.important_tokens: Dict[str, Dict[str, float]] = {}

    def filter_token(self, token: str, lang: str = 'en') -> bool:
        """
        Check if token should be included.

        Args:
            token: Token to check
            lang: Language code

        Returns:
            True if token should be included
        """
        # Clean token
        token = token.strip().lower()

        # Check length
        if len(token) < self.min_token_length:
            return False

        # Check if stopword
        if token in self.stopwords.get(lang, []):
            return False

        # Check for non-alphabetic tokens
        if not any(c.isalpha() for c in token):
            return False

        return True

    def extract_from_shap(
        self,
        shap_results: Dict[str, Any],
        lang: str = 'en'
    ) -> Dict[str, float]:
        """
        Extract important tokens from SHAP results.

        Args:
            shap_results: SHAP analysis results
            lang: Language code

        Returns:
            Dictionary of important tokens and scores
        """
        token_importance = shap_results.get('token_importance', {})

        # Filter and threshold
        filtered_tokens = {}

        for token, importance in token_importance.items():
            if importance < self.importance_threshold:
                continue

            if not self.filter_token(token, lang):
                continue

            filtered_tokens[token] = importance

        # Sort and take top-k
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        result = dict(sorted_tokens)
        self.important_tokens[lang] = result

        return result

    def extract_from_multiple_sources(
        self,
        sources: List[Dict[str, Any]],
        lang: str = 'en',
        aggregation: str = 'max'
    ) -> Dict[str, float]:
        """
        Extract tokens from multiple SHAP result sources.

        Args:
            sources: List of SHAP result dictionaries
            lang: Language code
            aggregation: How to aggregate ('max', 'mean', 'sum')

        Returns:
            Aggregated important tokens
        """
        all_tokens = defaultdict(list)

        for source in sources:
            token_importance = source.get('token_importance', {})

            for token, importance in token_importance.items():
                if self.filter_token(token, lang):
                    all_tokens[token].append(importance)

        # Aggregate
        aggregated = {}
        for token, scores in all_tokens.items():
            if aggregation == 'max':
                aggregated[token] = max(scores)
            elif aggregation == 'mean':
                aggregated[token] = np.mean(scores)
            elif aggregation == 'sum':
                aggregated[token] = sum(scores)

        # Filter and sort
        filtered = {
            k: v for k, v in aggregated.items()
            if v >= self.importance_threshold
        }

        sorted_tokens = sorted(
            filtered.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        result = dict(sorted_tokens)
        self.important_tokens[lang] = result

        return result

    def get_tokens_list(self, lang: str = 'en') -> List[str]:
        """
        Get list of important tokens for a language.

        Args:
            lang: Language code

        Returns:
            List of important tokens (ordered by importance)
        """
        return list(self.important_tokens.get(lang, {}).keys())

    def get_all_tokens(self) -> Dict[str, List[str]]:
        """
        Get important tokens for all languages.

        Returns:
            Dictionary mapping language to token list
        """
        return {
            lang: list(tokens.keys())
            for lang, tokens in self.important_tokens.items()
        }

    def save(self, output_path: str):
        """Save important tokens to file."""
        with open(output_path, 'w') as f:
            json.dump(self.important_tokens, f, indent=2)

    def load(self, input_path: str):
        """Load important tokens from file."""
        with open(input_path, 'r') as f:
            self.important_tokens = json.load(f)

    def merge_with(
        self,
        other_tokens: Dict[str, Dict[str, float]],
        aggregation: str = 'max'
    ):
        """
        Merge with another set of important tokens.

        Args:
            other_tokens: Other token importance dictionary
            aggregation: How to aggregate overlapping tokens
        """
        for lang, tokens in other_tokens.items():
            if lang not in self.important_tokens:
                self.important_tokens[lang] = {}

            for token, importance in tokens.items():
                if token in self.important_tokens[lang]:
                    if aggregation == 'max':
                        self.important_tokens[lang][token] = max(
                            self.important_tokens[lang][token],
                            importance
                        )
                    elif aggregation == 'mean':
                        self.important_tokens[lang][token] = (
                            self.important_tokens[lang][token] + importance
                        ) / 2
                else:
                    self.important_tokens[lang][token] = importance


def extract_important_tokens(
    shap_results: Dict[str, Any],
    lang: str = 'en',
    top_k: int = 50,
    threshold: float = 0.1,
    output_path: Optional[str] = None
) -> List[str]:
    """
    Convenience function to extract important tokens.

    Args:
        shap_results: SHAP analysis results
        lang: Language code
        top_k: Maximum number of tokens
        threshold: Minimum importance threshold
        output_path: Optional path to save results

    Returns:
        List of important tokens
    """
    extractor = TokenImportanceExtractor(
        importance_threshold=threshold,
        top_k=top_k
    )

    extractor.extract_from_shap(shap_results, lang)

    if output_path:
        extractor.save(output_path)

    return extractor.get_tokens_list(lang)


# Pre-defined important tokens based on sexism detection literature
PREDEFINED_SEXIST_TOKENS = {
    'en': [
        'woman', 'women', 'girl', 'girls', 'female', 'females',
        'bitch', 'slut', 'whore', 'hoe', 'cunt',
        'kitchen', 'cook', 'clean', 'sandwich',
        'emotional', 'crazy', 'hysterical',
        'sexy', 'hot', 'beautiful', 'ugly',
        'feminist', 'feminism', 'feminazi',
        'rape', 'abuse', 'harass', 'assault',
        'belong', 'place', 'role', 'deserve',
        'weak', 'inferior', 'stupid', 'dumb'
    ],
    'es': [
        'mujer', 'mujeres', 'chica', 'chicas', 'hembra',
        'puta', 'zorra', 'perra', 'golfa',
        'cocina', 'limpiar', 'casa', 'hogar',
        'emocional', 'loca', 'histérica',
        'sexy', 'guapa', 'fea',
        'feminista', 'feminismo',
        'violación', 'abuso', 'acoso',
        'pertenecer', 'lugar', 'papel', 'merecer',
        'débil', 'inferior', 'estúpida', 'tonta'
    ]
}


class CriticalTokenDetector:
    """
    ConfPO-style critical token detection.

    A token is critical if its log-probability is below the arithmetic mean
    of all token log-probabilities in the sequence:
        avg_logp = log(mean(exp(logp_i)))
        critical if logp_i < avg_logp
    """

    def __init__(
        self,
        min_token_length: int = 2,
        stopwords: Optional[Dict[str, List[str]]] = None,
    ):
        # Reuse stopwords from TokenImportanceExtractor defaults
        _default_extractor = TokenImportanceExtractor(stopwords=stopwords)
        self.stopwords = _default_extractor.stopwords
        self.min_token_length = min_token_length

    def _filter_token(self, token: str, lang: str = "en") -> bool:
        """Check if token should be considered (reuses existing filter logic)."""
        token_clean = token.strip().lower()
        if len(token_clean) < self.min_token_length:
            return False
        if token_clean in self.stopwords.get(lang, []):
            return False
        if not any(c.isalpha() for c in token_clean):
            return False
        return True

    def detect_critical(
        self,
        tokens: List[str],
        logprobs: np.ndarray,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Detect critical tokens using ConfPO arithmetic mean threshold.

        Args:
            tokens: List of token strings
            logprobs: Array of per-token log-probabilities
            lang: Language code for stopword filtering

        Returns:
            Dictionary with critical tokens, mask, avg_logp, fraction
        """
        logprobs = np.asarray(logprobs, dtype=np.float64)

        # Arithmetic mean in probability space, then back to log:
        # avg_logp = log(mean(exp(logp_i)))
        # Use log-sum-exp trick for numerical stability
        max_lp = logprobs.max()
        avg_logp = max_lp + np.log(np.mean(np.exp(logprobs - max_lp)))

        # A token is critical if its logprob < avg_logp
        critical_mask = logprobs < avg_logp

        # Filter: only keep meaningful tokens
        filtered_critical = []
        filtered_mask = np.zeros(len(tokens), dtype=bool)

        for i, (tok, is_crit) in enumerate(zip(tokens, critical_mask)):
            if is_crit and self._filter_token(tok, lang):
                filtered_critical.append(tok)
                filtered_mask[i] = True

        total_meaningful = sum(1 for t in tokens if self._filter_token(t, lang))
        critical_fraction = len(filtered_critical) / total_meaningful if total_meaningful > 0 else 0.0

        return {
            "critical_tokens": filtered_critical,
            "critical_mask": filtered_mask,
            "avg_logp": float(avg_logp),
            "critical_fraction": float(critical_fraction),
            "n_critical": len(filtered_critical),
            "n_meaningful": total_meaningful,
        }


class PreferenceTokenHighlighter:
    """
    Combines ConfPO critical token detection with TIS-DPO rank-based
    importance weighting for visual token highlighting.

    TIS-DPO weights: w_i = weight_min + (rank_i / (N-1)) * (weight_max - weight_min)
    where rank is ascending by logprob (lowest logprob = highest importance).
    """

    def __init__(
        self,
        weight_min: float = 0.7,
        weight_max: float = 1.3,
        min_token_length: int = 2,
        stopwords: Optional[Dict[str, List[str]]] = None,
    ):
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.critical_detector = CriticalTokenDetector(
            min_token_length=min_token_length,
            stopwords=stopwords,
        )

    def compute_importance_weights(
        self, logprobs: np.ndarray
    ) -> np.ndarray:
        """
        Compute TIS-DPO rank-based importance weights.

        Tokens are ranked by logprob ascending (lowest = rank 0 = most important).
        w_i = weight_min + (rank_i / (N-1)) * (weight_max - weight_min)

        Args:
            logprobs: Per-token log-probabilities

        Returns:
            Array of importance weights (higher = more important/uncertain)
        """
        logprobs = np.asarray(logprobs)
        n = len(logprobs)
        if n <= 1:
            return np.ones(n) * (self.weight_min + self.weight_max) / 2

        # Rank ascending by logprob: lowest logprob gets rank 0 (most important)
        ranks = logprobs.argsort().argsort()  # rank of each element
        # Invert: lowest logprob should get highest weight
        inv_ranks = (n - 1) - ranks

        weights = self.weight_min + (inv_ranks / (n - 1)) * (self.weight_max - self.weight_min)
        return weights

    def highlight(
        self,
        tokens: List[str],
        logprobs: np.ndarray,
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Produce combined critical detection + importance weighting.

        Args:
            tokens: Token strings
            logprobs: Per-token log-probabilities
            lang: Language code

        Returns:
            Dictionary with highlighted tokens, weights, and critical info
        """
        critical_result = self.critical_detector.detect_critical(tokens, logprobs, lang)
        weights = self.compute_importance_weights(logprobs)

        # Build per-token highlight info
        highlights = []
        for i, tok in enumerate(tokens):
            highlights.append({
                "token": tok,
                "logprob": float(logprobs[i]),
                "weight": float(weights[i]),
                "is_critical": bool(critical_result["critical_mask"][i]),
            })

        return {
            "highlights": highlights,
            "weights": weights,
            "critical_tokens": critical_result["critical_tokens"],
            "critical_fraction": critical_result["critical_fraction"],
            "avg_logp": critical_result["avg_logp"],
        }


if __name__ == "__main__":
    print("Token Importance module loaded successfully")
    print(f"Predefined tokens available for: {list(PREDEFINED_SEXIST_TOKENS.keys())}")

    # Test CriticalTokenDetector
    detector = CriticalTokenDetector()
    test_tokens = ["the", "woman", "should", "stay", "in", "kitchen", "and", "cook"]
    test_logprobs = np.array([-0.1, -2.5, -0.5, -1.8, -0.2, -3.0, -0.1, -2.0])
    result = detector.detect_critical(test_tokens, test_logprobs, "en")
    print(f"\nCritical tokens: {result['critical_tokens']}")
    print(f"Critical fraction: {result['critical_fraction']:.2f}")

    # Test PreferenceTokenHighlighter
    highlighter = PreferenceTokenHighlighter()
    highlight_result = highlighter.highlight(test_tokens, test_logprobs, "en")
    print(f"Importance weights: {highlight_result['weights']}")
