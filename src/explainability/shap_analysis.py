"""
SHAP Analysis for Sexism Detection

Computes SHAP values to understand token importance
for model predictions.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP analysis will be limited.")


class SHAPAnalyzer:
    """
    SHAP-based explainability analyzer for text classification.

    Computes feature importance using SHAP values for
    understanding model decisions.
    """

    def __init__(
        self,
        model,
        tokenizer,
        background_samples: int = 100,
        max_evals: int = 500
    ):
        """
        Initialize SHAP analyzer.

        Args:
            model: The model to analyze
            tokenizer: Tokenizer for the model
            background_samples: Number of background samples
            max_evals: Maximum evaluations for SHAP
        """
        self.model = model
        self.tokenizer = tokenizer
        self.background_samples = background_samples
        self.max_evals = max_evals
        self.explainer = None

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities for texts.

        Args:
            texts: List of input texts

        Returns:
            Array of prediction probabilities
        """
        import torch

        probas = []
        self.model.eval()

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True
                )

                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                if hasattr(outputs, 'logits'):
                    probs = torch.softmax(outputs.logits, dim=-1)
                else:
                    probs = torch.softmax(outputs, dim=-1)

                probas.append(probs.cpu().numpy()[0])

        return np.array(probas)

    def setup_explainer(self, background_texts: List[str]):
        """
        Setup SHAP explainer with background data.

        Args:
            background_texts: List of texts for background distribution
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap package is required for SHAP analysis")

        # Sample background if needed
        if len(background_texts) > self.background_samples:
            indices = np.random.choice(
                len(background_texts),
                self.background_samples,
                replace=False
            )
            background_texts = [background_texts[i] for i in indices]

        # Create explainer
        self.explainer = shap.Explainer(
            self._predict_proba,
            shap.maskers.Text(self.tokenizer),
            algorithm='partition'
        )

    def compute_shap_values(
        self,
        texts: List[str],
        class_index: int = 1,
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[List[str]]]:
        """
        Compute SHAP values for texts.

        Args:
            texts: List of texts to explain
            class_index: Class index to explain (1 = sexist)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (shap_values_list, tokens_list)
        """
        if self.explainer is None:
            raise ValueError("Explainer not setup. Call setup_explainer first.")

        shap_values_list = []
        tokens_list = []

        iterator = tqdm(texts, desc="Computing SHAP") if show_progress else texts

        for text in iterator:
            try:
                shap_values = self.explainer([text], max_evals=self.max_evals)

                # Get values for specified class
                values = shap_values.values[0, :, class_index]
                tokens = shap_values.data[0]

                shap_values_list.append(values)
                tokens_list.append(tokens)

            except Exception as e:
                print(f"Error computing SHAP for text: {str(e)[:50]}")
                shap_values_list.append(np.array([]))
                tokens_list.append([])

        return shap_values_list, tokens_list

    def aggregate_token_importance(
        self,
        shap_values_list: List[np.ndarray],
        tokens_list: List[List[str]],
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Aggregate SHAP values across all texts to get token importance.

        Args:
            shap_values_list: List of SHAP values arrays
            tokens_list: List of token lists
            normalize: Whether to normalize importance scores

        Returns:
            Dictionary mapping tokens to importance scores
        """
        token_importance = {}
        token_counts = {}

        for values, tokens in zip(shap_values_list, tokens_list):
            if len(values) == 0:
                continue

            for token, value in zip(tokens, values):
                # Clean token
                token = token.strip().lower()
                if not token or len(token) < 2:
                    continue

                if token not in token_importance:
                    token_importance[token] = 0.0
                    token_counts[token] = 0

                token_importance[token] += abs(value)
                token_counts[token] += 1

        # Average by count
        for token in token_importance:
            if token_counts[token] > 0:
                token_importance[token] /= token_counts[token]

        # Normalize if requested
        if normalize and token_importance:
            max_importance = max(token_importance.values())
            if max_importance > 0:
                token_importance = {
                    k: v / max_importance
                    for k, v in token_importance.items()
                }

        # Sort by importance
        token_importance = dict(
            sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return token_importance

    def analyze(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        class_name: str = "sexist",
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform complete SHAP analysis.

        Args:
            texts: List of texts to analyze
            labels: Optional labels for filtering
            class_name: Class name being analyzed
            output_path: Path to save results

        Returns:
            Analysis results dictionary
        """
        # Filter to correctly classified instances if labels provided
        if labels is not None:
            # Get predictions
            probs = self._predict_proba(texts)
            preds = np.argmax(probs, axis=1)

            label_map = {"NO": 0, "YES": 1}
            true_labels = [label_map.get(l, 0) for l in labels]

            # Filter to correct predictions
            correct_indices = [
                i for i, (p, t) in enumerate(zip(preds, true_labels))
                if p == t
            ]

            texts = [texts[i] for i in correct_indices]
            print(f"Analyzing {len(texts)} correctly classified instances")

        # Compute SHAP values
        shap_values, tokens = self.compute_shap_values(texts)

        # Aggregate importance
        token_importance = self.aggregate_token_importance(shap_values, tokens)

        results = {
            'token_importance': token_importance,
            'num_texts_analyzed': len(texts),
            'num_unique_tokens': len(token_importance),
            'class_analyzed': class_name
        }

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}")

        return results


def compute_shap_values(
    model,
    tokenizer,
    texts: List[str],
    labels: Optional[List[str]] = None,
    background_texts: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to compute SHAP values.

    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        texts: Texts to analyze
        labels: Optional labels
        background_texts: Background texts (defaults to subset of texts)
        output_path: Path to save results
        **kwargs: Additional arguments for SHAPAnalyzer

    Returns:
        Analysis results
    """
    analyzer = SHAPAnalyzer(model, tokenizer, **kwargs)

    # Setup with background
    background = background_texts or texts[:100]
    analyzer.setup_explainer(background)

    # Analyze
    results = analyzer.analyze(texts, labels, output_path=output_path)

    return results


if __name__ == "__main__":
    print("SHAP Analysis module loaded successfully")
    print(f"SHAP available: {SHAP_AVAILABLE}")
