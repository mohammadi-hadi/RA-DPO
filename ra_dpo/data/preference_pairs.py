"""
DPO Preference Pair Generation

Creates preference pairs for Direct Preference Optimization training
from multi-annotator labels and model predictions.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from datasets import Dataset


# Prompt templates
PROMPT_TEMPLATE_EN = """Binary Sexism Detection: A two-class (or binary) classification where systems have to decide whether or not a given tweet is sexist.

### Post: {POST}
### Class (YES or NO):"""

PROMPT_TEMPLATE_ES = """Detección Binaria de Sexismo: Una clasificación de dos clases (o binaria) donde los sistemas deben decidir si un tweet dado es sexista o no.

### Post: {POST}
### Clase (SI o NO):"""


@dataclass
class PreferencePair:
    """Single preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    confidence: float = 1.0
    agreement: float = 1.0
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class PreferencePairGenerator:
    """
    Generates preference pairs for DPO training from EXIST 2023 data.

    Supports multiple strategies:
    - Standard: Ground truth vs opposite
    - Confidence-weighted: Based on model prediction confidence
    - XAI-enhanced: With important tokens highlighted
    """

    def __init__(
        self,
        prompt_template_en: str = PROMPT_TEMPLATE_EN,
        prompt_template_es: str = PROMPT_TEMPLATE_ES,
        yes_labels: List[str] = None,
        no_labels: List[str] = None
    ):
        """
        Initialize the preference pair generator.

        Args:
            prompt_template_en: Template for English prompts
            prompt_template_es: Template for Spanish prompts
            yes_labels: Labels considered as "YES" (sexist)
            no_labels: Labels considered as "NO" (not sexist)
        """
        self.prompt_template_en = prompt_template_en
        self.prompt_template_es = prompt_template_es
        self.yes_labels = yes_labels or ["YES", "yes", "sexist", "Sexist", "SI", "si"]
        self.no_labels = no_labels or ["NO", "no", "not sexist", "Not Sexist", "not_sexist"]

    def get_prompt_template(self, lang: str) -> str:
        """Get prompt template for specified language."""
        return self.prompt_template_es if lang == 'es' else self.prompt_template_en

    def majority_vote(self, labels: List[str]) -> str:
        """Get majority vote label."""
        if not labels:
            return "NO"
        return Counter(labels).most_common(1)[0][0]

    def agreement_score(self, labels: List[str]) -> float:
        """Calculate annotator agreement score."""
        if not labels:
            return 0.0
        counts = Counter(labels)
        return max(counts.values()) / len(labels)

    def create_standard_pairs(
        self,
        df: pd.DataFrame,
        text_column: str = 'tweet',
        label_column: str = 'labels_task1',
        lang_column: str = 'lang'
    ) -> List[PreferencePair]:
        """
        Create standard preference pairs (ground truth vs opposite).

        Args:
            df: DataFrame with tweet data
            text_column: Column name for tweet text
            label_column: Column name for annotator labels
            lang_column: Column name for language

        Returns:
            List of PreferencePair objects
        """
        pairs = []

        for _, row in df.iterrows():
            text = row[text_column]
            labels = row[label_column]
            lang = row.get(lang_column, 'en')

            # Get majority vote as ground truth
            ground_truth = self.majority_vote(labels)
            agreement = self.agreement_score(labels)

            # Create prompt
            template = self.get_prompt_template(lang)
            prompt = template.format(POST=text)

            # Determine chosen and rejected
            chosen = ground_truth
            rejected = "NO" if ground_truth == "YES" else "YES"

            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                agreement=agreement,
                weight=agreement,  # Weight by agreement
                metadata={
                    'id': row.get('id', ''),
                    'lang': lang,
                    'original_labels': labels
                }
            )
            pairs.append(pair)

        return pairs

    def create_confidence_weighted_pairs(
        self,
        df: pd.DataFrame,
        predictions: List[str],
        confidences: List[float],
        text_column: str = 'tweet',
        label_column: str = 'labels_task1',
        lang_column: str = 'lang'
    ) -> List[PreferencePair]:
        """
        Create confidence-weighted preference pairs.

        Weights are calculated based on:
        - Model prediction confidence
        - Annotator agreement
        - Whether prediction matches ground truth

        Args:
            df: DataFrame with tweet data
            predictions: Model predictions for each instance
            confidences: Model confidence scores for each prediction
            text_column: Column name for tweet text
            label_column: Column name for annotator labels
            lang_column: Column name for language

        Returns:
            List of PreferencePair objects with confidence weights
        """
        pairs = []

        for idx, row in df.iterrows():
            text = row[text_column]
            labels = row[label_column]
            lang = row.get(lang_column, 'en')

            # Get prediction and confidence
            pred = predictions[idx] if idx < len(predictions) else "NO"
            confidence = confidences[idx] if idx < len(confidences) else 0.5

            # Get ground truth and agreement
            ground_truth = self.majority_vote(labels)
            agreement = self.agreement_score(labels)

            # Create prompt
            template = self.get_prompt_template(lang)
            prompt = template.format(POST=text)

            # Determine chosen/rejected based on model's prediction
            is_correct = (pred == ground_truth)

            if is_correct:
                # Model predicted correctly → reinforce: chosen=prediction, rejected=opposite
                chosen = pred
                rejected = "NO" if pred == "YES" else "YES"
                weight = confidence * agreement
            else:
                # Model predicted WRONG → correct: chosen=ground_truth, rejected=model's wrong answer
                chosen = ground_truth
                rejected = pred  # The model's wrong prediction becomes the rejected example
                weight = confidence * agreement * 1.5  # Stronger correction for errors

            pair = PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                confidence=confidence,
                agreement=agreement,
                weight=weight,
                metadata={
                    'id': row.get('id', ''),
                    'lang': lang,
                    'prediction': pred,
                    'is_correct': is_correct,
                    'original_labels': labels
                }
            )
            pairs.append(pair)

        return pairs

    def create_xai_enhanced_pairs(
        self,
        df: pd.DataFrame,
        predictions: List[str],
        confidences: List[float],
        important_tokens: Dict[str, List[str]],
        text_column: str = 'tweet',
        label_column: str = 'labels_task1',
        lang_column: str = 'lang',
        highlight_marker: str = "**"
    ) -> List[PreferencePair]:
        """
        Create XAI-enhanced preference pairs with important tokens highlighted.

        Args:
            df: DataFrame with tweet data
            predictions: Model predictions
            confidences: Model confidence scores
            important_tokens: Dict mapping language to list of important tokens
            text_column: Column name for tweet text
            label_column: Column name for annotator labels
            lang_column: Column name for language
            highlight_marker: Marker to use for highlighting (e.g., "**" for bold)

        Returns:
            List of PreferencePair objects with XAI-enhanced prompts
        """
        import re

        pairs = []

        for idx, row in df.iterrows():
            text = row[text_column]
            labels = row[label_column]
            lang = row.get(lang_column, 'en')

            # Get language-specific important tokens
            tokens = important_tokens.get(lang, [])

            # Highlight important tokens in text
            highlighted_text = text
            for token in tokens:
                if token.lower() in highlighted_text.lower():
                    # Use regex to preserve original case
                    pattern = re.compile(rf'\b({re.escape(token)})\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(
                        f'{highlight_marker}\\1{highlight_marker}',
                        highlighted_text
                    )

            # Get prediction and confidence
            pred = predictions[idx] if idx < len(predictions) else "NO"
            confidence = confidences[idx] if idx < len(confidences) else 0.5

            # Get ground truth and agreement
            ground_truth = self.majority_vote(labels)
            agreement = self.agreement_score(labels)

            # Create XAI-enhanced prompt
            template = self.get_prompt_template(lang)
            prompt = template.format(POST=highlighted_text)

            # Calculate weight
            is_correct = (pred == ground_truth)
            weight = confidence * agreement

            pair = PreferencePair(
                prompt=prompt,
                chosen=ground_truth,
                rejected="NO" if ground_truth == "YES" else "YES",
                confidence=confidence,
                agreement=agreement,
                weight=weight,
                metadata={
                    'id': row.get('id', ''),
                    'lang': lang,
                    'prediction': pred,
                    'is_correct': is_correct,
                    'original_text': text,
                    'highlighted_text': highlighted_text,
                    'original_labels': labels
                }
            )
            pairs.append(pair)

        return pairs

    def pairs_to_dataset(
        self,
        pairs: List[PreferencePair],
        include_weights: bool = True
    ) -> Dataset:
        """
        Convert preference pairs to HuggingFace Dataset.

        Args:
            pairs: List of PreferencePair objects
            include_weights: Whether to include weight column

        Returns:
            HuggingFace Dataset
        """
        data = {
            'prompt': [p.prompt for p in pairs],
            'chosen': [p.chosen for p in pairs],
            'rejected': [p.rejected for p in pairs],
        }

        if include_weights:
            data['weight'] = [p.weight for p in pairs]
            data['confidence'] = [p.confidence for p in pairs]
            data['agreement'] = [p.agreement for p in pairs]

        return Dataset.from_dict(data)


def create_dpo_pairs(
    df: pd.DataFrame,
    mode: str = 'standard',
    predictions: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    important_tokens: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Dataset:
    """
    Convenience function to create DPO preference pairs.

    Args:
        df: DataFrame with EXIST 2023 data
        mode: 'standard', 'confidence', or 'xai'
        predictions: Model predictions (required for confidence/xai modes)
        confidences: Model confidence scores (required for confidence/xai modes)
        important_tokens: Important tokens for XAI mode
        **kwargs: Additional arguments for PreferencePairGenerator

    Returns:
        HuggingFace Dataset with preference pairs
    """
    generator = PreferencePairGenerator()

    if mode == 'standard':
        pairs = generator.create_standard_pairs(df, **kwargs)
    elif mode == 'confidence':
        if predictions is None or confidences is None:
            raise ValueError("predictions and confidences required for confidence mode")
        pairs = generator.create_confidence_weighted_pairs(
            df, predictions, confidences, **kwargs
        )
    elif mode == 'xai':
        if predictions is None or confidences is None or important_tokens is None:
            raise ValueError(
                "predictions, confidences, and important_tokens required for xai mode"
            )
        pairs = generator.create_xai_enhanced_pairs(
            df, predictions, confidences, important_tokens, **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return generator.pairs_to_dataset(pairs)


if __name__ == "__main__":
    # Test preference pair generation
    test_data = pd.DataFrame({
        'id': ['1', '2', '3'],
        'tweet': [
            "This is a test tweet about women",
            "Another tweet mentioning equality",
            "Un tweet en español sobre mujeres"
        ],
        'labels_task1': [
            ['YES', 'YES', 'YES', 'NO', 'YES', 'YES'],
            ['NO', 'NO', 'NO', 'NO', 'YES', 'NO'],
            ['YES', 'NO', 'YES', 'YES', 'NO', 'YES']
        ],
        'lang': ['en', 'en', 'es']
    })

    generator = PreferencePairGenerator()

    # Test standard pairs
    pairs = generator.create_standard_pairs(test_data)
    print("Standard Pairs:")
    for pair in pairs:
        print(f"  Chosen: {pair.chosen}, Rejected: {pair.rejected}, "
              f"Agreement: {pair.agreement:.2f}")

    # Convert to dataset
    dataset = generator.pairs_to_dataset(pairs)
    print(f"\nDataset: {dataset}")
