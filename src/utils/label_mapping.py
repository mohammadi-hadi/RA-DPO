"""
Label Mapping Utilities

Maps LLM outputs to standard labels using various strategies
including exact matching, SBERT embeddings, and logistic regression.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import numpy as np


class LabelMapper:
    """
    Maps free-form LLM outputs to standard classification labels.

    Supports multiple mapping strategies:
    - Exact matching
    - Keyword matching
    - Embedding-based similarity (SBERT)
    """

    def __init__(
        self,
        positive_labels: List[str] = None,
        negative_labels: List[str] = None,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the label mapper.

        Args:
            positive_labels: Labels indicating sexist content
            negative_labels: Labels indicating non-sexist content
            use_embeddings: Whether to use SBERT embeddings
            embedding_model: SBERT model name for embeddings
        """
        self.positive_labels = positive_labels or [
            'yes', 'sexist', 'sexism', 'si', 'sí',
            'true', 'positive', '1', 'offensive',
            'misogynistic', 'discriminatory'
        ]

        self.negative_labels = negative_labels or [
            'no', 'not sexist', 'not_sexist', 'non-sexist',
            'false', 'negative', '0', 'neutral',
            'not offensive', 'clean'
        ]

        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.label_embeddings = None

    def _load_embedding_model(self):
        """Load SBERT model for embedding-based matching."""
        if self.embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

                # Pre-compute label embeddings
                all_labels = self.positive_labels + self.negative_labels
                self.label_embeddings = self.embedding_model.encode(all_labels)

            except ImportError:
                print("Warning: sentence-transformers not installed. "
                      "Falling back to keyword matching.")
                self.use_embeddings = False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Lowercase
        text = text.lower().strip()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _exact_match(self, text: str) -> Optional[str]:
        """Try exact matching against known labels."""
        normalized = self._normalize_text(text)

        for label in self.positive_labels:
            if normalized == label.lower() or normalized.startswith(label.lower()):
                return "YES"

        for label in self.negative_labels:
            if normalized == label.lower() or normalized.startswith(label.lower()):
                return "NO"

        return None

    def _keyword_match(self, text: str) -> Optional[str]:
        """Try keyword matching for ambiguous outputs."""
        normalized = self._normalize_text(text)

        # Check for positive keywords
        positive_keywords = ['sexist', 'yes', 'si', 'offensive', 'discriminat']
        for keyword in positive_keywords:
            if keyword in normalized:
                return "YES"

        # Check for negative keywords
        negative_keywords = ['not sexist', 'no', 'neutral', 'clean', 'not offensive']
        for keyword in negative_keywords:
            if keyword in normalized:
                return "NO"

        return None

    def _embedding_match(self, text: str) -> str:
        """Use SBERT embeddings for similarity-based matching."""
        self._load_embedding_model()

        if self.embedding_model is None:
            # Fallback if embeddings not available
            return self.map_label(text, use_embeddings=False)

        # Get embedding for text
        text_embedding = self.embedding_model.encode([text])[0]

        # Compute similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            [text_embedding],
            self.label_embeddings
        )[0]

        # Get best match
        best_idx = np.argmax(similarities)
        best_label = (self.positive_labels + self.negative_labels)[best_idx]

        # Map to YES/NO
        if best_label in self.positive_labels:
            return "YES"
        else:
            return "NO"

    def map_label(
        self,
        text: str,
        use_embeddings: Optional[bool] = None,
        default: str = "NO"
    ) -> str:
        """
        Map LLM output to standard label.

        Args:
            text: LLM output text
            use_embeddings: Override embedding usage
            default: Default label if no match found

        Returns:
            Mapped label ("YES" or "NO")
        """
        if not text or not isinstance(text, str):
            return default

        # Try exact matching first
        result = self._exact_match(text)
        if result is not None:
            return result

        # Try keyword matching
        result = self._keyword_match(text)
        if result is not None:
            return result

        # Try embedding matching if enabled
        use_emb = use_embeddings if use_embeddings is not None else self.use_embeddings
        if use_emb:
            return self._embedding_match(text)

        # Return default
        return default

    def map_batch(
        self,
        texts: List[str],
        use_embeddings: Optional[bool] = None
    ) -> List[str]:
        """
        Map a batch of LLM outputs.

        Args:
            texts: List of LLM outputs
            use_embeddings: Override embedding usage

        Returns:
            List of mapped labels
        """
        return [self.map_label(text, use_embeddings) for text in texts]


def map_llm_output(
    output: str,
    default: str = "NO",
    use_embeddings: bool = False
) -> str:
    """
    Convenience function to map a single LLM output.

    Args:
        output: LLM output string
        default: Default label
        use_embeddings: Whether to use embedding matching

    Returns:
        Mapped label
    """
    mapper = LabelMapper(use_embeddings=use_embeddings)
    return mapper.map_label(output, default=default)


def extract_label_from_generation(
    generated_text: str,
    prompt: str
) -> str:
    """
    Extract label from generated text by removing prompt.

    Args:
        generated_text: Full generated text including prompt
        prompt: Original prompt

    Returns:
        Extracted answer portion
    """
    # Remove prompt from beginning
    if generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        # Try to find answer after common markers
        markers = ['Answer:', 'Class:', 'Respuesta:', 'Clase:', '###']
        answer = generated_text

        for marker in markers:
            if marker in generated_text:
                parts = generated_text.split(marker)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    break

    # Clean up
    answer = answer.split('\n')[0].strip()

    return answer


if __name__ == "__main__":
    # Test label mapping
    mapper = LabelMapper()

    test_outputs = [
        "yes",
        "YES",
        "No, this is not sexist",
        "This tweet is sexist because...",
        "Not offensive",
        "SI",
        "no es sexista"
    ]

    print("Label Mapping Tests:")
    for output in test_outputs:
        mapped = mapper.map_label(output)
        print(f"  '{output}' -> {mapped}")
