"""
Text Preprocessing Utilities

Handles text preprocessing for social media content including
URL removal, mention handling, and normalization.
"""

import re
import unicodedata
from typing import List, Optional
import html


class TextPreprocessor:
    """
    Preprocessor for social media text data.

    Handles common preprocessing steps like URL removal, mention handling,
    and text normalization while preserving important content.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        lowercase: bool = False,
        remove_emojis: bool = False,
        normalize_unicode: bool = True,
        min_length: int = 1
    ):
        """
        Initialize preprocessor with configuration.

        Args:
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions from text
            remove_hashtags: Remove #hashtags (or just the # symbol)
            lowercase: Convert text to lowercase
            remove_emojis: Remove emoji characters
            normalize_unicode: Normalize unicode characters
            min_length: Minimum text length after preprocessing
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.remove_emojis = remove_emojis
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length

        # Compile regex patterns
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|bit\.ly/\S+|t\.co/\S+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        self.whitespace_pattern = re.compile(r'\s+')

    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)

        # Handle mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub(' ', text)

        # Handle hashtags
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub(r'\1', text)  # Keep word, remove #
        else:
            # Just remove the # symbol but keep the word
            text = re.sub(r'#(\w+)', r'\1', text)

        # Remove emojis
        if self.remove_emojis:
            text = self.emoji_pattern.sub(' ', text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()

        # Check minimum length
        if len(text) < self.min_length:
            return ""

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of texts to preprocess

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def preprocess_tweet(
    text: str,
    remove_urls: bool = True,
    lowercase: bool = False,
    normalize: bool = True
) -> str:
    """
    Convenience function for preprocessing a single tweet.

    Args:
        text: Tweet text
        remove_urls: Whether to remove URLs
        lowercase: Whether to lowercase
        normalize: Whether to normalize unicode

    Returns:
        Preprocessed tweet text
    """
    preprocessor = TextPreprocessor(
        remove_urls=remove_urls,
        lowercase=lowercase,
        normalize_unicode=normalize
    )
    return preprocessor.preprocess(text)


def clean_for_tokenization(text: str) -> str:
    """
    Minimal cleaning for tokenization - preserves most content.

    Args:
        text: Input text

    Returns:
        Cleaned text suitable for tokenization
    """
    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions from text."""
    return re.findall(r'@(\w+)', text)


def extract_hashtags(text: str) -> List[str]:
    """Extract #hashtags from text."""
    return re.findall(r'#(\w+)', text)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.findall(text)


if __name__ == "__main__":
    # Test preprocessing
    test_texts = [
        "This is a test tweet with a URL https://example.com and @mention #hashtag",
        "Another tweet with emojis 😀🎉 and special chars: &amp; &lt;",
        "Spanish text: ¡Hola! ¿Cómo estás? #español",
    ]

    preprocessor = TextPreprocessor()

    for text in test_texts:
        processed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
