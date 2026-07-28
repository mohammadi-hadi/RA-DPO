"""
EXIST 2023 Dataset Loading Utilities

Handles loading and parsing of the EXIST 2023 sexism detection dataset
with support for multi-annotator labels and various aggregation strategies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class EXISTInstance:
    """Single instance from EXIST 2023 dataset."""
    id: str
    lang: str
    tweet: str
    labels_task1: List[str]
    labels_task2: Optional[List[str]] = None
    labels_task3: Optional[List[str]] = None
    annotator_genders: Optional[List[str]] = None
    annotator_ages: Optional[List[str]] = None
    split: Optional[str] = None


class EXISTDataLoader:
    """
    Loader for EXIST 2023 dataset with multi-annotator label handling.

    Attributes:
        data_path: Path to the JSON data file
        data: Loaded dataset as dictionary
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader.

        Args:
            data_path: Path to EXIST2023_training.json or similar file
        """
        self.data_path = Path(data_path)
        self.data: Dict = {}
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> Dict:
        """Load the dataset from JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert loaded data to pandas DataFrame."""
        if not self.data:
            self.load()

        records = []
        for instance_id, instance_data in self.data.items():
            record = {
                'id': instance_id,
                'lang': instance_data.get('lang', ''),
                'tweet': instance_data.get('tweet', ''),
                'labels_task1': instance_data.get('labels_task1', []),
                'labels_task2': instance_data.get('labels_task2', []),
                'labels_task3': instance_data.get('labels_task3', []),
                'annotator_genders': instance_data.get('gender_annotators', []),
                'annotator_ages': instance_data.get('age_annotators', []),
                'split': instance_data.get('split', ''),
                'num_annotators': instance_data.get('number_annotators', 0),
            }
            records.append(record)

        self._df = pd.DataFrame(records)
        return self._df

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self._df is None:
            self.to_dataframe()

        df = self._df

        # Apply majority vote for label distribution
        df['majority_label'] = df['labels_task1'].apply(majority_vote)

        stats = {
            'total_instances': len(df),
            'languages': df['lang'].value_counts().to_dict(),
            'label_distribution': df['majority_label'].value_counts().to_dict(),
            'avg_annotators': df['num_annotators'].mean(),
            'label_agreement': df['labels_task1'].apply(agreement_score).mean(),
        }

        # Per-language statistics
        for lang in df['lang'].unique():
            lang_df = df[df['lang'] == lang]
            stats[f'{lang}_instances'] = len(lang_df)
            stats[f'{lang}_label_dist'] = lang_df['majority_label'].value_counts().to_dict()

        return stats

    def split_by_language(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset by language (English and Spanish)."""
        if self._df is None:
            self.to_dataframe()

        en_df = self._df[self._df['lang'] == 'en'].copy()
        es_df = self._df[self._df['lang'] == 'es'].copy()

        return en_df, es_df

    def create_train_val_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.

        Args:
            df: DataFrame to split (uses loaded data if None)
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            stratify: Whether to stratify by majority label
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df is None:
            if self._df is None:
                self.to_dataframe()
            df = self._df.copy()

        # Add majority label for stratification
        df['majority_label'] = df['labels_task1'].apply(majority_vote)

        stratify_col = df['majority_label'] if stratify else None

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )

        # Second split: train vs val
        val_proportion = val_size / (train_size + val_size)
        stratify_col = train_val_df['majority_label'] if stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_proportion,
            stratify=stratify_col,
            random_state=random_state
        )

        return train_df, val_df, test_df


def majority_vote(labels: List[str]) -> str:
    """
    Get majority vote from list of annotator labels.

    Args:
        labels: List of labels from multiple annotators

    Returns:
        Most common label
    """
    if not labels:
        return "NO"  # Default
    return Counter(labels).most_common(1)[0][0]


def soft_labels(labels: List[str]) -> Dict[str, float]:
    """
    Convert annotator labels to soft probability distribution.

    Args:
        labels: List of labels from multiple annotators

    Returns:
        Dictionary mapping each label to its proportion
    """
    if not labels:
        return {"YES": 0.0, "NO": 1.0}

    counts = Counter(labels)
    total = sum(counts.values())
    return {label: count / total for label, count in counts.items()}


def agreement_score(labels: List[str]) -> float:
    """
    Calculate annotator agreement score.

    Args:
        labels: List of labels from multiple annotators

    Returns:
        Agreement score between 0 and 1 (1 = perfect agreement)
    """
    if not labels:
        return 0.0

    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def load_exist_dataset(
    data_path: Union[str, Path],
    return_splits: bool = False,
    language: Optional[str] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Convenience function to load EXIST 2023 dataset.

    Args:
        data_path: Path to the JSON data file
        return_splits: If True, return train/val/test splits
        language: Filter by language ('en' or 'es'), None for all

    Returns:
        DataFrame or tuple of DataFrames if return_splits=True
    """
    loader = EXISTDataLoader(data_path)
    df = loader.to_dataframe()

    if language:
        df = df[df['lang'] == language].copy()

    if return_splits:
        return loader.create_train_val_test_split(df)

    return df


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "EXIST2023_training.json"

    loader = EXISTDataLoader(data_path)
    df = loader.to_dataframe()

    print(f"Loaded {len(df)} instances")
    print("\nStatistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
