"""
Evaluation Metrics

Computes standard classification metrics for sexism detection evaluation.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def compute_metrics(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    labels: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names for report
        average: Averaging method for multi-class

    Returns:
        Dictionary of metric scores
    """
    # Convert string labels to numeric if needed
    label_map = {"NO": 0, "YES": 1, "no": 0, "yes": 1}

    if isinstance(y_true[0], str):
        y_true = [label_map.get(y, 0) for y in y_true]
    if isinstance(y_pred[0], str):
        y_pred = [label_map.get(y, 0) for y in y_pred]

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }

    # Per-class metrics
    for cls in [0, 1]:
        cls_name = ['NO', 'YES'][cls]
        metrics[f'f1_{cls_name}'] = f1_score(
            y_true, y_pred, average=None
        )[cls] if cls < len(set(y_true)) else 0.0
        metrics[f'precision_{cls_name}'] = precision_score(
            y_true, y_pred, average=None
        )[cls] if cls < len(set(y_true)) else 0.0
        metrics[f'recall_{cls_name}'] = recall_score(
            y_true, y_pred, average=None
        )[cls] if cls < len(set(y_true)) else 0.0

    return metrics


def evaluate_predictions(
    y_true: List[Union[str, int]],
    y_pred: List[Union[str, int]],
    confidences: Optional[List[float]] = None,
    return_report: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Optional prediction confidences
        return_report: Whether to include classification report

    Returns:
        Evaluation results dictionary
    """
    metrics = compute_metrics(y_true, y_pred)

    # Add confidence statistics if provided
    if confidences is not None:
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['std_confidence'] = np.std(confidences)
        metrics['min_confidence'] = np.min(confidences)
        metrics['max_confidence'] = np.max(confidences)

        # Confidence by correctness
        correct_mask = [t == p for t, p in zip(y_true, y_pred)]
        correct_conf = [c for c, m in zip(confidences, correct_mask) if m]
        wrong_conf = [c for c, m in zip(confidences, correct_mask) if not m]

        if correct_conf:
            metrics['avg_confidence_correct'] = np.mean(correct_conf)
        if wrong_conf:
            metrics['avg_confidence_wrong'] = np.mean(wrong_conf)

    # Add classification report
    if return_report:
        label_map = {"NO": 0, "YES": 1, "no": 0, "yes": 1}
        y_true_num = [label_map.get(y, y) for y in y_true]
        y_pred_num = [label_map.get(y, y) for y in y_pred]

        metrics['classification_report'] = classification_report(
            y_true_num, y_pred_num,
            target_names=['NO', 'YES'],
            output_dict=True
        )

        metrics['confusion_matrix'] = confusion_matrix(
            y_true_num, y_pred_num
        ).tolist()

    return metrics


def generate_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics_to_show: List[str] = None
) -> pd.DataFrame:
    """
    Generate a comparison table of results across approaches.

    Args:
        results: Nested dict of {approach: {lang: {metric: value}}}
        metrics_to_show: List of metrics to include

    Returns:
        DataFrame with comparison table
    """
    metrics_to_show = metrics_to_show or ['f1_macro', 'accuracy']

    rows = []
    for approach, lang_results in results.items():
        row = {'Approach': approach}

        for lang in ['en', 'es']:
            lang_data = lang_results.get(lang, {})
            for metric in metrics_to_show:
                col_name = f'{lang.upper()} ({metric})'
                row[col_name] = lang_data.get(metric, '-')

        rows.append(row)

    df = pd.DataFrame(rows)

    # Format numeric columns
    for col in df.columns:
        if col != 'Approach' and df[col].dtype != object:
            df[col] = df[col].apply(
                lambda x: f'{x:.4f}' if isinstance(x, float) else x
            )

    return df


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    # Group metrics
    main_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    confidence_metrics = ['avg_confidence', 'std_confidence']
    class_metrics = ['f1_YES', 'f1_NO', 'precision_YES', 'recall_YES']

    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")

    print("-" * 50)

    for metric in confidence_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")

    print("-" * 50)

    for metric in class_metrics:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]:.4f}")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test metrics
    y_true = ["YES", "NO", "YES", "YES", "NO", "NO"]
    y_pred = ["YES", "NO", "NO", "YES", "NO", "YES"]
    confidences = [0.9, 0.8, 0.6, 0.95, 0.7, 0.55]

    results = evaluate_predictions(y_true, y_pred, confidences)
    print_metrics(results, "Test Evaluation")
