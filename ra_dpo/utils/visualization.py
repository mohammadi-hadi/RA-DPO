"""
Visualization Utilities

Creates plots for results analysis, confusion matrices,
and token importance visualization.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_results(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot comparison of results across approaches.

    Args:
        results: Dict of {approach: {metric: value}}
        metrics: Metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    metrics = metrics or ['f1_macro', 'accuracy']

    # Prepare data
    approaches = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(approaches))

    for idx, metric in enumerate(metrics):
        values = [results[app].get(metric, 0) for app in approaches]

        bars = axes[idx].bar(approaches, values, color=colors)
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xticklabels(approaches, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_language_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = 'f1_macro',
    title: str = "Performance by Language",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot comparison across languages.

    Args:
        results: Dict of {approach: {lang: {metric: value}}}
        metric: Metric to compare
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    approaches = list(results.keys())
    languages = ['en', 'es']

    x = np.arange(len(approaches))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    for i, lang in enumerate(languages):
        values = [
            results[app].get(lang, {}).get(metric, 0)
            for app in approaches
        ]
        offset = width * i - width / 2
        bars = ax.bar(x + offset, values, width, label=lang.upper())

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        normalize: Whether to normalize values
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    labels = labels or ['NO', 'YES']
    cm = np.array(confusion_matrix)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_token_importance(
    token_importance: Dict[str, float],
    top_k: int = 20,
    title: str = "Token Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot token importance scores.

    Args:
        token_importance: Dict mapping tokens to importance scores
        top_k: Number of top tokens to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    # Sort and get top-k
    sorted_tokens = sorted(
        token_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    tokens = [t[0] for t in sorted_tokens]
    importance = [t[1] for t in sorted_tokens]

    fig, ax = plt.subplots(figsize=figsize)

    # Color based on positive/negative importance
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in importance]

    bars = ax.barh(range(len(tokens)), importance, color=colors)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confidence_distribution(
    confidences: List[float],
    labels: Optional[List[str]] = None,
    predictions: Optional[List[str]] = None,
    title: str = "Confidence Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot distribution of prediction confidences.

    Args:
        confidences: List of confidence scores
        labels: True labels for coloring
        predictions: Predicted labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Overall distribution
    axes[0].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Overall Confidence Distribution')
    axes[0].axvline(
        x=np.mean(confidences),
        color='red',
        linestyle='--',
        label=f'Mean: {np.mean(confidences):.3f}'
    )
    axes[0].legend()

    # By correctness if labels provided
    if labels and predictions:
        correct = [c for c, l, p in zip(confidences, labels, predictions) if l == p]
        wrong = [c for c, l, p in zip(confidences, labels, predictions) if l != p]

        axes[1].hist(correct, bins=15, alpha=0.7, label='Correct', color='green')
        axes[1].hist(wrong, bins=15, alpha=0.7, label='Wrong', color='red')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Confidence by Correctness')
        axes[1].legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_results_summary_figure(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None
):
    """
    Create comprehensive results summary figure.

    Args:
        all_results: Complete results dictionary
        save_path: Path to save figure
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    fig = plt.figure(figsize=(16, 10))

    # Main comparison plot
    ax1 = fig.add_subplot(2, 2, 1)
    approaches = list(all_results.keys())

    en_f1 = [all_results[a].get('en', {}).get('f1_macro', 0) for a in approaches]
    es_f1 = [all_results[a].get('es', {}).get('f1_macro', 0) for a in approaches]

    x = np.arange(len(approaches))
    width = 0.35

    ax1.bar(x - width/2, en_f1, width, label='English', color='steelblue')
    ax1.bar(x + width/2, es_f1, width, label='Spanish', color='darkorange')
    ax1.set_ylabel('F1-Macro')
    ax1.set_title('Performance Comparison by Language')
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Average performance
    ax2 = fig.add_subplot(2, 2, 2)
    avg_f1 = [(en + es) / 2 for en, es in zip(en_f1, es_f1)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(approaches)))

    bars = ax2.barh(approaches, avg_f1, color=colors)
    ax2.set_xlabel('Average F1-Macro')
    ax2.set_title('Overall Performance Ranking')
    ax2.set_xlim(0, 1)

    for bar, val in zip(bars, avg_f1):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center')

    # Improvement over baseline
    ax3 = fig.add_subplot(2, 2, 3)
    baseline_avg = avg_f1[0] if avg_f1 else 0
    improvements = [(f - baseline_avg) * 100 for f in avg_f1]

    colors = ['green' if i > 0 else 'red' for i in improvements]
    ax3.bar(approaches, improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Improvement over Baseline (%)')
    ax3.set_title('Relative Performance Improvement')
    ax3.set_xticklabels(approaches, rotation=45, ha='right')

    # Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    table_data = [
        ['Approach', 'EN F1', 'ES F1', 'Avg F1']
    ]
    for i, approach in enumerate(approaches):
        table_data.append([
            approach,
            f'{en_f1[i]:.4f}',
            f'{es_f1[i]:.4f}',
            f'{avg_f1[i]:.4f}'
        ])

    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    plt.suptitle('Sexism Detection Results Summary', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =========================================================================
# Reliability Inference Visualization Functions
# =========================================================================


def plot_reliability_distribution(
    reliability_scores: np.ndarray,
    correct: np.ndarray,
    threshold: float = 0.6,
    title: str = "Reliability Score Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot histogram of reliability scores colored by correct/incorrect.

    Args:
        reliability_scores: Array of R(x) scores
        correct: Boolean array of prediction correctness
        threshold: Decision threshold
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    correct = np.asarray(correct, dtype=bool)
    reliability_scores = np.asarray(reliability_scores)

    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(0, 1, 30)
    ax.hist(
        reliability_scores[correct], bins=bins, alpha=0.7,
        label="Correct", color="#2ecc71", edgecolor="black", linewidth=0.5,
    )
    ax.hist(
        reliability_scores[~correct], bins=bins, alpha=0.7,
        label="Incorrect", color="#e74c3c", edgecolor="black", linewidth=0.5,
    )

    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Reliability Score R(x)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_abstention_curve(
    sweep_results: List[Dict[str, float]],
    title: str = "Abstention Curve: Coverage vs Quality",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot coverage vs accuracy/F1 at varying thresholds.

    Args:
        sweep_results: List of dicts with threshold, coverage, accuracy, f1
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    thresholds = [r["threshold"] for r in sweep_results]
    coverages = [r["coverage"] for r in sweep_results]
    accuracies = [r["accuracy"] for r in sweep_results]
    f1s = [r["f1"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=figsize)

    color_acc = "#2c3e50"
    color_cov = "#3498db"
    color_f1 = "#e67e22"

    ax1.plot(thresholds, accuracies, "o-", color=color_acc, label="Accuracy", linewidth=2)
    ax1.plot(thresholds, f1s, "s-", color=color_f1, label="F1 (acc*cov harmonic)", linewidth=2)
    ax1.set_xlabel("Reliability Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverages, "^--", color=color_cov, label="Coverage", linewidth=2)
    ax2.set_ylabel("Coverage", color=color_cov)
    ax2.set_ylim(0, 1.05)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_diagram(
    cal_before: Dict[str, Any],
    cal_after: Dict[str, Any],
    title: str = "Calibration Diagram",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Plot reliability diagram before and after temperature scaling.

    Args:
        cal_before: Calibration metrics before scaling
        cal_after: Calibration metrics after scaling
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, cal_data, subtitle in [
        (axes[0], cal_before, f"Before (ECE={cal_before['ece']:.4f})"),
        (axes[1], cal_after, f"After (ECE={cal_after['ece']:.4f})"),
    ]:
        centers = cal_data["bin_centers"]
        accs = cal_data["bin_accuracies"]
        counts = cal_data["bin_counts"]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

        # Bar chart of accuracy per bin
        width = 1.0 / len(centers) * 0.8
        ax.bar(centers, accs, width=width, alpha=0.6, color="#3498db",
               edgecolor="black", linewidth=0.5, label="Observed")

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_critical_token_frequency(
    token_counts: Dict[str, int],
    top_k: int = 30,
    title: str = "Most Frequent Critical Tokens",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot bar chart of top-k most frequently critical tokens.

    Args:
        token_counts: Dict mapping token -> frequency as critical
        top_k: Number of tokens to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    tokens = [t[0] for t in sorted_tokens]
    counts = [t[1] for t in sorted_tokens]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(tokens)))

    ax.barh(range(len(tokens)), counts, color=colors)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency as Critical Token")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_token_highlighting_example(
    tokens: List[str],
    weights: List[float],
    is_critical: List[bool],
    title: str = "Token Importance Heatmap",
    figsize: Tuple[int, int] = (14, 3),
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap for a single example showing token importance.

    Args:
        tokens: Token strings
        weights: Importance weights per token
        is_critical: Whether each token is critical
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    # Limit tokens for readability
    max_tokens = 50
    tokens = tokens[:max_tokens]
    weights = weights[:max_tokens]
    is_critical = is_critical[:max_tokens]

    fig, ax = plt.subplots(figsize=figsize)

    weights_arr = np.array(weights).reshape(1, -1)
    im = ax.imshow(weights_arr, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.5)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks([])

    # Mark critical tokens with a border
    for i, crit in enumerate(is_critical):
        if crit:
            ax.add_patch(plt.Rectangle(
                (i - 0.5, -0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=2
            ))

    plt.colorbar(im, ax=ax, label="Importance Weight", shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_components(
    calibrated_confs: np.ndarray,
    agreement_scores: np.ndarray,
    critical_fractions: np.ndarray,
    correct: np.ndarray,
    title: str = "Reliability Score Components",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
):
    """
    Plot scatter matrix of reliability score components.

    Args:
        calibrated_confs: Calibrated confidence values
        agreement_scores: Agreement values
        critical_fractions: Critical token fractions
        correct: Boolean correctness array
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    correct = np.asarray(correct, dtype=bool)
    colors = np.where(correct, "#2ecc71", "#e74c3c")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    pairs = [
        (calibrated_confs, agreement_scores, "Calibrated Conf.", "Agreement"),
        (calibrated_confs, 1 - critical_fractions, "Calibrated Conf.", "1 - Crit. Fraction"),
        (agreement_scores, 1 - critical_fractions, "Agreement", "1 - Crit. Fraction"),
    ]

    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(x, y, c=colors, alpha=0.5, s=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=8, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=8, label='Incorrect'),
    ]
    axes[2].legend(handles=legend_elements, loc="lower left")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_reliability_summary_figure(
    reliability_scores: np.ndarray,
    correct: np.ndarray,
    sweep_results: List[Dict[str, float]],
    cal_before: Dict[str, Any],
    cal_after: Dict[str, Any],
    calibrated_confs: np.ndarray,
    agreement_scores: np.ndarray,
    critical_fractions: np.ndarray,
    threshold: float = 0.6,
    title: str = "Reliability Analysis Summary",
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
):
    """
    Create comprehensive 2x3 publication-ready summary figure.

    Args:
        reliability_scores: R(x) scores
        correct: Boolean correctness array
        sweep_results: Threshold sweep results
        cal_before: Calibration metrics before
        cal_after: Calibration metrics after
        calibrated_confs: Calibrated confidences
        agreement_scores: Agreement scores
        critical_fractions: Critical fractions
        threshold: Decision threshold
        title: Plot title
        figsize: Figure size
        save_path: Path to save
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return

    correct = np.asarray(correct, dtype=bool)
    reliability_scores = np.asarray(reliability_scores)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # (0,0) Reliability distribution
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 25)
    ax.hist(reliability_scores[correct], bins=bins, alpha=0.7,
            label="Correct", color="#2ecc71", edgecolor="black", linewidth=0.3)
    ax.hist(reliability_scores[~correct], bins=bins, alpha=0.7,
            label="Incorrect", color="#e74c3c", edgecolor="black", linewidth=0.3)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("R(x)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Reliability Distribution")
    ax.legend(fontsize=8)

    # (0,1) Abstention curve
    ax = axes[0, 1]
    thresholds = [r["threshold"] for r in sweep_results]
    coverages = [r["coverage"] for r in sweep_results]
    accuracies = [r["accuracy"] for r in sweep_results]
    ax.plot(thresholds, accuracies, "o-", color="#2c3e50", label="Accuracy", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(thresholds, coverages, "^--", color="#3498db", label="Coverage", linewidth=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Coverage")
    ax.set_title("(b) Abstention Curve")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    # (0,2) Calibration before
    ax = axes[0, 2]
    centers = cal_before["bin_centers"]
    accs = cal_before["bin_accuracies"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    width = 1.0 / len(centers) * 0.8
    ax.bar(centers, accs, width=width, alpha=0.6, color="#3498db",
           edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"(c) Calibration Before (ECE={cal_before['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (1,0) Calibration after
    ax = axes[1, 0]
    centers = cal_after["bin_centers"]
    accs = cal_after["bin_accuracies"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.bar(centers, accs, width=width, alpha=0.6, color="#2ecc71",
           edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"(d) Calibration After (ECE={cal_after['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (1,1) Component scatter: confidence vs agreement
    ax = axes[1, 1]
    colors_arr = np.where(correct, "#2ecc71", "#e74c3c")
    ax.scatter(calibrated_confs, agreement_scores, c=colors_arr, alpha=0.4, s=8)
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("Agreement Score")
    ax.set_title("(e) Confidence vs Agreement")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # (1,2) Component scatter: confidence vs 1-critical_fraction
    ax = axes[1, 2]
    ax.scatter(calibrated_confs, 1 - critical_fractions, c=colors_arr, alpha=0.4, s=8)
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("1 - Critical Fraction")
    ax.set_title("(f) Confidence vs Token Stability")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Visualization module loaded. Plotting available: {PLOTTING_AVAILABLE}")
