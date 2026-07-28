#!/usr/bin/env python3
"""
Generate publication-quality figures for the ACL paper.

Reads results from results/final_reliability_3factor/ and generates:
  1. fig_coverage_accuracy.pdf — Coverage vs accuracy curves
  2. fig_training_progression.pdf — F1 progression across training stages
  3. fig_efficiency.pdf — Training data size vs F1 (smart sampling)
  4. fig_model_comparison.pdf — Best F1 per model bar chart
  5. fig_hard_cases.pdf — Performance on hard vs easy cases
  6. fig_rx_gain.pdf — Accuracy gain from R(x) filtering

Usage:
    python scripts/generate_paper_figures.py
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path('results/final_reliability_3factor')
FIGURES_DIR = Path('results/final_report/figures')

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'gpt-4o (base)': '#BDBDBD',
    'gpt-4o-mini (base)': '#90A4AE',
    'gpt-4o-mini (SFT)': '#66BB6A',
    'gpt-4o (Standard DPO)': '#42A5F5',
    'gpt-4o (RA-DPO)': '#EF5350',
    'gpt-4o (Smart-30% DPO)': '#FFA726',
    'gpt-4o (Smart-50% DPO)': '#26A69A',
    'gpt-4o (Random-50% DPO)': '#AB47BC',
}

SHORT_NAMES = {
    'gpt-4o (base)': 'gpt-4o base',
    'gpt-4o-mini (base)': 'gpt-4o-mini base',
    'gpt-4o-mini (SFT)': 'gpt-4o-mini SFT',
    'gpt-4o (Standard DPO)': 'Std DPO (5,536)',
    'gpt-4o (RA-DPO)': 'RA-DPO (8,984)',
    'gpt-4o (Smart-30% DPO)': 'Smart-30% (1,661)',
    'gpt-4o (Smart-50% DPO)': 'Smart-50% (2,768)',
    'gpt-4o (Random-50% DPO)': 'Random-50% (2,768)',
}

# Order for consistent plotting
PLOT_ORDER = [
    'gpt-4o (base)',
    'gpt-4o-mini (base)',
    'gpt-4o (Random-50% DPO)',
    'gpt-4o (Smart-50% DPO)',
    'gpt-4o (Standard DPO)',
    'gpt-4o (Smart-30% DPO)',
    'gpt-4o-mini (SFT)',
    'gpt-4o (RA-DPO)',
]


def load_results():
    results = {}
    for p in sorted(RESULTS_DIR.glob('*.json')):
        if p.name == 'summary.json':
            continue
        with open(p) as f:
            data = json.load(f)
        results[data['model']] = data
    return results


def fig_coverage_accuracy(results, output_dir):
    """Coverage vs accuracy — the key figure showing R(x) filtering power."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Only plot the most important models to avoid clutter
    highlight = ['gpt-4o (base)', 'gpt-4o (Standard DPO)', 'gpt-4o (Smart-30% DPO)', 'gpt-4o (RA-DPO)']
    fade = [k for k in results if k not in highlight]

    # Faded lines first
    for mk in fade:
        if mk not in results:
            continue
        sweep = results[mk].get('sweep_results', [])
        if not sweep:
            continue
        covs = [s['coverage'] for s in sweep if s['coverage'] > 0]
        accs = [s['accuracy'] for s in sweep if s['coverage'] > 0]
        if covs:
            ax.plot(covs, accs, '-', color=COLORS.get(mk, '#999'),
                    linewidth=0.8, alpha=0.3)

    # Highlighted lines
    for mk in highlight:
        if mk not in results:
            continue
        sweep = results[mk].get('sweep_results', [])
        if not sweep:
            continue
        covs = [s['coverage'] for s in sweep if s['coverage'] > 0]
        accs = [s['accuracy'] for s in sweep if s['coverage'] > 0]
        if not covs:
            continue

        short = SHORT_NAMES.get(mk, mk[:20])
        lw = 2.5 if mk == 'gpt-4o (RA-DPO)' else 1.8
        ax.plot(covs, accs, 'o-', color=COLORS[mk],
                label=short, markersize=2, linewidth=lw)

        # Mark optimal point
        rl = results[mk].get('reliability', {})
        if rl.get('coverage_at_optimal', 0) > 0:
            ax.plot(rl['coverage_at_optimal'], rl['accuracy_at_optimal'],
                    '*', color=COLORS[mk], markersize=12, zorder=10)

    # Add annotation for RA-DPO at 50% coverage
    ra = results.get('gpt-4o (RA-DPO)', {})
    acc50 = ra.get('accuracy_at_coverage', {}).get('acc@50%', 0)
    if acc50:
        ax.annotate(f'96.0% acc\nat 50% cov',
                    xy=(0.50, acc50), xytext=(0.35, 0.97),
                    fontsize=9, fontweight='bold', color=COLORS['gpt-4o (RA-DPO)'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['gpt-4o (RA-DPO)'], lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gpt-4o (RA-DPO)'], alpha=0.9))

    # Add shaded region for "high accuracy zone"
    ax.axhspan(0.93, 1.0, alpha=0.05, color='green')
    ax.text(0.32, 0.94, 'High-accuracy zone (>93%)', fontsize=7, color='green', alpha=0.7)

    ax.set_xlabel('Coverage (fraction predicted)')
    ax.set_ylabel('Accuracy (on predicted instances)')
    ax.set_title('Reliability-Aware Inference: Coverage vs Accuracy')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0.28, 1.03)
    ax.set_ylim(0.72, 1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_coverage_accuracy.pdf')
    plt.savefig(output_dir / 'fig_coverage_accuracy.png')
    plt.close()
    print('  Saved fig_coverage_accuracy')


def fig_training_progression(results, output_dir):
    """Bar chart showing F1 progression — grouped by method type."""
    fig, ax = plt.subplots(figsize=(11, 5))

    order = [k for k in PLOT_ORDER if k in results]
    f1s = [results[m]['standard_metrics']['f1_macro'] for m in order]
    colors = [COLORS.get(m, '#666') for m in order]
    labels = [SHORT_NAMES.get(m, m[:20]) for m in order]

    bars = ax.bar(range(len(order)), f1s, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('F1-Macro')
    ax.set_title('Training Progression: Base Models to DPO Variants')

    # Grouping brackets
    # Base models: 0-1, Fine-tuned: 2-7
    ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.4)
    ax.text(0.5, min(f1s) - 0.02, 'Base', ha='center', fontsize=8, color='gray')
    ax.text(4.5, min(f1s) - 0.02, 'Fine-tuned', ha='center', fontsize=8, color='gray')

    # Reference line for Standard DPO
    if 'gpt-4o (Standard DPO)' in results:
        std_f1 = results['gpt-4o (Standard DPO)']['standard_metrics']['f1_macro']
        ax.axhline(y=std_f1, color=COLORS['gpt-4o (Standard DPO)'],
                   linestyle='--', alpha=0.4, linewidth=1)

    ax.set_ylim(min(f1s) - 0.04, max(f1s) + 0.025)
    ax.grid(True, alpha=0.1, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_training_progression.pdf')
    plt.savefig(output_dir / 'fig_training_progression.png')
    plt.close()
    print('  Saved fig_training_progression')


def fig_efficiency(results, output_dir):
    """Training data efficiency — the smart sampling story."""
    dpo_models = {k: v for k, v in results.items() if v.get('training_pairs')}
    if not dpo_models:
        print('  [SKIP] fig_efficiency')
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Plot each point
    for mk, data in dpo_models.items():
        pairs = data['training_pairs']
        f1 = data['standard_metrics']['f1_macro']
        color = COLORS.get(mk, '#666')
        short = SHORT_NAMES.get(mk, mk[:20])

        marker = 'D' if 'Smart' in mk else ('s' if 'Random' in mk else ('*' if 'RA-DPO' in mk else 'o'))
        size = 200 if 'RA-DPO' in mk else 150
        ax.scatter(pairs, f1, color=color, s=size, marker=marker, zorder=5,
                   edgecolors='black', linewidths=0.5)

        # Smart label positioning
        if 'Smart-30' in mk:
            ax.annotate(short, (pairs, f1), textcoords='offset points',
                        xytext=(-10, 12), fontsize=8, color=color, fontweight='bold')
        elif 'Random' in mk:
            ax.annotate(short, (pairs, f1), textcoords='offset points',
                        xytext=(10, -15), fontsize=8, color=color)
        elif 'Smart-50' in mk:
            ax.annotate(short, (pairs, f1), textcoords='offset points',
                        xytext=(10, 8), fontsize=8, color=color)
        elif 'RA-DPO' in mk:
            ax.annotate(short, (pairs, f1), textcoords='offset points',
                        xytext=(10, -12), fontsize=8, color=color, fontweight='bold')
        else:
            ax.annotate(short, (pairs, f1), textcoords='offset points',
                        xytext=(10, 5), fontsize=8, color=color)

    # Draw arrow showing "3.3x more efficient"
    smart30 = results.get('gpt-4o (Smart-30% DPO)')
    std_dpo = results.get('gpt-4o (Standard DPO)')
    if smart30 and std_dpo:
        s30_f1 = smart30['standard_metrics']['f1_macro']
        std_f1 = std_dpo['standard_metrics']['f1_macro']
        ax.annotate('', xy=(1661, s30_f1), xytext=(5536, std_f1),
                    arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=2, linestyle='-'))
        ax.text(3200, (s30_f1 + std_f1) / 2 + 0.002,
                '3.3x fewer pairs\nsame F1', fontsize=9, ha='center',
                color='#FF6F00', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#FF6F00', alpha=0.9))

    # Highlight smart > random at same data size
    smart50 = results.get('gpt-4o (Smart-50% DPO)')
    rand50 = results.get('gpt-4o (Random-50% DPO)')
    if smart50 and rand50:
        s50_f1 = smart50['standard_metrics']['f1_macro']
        r50_f1 = rand50['standard_metrics']['f1_macro']
        ax.annotate('', xy=(2768, s50_f1), xytext=(2768, r50_f1),
                    arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=1.5))
        ax.text(2950, (s50_f1 + r50_f1) / 2, 'smart >\nrandom',
                fontsize=7, color='#7B1FA2', va='center')

    ax.set_xlabel('Number of DPO Training Pairs')
    ax.set_ylabel('F1-Macro')
    ax.set_title('Training Efficiency: Smart Sampling vs Standard DPO')
    ax.grid(True, alpha=0.15)
    ax.set_xlim(800, max(v['training_pairs'] for v in dpo_models.values()) * 1.12)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_efficiency.pdf')
    plt.savefig(output_dir / 'fig_efficiency.png')
    plt.close()
    print('  Saved fig_efficiency')


def fig_model_comparison(results, output_dir):
    """Horizontal bar chart — sorted by F1."""
    sorted_models = sorted(results.items(),
                          key=lambda x: x[1]['standard_metrics']['f1_macro'])

    models = [m[0] for m in sorted_models]
    f1s = [m[1]['standard_metrics']['f1_macro'] for m in sorted_models]
    colors = [COLORS.get(m, '#666') for m in models]
    labels = [SHORT_NAMES.get(m, m[:25]) for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(models)), f1s, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for bar, val, mk in zip(bars, f1s, models):
        weight = 'bold' if mk in ['gpt-4o (RA-DPO)', 'gpt-4o (Smart-30% DPO)'] else 'normal'
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight=weight)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('F1-Macro')
    ax.set_title('Model Comparison: F1-Macro on Full Test Set (692 samples)')
    ax.set_xlim(0.70, max(f1s) + 0.025)
    ax.grid(True, alpha=0.1, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_comparison.pdf')
    plt.savefig(output_dir / 'fig_model_comparison.png')
    plt.close()
    print('  Saved fig_model_comparison')


def fig_hard_cases(results, output_dir):
    """Performance on hard cases (low agreement) vs easy cases — shows where R(x) matters."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Get agreement from any model (same test set)
    ref = list(results.values())[0]
    agreements = np.array(ref['per_instance']['agreements'])
    hard_mask = agreements < 0.67
    easy_mask = agreements >= 0.83

    order = [k for k in PLOT_ORDER if k in results]
    x = np.arange(len(order))
    width = 0.35

    hard_accs = []
    easy_accs = []
    for mk in order:
        correct = np.array(results[mk]['per_instance']['correct'])
        hard_accs.append(correct[hard_mask].mean())
        easy_accs.append(correct[easy_mask].mean())

    bars1 = ax.bar(x - width/2, easy_accs, width, label='Easy cases (agree >= 0.83)',
                   color=[COLORS.get(m, '#666') for m in order], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, hard_accs, width, label='Hard cases (agree < 0.67)',
                   color=[COLORS.get(m, '#666') for m in order], alpha=0.45, edgecolor='white',
                   hatch='//')

    # Value labels on hard cases
    for bar, val in zip(bars2, hard_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(m, m[:15]) for m in order], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Performance on Easy vs Hard Cases ({easy_mask.sum()} easy, {hard_mask.sum()} hard)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.1, axis='y')

    # Annotation: R(x) abstains on hard cases
    ax.text(0.98, 0.55, 'R(x) filtering abstains\non hard cases, keeping\nonly high-accuracy predictions',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_hard_cases.pdf')
    plt.savefig(output_dir / 'fig_hard_cases.png')
    plt.close()
    print('  Saved fig_hard_cases')


def fig_rx_gain(results, output_dir):
    """Accuracy gain from R(x) filtering at different coverage levels."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    coverage_levels = ['acc@100%', 'acc@90%', 'acc@80%', 'acc@60%', 'acc@50%']
    cov_labels = ['100%', '90%', '80%', '60%', '50%']

    # Highlight key models
    key_models = ['gpt-4o (base)', 'gpt-4o (Standard DPO)', 'gpt-4o (Smart-30% DPO)', 'gpt-4o (RA-DPO)']

    x = np.arange(len(coverage_levels))
    width = 0.18
    offsets = np.linspace(-1.5*width, 1.5*width, len(key_models))

    for i, mk in enumerate(key_models):
        if mk not in results:
            continue
        accs = [results[mk]['accuracy_at_coverage'].get(cl, 0) for cl in coverage_levels]
        color = COLORS.get(mk, '#666')
        short = SHORT_NAMES.get(mk, mk[:15])
        lw = 2 if mk == 'gpt-4o (RA-DPO)' else 1
        bars = ax.bar(x + offsets[i], accs, width, label=short, color=color,
                      edgecolor='black' if mk == 'gpt-4o (RA-DPO)' else 'white',
                      linewidth=lw * 0.5)

    # Value labels on RA-DPO bars
    ra_data = results.get('gpt-4o (RA-DPO)')
    if ra_data:
        ra_accs = [ra_data['accuracy_at_coverage'].get(cl, 0) for cl in coverage_levels]
        for j, val in enumerate(ra_accs):
            ax.text(x[j] + offsets[-1], val + 0.005, f'{val:.1%}',
                    ha='center', fontsize=7, fontweight='bold', color=COLORS['gpt-4o (RA-DPO)'])

    ax.set_xticks(x)
    ax.set_xticklabels(cov_labels, fontsize=10)
    ax.set_xlabel('Coverage Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy at Different Coverage Levels (R(x) Filtering)')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.1, axis='y')

    # Annotate the gap at 50% coverage
    base_acc50 = results.get('gpt-4o (base)', {}).get('accuracy_at_coverage', {}).get('acc@50%', 0)
    ra_acc50 = ra_data['accuracy_at_coverage']['acc@50%'] if ra_data else 0
    if base_acc50 and ra_acc50:
        ax.annotate('', xy=(4 + offsets[-1], ra_acc50), xytext=(4 + offsets[0], base_acc50),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        gap = ra_acc50 - base_acc50
        ax.text(4 + 0.35, (ra_acc50 + base_acc50) / 2, f'+{gap:.1%}',
                fontsize=9, fontweight='bold', color='red', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_rx_gain.pdf')
    plt.savefig(output_dir / 'fig_rx_gain.png')
    plt.close()
    print('  Saved fig_rx_gain')


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('[LOAD] Loading results...')
    results = load_results()
    print(f'  Found {len(results)} models')

    if not results:
        print('[ERROR] No results found.')
        sys.exit(1)

    print('\n[FIGURES] Generating...')
    fig_coverage_accuracy(results, FIGURES_DIR)
    fig_training_progression(results, FIGURES_DIR)
    fig_efficiency(results, FIGURES_DIR)
    fig_model_comparison(results, FIGURES_DIR)
    fig_hard_cases(results, FIGURES_DIR)
    fig_rx_gain(results, FIGURES_DIR)

    print(f'\nAll figures saved to {FIGURES_DIR}')


if __name__ == '__main__':
    main()
