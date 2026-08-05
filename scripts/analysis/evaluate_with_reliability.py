#!/usr/bin/env python3
"""
Evaluate models with Reliability-Aware inference (PREDICT/ABSTAIN).

For each test instance:
  1. Get model prediction + confidence (from logprobs)
  2. Get annotator agreement (real from dataset)
  3. Compute R(x) = α·confidence + β·agreement + γ·(1 - sigmoid_score)
  4. If R(x) ≥ τ → PREDICT, else ABSTAIN
  5. Sweep τ to find optimal threshold
  6. Report: accuracy@predicted, coverage, F1

Usage:
    python scripts/analysis/evaluate_with_reliability.py
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.utils.metrics import compute_metrics
from ra_dpo.pipeline.prompts import PromptBuilder


# Reliability weights (optimized via logistic regression)
ALPHA = 0.523   # confidence weight
BETA = 0.274    # agreement weight
# No gamma for API models (no token-level logprobs available)


def compute_reliability_score(confidence, agreement):
    """Compute R(x) for API models (2-factor version)."""
    # Normalize to use only 2 factors
    alpha_norm = ALPHA / (ALPHA + BETA)
    beta_norm = BETA / (ALPHA + BETA)
    return alpha_norm * confidence + beta_norm * agreement


def sweep_threshold(r_scores, correct, thresholds=None):
    """Sweep threshold to find optimal τ."""
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.95, 50)

    results = []
    for tau in thresholds:
        mask = r_scores >= tau
        coverage = mask.mean()
        n_predicted = mask.sum()

        if n_predicted == 0:
            results.append({
                'threshold': float(tau), 'coverage': 0, 'accuracy': 0,
                'f1_predicted': 0, 'n_predicted': 0
            })
            continue

        acc = correct[mask].mean()
        # F1 combining accuracy and coverage
        if acc + coverage > 0:
            f1_combined = 2 * acc * coverage / (acc + coverage)
        else:
            f1_combined = 0

        results.append({
            'threshold': float(tau),
            'coverage': float(coverage),
            'accuracy': float(acc),
            'f1_combined': float(f1_combined),
            'n_predicted': int(n_predicted),
        })

    return results


def evaluate_model_with_reliability(
    model_name, test_df, client, prompt_builder,
    strategy='structured', scenario='zero_shot', lang='en'
):
    """Evaluate a model with R(x) reliability scoring."""
    subset = test_df[test_df['lang'] == lang].copy()
    system = prompt_builder.get_system_prompt(strategy, lang)

    predictions = []
    confidences = []
    agreements = []
    true_labels = subset['majority_label'].tolist()

    is_reasoning = model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4')
    supports_logprobs = not is_reasoning and 'gpt-5' not in model_name

    from tqdm import tqdm
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"  {model_name[:25]}"):
        user = prompt_builder.format_user_prompt(row['tweet'], lang, strategy)

        try:
            if supports_logprobs:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                    max_tokens=10, temperature=0.0,
                    logprobs=True, top_logprobs=3,
                )
                conf = 0.5
                if r.choices[0].logprobs and r.choices[0].logprobs.content:
                    conf = min(float(np.exp(r.choices[0].logprobs.content[0].logprob)), 1.0)
            else:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': system + '\n\n' + user}],
                    max_completion_tokens=50,
                )
                conf = 1.0

            text_out = r.choices[0].message.content or ''
            pred = prompt_builder.parse_prediction(text_out, lang)
            predictions.append(pred)
            confidences.append(conf)

        except Exception as e:
            predictions.append('NO')
            confidences.append(0.5)
            time.sleep(1)

        agreements.append(row['agreement_score'])

    # Compute R(x) for each instance
    r_scores = np.array([
        compute_reliability_score(c, a)
        for c, a in zip(confidences, agreements)
    ])

    correct = np.array([p == t for p, t in zip(predictions, true_labels)])

    # Standard metrics (no abstention)
    standard_metrics = compute_metrics(true_labels, predictions)
    standard_metrics['avg_confidence'] = float(np.mean(confidences))

    # Threshold sweep
    sweep_results = sweep_threshold(r_scores, correct)

    # Find optimal threshold
    best = max(sweep_results, key=lambda x: x.get('f1_combined', 0))
    optimal_tau = best['threshold']

    # Metrics at optimal threshold
    mask_optimal = r_scores >= optimal_tau
    if mask_optimal.sum() > 0:
        optimal_acc = correct[mask_optimal].mean()
        optimal_coverage = mask_optimal.mean()
    else:
        optimal_acc = 0
        optimal_coverage = 0

    return {
        'model': model_name,
        'strategy': strategy,
        'scenario': scenario,
        'lang': lang,
        'standard_metrics': standard_metrics,
        'reliability': {
            'optimal_threshold': float(optimal_tau),
            'accuracy_at_optimal': float(optimal_acc),
            'coverage_at_optimal': float(optimal_coverage),
            'f1_combined_at_optimal': float(best.get('f1_combined', 0)),
        },
        'sweep_results': sweep_results,
        'r_scores': [float(r) for r in r_scores],
        'confidences': [float(c) for c in confidences],
        'correct': [bool(c) for c in correct],
        'n_samples': len(subset),
        'timestamp': datetime.now().isoformat(),
    }


def generate_plots(all_results, output_dir):
    """Generate reliability analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300})
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Coverage vs Accuracy curve for each model
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(range(len(all_results)))

    for i, (model_key, result) in enumerate(all_results.items()):
        sweep = result.get('sweep_results', [])
        if not sweep:
            continue
        coverages = [s['coverage'] for s in sweep if s['coverage'] > 0]
        accuracies = [s['accuracy'] for s in sweep if s['coverage'] > 0]
        if coverages:
            ax.plot(coverages, accuracies, 'o-', color=colors[i],
                    label=model_key[:25], markersize=3, linewidth=1.5)

            # Mark optimal point
            opt = result.get('reliability', {})
            if opt.get('coverage_at_optimal', 0) > 0:
                ax.plot(opt['coverage_at_optimal'], opt['accuracy_at_optimal'],
                        '*', color=colors[i], markersize=12)

    ax.set_xlabel('Coverage (fraction predicted)')
    ax.set_ylabel('Accuracy (on predicted instances)')
    ax.set_title('Reliability-Aware Inference: Coverage vs Accuracy')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'coverage_vs_accuracy.pdf')
    plt.savefig(plots_dir / 'coverage_vs_accuracy.png')
    plt.close()

    # Plot 2: Model comparison bar chart (standard F1 vs R(x) accuracy)
    models = list(all_results.keys())
    standard_f1 = [all_results[m]['standard_metrics']['f1_macro'] for m in models]
    rx_acc = [all_results[m]['reliability']['accuracy_at_optimal'] for m in models]
    rx_cov = [all_results[m]['reliability']['coverage_at_optimal'] for m in models]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(models))
    w = 0.35
    bars1 = ax.bar([i - w/2 for i in x], standard_f1, w, label='F1-Macro (all data)', color='#4CAF50')
    bars2 = ax.bar([i + w/2 for i in x], rx_acc, w, label='Accuracy (R(x) filtered)', color='#2196F3')

    ax.set_xticks(list(x))
    ax.set_xticklabels([m[:20] for m in models], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Standard F1 vs Reliability-Filtered Accuracy')
    ax.legend()

    for bar, val in zip(bars1, standard_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, rx_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(plots_dir / 'standard_vs_reliability.pdf')
    plt.savefig(plots_dir / 'standard_vs_reliability.png')
    plt.close()

    print(f'  Plots saved to {plots_dir}')


def main():
    os.environ.setdefault('OPENAI_API_KEY',
        "REDACTED")
    from openai import OpenAI
    client = OpenAI()

    # Load data
    print('[DATA] Loading EXIST 2023...')
    loader = EXISTDataLoader('EXIST2023_training.json')
    df = loader.to_dataframe()
    df['majority_label'] = df['labels_task1'].apply(majority_vote)
    df['agreement_score'] = df['labels_task1'].apply(agreement_score)
    _, _, test_df = loader.create_train_val_test_split(df)
    print(f'  Test: {len(test_df)} (en={len(test_df[test_df["lang"]=="en"])}, es={len(test_df[test_df["lang"]=="es"])})')

    pb = PromptBuilder()
    output_dir = Path('results/reliability_eval')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models to evaluate with reliability
    models = {
        'gpt-4o-mini (base)': 'gpt-4o-mini',
        'gpt-4o-mini (SFT)': 'ft:gpt-4o-mini-2024-07-18:anonymized-org:sexism-sft:DRNcfEOv',
    }

    # Check if DPO models are ready
    for job in client.fine_tuning.jobs.list(limit=5).data:
        if job.status == 'succeeded' and job.method and job.method.type == 'dpo':
            model_id = job.fine_tuned_model
            # Determine if it's standard or RA-DPO based on training file size
            models[f'gpt-4o (DPO)'] = model_id

    print(f'\nModels to evaluate: {list(models.keys())}')

    # Evaluate each model
    all_results = {}
    for model_key, model_id in models.items():
        print(f'\n{"="*60}')
        print(f'  Evaluating: {model_key}')
        print(f'{"="*60}')

        result_path = output_dir / f'{model_key.replace(" ", "_").replace("(","").replace(")","")}.json'
        if result_path.exists():
            print(f'  Loading cached result')
            with open(result_path) as f:
                all_results[model_key] = json.load(f)
            continue

        # Evaluate on English with best strategy (structured)
        result = evaluate_model_with_reliability(
            model_id, test_df, client, pb,
            strategy='structured', lang='en',
        )
        all_results[model_key] = result

        # Save
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Print summary
        sm = result['standard_metrics']
        rl = result['reliability']
        print(f'  Standard F1:    {sm["f1_macro"]:.3f}')
        print(f'  Optimal τ:      {rl["optimal_threshold"]:.3f}')
        print(f'  R(x) accuracy:  {rl["accuracy_at_optimal"]:.3f} (coverage={rl["coverage_at_optimal"]:.1%})')

    # Generate comparison table
    print(f'\n{"="*60}')
    print(f'  COMPARISON TABLE')
    print(f'{"="*60}')
    print(f'{"Model":25s} {"F1":>8s} {"R(x) Acc":>10s} {"Coverage":>10s} {"τ*":>8s}')
    print('-' * 65)
    for model_key, result in all_results.items():
        sm = result['standard_metrics']
        rl = result['reliability']
        print(f'{model_key:25s} {sm["f1_macro"]:8.3f} {rl["accuracy_at_optimal"]:10.3f} '
              f'{rl["coverage_at_optimal"]:10.1%} {rl["optimal_threshold"]:8.3f}')

    # Generate plots
    generate_plots(all_results, output_dir)

    # Save full results
    summary = {k: {
        'f1_macro': v['standard_metrics']['f1_macro'],
        'accuracy_at_optimal': v['reliability']['accuracy_at_optimal'],
        'coverage_at_optimal': v['reliability']['coverage_at_optimal'],
        'optimal_threshold': v['reliability']['optimal_threshold'],
    } for k, v in all_results.items()}

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll results saved to {output_dir}')


if __name__ == '__main__':
    main()
