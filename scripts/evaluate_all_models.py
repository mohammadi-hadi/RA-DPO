#!/usr/bin/env python3
"""
Evaluate ALL fine-tuned models on the same test set with 3-factor R(x).

For each model:
  1. Run inference on 692 test samples (EN+ES, structured prompt)
  2. Extract confidence from logprobs
  3. Look up annotator agreement + sigmoid token scores
  4. Compute R(x) = alpha*conf + beta*agree + gamma*(1-sigmoid_score)
  5. Optimize weights via logistic regression
  6. Sweep threshold for coverage-accuracy trade-off
  7. Save all results (including per-instance data)

Usage:
    python scripts/evaluate_all_models.py
    python scripts/evaluate_all_models.py --models "Smart-30%,Random-50%"
"""

import json, os, sys, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.utils.metrics import compute_metrics
from ra_dpo.pipeline.prompts import PromptBuilder
from ra_dpo.utils.weight_optimizer import WeightOptimizer

if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('OPENAI_API_KEY must be set in the environment')

from openai import OpenAI

OUTPUT_DIR = Path('results/final_reliability_3factor')
TOKEN_SCORES_PATH = Path('results/token_scores/token_scores_cache.json')

# Model registry: display_name -> (model_id, training_pairs)
MODEL_REGISTRY = {
    'gpt-4o (base)': ('gpt-4o', None),
    'gpt-4o-mini (base)': ('gpt-4o-mini', None),
    'gpt-4o-mini (SFT)': ('ft:gpt-4o-mini-2024-07-18:anonymized-org:sexism-sft:DRNcfEOv', None),
    'gpt-4o (Standard DPO)': ('ft:gpt-4o-2024-08-06:anonymized-org::DRjUFQsX', 5536),
    'gpt-4o (RA-DPO)': ('ft:gpt-4o-2024-08-06:anonymized-org::DRiklNmA', 8984),
    'gpt-4o (Smart-30% DPO)': ('ft:gpt-4o-2024-08-06:anonymized-org::DSGdXlvn', 1661),
    'gpt-4o (Random-50% DPO)': ('ft:gpt-4o-2024-08-06:anonymized-org::DSIB40dK', 2768),
}

SMART50_JOB_ID = 'ftjob-32DMFxnVPS2McyMvQEcThFap'


def load_token_scores(test_df, cache_path):
    """Load sigmoid scores for test samples from the cache."""
    with open(cache_path) as f:
        cache = json.load(f)

    scores = {}
    for idx, row in test_df.iterrows():
        tweet_id = str(row.get('id', idx))
        # Test samples are keyed as test_{id}
        key = f'test_{tweet_id}'
        if key in cache:
            scores[tweet_id] = cache[key]['sigmoid_score']
        else:
            # Try raw id
            if tweet_id in cache:
                scores[tweet_id] = cache[tweet_id]['sigmoid_score']
            else:
                scores[tweet_id] = 0.5  # neutral default

    found = sum(1 for v in scores.values() if v != 0.5)
    print(f'  Token scores: {found}/{len(test_df)} found in cache')
    return scores


def evaluate_model(model_id, test_df, client, prompt_builder, token_scores):
    """Evaluate a single model on the test set."""
    from tqdm import tqdm

    system = prompt_builder.get_system_prompt('structured', 'en')
    predictions = []
    confidences = []
    agreements = []
    sigmoid_scores = []
    true_labels = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f'  {model_id[:40]}'):
        tweet_id = str(row.get('id', _))
        text = row['tweet']
        lang = row.get('lang', 'en')

        # Use language-appropriate system prompt
        sys_prompt = prompt_builder.get_system_prompt('structured', lang)
        user_prompt = prompt_builder.format_user_prompt(text, lang, 'structured')

        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                max_tokens=10, temperature=0.0,
                logprobs=True, top_logprobs=3,
            )

            text_out = r.choices[0].message.content or ''
            pred = prompt_builder.parse_prediction(text_out, lang)

            conf = 0.5
            if r.choices[0].logprobs and r.choices[0].logprobs.content:
                conf = min(float(np.exp(r.choices[0].logprobs.content[0].logprob)), 1.0)

        except Exception as e:
            pred = 'NO'
            conf = 0.5
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(3)

        predictions.append(pred)
        confidences.append(conf)
        agreements.append(float(row['agreement_score']))
        sigmoid_scores.append(token_scores.get(tweet_id, 0.5))
        true_labels.append(row['majority_label'])

    return {
        'predictions': predictions,
        'confidences': np.array(confidences),
        'agreements': np.array(agreements),
        'sigmoid_scores': np.array(sigmoid_scores),
        'true_labels': true_labels,
        'correct': np.array([p == t for p, t in zip(predictions, true_labels)]),
    }


def sweep_threshold(r_scores, correct, thresholds=None):
    """Sweep threshold to find coverage-accuracy trade-off."""
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.95, 50)

    results = []
    for tau in thresholds:
        mask = r_scores >= tau
        coverage = mask.mean()
        n_pred = mask.sum()

        if n_pred == 0:
            results.append({'threshold': float(tau), 'coverage': 0, 'accuracy': 0,
                            'f1_combined': 0, 'n_predicted': 0})
            continue

        acc = correct[mask].mean()
        f1 = 2 * acc * coverage / (acc + coverage) if (acc + coverage) > 0 else 0

        results.append({
            'threshold': float(tau),
            'coverage': float(coverage),
            'accuracy': float(acc),
            'f1_combined': float(f1),
            'n_predicted': int(n_pred),
        })
    return results


def accuracy_at_coverage_levels(r_scores, correct, levels=[1.0, 0.9, 0.8, 0.6, 0.5]):
    """Get accuracy at specific coverage levels."""
    result = {}
    sorted_indices = np.argsort(-r_scores)  # highest R(x) first

    for level in levels:
        n = max(1, int(len(r_scores) * level))
        top_idx = sorted_indices[:n]
        acc = correct[top_idx].mean()
        actual_cov = n / len(r_scores)
        result[f'acc@{int(level*100)}%'] = float(acc)
    return result


def process_model(model_key, model_id, training_pairs, test_df, client,
                  prompt_builder, token_scores, output_dir):
    """Full evaluation pipeline for a single model."""
    result_path = output_dir / f'{model_key.replace(" ", "_").replace("(","").replace(")","").replace("%","pct")}.json'

    if result_path.exists():
        print(f'  [CACHED] {model_key}')
        with open(result_path) as f:
            return json.load(f)

    print(f'\n{"="*60}')
    print(f'  Evaluating: {model_key} ({model_id[:50]})')
    print(f'{"="*60}')

    # Run inference
    eval_data = evaluate_model(model_id, test_df, client, prompt_builder, token_scores)

    # Standard metrics
    standard_metrics = compute_metrics(eval_data['true_labels'], eval_data['predictions'])
    standard_metrics['avg_confidence'] = float(eval_data['confidences'].mean())

    # Optimize 3-factor weights
    optimizer = WeightOptimizer(C=1.0)
    opt_weights = optimizer.fit(
        eval_data['confidences'],
        eval_data['agreements'],
        eval_data['sigmoid_scores'],  # passed as critical_fractions
        eval_data['correct'],
    )

    alpha = opt_weights.confidence_weight
    beta = opt_weights.agreement_weight
    gamma = opt_weights.token_weight

    # Compute R(x) with optimized weights
    r_scores = (alpha * eval_data['confidences'] +
                beta * eval_data['agreements'] +
                gamma * (1.0 - eval_data['sigmoid_scores']))

    # Sweep threshold
    sweep_results = sweep_threshold(r_scores, eval_data['correct'])

    # Best threshold
    best = max(sweep_results, key=lambda x: x.get('f1_combined', 0))

    # Accuracy at coverage levels
    acc_at_cov = accuracy_at_coverage_levels(r_scores, eval_data['correct'])

    result = {
        'model': model_key,
        'model_id': model_id,
        'training_pairs': training_pairs,
        'standard_metrics': standard_metrics,
        'optimized_weights': {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
        },
        'reliability': {
            'optimal_threshold': float(best['threshold']),
            'accuracy_at_optimal': float(best.get('accuracy', 0)),
            'coverage_at_optimal': float(best.get('coverage', 0)),
            'f1_combined_at_optimal': float(best.get('f1_combined', 0)),
        },
        'accuracy_at_coverage': acc_at_cov,
        'sweep_results': sweep_results,
        'per_instance': {
            'predictions': eval_data['predictions'],
            'confidences': [float(c) for c in eval_data['confidences']],
            'agreements': [float(a) for a in eval_data['agreements']],
            'sigmoid_scores': [float(s) for s in eval_data['sigmoid_scores']],
            'correct': [bool(c) for c in eval_data['correct']],
        },
        'n_samples': len(test_df),
        'timestamp': datetime.now().isoformat(),
    }

    # Save
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary
    sm = result['standard_metrics']
    rl = result['reliability']
    print(f'  F1-Macro:    {sm["f1_macro"]:.4f}')
    print(f'  Accuracy:    {sm["accuracy"]:.4f}')
    print(f'  Weights:     alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}')
    print(f'  Optimal tau: {rl["optimal_threshold"]:.3f}')
    print(f'  R(x) acc:    {rl["accuracy_at_optimal"]:.3f} (cov={rl["coverage_at_optimal"]:.1%})')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model keys to evaluate (default: all)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Check if smart50 is done
    try:
        job = client.fine_tuning.jobs.retrieve(SMART50_JOB_ID)
        if job.status == 'succeeded' and job.fine_tuned_model:
            print(f'[INFO] Smart-50% DPO completed: {job.fine_tuned_model}')
            MODEL_REGISTRY['gpt-4o (Smart-50% DPO)'] = (job.fine_tuned_model, 2768)
        else:
            print(f'[INFO] Smart-50% DPO status: {job.status} (will use placeholder)')
    except Exception as e:
        print(f'[WARN] Could not check smart50 status: {e}')

    # Filter models if specified
    models = MODEL_REGISTRY
    if args.models:
        keys = [k.strip() for k in args.models.split(',')]
        models = {k: v for k, v in MODEL_REGISTRY.items()
                  if any(filt.lower() in k.lower() for filt in keys)}

    # Load data
    print('\n[DATA] Loading EXIST 2023...')
    loader = EXISTDataLoader('EXIST2023_training.json')
    df = loader.to_dataframe()
    df['majority_label'] = df['labels_task1'].apply(majority_vote)
    df['agreement_score'] = df['labels_task1'].apply(agreement_score)
    _, _, test_df = loader.create_train_val_test_split(df)
    print(f'  Test: {len(test_df)} samples (en={len(test_df[test_df["lang"]=="en"])}, es={len(test_df[test_df["lang"]=="es"])})')

    # Load token scores
    print('\n[TOKEN] Loading sigmoid scores...')
    token_scores = load_token_scores(test_df, TOKEN_SCORES_PATH)

    # Initialize prompt builder
    pb = PromptBuilder()

    # Evaluate each model
    all_results = {}
    for model_key, (model_id, training_pairs) in models.items():
        result = process_model(
            model_key, model_id, training_pairs,
            test_df, client, pb, token_scores, OUTPUT_DIR
        )
        all_results[model_key] = result

    # Print comparison table
    print(f'\n{"="*80}')
    print(f'  COMPARISON TABLE (3-Factor R(x))')
    print(f'{"="*80}')
    print(f'{"Model":30s} {"Pairs":>6s} {"F1":>8s} {"Acc":>8s} {"R(x)Acc":>10s} {"Cov":>8s} {"alpha":>7s} {"beta":>7s} {"gamma":>7s}')
    print('-' * 100)
    for mk, r in all_results.items():
        sm = r['standard_metrics']
        rl = r['reliability']
        w = r['optimized_weights']
        pairs = str(r.get('training_pairs', '-') or '-')
        print(f'{mk:30s} {pairs:>6s} {sm["f1_macro"]:8.4f} {sm["accuracy"]:8.4f} '
              f'{rl["accuracy_at_optimal"]:10.4f} {rl["coverage_at_optimal"]:8.1%} '
              f'{w["alpha"]:7.3f} {w["beta"]:7.3f} {w["gamma"]:7.3f}')

    # Save summary
    summary = {}
    for mk, r in all_results.items():
        summary[mk] = {
            'f1_macro': r['standard_metrics']['f1_macro'],
            'accuracy': r['standard_metrics']['accuracy'],
            'training_pairs': r.get('training_pairs'),
            'optimized_weights': r['optimized_weights'],
            'accuracy_at_coverage': r['accuracy_at_coverage'],
            'reliability': r['reliability'],
        }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll results saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
