#!/usr/bin/env python3
"""
Compute per-token importance scores for all training data using gpt-4o-mini.
Implements the perplexity-based approach that approximates the sigmoid formula:
    sigmoid_score = mean(sigmoid(k × (T - p_i)))

For each tweet:
1. Ask gpt-4o-mini to rate each word's sexism relevance (0-1)
2. Capture logprobs of the model's score assignments
3. Compute critical_fraction and sigmoid_score
4. Save all raw logprobs for reproducibility

Saves results incrementally to resume on disconnect.
"""

import json, os, sys, time, numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from tqdm import tqdm

if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('OPENAI_API_KEY must be set in the environment')

from openai import OpenAI

MODEL = 'gpt-4o-mini'
OUTPUT_DIR = Path('results/token_scores')
CACHE_FILE = OUTPUT_DIR / 'token_scores_cache.json'
STEEPNESS = 10  # k parameter


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def get_token_score(client, text, model=MODEL):
    """Get token importance scores and logprobs for a tweet."""
    prompt = (
        'For each word in the text below, output the word followed by its '
        'sexism relevance score (0.0=irrelevant, 1.0=highly relevant for '
        'sexism detection). Output as a list, one word per line in format: '
        'word|score\n\nText: ' + text
    )

    r = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=300, temperature=0.0,
        logprobs=True, top_logprobs=3,
    )

    content = r.choices[0].message.content or ''
    tokens_data = r.choices[0].logprobs.content if r.choices[0].logprobs else []

    # Parse word|score pairs
    word_scores = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                word = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    word_scores[word] = score
                except ValueError:
                    pass

    # Extract logprobs for score digits
    score_logprobs = []
    for t in tokens_data:
        tok = t.token.strip()
        # Check if this is a numeric token (part of a score)
        if tok and all(c in '0123456789.' for c in tok):
            score_logprobs.append(t.logprob)

    # Compute sigmoid score from logprobs (exact formula from report)
    if score_logprobs:
        lps = np.array(score_logprobs)
        probs = np.exp(lps)
        T = np.mean(probs)
        sw = sigmoid(STEEPNESS * (T - probs))
        sigmoid_score = float(np.mean(sw))
    else:
        sigmoid_score = 0.5

    # Compute critical fraction from word scores
    if word_scores:
        vals = list(word_scores.values())
        critical_fraction = sum(1 for v in vals if v > 0.5) / len(vals)
    else:
        critical_fraction = 0.5

    return {
        'word_scores': word_scores,
        'critical_fraction': critical_fraction,
        'sigmoid_score': sigmoid_score,
        'score_logprobs': [float(lp) for lp in score_logprobs],
        'n_words': len(word_scores),
        'raw_response': content,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f'Loaded {len(cache)} cached scores')

    # Load data
    loader = EXISTDataLoader('EXIST2023_training.json')
    df = loader.to_dataframe()
    df['majority_label'] = df['labels_task1'].apply(majority_vote)
    df['agreement_score'] = df['labels_task1'].apply(agreement_score)
    train_df, val_df, test_df = loader.create_train_val_test_split(df)

    client = OpenAI()

    # Process training data
    print(f'\nProcessing {len(train_df)} training samples...')
    save_counter = 0

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Token scoring'):
        tweet_id = str(row.get('id', idx))

        if tweet_id in cache:
            continue

        try:
            result = get_token_score(client, row['tweet'])
            result['tweet_id'] = tweet_id
            result['agreement'] = float(row['agreement_score'])
            result['majority_label'] = row['majority_label']
            result['lang'] = row.get('lang', 'en')
            cache[tweet_id] = result
            save_counter += 1

            # Save every 100 samples
            if save_counter % 100 == 0:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f)
                print(f'  Saved {len(cache)} scores')

        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(5)
            continue

    # Final save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    # Also process validation and test data
    for split_name, split_df in [('val', val_df), ('test', test_df)]:
        print(f'\nProcessing {len(split_df)} {split_name} samples...')
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'{split_name}'):
            tweet_id = f'{split_name}_{row.get("id", idx)}'
            if tweet_id in cache:
                continue
            try:
                result = get_token_score(client, row['tweet'])
                result['tweet_id'] = tweet_id
                result['agreement'] = float(row['agreement_score'])
                result['majority_label'] = row['majority_label']
                result['lang'] = row.get('lang', 'en')
                cache[tweet_id] = result
            except:
                time.sleep(2)
                continue

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

    # Summary
    train_scores = [v for k, v in cache.items() if not k.startswith('val_') and not k.startswith('test_')]
    if train_scores:
        sig_scores = [s['sigmoid_score'] for s in train_scores]
        crit_fracs = [s['critical_fraction'] for s in train_scores]
        print(f'\n=== SUMMARY ===')
        print(f'Total cached: {len(cache)}')
        print(f'Training samples: {len(train_scores)}')
        print(f'Sigmoid score: mean={np.mean(sig_scores):.3f} std={np.std(sig_scores):.3f}')
        print(f'Critical fraction: mean={np.mean(crit_fracs):.3f} std={np.std(crit_fracs):.3f}')

    print(f'\nAll scores saved to {CACHE_FILE}')


if __name__ == '__main__':
    main()
