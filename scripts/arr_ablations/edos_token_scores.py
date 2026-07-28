"""Token-uncertainty (sigmoid_score) plug-in for the EDOS track.

Produces per-item token-importance scores and the R(x) `sigmoid_score`
component for EDOS texts, mirroring what scripts/compute_token_scores.py did
for EXIST 2023. Two backends:

  --backend openai
      EXACT port of scripts/compute_token_scores.py to EDOS text:
      gpt-4o-mini rates each word's sexism relevance ("word|score" lines,
      temperature 0, logprobs on), the logprobs of the numeric score tokens
      give a per-word probability stream p_i, and
          sigmoid_score = mean( sigmoid( k * (T - p_i) ) ),  k=10, T=mean(p_i)
      critical_fraction = fraction of word scores > 0.5.
      DO NOT run while the OpenAI quota is exhausted — the port is provided
      so the two tracks stay method-identical once quota returns.

  --backend local
      Same word-masking importance idea scored with the LOCAL
      Llama-3.2-3B-Instruct YES/NO probabilities instead of the API model.
      For every item:
        1. p_full  = P(YES | structured prompt, full text)   (normalized over
           YES/NO first-answer tokens, exactly as in the EXIST local
           pipeline's stage_02/stage_06 inference).
        2. For each word w_i: p_i = P(YES | text with w_i removed), and
           conf_i = max(p_i, 1 - p_i)  (the masked-classification confidence,
           the local analogue of the API score-token probability stream).
        3. word_scores[w_i] = |p_full - p_i|      (importance, in [0, 1])
           sigmoid_score    = mean( sigmoid( k * (T - conf_i) ) ), T=mean(conf_i)
           critical_fraction = fraction of word_scores > 0.5
      Masked variants are batched (left padding) so each item costs
      ceil((n_words + 1) / batch_size) forward passes.

      --approx single_pass
      Cheaper approximation behind a flag (recommended if the full masking
      run exceeds the time budget): ONE forward pass per item. The per-token
      probabilities of the text tokens under the model (teacher forcing,
      inside the same structured prompt) form the probability stream p_i and
          sigmoid_score = mean( sigmoid( k * (T - p_i) ) ), T = mean(p_i)
      word_scores[token] = 1 - p_i (surprisal-style importance proxy).
      This keeps the exact sigmoid aggregation formula while reducing cost
      from ~(n_words+1) to 1 forward pass per item.

Cache format mirrors results/token_scores/token_scores_cache.json:
one JSON dict keyed by rewire_id with fields word_scores, critical_fraction,
sigmoid_score, n_words, agreement, majority_label, gold_label, lang, split,
backend. Saved incrementally to resume on disconnect. Each backend writes its
OWN cache file (token_scores_<backend>.json) so methods cannot silently mix;
configs/edos_pipeline.yaml -> rx.token_scores_backend selects the canonical
cache that edos_pipeline.py consumes.

Measured on M4 Max (MPS, fp32, batch 16): masking = 2.21 s/item on short
items (~12 prompts/item); at the corpus mean of 24.3 prompts/item that is
~4.4 s/item -> ~17 h for train+test (18,000 items). Over the 8 h budget, so
`--approx single_pass` is the recommended backend for the light track.

Usage:
    python scripts/arr_ablations/edos_token_scores.py --backend local \
        --splits train test [--limit 3] [--device mps] [--batch-size 16]
    python scripts/arr_ablations/edos_token_scores.py --backend local \
        --approx single_pass --splits train test      # RECOMMENDED
    python scripts/arr_ablations/edos_token_scores.py --backend openai \
        --splits train test          # only when API quota is available
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.edos_loader import EDOSDataLoader  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "edos_pipeline.yaml"
STEEPNESS = 10  # k parameter — same as scripts/compute_token_scores.py


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


BACKEND_TAGS = ("openai", "local_masking", "local_single_pass")


def backend_tag(backend: str, approx: str | None = None) -> str:
    if backend == "openai":
        return "openai"
    return "local_single_pass" if approx == "single_pass" else "local_masking"


def cache_path(tag: str | None = None) -> Path:
    """Cache file for one backend. Default: the canonical backend selected
    by configs/edos_pipeline.yaml -> rx.token_scores_backend."""
    cfg = load_config()["rx"]
    tag = tag or cfg["token_scores_backend"]
    if tag not in BACKEND_TAGS:
        raise KeyError(f"unknown token-scores backend {tag!r}")
    return ROOT / cfg["token_scores_dir"] / f"token_scores_{tag}.json"


def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def sigmoid_score_from_probs(probs: np.ndarray) -> float:
    """mean(sigmoid(k * (T - p_i))) with T = mean(p_i) — the exact formula of
    scripts/compute_token_scores.py."""
    if len(probs) == 0:
        return 0.5
    T = float(np.mean(probs))
    return float(np.mean(sigmoid(STEEPNESS * (T - probs))))


# --------------------------------------------------------------------------
# Plug-in interface used by scripts/arr_ablations/edos_pipeline.py
# --------------------------------------------------------------------------

def load_cache(tag: str | None = None) -> dict:
    p = cache_path(tag)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def load_sigmoid_scores(rewire_ids, tag: str | None = None) -> np.ndarray:
    """Return sigmoid_scores aligned with `rewire_ids` from the canonical
    (config-selected) backend cache. Raises if any id is missing."""
    cache = load_cache(tag)
    missing = [r for r in rewire_ids if r not in cache]
    if missing:
        raise KeyError(
            f"{len(missing)} rewire_ids missing from {cache_path(tag)} "
            f"(first: {missing[:3]}). Run edos_token_scores.py first.")
    return np.asarray([cache[r]["sigmoid_score"] for r in rewire_ids],
                      dtype=float)


def save_cache(cache: dict, tag: str | None = None) -> None:
    p = cache_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(cache, f)


# --------------------------------------------------------------------------
# Backend: openai — exact port of scripts/compute_token_scores.py
# --------------------------------------------------------------------------

def get_token_score_openai(client, text: str, model: str = "gpt-4o-mini") -> dict:
    """Verbatim method port (prompt, parsing, formulas) of
    scripts/compute_token_scores.py::get_token_score, applied to EDOS text."""
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

    word_scores = {}
    for line in content.strip().split('\n'):
        line = line.strip()
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                word = parts[0].strip()
                try:
                    word_scores[word] = float(parts[1].strip())
                except ValueError:
                    pass

    score_logprobs = []
    for t in tokens_data:
        tok = t.token.strip()
        if tok and all(c in '0123456789.' for c in tok):
            score_logprobs.append(t.logprob)

    if score_logprobs:
        probs = np.exp(np.array(score_logprobs))
        sig = sigmoid_score_from_probs(probs)
    else:
        sig = 0.5

    if word_scores:
        vals = list(word_scores.values())
        critical_fraction = sum(1 for v in vals if v > 0.5) / len(vals)
    else:
        critical_fraction = 0.5

    return {
        'word_scores': word_scores,
        'critical_fraction': critical_fraction,
        'sigmoid_score': sig,
        'score_logprobs': [float(lp) for lp in score_logprobs],
        'n_words': len(word_scores),
        'raw_response': content,
    }


def run_openai(df, cache, tag, save_every=100):
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY must be set for --backend openai')
    from openai import OpenAI
    from tqdm import tqdm
    client = OpenAI()
    n_new = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc='openai token scoring'):
        rid = row['rewire_id']
        if rid in cache:
            continue
        try:
            result = get_token_score_openai(client, row['text'])
        except Exception as e:  # rate limits etc. — resume-safe, like the original
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(5)
            continue
        result.update(_meta(row, backend=tag))
        cache[rid] = result
        n_new += 1
        if n_new % save_every == 0:
            save_cache(cache, tag)
    save_cache(cache, tag)
    return n_new


# --------------------------------------------------------------------------
# Backend: local — word-masking with Llama YES/NO probabilities
# --------------------------------------------------------------------------

def _meta(row, backend: str) -> dict:
    return {
        'rewire_id': row['rewire_id'],
        'agreement': float(row['agreement_score']),
        'majority_label': row['majority_label'],
        'gold_label': row['gold_label'],
        'lang': row['lang'],
        'split': row['split'],
        'backend': backend,
    }


def load_local_model(device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = load_config()
    model_id = cfg["model"]["id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so logits[:, -1] is the next-answer position
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    if device == "cpu":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()
    return tok, model


def yes_no_ids(tok):
    """Identical first-answer-token id sets to the EXIST local pipeline."""
    yes, no = [], []
    for t in [" YES", "YES", " Yes", "Yes", " yes", "yes"]:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            yes.append(ids[0])
    for t in [" NO", "NO", " No", "No", " no", "no"]:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids:
            no.append(ids[0])
    return list(dict.fromkeys(yes)), list(dict.fromkeys(no))


def build_prompt(tok, pb, strategy: str, text: str) -> str:
    messages = [
        {"role": "system", "content": pb.get_system_prompt(strategy, "en")},
        {"role": "user", "content": pb.format_user_prompt(text, "en", strategy)},
    ]
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   tokenize=False)


def p_yes_batch(model, tok, prompts, yes_ids, no_ids, device, batch_size):
    """Normalized P(YES) for each prompt, batched with left padding."""
    import torch
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = probs[:, yes_ids].sum(dim=-1)
        p_no = probs[:, no_ids].sum(dim=-1)
        total = p_yes + p_no
        norm = torch.where(total > 0, p_yes / total,
                           torch.full_like(total, 0.5))
        out.extend(norm.cpu().tolist())
    return out


def get_token_score_local_masking(model, tok, pb, strategy, text,
                                  yes_ids_, no_ids_, device, batch_size) -> dict:
    words = text.split()
    prompts = [build_prompt(tok, pb, strategy, text)]
    for i in range(len(words)):
        masked = " ".join(words[:i] + words[i + 1:])
        prompts.append(build_prompt(tok, pb, strategy, masked))
    p = p_yes_batch(model, tok, prompts, yes_ids_, no_ids_, device, batch_size)
    p_full, p_masked = p[0], np.asarray(p[1:], dtype=float)

    word_scores = {w: float(abs(p_full - p_masked[i]))
                   for i, w in enumerate(words)}
    conf = np.maximum(p_masked, 1.0 - p_masked)   # masked-classification confidence
    sig = sigmoid_score_from_probs(conf)
    vals = list(word_scores.values())
    critical_fraction = (sum(1 for v in vals if v > 0.5) / len(vals)
                         if vals else 0.5)
    return {
        'word_scores': word_scores,
        'critical_fraction': critical_fraction,
        'sigmoid_score': sig,
        'p_yes_full': float(p_full),
        'n_words': len(words),
    }


def get_token_score_local_single_pass(model, tok, pb, strategy, text,
                                      device) -> dict:
    """One forward pass: per-token teacher-forced probabilities of the text
    region inside the structured prompt form the probability stream."""
    import torch
    prompt = build_prompt(tok, pb, strategy, text)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    # locate the text tokens inside the prompt
    text_ids = tok(text, add_special_tokens=False).input_ids
    full = ids[0].tolist()
    start = None
    for j in range(len(full) - len(text_ids) + 1):
        if full[j:j + len(text_ids)] == text_ids:
            start = j
            break
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    if start is None or start == 0:   # fallback: whole prompt region
        start, text_ids = 1, full[1:]
    # P(token_t | prefix) comes from logits at position t-1
    probs = np.array([
        float(logprobs[start + k - 1, full[start + k]].exp())
        for k in range(len(text_ids))
    ])
    sig = sigmoid_score_from_probs(probs)
    toks = tok.convert_ids_to_tokens(text_ids)
    word_scores = {f"{k}:{t}": float(1.0 - probs[k]) for k, t in enumerate(toks)}
    vals = list(word_scores.values())
    critical_fraction = (sum(1 for v in vals if v > 0.5) / len(vals)
                         if vals else 0.5)
    return {
        'word_scores': word_scores,
        'critical_fraction': critical_fraction,
        'sigmoid_score': sig,
        'n_words': len(toks),
    }


def run_local(df, cache, tag, device, batch_size, approx, save_every=100):
    from tqdm import tqdm
    from src.pipeline.prompts import PromptBuilder
    cfg = load_config()
    strategy = cfg["prompt"]["strategy"]
    tok, model = load_local_model(device)
    y_ids, n_ids = yes_no_ids(tok)
    pb = PromptBuilder()

    n_new, t0 = 0, time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc=tag):
        rid = row['rewire_id']
        if rid in cache:
            continue
        if approx == "single_pass":
            result = get_token_score_local_single_pass(
                model, tok, pb, strategy, row['text'], device)
        else:
            result = get_token_score_local_masking(
                model, tok, pb, strategy, row['text'],
                y_ids, n_ids, device, batch_size)
        result.update(_meta(row, backend=tag))
        cache[rid] = result
        n_new += 1
        if n_new % save_every == 0:
            save_cache(cache, tag)
    save_cache(cache, tag)

    if n_new:
        per_item = (time.time() - t0) / n_new
        print(f"\n{tag}: {n_new} items scored, "
              f"{per_item:.2f} s/item on {device}")
        for name, n in [("train (14,000)", 14000), ("test (4,000)", 4000),
                        ("train+test (18,000)", 18000)]:
            print(f"  extrapolated {name}: {per_item * n / 3600:.2f} h")
    return n_new


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["openai", "local"], required=True)
    ap.add_argument("--approx", choices=["single_pass"], default=None,
                    help="local backend only: 1-forward-pass approximation")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N items (smoke tests)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    loader = EDOSDataLoader(ROOT / "data" / "edos")
    df = loader.to_dataframe()
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    tag = backend_tag(args.backend, args.approx)
    print(f"scoring {len(df)} items (splits={args.splits}, backend={tag})")

    cache = load_cache(tag)
    print(f"cache: {len(cache)} existing entries at {cache_path(tag)}")

    if args.backend == "openai":
        n = run_openai(df, cache, tag)
    else:
        n = run_local(df, cache, tag, args.device, args.batch_size,
                      args.approx)
    print(f"done: {n} new entries, cache now {len(cache)}")

    sigs = [v["sigmoid_score"] for v in cache.values()]
    if sigs:
        print(f"sigmoid_score: mean={np.mean(sigs):.4f} std={np.std(sigs):.4f}")


if __name__ == "__main__":
    main()
