"""
OpenAI API runner with Batch API support and rate limiting.

Reads API key from OPENAI_API_KEY environment variable (never hardcoded).
Supports reasoning models (o3/o4) with special parameter handling.
"""

import json
import os
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .prompts import PromptBuilder, STRATEGY_MAX_TOKENS
from .config import ExperimentConfig
from .results_manager import ResultsManager


class TokenBucketRateLimiter:
    """Simple rate limiter based on requests per minute."""

    def __init__(self, rpm: int = 500):
        self.min_interval = 60.0 / rpm
        self.last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class OpenAIRunner:
    """
    Runs experiments using OpenAI API models.

    Two modes:
    - Batch API for large runs (>100 samples): 50% cost discount, async.
    - Individual calls with rate limiting for small/interactive runs.
    """

    def __init__(self, config: ExperimentConfig, results_mgr: ResultsManager):
        self.config = config
        self.results_mgr = results_mgr
        self.prompt_builder = PromptBuilder()

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def is_reasoning_model(self, model: str) -> bool:
        return model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    def supports_logprobs(self, model: str) -> bool:
        """GPT-5+ and reasoning models don't support logprobs."""
        if self.is_reasoning_model(model):
            return False
        if "gpt-5" in model:
            return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        model: str,
        strategy: str,
        scenario: str,
        lang: str,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run a single experiment configuration and return metrics."""
        system_prompt = self.prompt_builder.get_system_prompt(strategy, lang)

        examples = None
        if scenario == "few_shot":
            examples = self._select_few_shot_examples(train_df, lang)

        test_subset = test_df[test_df["lang"] == lang].copy() if lang != "both" else test_df.copy()
        if self.config.max_samples:
            test_subset = test_subset.head(self.config.max_samples)

        n_samples = len(test_subset)

        if self.config.openai_use_batch_api and n_samples > 100 and self.supports_logprobs(model):
            predictions, confidences = self._run_batch_api(
                model, system_prompt, test_subset, lang, strategy, examples
            )
        else:
            predictions, confidences = self._run_individual(
                model, system_prompt, test_subset, lang, strategy, examples
            )

        true_labels = test_subset["majority_label"].tolist()
        from src.utils.metrics import compute_metrics
        metrics = compute_metrics(true_labels, predictions)
        metrics["avg_confidence"] = float(np.mean(confidences)) if confidences else 0.0

        return {
            "model": model,
            "strategy": strategy,
            "scenario": scenario,
            "lang": lang,
            "metrics": metrics,
            "predictions": predictions,
            "confidences": [float(c) for c in confidences],
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Batch API mode
    # ------------------------------------------------------------------

    def _run_batch_api(
        self, model, system_prompt, test_df, lang, strategy, examples
    ) -> Tuple[List[str], List[float]]:
        """Use OpenAI Batch API for large runs."""
        max_tokens = self.prompt_builder.get_max_tokens(strategy)

        # Build JSONL requests
        requests = []
        id_order = []
        for idx, row in test_df.iterrows():
            user_prompt = self.prompt_builder.format_user_prompt(
                row["tweet"], lang, strategy, examples
            )
            custom_id = str(idx)
            id_order.append(custom_id)

            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

        # Write to temp JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
            jsonl_path = f.name

        try:
            # Upload file
            with open(jsonl_path, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="batch")

            # Create batch
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            # Poll until complete
            print(f"    Batch {batch.id} submitted ({len(requests)} requests). Polling...")
            while True:
                batch = self.client.batches.retrieve(batch.id)
                if batch.status in ("completed", "failed", "expired", "cancelled"):
                    break
                time.sleep(30)

            if batch.status != "completed":
                print(f"    Batch failed: {batch.status}")
                return ["NO"] * len(id_order), [0.5] * len(id_order)

            # Download results
            output_file = self.client.files.content(batch.output_file_id)
            result_map = {}
            for line in output_file.text.strip().split("\n"):
                item = json.loads(line)
                cid = item["custom_id"]
                resp = item.get("response", {}).get("body", {})
                result_map[cid] = resp

            # Parse in original order
            predictions = []
            confidences = []
            for cid in id_order:
                resp = result_map.get(cid, {})
                pred, conf = self._parse_api_response(resp, strategy)
                predictions.append(pred)
                confidences.append(conf)

            return predictions, confidences

        finally:
            os.unlink(jsonl_path)

    # ------------------------------------------------------------------
    # Individual call mode
    # ------------------------------------------------------------------

    def _run_individual(
        self, model, system_prompt, test_df, lang, strategy, examples
    ) -> Tuple[List[str], List[float]]:
        """Individual API calls with rate limiting."""
        predictions = []
        confidences = []
        rate_limiter = TokenBucketRateLimiter(rpm=self.config.openai_rate_limit_rpm)

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"    {model}"):
            rate_limiter.wait()
            pred, conf = self._predict_single(
                model, system_prompt, row["tweet"], lang, strategy, examples
            )
            predictions.append(pred)
            confidences.append(conf)

        return predictions, confidences

    def _predict_single(
        self, model, system_prompt, text, lang, strategy, examples, max_retries=3
    ) -> Tuple[str, float]:
        """Single prediction with retry and exponential backoff."""
        user_prompt = self.prompt_builder.format_user_prompt(text, lang, strategy, examples)
        max_tokens = self.prompt_builder.get_max_tokens(strategy)

        for attempt in range(max_retries):
            try:
                if self.is_reasoning_model(model):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
                        ],
                        max_completion_tokens=max_tokens + 50,
                    )
                    text_out = response.choices[0].message.content or ""
                    pred = self.prompt_builder.parse_prediction(text_out, lang)
                    return pred, 1.0
                elif not self.supports_logprobs(model):
                    # GPT-5+ models: no logprobs, no temperature
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_completion_tokens=max_tokens,
                    )
                    text_out = response.choices[0].message.content or ""
                    pred = self.prompt_builder.parse_prediction(text_out, lang)
                    return pred, 1.0
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                        logprobs=True,
                        top_logprobs=5,
                    )
                    return self._parse_api_response_obj(response, strategy, lang)

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                    time.sleep(wait)
                elif attempt == max_retries - 1:
                    return "NO", 0.5

        return "NO", 0.5

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_api_response_obj(self, response, strategy, lang="en") -> Tuple[str, float]:
        """Parse a live API response object."""
        text_out = response.choices[0].message.content or ""
        pred = self.prompt_builder.parse_prediction(text_out, lang)

        confidence = 0.5
        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            logprob = choice.logprobs.content[0].logprob
            confidence = min(float(np.exp(logprob)), 1.0)

        return pred, confidence

    def _parse_api_response(self, resp_body: dict, strategy: str) -> Tuple[str, float]:
        """Parse a Batch API response body dict."""
        choices = resp_body.get("choices", [])
        if not choices:
            return "NO", 0.5

        choice = choices[0]
        text_out = choice.get("message", {}).get("content", "")
        pred = self.prompt_builder.parse_prediction(text_out)

        confidence = 0.5
        logprobs_data = choice.get("logprobs", {})
        content = logprobs_data.get("content", []) if logprobs_data else []
        if content:
            logprob = content[0].get("logprob", -0.69)
            confidence = min(float(np.exp(logprob)), 1.0)

        return pred, confidence

    # ------------------------------------------------------------------
    # Few-shot example selection
    # ------------------------------------------------------------------

    def _select_few_shot_examples(
        self, train_df: pd.DataFrame, lang: str
    ) -> List[Dict[str, str]]:
        """Select few-shot examples using high-agreement strategy."""
        from src.data.data_loader import agreement_score

        lang_df = train_df[train_df["lang"] == lang].copy()
        lang_df["_agree"] = lang_df["labels_task1"].apply(agreement_score)
        lang_df = lang_df.sort_values("_agree", ascending=False)

        examples = []
        n_per_class = self.config.few_shot_examples_per_class
        for label in ["YES", "NO"]:
            subset = lang_df[lang_df["majority_label"] == label].head(n_per_class)
            for _, row in subset.iterrows():
                examples.append({"text": row["tweet"], "label": label})

        return examples[:self.config.few_shot_num_examples]
