"""
Local HuggingFace model runner for M4 Max with MPS memory management.

Loads one model at a time, runs all prompt/scenario combinations,
then explicitly frees memory before loading the next model.
"""

import gc
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .prompts import PromptBuilder
from .config import ExperimentConfig
from .results_manager import ResultsManager


class LocalModelRunner:
    """
    Runs zero-shot and few-shot experiments using local HuggingFace models.

    Memory strategy for M4 Max 64 GB:
    - fp16 for all models (no bitsandbytes on MPS)
    - Explicit unload between models via unload_model()
    - Batch size 1 for generation (sequential)
    """

    def __init__(self, config: ExperimentConfig, results_mgr: ResultsManager):
        self.config = config
        self.results_mgr = results_mgr
        self.prompt_builder = PromptBuilder()

        self.model = None
        self.tokenizer = None
        self.current_model_name: Optional[str] = None
        self.yes_token_ids: List[int] = []
        self.no_token_ids: List[int] = []

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self, hf_id: str, quantize: bool = False):
        """Load a HuggingFace model for inference."""
        self.unload_model()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.models.sft_trainer import _detect_device, _get_torch_dtype

        device = _detect_device()
        dtype = _get_torch_dtype(device)
        print(f"  Loading {hf_id} (device={device}, dtype={dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        if device == "mps":
            self.model = self.model.to(device)

        self.model.eval()
        self.current_model_name = hf_id
        self._setup_answer_tokens()

    def unload_model(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        self.yes_token_ids = []
        self.no_token_ids = []
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _setup_answer_tokens(self):
        """Map YES/NO answer tokens to IDs for confidence extraction."""
        yes_tokens = ["yes", "Yes", "YES", "SI", "Si", "si", "sexist", "Sexist"]
        no_tokens = ["no", "No", "NO", "not", "Not"]

        self.yes_token_ids = list(set(
            tid for t in yes_tokens
            for tid in self.tokenizer.encode(t, add_special_tokens=False)
        ))
        self.no_token_ids = list(set(
            tid for t in no_tokens
            for tid in self.tokenizer.encode(t, add_special_tokens=False)
        ))

    # ------------------------------------------------------------------
    # Experiment execution
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        strategy: str,
        scenario: str,
        lang: str,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run a single experiment with the currently loaded model."""
        examples = None
        if scenario == "few_shot":
            examples = self._select_few_shot_examples(train_df, lang)

        test_subset = test_df[test_df["lang"] == lang].copy() if lang != "both" else test_df.copy()
        if self.config.max_samples:
            test_subset = test_subset.head(self.config.max_samples)

        predictions, confidences = [], []

        for _, row in tqdm(
            test_subset.iterrows(), total=len(test_subset),
            desc=f"    {strategy}/{scenario}/{lang}"
        ):
            prompt = self.prompt_builder.format_local_prompt(
                row["tweet"], lang, strategy, examples
            )
            pred, conf = self._predict(prompt, lang, strategy)
            predictions.append(pred)
            confidences.append(conf)

        true_labels = test_subset["majority_label"].tolist()
        from src.utils.metrics import compute_metrics
        metrics = compute_metrics(true_labels, predictions)
        metrics["avg_confidence"] = float(np.mean(confidences)) if confidences else 0.0

        return {
            "model": self.current_model_name,
            "strategy": strategy,
            "scenario": scenario,
            "lang": lang,
            "metrics": metrics,
            "predictions": predictions,
            "confidences": [float(c) for c in confidences],
            "n_samples": len(test_subset),
            "timestamp": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(self, prompt: str, lang: str, strategy: str) -> Tuple[str, float]:
        """Generate prediction and extract confidence from logits."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        max_new = 5 if strategy != "cot" else 200

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # For CoT, parse the full generated text
        if strategy == "cot":
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            text_out = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            pred = self.prompt_builder.parse_prediction(text_out, lang)
            # Approximate confidence from first token
            confidence = self._extract_confidence_from_scores(outputs.scores)
            return pred, confidence

        # For direct-answer strategies, use first token logits
        confidence = self._extract_confidence_from_scores(outputs.scores)

        first_token_logits = outputs.scores[0][0]
        all_probs = torch.softmax(first_token_logits, dim=-1)

        yes_prob = sum(
            all_probs[tid].item() for tid in self.yes_token_ids if tid < len(all_probs)
        )
        no_prob = sum(
            all_probs[tid].item() for tid in self.no_token_ids if tid < len(all_probs)
        )

        total = yes_prob + no_prob
        if total > 0:
            yes_norm = yes_prob / total
        else:
            yes_norm = 0.5

        prediction = "YES" if yes_norm > 0.5 else "NO"
        confidence = max(yes_norm, 1.0 - yes_norm)
        return prediction, confidence

    def _extract_confidence_from_scores(self, scores) -> float:
        """Extract confidence from the first generation step scores."""
        if not scores:
            return 0.5
        first_logits = scores[0][0]
        probs = torch.softmax(first_logits, dim=-1)
        max_prob = probs.max().item()
        return min(max_prob, 1.0)

    # ------------------------------------------------------------------
    # Few-shot
    # ------------------------------------------------------------------

    def _select_few_shot_examples(
        self, train_df: pd.DataFrame, lang: str
    ) -> List[Dict[str, str]]:
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
