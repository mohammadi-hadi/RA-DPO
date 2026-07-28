"""
Experiment configuration with YAML loading and CLI override support.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class ExperimentConfig:
    """Full experiment matrix configuration."""

    # Data
    data_path: str = "EXIST2023_training.json"
    test_data_path: str = "EXIST 2023 Dataset/test/EXIST2023_test_clean.json"
    languages: List[str] = field(default_factory=lambda: ["en", "es"])
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42

    # OpenAI models
    openai_models: List[str] = field(default_factory=lambda: [
        "gpt-4o-mini", "gpt-4o",
        "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano",
        "o3-mini", "o4-mini",
    ])

    # Local HuggingFace models
    local_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "TinyLlama-1.1B": {
            "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantize": False,
            "train": False,
        },
        "Phi-3-mini-3.8B": {
            "hf_id": "microsoft/Phi-3-mini-4k-instruct",
            "quantize": False,
            "train": False,
        },
        "Mistral-7B": {
            "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "quantize": False,
            "train": True,
        },
        "Llama-3.1-8B": {
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "quantize": False,
            "train": True,
        },
        "Qwen2.5-7B": {
            "hf_id": "Qwen/Qwen2.5-7B-Instruct",
            "quantize": False,
            "train": False,
        },
        "Gemma-2-9B": {
            "hf_id": "google/gemma-2-9b-it",
            "quantize": False,
            "train": False,
        },
    })

    # Prompt strategies
    prompt_strategies: List[str] = field(default_factory=lambda: [
        "basic", "definition", "cot", "persona", "structured",
    ])

    # Scenarios
    scenarios: List[str] = field(default_factory=lambda: [
        "zero_shot", "few_shot",
    ])
    training_scenarios: List[str] = field(default_factory=lambda: [
        "sft", "dpo", "confidence_dpo", "xai_dpo",
    ])

    # Efficiency experiment
    efficiency_fractions: List[float] = field(default_factory=lambda: [
        0.10, 0.25, 0.50, 0.75, 1.00,
    ])
    efficiency_sampling: List[str] = field(default_factory=lambda: [
        "random", "smart",
    ])
    efficiency_models: List[str] = field(default_factory=lambda: [
        "Mistral-7B", "Llama-3.1-8B",
    ])

    # Few-shot settings
    few_shot_num_examples: int = 5
    few_shot_examples_per_class: int = 3
    few_shot_selection: str = "high_agreement"

    # Output
    output_dir: str = "./results/experiments"
    resume: bool = True
    max_samples: Optional[int] = None

    # OpenAI settings
    openai_batch_size: int = 50
    openai_rate_limit_rpm: int = 500
    openai_use_batch_api: bool = True

    # Stages to run
    stages: List[str] = field(default_factory=lambda: [
        "openai", "local", "training", "efficiency", "reliability", "report",
    ])

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[argparse.Namespace] = None) -> "ExperimentConfig":
        """Load config from YAML, apply CLI overrides."""
        config = cls()

        yaml_path = Path(path)
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
            config = cls._apply_yaml(config, data)

        if overrides is not None:
            config = cls._apply_overrides(config, overrides)

        return config

    @classmethod
    def _apply_yaml(cls, config: "ExperimentConfig", data: dict) -> "ExperimentConfig":
        """Apply YAML data to config."""
        flat_map = {
            "experiment.seed": "seed",
            "data.train_file": "data_path",
            "data.test_file": "test_data_path",
            "data.languages": "languages",
            "data.train_split": "train_split",
            "data.val_split": "val_split",
            "data.test_split": "test_split",
            "output.base_dir": "output_dir",
            "output.resume": "resume",
            "openai.batch_api": "openai_use_batch_api",
            "openai.rate_limit_rpm": "openai_rate_limit_rpm",
            "few_shot.num_examples": "few_shot_num_examples",
            "few_shot.examples_per_class": "few_shot_examples_per_class",
            "few_shot.selection_strategy": "few_shot_selection",
        }

        for yaml_key, attr in flat_map.items():
            parts = yaml_key.split(".")
            val = data
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    val = None
                    break
            if val is not None:
                setattr(config, attr, val)

        # Direct list/dict fields
        if "openai" in data and "models" in data["openai"]:
            config.openai_models = data["openai"]["models"]
        if "local_models" in data:
            config.local_models = data["local_models"]
        if "prompt_strategies" in data:
            config.prompt_strategies = data["prompt_strategies"]
        if "scenarios" in data:
            config.scenarios = data["scenarios"]
        if "training_scenarios" in data:
            config.training_scenarios = data["training_scenarios"]
        if "efficiency" in data:
            eff = data["efficiency"]
            if "fractions" in eff:
                config.efficiency_fractions = eff["fractions"]
            if "sampling_strategies" in eff:
                config.efficiency_sampling = eff["sampling_strategies"]
            if "models" in eff:
                config.efficiency_models = eff["models"]

        return config

    @classmethod
    def _apply_overrides(cls, config: "ExperimentConfig", args: argparse.Namespace) -> "ExperimentConfig":
        """Apply CLI argument overrides."""
        if hasattr(args, "stages") and args.stages:
            config.stages = args.stages
        if hasattr(args, "openai_models") and args.openai_models:
            config.openai_models = args.openai_models
        if hasattr(args, "local_models") and args.local_models:
            # Filter local_models dict to only requested keys
            config.local_models = {
                k: v for k, v in config.local_models.items()
                if k in args.local_models
            }
        if hasattr(args, "prompt_strategies") and args.prompt_strategies:
            config.prompt_strategies = args.prompt_strategies
        if hasattr(args, "max_samples") and args.max_samples is not None:
            config.max_samples = args.max_samples
        if hasattr(args, "resume") and args.resume is not None:
            config.resume = args.resume
        if hasattr(args, "output_dir") and args.output_dir:
            config.output_dir = args.output_dir
        if hasattr(args, "config") and args.config:
            pass  # Already loaded from YAML

        return config

    @property
    def trainable_models(self) -> Dict[str, Dict[str, Any]]:
        """Return only models marked for training."""
        return {k: v for k, v in self.local_models.items() if v.get("train", False)}

    @property
    def reasoning_models(self) -> List[str]:
        """OpenAI reasoning models that need special handling."""
        return [m for m in self.openai_models if m.startswith("o3") or m.startswith("o4")]
