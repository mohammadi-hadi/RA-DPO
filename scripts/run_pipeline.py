#!/usr/bin/env python3
"""
Unified Sexism Detection Experiment Pipeline

Single entry point replacing all previous experiment scripts.

Usage:
    python scripts/run_pipeline.py                                   # Run everything
    python scripts/run_pipeline.py --stages openai local             # Only API + local models
    python scripts/run_pipeline.py --stages training                 # Only SFT/DPO training
    python scripts/run_pipeline.py --stages efficiency               # Only data efficiency
    python scripts/run_pipeline.py --stages report                   # Only generate reports
    python scripts/run_pipeline.py --openai-models gpt-4.1-mini     # Specific OpenAI model
    python scripts/run_pipeline.py --local-models TinyLlama-1.1B    # Specific local model
    python scripts/run_pipeline.py --prompt-strategies basic persona # Specific prompts
    python scripts/run_pipeline.py --max-samples 50                  # Debug mode
    python scripts/run_pipeline.py --resume                          # Resume from checkpoints
    python scripts/run_pipeline.py --no-resume                       # Force re-run everything
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sexism detection experiment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="configs/experiment_config.yaml",
        help="Path to experiment config YAML (default: configs/experiment_config.yaml)",
    )
    parser.add_argument(
        "--stages", nargs="+",
        choices=["openai", "local", "training", "efficiency", "reliability", "report"],
        help="Stages to run (default: all)",
    )
    parser.add_argument(
        "--openai-models", nargs="+", dest="openai_models",
        help="Subset of OpenAI models to run",
    )
    parser.add_argument(
        "--local-models", nargs="+", dest="local_models",
        help="Subset of local models to run (use keys from config, e.g. TinyLlama-1.1B)",
    )
    parser.add_argument(
        "--prompt-strategies", nargs="+", dest="prompt_strategies",
        choices=["basic", "definition", "cot", "persona", "structured"],
        help="Subset of prompt strategies to test",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, dest="max_samples",
        help="Limit test samples per experiment (for debugging)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, dest="output_dir",
        help="Override results output directory",
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", action="store_true", default=None,
        help="Resume from checkpoints (skip completed experiments)",
    )
    resume_group.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Force re-run all experiments",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    from ra_dpo.pipeline.config import ExperimentConfig
    from ra_dpo.pipeline.orchestrator import PipelineOrchestrator

    config = ExperimentConfig.from_yaml(args.config, overrides=args)
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
