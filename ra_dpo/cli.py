#!/usr/bin/env python3
"""Command-line entry point for the RA-DPO experiment pipeline.

Installed as the ``ra-dpo`` console command. Run it from the repository root
so the default config paths (``configs/*.yaml``) resolve.

Usage:
    ra-dpo                                   # run everything
    ra-dpo --stages openai local             # only API + local models
    ra-dpo --stages training                 # only SFT/DPO training
    ra-dpo --stages efficiency               # only data efficiency
    ra-dpo --stages report                   # only generate reports
    ra-dpo --openai-models gpt-4.1-mini      # specific OpenAI model
    ra-dpo --local-models TinyLlama-1.1B     # specific local model
    ra-dpo --prompt-strategies basic persona # specific prompts
    ra-dpo --max-samples 50                  # small-sample debug run
    ra-dpo --resume                          # resume from checkpoints
    ra-dpo --no-resume                       # force re-run everything

``python scripts/run_pipeline.py`` from a clone is equivalent to ``ra-dpo``.
"""

import argparse


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
