"""
Main pipeline orchestrator.

Coordinates all experiment stages:
  1. Load data, create splits
  2. OpenAI models: zero-shot & few-shot across prompt strategies
  3. Local models: zero-shot & few-shot across prompt strategies
  4. Training: SFT → DPO → Conf-DPO → XAI-DPO for selected models
  5. Efficiency: DPO data efficiency experiment (10-100%)
  6. Reliability: Agreement predictor + weight optimization
  7. Report: Generate tables, plots, LaTeX
"""

import time
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import ExperimentConfig
from .results_manager import ResultsManager


class PipelineOrchestrator:
    """Runs the full experiment pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_mgr = ResultsManager(config.output_dir)

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def run(self):
        """Execute all requested stages in order."""
        stages = self.config.stages
        t0 = time.time()

        print("=" * 70)
        print("  Sexism Detection — Comprehensive Experiment Pipeline")
        print("=" * 70)
        print(f"  Stages: {stages}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Resume: {self.config.resume}")
        if self.config.max_samples:
            print(f"  Max samples: {self.config.max_samples} (debug mode)")
        print("=" * 70)

        self._load_data()

        if "openai" in stages:
            self._run_openai_stage()
        if "local" in stages:
            self._run_local_stage()
        if "training" in stages:
            self._run_training_stage()
        if "efficiency" in stages:
            self._run_efficiency_stage()
        if "reliability" in stages:
            self._run_reliability_stage()
        if "report" in stages:
            self._run_report_stage()

        elapsed = time.time() - t0
        print(f"\nPipeline complete in {elapsed / 60:.1f} minutes.")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score

        print("\n[DATA] Loading EXIST 2023 dataset...")
        loader = EXISTDataLoader(self.config.data_path)
        df = loader.to_dataframe()
        df["majority_label"] = df["labels_task1"].apply(majority_vote)
        df["agreement_score"] = df["labels_task1"].apply(agreement_score)

        self.train_df, self.val_df, self.test_df = loader.create_train_val_test_split(
            df,
            train_size=self.config.train_split,
            val_size=self.config.val_split,
            test_size=self.config.test_split,
            random_state=self.config.seed,
        )

        for split_name, split_df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            en = len(split_df[split_df["lang"] == "en"])
            es = len(split_df[split_df["lang"] == "es"])
            print(f"  {split_name}: {len(split_df)} (en={en}, es={es})")

    # ------------------------------------------------------------------
    # Stage: OpenAI
    # ------------------------------------------------------------------

    def _run_openai_stage(self):
        from .openai_runner import OpenAIRunner

        print("\n" + "=" * 70)
        print("  STAGE: OpenAI API Models")
        print("=" * 70)

        runner = OpenAIRunner(self.config, self.results_mgr)

        total = len(self.config.openai_models) * len(self.config.prompt_strategies) * len(self.config.scenarios) * len(self.config.languages)
        done = 0

        for model in self.config.openai_models:
            print(f"\n  Model: {model}")
            for strategy in self.config.prompt_strategies:
                for scenario in self.config.scenarios:
                    for lang in self.config.languages:
                        done += 1
                        key = f"openai/{model}/{strategy}/{scenario}/{lang}"

                        if self.config.resume and self.results_mgr.exists(key):
                            print(f"    [{done}/{total}] SKIP {key}")
                            continue

                        print(f"    [{done}/{total}] {key}")
                        try:
                            result = runner.run_experiment(
                                model=model, strategy=strategy, scenario=scenario,
                                lang=lang, test_df=self.test_df, train_df=self.train_df,
                            )
                            self.results_mgr.save(key, result)
                            f1 = result["metrics"].get("f1_macro", 0)
                            print(f"      → F1={f1:.4f}")
                        except Exception as e:
                            print(f"      → ERROR: {e}")

    # ------------------------------------------------------------------
    # Stage: Local models
    # ------------------------------------------------------------------

    def _run_local_stage(self):
        from .local_runner import LocalModelRunner

        print("\n" + "=" * 70)
        print("  STAGE: Local HuggingFace Models")
        print("=" * 70)

        runner = LocalModelRunner(self.config, self.results_mgr)

        for model_key, model_cfg in self.config.local_models.items():
            hf_id = model_cfg["hf_id"]

            # Check if all experiments for this model are done
            all_done = all(
                self.results_mgr.exists(f"local/{model_key}/{s}/{sc}/{l}")
                for s in self.config.prompt_strategies
                for sc in self.config.scenarios
                for l in self.config.languages
            )
            if self.config.resume and all_done:
                print(f"\n  {model_key}: all experiments complete, skipping")
                continue

            print(f"\n  Loading {model_key} ({hf_id})...")
            try:
                runner.load_model(hf_id, quantize=model_cfg.get("quantize", False))
            except Exception as e:
                print(f"    ERROR loading model: {e}")
                continue

            for strategy in self.config.prompt_strategies:
                for scenario in self.config.scenarios:
                    for lang in self.config.languages:
                        key = f"local/{model_key}/{strategy}/{scenario}/{lang}"

                        if self.config.resume and self.results_mgr.exists(key):
                            print(f"    SKIP {key}")
                            continue

                        print(f"    {key}")
                        try:
                            result = runner.run_experiment(
                                strategy=strategy, scenario=scenario,
                                lang=lang, test_df=self.test_df, train_df=self.train_df,
                            )
                            self.results_mgr.save(key, result)
                            f1 = result["metrics"].get("f1_macro", 0)
                            print(f"      → F1={f1:.4f}")
                        except Exception as e:
                            print(f"      → ERROR: {e}")

            runner.unload_model()

    # ------------------------------------------------------------------
    # Stage: Training
    # ------------------------------------------------------------------

    def _run_training_stage(self):
        from .training_runner import TrainingRunner

        print("\n" + "=" * 70)
        print("  STAGE: SFT / DPO Training")
        print("=" * 70)

        runner = TrainingRunner(self.config, self.results_mgr)

        for model_key, model_cfg in self.config.trainable_models.items():
            print(f"\n  Training {model_key} ({model_cfg['hf_id']})")
            try:
                runner.run_training_for_model(
                    model_key=model_key, model_cfg=model_cfg,
                    train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # Stage: Efficiency
    # ------------------------------------------------------------------

    def _run_efficiency_stage(self):
        from .efficiency_experiment import EfficiencyExperiment

        print("\n" + "=" * 70)
        print("  STAGE: Data Efficiency Experiment")
        print("=" * 70)

        runner = EfficiencyExperiment(self.config, self.results_mgr)

        for model_key in self.config.efficiency_models:
            model_cfg = self.config.local_models.get(model_key)
            if model_cfg is None:
                print(f"  WARNING: {model_key} not in local_models, skipping")
                continue
            print(f"\n  Efficiency experiment for {model_key}")
            try:
                runner.run(
                    model_key=model_key, model_cfg=model_cfg,
                    train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # Stage: Reliability
    # ------------------------------------------------------------------

    def _run_reliability_stage(self):
        print("\n" + "=" * 70)
        print("  STAGE: Reliability Analysis")
        print("=" * 70)

        key = "reliability/agreement_predictor"
        if self.config.resume and self.results_mgr.exists(key):
            print("  Agreement predictor already trained, skipping")
        else:
            print("  Training agreement predictor...")
            try:
                from src.models.agreement_predictor import train_agreement_predictor
                result = train_agreement_predictor(
                    train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                    output_dir=str(self.config.output_dir + "/models/agreement_predictor"),
                )
                result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(key, result)
            except Exception as e:
                print(f"    ERROR: {e}")

        key = "reliability/weight_optimization"
        if self.config.resume and self.results_mgr.exists(key):
            print("  Weight optimization already done, skipping")
        else:
            print("  Optimizing reliability weights...")
            try:
                import numpy as np
                from src.utils.weight_optimizer import optimize_reliability_weights

                # Use validation data for weight optimization
                val_conf = np.random.uniform(0.5, 1.0, len(self.val_df))  # Placeholder
                val_agree = self.val_df["agreement_score"].values
                val_crit = np.random.uniform(0.0, 0.5, len(self.val_df))  # Placeholder
                val_correct = np.ones(len(self.val_df))  # Placeholder

                result = optimize_reliability_weights(
                    val_conf, val_agree, val_crit, val_correct,
                )
                result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(key, result)
            except Exception as e:
                print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # Stage: Report
    # ------------------------------------------------------------------

    def _run_report_stage(self):
        print("\n" + "=" * 70)
        print("  STAGE: Report Generation")
        print("=" * 70)

        # Generate comparison CSV/LaTeX
        df = self.results_mgr.to_dataframe()
        if df.empty:
            print("  No results to report.")
            return

        print(f"  Total experiments: {len(df)}")

        # Main comparison table
        csv_df = self.results_mgr.generate_comparison_csv()
        print(f"  Saved comparison CSV ({len(csv_df)} rows)")

        # LaTeX table
        if not df.empty:
            self.results_mgr.generate_latex_table(df)
            print("  Saved LaTeX table")

        # Generate plots
        self._generate_plots(df)

    def _generate_plots(self, df: pd.DataFrame):
        """Generate all publication-ready plots."""
        from pathlib import Path

        plots_dir = Path(self.config.output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("  matplotlib/seaborn not available, skipping plots")
            return

        # Set paper style
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

        # Plot 1: Model comparison bar chart
        if "f1_macro" in df.columns and "model" in df.columns:
            self._plot_model_comparison(df, plots_dir)

        # Plot 2: Prompt strategy heatmap
        if "strategy" in df.columns and "f1_macro" in df.columns:
            self._plot_strategy_heatmap(df, plots_dir)

        # Plot 3: Language comparison
        if "lang" in df.columns:
            self._plot_language_comparison(df, plots_dir)

        # Plot 4: Efficiency curves
        eff_df = self.results_mgr.to_dataframe("efficiency/")
        if not eff_df.empty:
            self._plot_efficiency_curves(eff_df, plots_dir)

        print(f"  Plots saved to {plots_dir}")

    def _plot_model_comparison(self, df, plots_dir):
        import matplotlib.pyplot as plt

        # Best prompt strategy per model (zero-shot)
        zs = df[(df.get("scenario", "") == "zero_shot") | (df.get("scenario") is None)]
        if zs.empty:
            zs = df
        grouped = zs.groupby("model")["f1_macro"].max().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.Set2(range(len(grouped)))
        bars = ax.bar(range(len(grouped)), grouped.values, color=colors)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right")
        ax.set_ylabel("F1-Macro")
        ax.set_title("Model Comparison (Best Prompt Strategy)")
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / "model_comparison_f1.pdf")
        plt.savefig(plots_dir / "model_comparison_f1.png")
        plt.close()

    def _plot_strategy_heatmap(self, df, plots_dir):
        import matplotlib.pyplot as plt
        import seaborn as sns

        pivot = df.pivot_table(values="f1_macro", index="model", columns="strategy", aggfunc="mean")
        if pivot.empty:
            return

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, vmin=0.5, vmax=1.0)
        ax.set_title("F1-Macro by Model and Prompt Strategy")
        plt.tight_layout()
        plt.savefig(plots_dir / "prompt_strategy_heatmap.pdf")
        plt.savefig(plots_dir / "prompt_strategy_heatmap.png")
        plt.close()

    def _plot_language_comparison(self, df, plots_dir):
        import matplotlib.pyplot as plt

        lang_pivot = df.pivot_table(values="f1_macro", index="model", columns="lang", aggfunc="mean")
        if lang_pivot.empty or lang_pivot.shape[1] < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(lang_pivot))
        w = 0.35
        ax.bar([i - w / 2 for i in x], lang_pivot.get("en", []), w, label="English", color="#4CAF50")
        ax.bar([i + w / 2 for i in x], lang_pivot.get("es", []), w, label="Spanish", color="#2196F3")
        ax.set_xticks(list(x))
        ax.set_xticklabels(lang_pivot.index, rotation=45, ha="right")
        ax.set_ylabel("F1-Macro")
        ax.set_title("Performance by Language")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "language_comparison.pdf")
        plt.savefig(plots_dir / "language_comparison.png")
        plt.close()

    def _plot_efficiency_curves(self, df, plots_dir):
        import matplotlib.pyplot as plt

        # This is a placeholder — real data would come from efficiency results
        print("  Efficiency plots: will be generated from efficiency experiment results")
