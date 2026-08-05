"""
Data efficiency experiment for DPO training.

Trains on 10%, 25%, 50%, 75%, 100% of data with two sampling strategies:
- random: uniform random subset
- smart: prioritize high-disagreement instances (where DPO correction is most valuable)

Goal: show confidence-weighted DPO achieves similar/better results with less data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import ExperimentConfig
from .results_manager import ResultsManager
from .training_runner import _checkpoint_exists


class EfficiencyExperiment:
    """Measures DPO data efficiency with random vs smart sampling."""

    def __init__(self, config: ExperimentConfig, results_mgr: ResultsManager):
        self.config = config
        self.results_mgr = results_mgr

    def run(
        self,
        model_key: str,
        model_cfg: Dict[str, Any],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Run efficiency experiment for one model."""
        hf_id = model_cfg["hf_id"]

        for fraction in self.config.efficiency_fractions:
            for sampling in self.config.efficiency_sampling:
                pct = int(fraction * 100)
                experiment_key = f"efficiency/{model_key}/{pct}pct/{sampling}"

                if self.config.resume and self.results_mgr.exists(experiment_key):
                    print(f"    [SKIP] {experiment_key}")
                    continue

                print(f"    [{pct}% / {sampling}] Training...")

                subset_df = self._create_subset(train_df, fraction, sampling)
                base_dir = Path(self.config.output_dir) / "efficiency" / model_key / f"{pct}pct" / sampling

                # SFT on subset (shared between DPO variants)
                sft_dir = str(base_dir / "sft")
                if not _checkpoint_exists(sft_dir):
                    from ra_dpo.models.sft_trainer import train_sft
                    train_sft(
                        train_df=subset_df, val_df=val_df,
                        model_name=hf_id, output_dir=sft_dir,
                    )

                # Standard DPO on subset
                dpo_dir = str(base_dir / "dpo")
                dpo_metrics = self._train_and_get_metrics(
                    "dpo", hf_id, sft_dir, subset_df, val_df, test_df, dpo_dir
                )

                # Confidence DPO on subset
                conf_dir = str(base_dir / "confidence_dpo")
                conf_metrics = self._train_and_get_metrics(
                    "confidence_dpo", hf_id, sft_dir, subset_df, val_df, test_df, conf_dir
                )

                result = {
                    "fraction": fraction,
                    "sampling": sampling,
                    "n_train_samples": len(subset_df),
                    "n_total_train": len(train_df),
                    "dpo": dpo_metrics,
                    "confidence_dpo": conf_metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                self.results_mgr.save(experiment_key, result)

    def _create_subset(
        self, train_df: pd.DataFrame, fraction: float, sampling: str
    ) -> pd.DataFrame:
        """Create a training subset using the specified sampling strategy."""
        n = max(int(len(train_df) * fraction), 10)

        if sampling == "random":
            return train_df.sample(n=min(n, len(train_df)), random_state=self.config.seed)

        elif sampling == "smart":
            from ra_dpo.data.data_loader import agreement_score

            df = train_df.copy()
            if "agreement_score" not in df.columns:
                df["agreement_score"] = df["labels_task1"].apply(agreement_score)

            # 60% hard cases (low agreement), 40% easy cases (high agreement)
            hard_threshold = 5 / 6  # Less than unanimous
            hard_pool = df[df["agreement_score"] < hard_threshold]
            easy_pool = df[df["agreement_score"] >= hard_threshold]

            n_hard = min(int(n * 0.6), len(hard_pool))
            n_easy = min(n - n_hard, len(easy_pool))

            # If not enough hard cases, fill with easy
            if n_hard + n_easy < n:
                n_easy = min(n - n_hard, len(easy_pool))
                if n_hard + n_easy < n:
                    n_hard = min(n - n_easy, len(hard_pool))

            hard_sample = hard_pool.sample(n=n_hard, random_state=self.config.seed)
            easy_sample = easy_pool.sample(n=n_easy, random_state=self.config.seed)
            return pd.concat([hard_sample, easy_sample])

        raise ValueError(f"Unknown sampling strategy: {sampling}")

    def _train_and_get_metrics(
        self,
        method: str,
        hf_id: str,
        sft_dir: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Train a DPO variant and return metrics."""
        sft_path = sft_dir if _checkpoint_exists(sft_dir) else None

        if not _checkpoint_exists(output_dir):
            if method == "dpo":
                from ra_dpo.models.dpo_trainer import train_dpo
                train_dpo(
                    train_df=train_df, val_df=val_df,
                    sft_model_path=sft_path,
                    model_name=hf_id, output_dir=output_dir,
                )
            elif method == "confidence_dpo":
                from ra_dpo.models.confidence_dpo import train_confidence_dpo
                train_confidence_dpo(
                    train_df=train_df, val_df=val_df,
                    sft_model_path=sft_path,
                    model_name=hf_id, output_dir=output_dir,
                )

        return {"output_dir": output_dir, "method": method}
