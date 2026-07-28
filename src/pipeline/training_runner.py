"""
Training runner for SFT and DPO variants.

Orchestrates: SFT → DPO → Confidence-DPO → XAI-DPO for trainable models.
Reuses existing src/models/ trainers.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import ExperimentConfig
from .results_manager import ResultsManager


def _checkpoint_exists(output_dir: str) -> bool:
    """Check if a valid model checkpoint exists."""
    p = Path(output_dir)
    if not p.exists():
        return False
    sentinel_files = ["config.json", "adapter_config.json", "model.safetensors", "pytorch_model.bin"]
    return any((p / f).exists() for f in sentinel_files)


class TrainingRunner:
    """Orchestrates SFT and DPO training stages for local models."""

    def __init__(self, config: ExperimentConfig, results_mgr: ResultsManager):
        self.config = config
        self.results_mgr = results_mgr

    def run_training_for_model(
        self,
        model_key: str,
        model_cfg: Dict[str, Any],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Run full training pipeline for one model."""
        hf_id = model_cfg["hf_id"]
        base_output = Path(self.config.output_dir) / "training" / model_key

        # Step 1: SFT
        sft_dir = str(base_output / "sft")
        sft_key = f"training/{model_key}/sft"

        if not self.config.resume or not self.results_mgr.exists(sft_key):
            if not _checkpoint_exists(sft_dir):
                print(f"  [SFT] Training {model_key}...")
                from src.models.sft_trainer import train_sft
                sft_result = train_sft(
                    train_df=train_df, val_df=val_df,
                    model_name=hf_id, output_dir=sft_dir,
                )
                sft_result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(sft_key, sft_result)
            else:
                print(f"  [SFT] Checkpoint exists at {sft_dir}, skipping training")
                self.results_mgr.save(sft_key, {"skipped": True, "output_dir": sft_dir})
        else:
            print(f"  [SFT] Already recorded, skipping")

        # Step 2: Standard DPO
        dpo_dir = str(base_output / "dpo")
        dpo_key = f"training/{model_key}/dpo"

        if not self.config.resume or not self.results_mgr.exists(dpo_key):
            if not _checkpoint_exists(dpo_dir):
                print(f"  [DPO] Training {model_key}...")
                from src.models.dpo_trainer import train_dpo
                sft_path = sft_dir if _checkpoint_exists(sft_dir) else None
                dpo_result = train_dpo(
                    train_df=train_df, val_df=val_df,
                    sft_model_path=sft_path,
                    model_name=hf_id, output_dir=dpo_dir,
                )
                dpo_result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(dpo_key, dpo_result)
            else:
                print(f"  [DPO] Checkpoint exists, skipping")
                self.results_mgr.save(dpo_key, {"skipped": True, "output_dir": dpo_dir})
        else:
            print(f"  [DPO] Already recorded, skipping")

        # Step 3: Confidence-Weighted DPO
        conf_dir = str(base_output / "confidence_dpo")
        conf_key = f"training/{model_key}/confidence_dpo"

        if not self.config.resume or not self.results_mgr.exists(conf_key):
            if not _checkpoint_exists(conf_dir):
                print(f"  [Conf-DPO] Training {model_key}...")
                from src.models.confidence_dpo import train_confidence_dpo
                sft_path = sft_dir if _checkpoint_exists(sft_dir) else None
                conf_result = train_confidence_dpo(
                    train_df=train_df, val_df=val_df,
                    sft_model_path=sft_path,
                    model_name=hf_id, output_dir=conf_dir,
                )
                conf_result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(conf_key, conf_result)
            else:
                print(f"  [Conf-DPO] Checkpoint exists, skipping")
                self.results_mgr.save(conf_key, {"skipped": True, "output_dir": conf_dir})
        else:
            print(f"  [Conf-DPO] Already recorded, skipping")

        # Step 4: XAI-Enhanced DPO
        xai_dir = str(base_output / "xai_dpo")
        xai_key = f"training/{model_key}/xai_dpo"

        if not self.config.resume or not self.results_mgr.exists(xai_key):
            if not _checkpoint_exists(xai_dir):
                print(f"  [XAI-DPO] Training {model_key}...")
                from src.models.xai_dpo import train_xai_dpo
                sft_path = sft_dir if _checkpoint_exists(sft_dir) else None
                xai_result = train_xai_dpo(
                    train_df=train_df, val_df=val_df,
                    sft_model_path=sft_path,
                    model_name=hf_id, output_dir=xai_dir,
                )
                xai_result["timestamp"] = datetime.now().isoformat()
                self.results_mgr.save(xai_key, xai_result)
            else:
                print(f"  [XAI-DPO] Checkpoint exists, skipping")
                self.results_mgr.save(xai_key, {"skipped": True, "output_dir": xai_dir})
        else:
            print(f"  [XAI-DPO] Already recorded, skipping")

        print(f"  Training complete for {model_key}")
