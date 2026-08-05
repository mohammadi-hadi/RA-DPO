"""Train multiple agreement regressor backbones on the same train/val/test split
and compare metrics. Saves per-model predictions, targets, metrics, configs, and
checkpoints under results/agreement_predictor_comparison/ so the paper can later
be updated with the best one without re-running anything.

Does NOT overwrite the canonical pred_agreement.npy used by the current paper.

Usage:
    # Train all models in the default registry
    venv/bin/python scripts/analysis/compare_agreement_predictors.py

    # Train a single model
    venv/bin/python scripts/analysis/compare_agreement_predictors.py --only xlmr-base

    # Smoke test (1 epoch, mBERT)
    venv/bin/python scripts/analysis/compare_agreement_predictors.py --only mbert --epochs 1 --smoke

    # Add a custom model
    venv/bin/python scripts/analysis/compare_agreement_predictors.py --hf microsoft/mdeberta-v3-base --shortname mdeberta-v3-base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.models.agreement_predictor import AgreementPredictor


# Registered models. shortname -> (hf_id, default_batch_size, default_lr)
REGISTRY = {
    "mbert":            ("bert-base-multilingual-cased", 32, 3e-5),
    "xlmr-base":        ("xlm-roberta-base",             32, 3e-5),
    "mdeberta-v3-base": ("microsoft/mdeberta-v3-base",   32, 3e-5),
    "xlmr-large":       ("xlm-roberta-large",            16, 2e-5),
}

# Default order — small -> large so partial completions still produce useful data.
DEFAULT_ORDER = ["mbert", "xlmr-base", "mdeberta-v3-base", "xlmr-large"]

OUT_ROOT = ROOT / "results" / "agreement_predictor_comparison"
SEED = 42


def device_info() -> dict:
    """Report device, RAM, and PyTorch flags."""
    info = {
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if info["mps_available"]:
        info["device"] = "mps"
    elif info["cuda_available"]:
        info["device"] = "cuda"
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        info["device"] = "cpu"
    return info


def set_seeds(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_splits():
    """Reproduce the canonical train/val/test split used by the baseline."""
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, val_df, test_df = loader.create_train_val_test_split(df)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def save_split_index(split_df, path: Path, name: str):
    """Save instance order so future re-evaluation can verify alignment."""
    rows = []
    for i, row in split_df.iterrows():
        rows.append({
            "row_index": int(i),
            "id_or_idx": int(row.get("id", row.get("id_EXIST", i))) if isinstance(
                row.get("id", row.get("id_EXIST", i)), (int, np.integer)
            ) else str(row.get("id", row.get("id_EXIST", i))),
            "agreement_true": float(row["agreement_score"]),
            "tweet_first40": str(row["tweet"])[:40].replace("\n", " "),
        })
    with open(path, "w") as f:
        json.dump({"split": name, "n": len(rows), "rows": rows}, f, indent=2)


def train_one(shortname: str, hf_id: str, epochs: int, max_length: int,
              splits: tuple, smoke: bool = False,
              best_metric: str = "mse") -> dict:
    """Train one regressor, save predictions, metrics, checkpoint."""
    train_df, val_df, test_df = splits
    out_dir = OUT_ROOT / shortname
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)

    batch_size = REGISTRY.get(shortname, (hf_id, 32, 3e-5))[1]
    learning_rate = REGISTRY.get(shortname, (hf_id, 32, 3e-5))[2]

    config = {
        "shortname": shortname,
        "hf_id": hf_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "best_metric": best_metric,
        "seed": SEED,
        "smoke_test": smoke,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "device_info": device_info(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"[{shortname}] {hf_id}")
    print(f"  epochs={epochs} batch={batch_size} lr={learning_rate}"
          f" maxlen={max_length} best_metric={best_metric}"
          f" device={config['device_info']['device']}")
    print(f"{'=' * 70}", flush=True)

    set_seeds(SEED)
    t0 = time.time()
    predictor = AgreementPredictor(model_name=hf_id, max_length=max_length)
    train_result = predictor.train(
        train_df=train_df, val_df=val_df,
        output_dir=str(model_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=2 if not smoke else 0,
        metric_for_best_model=best_metric,
    )
    train_time = time.time() - t0
    print(f"[{shortname}] training time: {train_time / 60:.1f} min", flush=True)

    # Evaluate on test and val
    test_metrics = predictor.evaluate(test_df)
    val_metrics  = predictor.evaluate(val_df)

    preds_test = predictor.predict(test_df["tweet"].tolist())
    preds_val  = predictor.predict(val_df["tweet"].tolist())
    true_test  = test_df["agreement_score"].to_numpy()
    true_val   = val_df["agreement_score"].to_numpy()

    np.save(out_dir / "predictions_test.npy", preds_test)
    np.save(out_dir / "predictions_val.npy",  preds_val)
    np.save(out_dir / "targets_test.npy",     true_test)
    np.save(out_dir / "targets_val.npy",      true_val)

    save_split_index(test_df, out_dir / "test_order.json", "test")
    save_split_index(val_df,  out_dir / "val_order.json",  "val")

    metrics = {
        "shortname": shortname,
        "hf_id": hf_id,
        "test": test_metrics,
        "val": val_metrics,
        "train_loss_final": float(train_result.get("train_loss", -1)),
        "train_time_seconds": train_time,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{shortname}] TEST: r={test_metrics['pearson_r']:.4f} "
          f"MAE={test_metrics['mae']:.4f} "
          f"Spearman={test_metrics['spearman_rho']:.4f}", flush=True)

    # Free GPU memory before next model
    del predictor
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc
    gc.collect()

    return metrics


def write_comparison(all_metrics: list[dict]):
    """Write comparison.csv, comparison.md, manifest.json."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "device_info": device_info(),
        "models": [m["shortname"] for m in all_metrics],
        "baseline_reference": "results/unified_gpt4o/predicted_agreement/pred_agreement.npy",
    }
    with open(OUT_ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # CSV
    import csv
    fields = ["shortname", "hf_id",
              "test_pearson_r", "test_spearman_rho", "test_mae", "test_mse",
              "val_pearson_r", "val_spearman_rho", "val_mae", "val_mse",
              "train_time_min"]
    rows = []
    for m in all_metrics:
        rows.append({
            "shortname": m["shortname"],
            "hf_id": m["hf_id"],
            "test_pearson_r":   round(m["test"]["pearson_r"],   4),
            "test_spearman_rho":round(m["test"]["spearman_rho"],4),
            "test_mae":         round(m["test"]["mae"],         4),
            "test_mse":         round(m["test"]["mse"],         4),
            "val_pearson_r":    round(m["val"]["pearson_r"],    4),
            "val_spearman_rho": round(m["val"]["spearman_rho"], 4),
            "val_mae":          round(m["val"]["mae"],          4),
            "val_mse":          round(m["val"]["mse"],          4),
            "train_time_min":   round(m["train_time_seconds"] / 60, 1),
        })
    with open(OUT_ROOT / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    lines = [
        "# Agreement regressor comparison",
        "",
        f"Generated {manifest['generated_at']} on device "
        f"`{manifest['device_info']['device']}` "
        f"(PyTorch {manifest['device_info']['torch_version']}).",
        "",
        "**Current paper baseline (mBERT, 3 epochs):** "
        "Pearson r ≈ 0.27, MAE ≈ 0.138, Spearman ≈ 0.27 — see "
        "`results/unified_gpt4o/predicted_agreement/pred_agreement.npy`.",
        "",
        "## Test set (n=692)",
        "",
        "| shortname | HF id | Pearson r | Spearman ρ | MAE | MSE | Train (min) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    # Sort best-first by test Pearson r
    rows_sorted = sorted(rows, key=lambda r: -r["test_pearson_r"])
    for r in rows_sorted:
        lines.append(
            f"| {r['shortname']} | `{r['hf_id']}` | "
            f"{r['test_pearson_r']:.4f} | {r['test_spearman_rho']:.4f} | "
            f"{r['test_mae']:.4f} | {r['test_mse']:.4f} | "
            f"{r['train_time_min']:.1f} |"
        )

    lines += [
        "",
        "## Validation set (n=692)",
        "",
        "| shortname | Pearson r | Spearman ρ | MAE | MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['shortname']} | "
            f"{r['val_pearson_r']:.4f} | {r['val_spearman_rho']:.4f} | "
            f"{r['val_mae']:.4f} | {r['val_mse']:.4f} |"
        )

    lines += [
        "",
        "## How to use these outputs",
        "",
        "Each model directory under `results/agreement_predictor_comparison/`",
        "contains:",
        "",
        "- `config.json`     — hyperparameters + device info",
        "- `metrics.json`    — full metrics (val + test)",
        "- `predictions_test.npy` (692 floats, same order as the baseline)",
        "- `predictions_val.npy`  (692 floats)",
        "- `targets_test.npy`     (true agreement scores)",
        "- `targets_val.npy`",
        "- `test_order.json` / `val_order.json` — instance IDs + first 40 chars",
        "  of each tweet for ordering verification",
        "- `model/`        — saved checkpoint, reloadable via "
        "`AgreementPredictor.load(path)`",
        "",
        "**To swap a new regressor into the paper later:**",
        "",
        "1. Copy `predictions_test.npy` over `results/unified_gpt4o/"
        "predicted_agreement/pred_agreement.npy`.",
        "2. Re-run `scripts/analysis/predict_agreement_and_reeval.py` to "
        "refit OOF α/β/γ and recompute coverage-accuracy.",
        "3. Regenerate `tables/tab_coverage.tex` from the new numbers.",
        "4. Update body prose in the paper source",
        "   to reflect the new r / MAE / +pp numbers.",
        "",
    ]
    with open(OUT_ROOT / "comparison.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote: {OUT_ROOT / 'comparison.md'}")
    print(f"       {OUT_ROOT / 'comparison.csv'}")
    print(f"       {OUT_ROOT / 'manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(REGISTRY.keys()),
                    help="Run only this registered model")
    ap.add_argument("--hf", help="Custom HuggingFace id (must also pass --shortname)")
    ap.add_argument("--shortname", help="Output dir name for --hf model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: implies 1 epoch, no early stopping")
    ap.add_argument("--order", nargs="+", default=DEFAULT_ORDER,
                    help="Order of models to train (registry shortnames)")
    ap.add_argument("--best-metric", default="mse",
                    choices=["mse", "mae", "pearson_r", "spearman_rho"],
                    help="Validation metric for early stopping and best-checkpoint selection")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT = {OUT_ROOT}")
    print(f"device   = {device_info()}")

    splits = load_splits()
    print(f"splits   = train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")

    all_metrics = []
    if args.hf and args.shortname:
        m = train_one(args.shortname, args.hf, args.epochs,
                      args.max_length, splits, smoke=args.smoke,
                      best_metric=args.best_metric)
        all_metrics.append(m)
    elif args.only:
        hf_id = REGISTRY[args.only][0]
        m = train_one(args.only, hf_id, args.epochs,
                      args.max_length, splits, smoke=args.smoke,
                      best_metric=args.best_metric)
        all_metrics.append(m)
    else:
        for shortname in args.order:
            if shortname not in REGISTRY:
                print(f"WARN: {shortname} not in REGISTRY, skipping")
                continue
            hf_id = REGISTRY[shortname][0]
            try:
                m = train_one(shortname, hf_id, args.epochs,
                              args.max_length, splits, smoke=args.smoke,
                              best_metric=args.best_metric)
                all_metrics.append(m)
                # Write incremental comparison after each model so partial
                # results are useful if a later one fails or is interrupted.
                write_comparison(all_metrics)
            except Exception as e:
                print(f"FAIL [{shortname}]: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

    if all_metrics:
        write_comparison(all_metrics)


if __name__ == "__main__":
    main()
