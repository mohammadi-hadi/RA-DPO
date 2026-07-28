#!/usr/bin/env python3
"""
Full training pipeline: SFT → DPO → Conf-DPO with evaluation at each stage.
Runs on M4 Max 64GB with MPS backend.

Usage:
    python scripts/run_full_training.py                          # Both models
    python scripts/run_full_training.py --model Mistral-7B       # Single model
    python scripts/run_full_training.py --max-steps 100          # Quick test
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from src.utils.metrics import compute_metrics


MODELS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}

RESULTS_DIR = Path("results/experiments")
MODELS_DIR = Path("results/experiments/training")


def free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save_result(key, data):
    """Save result atomically."""
    parts = key.strip("/").split("/")
    path = RESULTS_DIR / "/".join(parts[:-1]) / f"{parts[-1]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)
    print(f"    Saved: {path}")


def checkpoint_exists(output_dir):
    p = Path(output_dir)
    if not p.exists():
        return False
    return any((p / f).exists() for f in [
        "config.json", "adapter_config.json", "model.safetensors", "pytorch_model.bin"
    ])


def evaluate_trained_model(model_name, model_path, test_df, sft_path=None):
    """Load a trained model and evaluate on test data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.models.sft_trainer import _detect_device, _get_torch_dtype

    device = _detect_device()
    dtype = _get_torch_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    )
    if device == "mps":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Setup yes/no tokens
    yes_tokens = ["yes", "Yes", "YES", "SI", "Si", "si"]
    no_tokens = ["no", "No", "NO"]
    yes_ids = list(set(tid for t in yes_tokens for tid in tokenizer.encode(t, add_special_tokens=False)))
    no_ids = list(set(tid for t in no_tokens for tid in tokenizer.encode(t, add_special_tokens=False)))

    prompt_en = "Binary Sexism Detection: Classify whether this post is sexist.\n\n### Post: {text}\n### Answer (YES or NO):"
    prompt_es = "Detección Binaria de Sexismo: Clasifica si este post es sexista.\n\n### Post: {text}\n### Respuesta (SI o NO):"

    predictions, confidences = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="    Evaluating"):
        template = prompt_es if row.get("lang") == "es" else prompt_en
        prompt = template.format(text=row["tweet"])

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, output_scores=True,
                return_dict_in_generate=True, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        logits = outputs.scores[0][0]
        probs = torch.softmax(logits, dim=-1)
        yes_prob = sum(probs[tid].item() for tid in yes_ids if tid < len(probs))
        no_prob = sum(probs[tid].item() for tid in no_ids if tid < len(probs))
        total = yes_prob + no_prob
        if total > 0:
            yes_norm = yes_prob / total
        else:
            yes_norm = 0.5

        predictions.append("YES" if yes_norm > 0.5 else "NO")
        confidences.append(max(yes_norm, 1.0 - yes_norm))

    true_labels = test_df["majority_label"].tolist()
    metrics = compute_metrics(true_labels, predictions)
    metrics["avg_confidence"] = float(np.mean(confidences))

    # Clean up
    del model, base_model
    free_memory()

    return metrics, predictions, confidences


def run_pipeline_for_model(model_key, hf_id, train_df, val_df, test_df, max_steps=500):
    """Run SFT → DPO → Conf-DPO for one model."""
    base = MODELS_DIR / model_key
    results_summary = {}

    # ═══════════════════════════════════════
    # Step 1: SFT
    # ═══════════════════════════════════════
    sft_dir = str(base / "sft")
    print(f"\n{'='*60}")
    print(f"  [{model_key}] Step 1: SFT Training")
    print(f"{'='*60}")

    if not checkpoint_exists(sft_dir):
        from src.models.sft_trainer import train_sft
        t0 = time.time()
        train_sft(
            train_df=train_df, val_df=val_df,
            model_name=hf_id, output_dir=sft_dir,
            num_epochs=5, batch_size=2, max_seq_length=512,
        )
        elapsed = time.time() - t0
        print(f"    SFT completed in {elapsed/60:.1f} min")
        free_memory()
    else:
        print(f"    SFT checkpoint exists, skipping")

    # Evaluate SFT
    print(f"    Evaluating SFT model...")
    try:
        sft_metrics, _, _ = evaluate_trained_model(hf_id, sft_dir, test_df)
        results_summary["sft"] = sft_metrics
        save_result(f"training/{model_key}/sft_eval", {
            "metrics": sft_metrics, "model": hf_id, "method": "sft",
            "timestamp": datetime.now().isoformat()
        })
        print(f"    SFT → F1={sft_metrics['f1_macro']:.4f}, Acc={sft_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"    SFT eval failed: {e}")
    free_memory()

    # ═══════════════════════════════════════
    # Step 2: Standard DPO
    # ═══════════════════════════════════════
    dpo_dir = str(base / "dpo")
    print(f"\n{'='*60}")
    print(f"  [{model_key}] Step 2: Standard DPO")
    print(f"{'='*60}")

    if not checkpoint_exists(dpo_dir):
        from src.models.dpo_trainer import train_dpo
        t0 = time.time()
        train_dpo(
            train_df=train_df, val_df=val_df,
            sft_model_path=sft_dir if checkpoint_exists(sft_dir) else None,
            model_name=hf_id, output_dir=dpo_dir,
            beta=0.1, max_steps=max_steps, batch_size=2,
        )
        elapsed = time.time() - t0
        print(f"    DPO completed in {elapsed/60:.1f} min")
        free_memory()
    else:
        print(f"    DPO checkpoint exists, skipping")

    # Evaluate DPO
    print(f"    Evaluating DPO model...")
    try:
        dpo_metrics, _, _ = evaluate_trained_model(hf_id, dpo_dir, test_df)
        results_summary["dpo"] = dpo_metrics
        save_result(f"training/{model_key}/dpo_eval", {
            "metrics": dpo_metrics, "model": hf_id, "method": "dpo",
            "timestamp": datetime.now().isoformat()
        })
        print(f"    DPO → F1={dpo_metrics['f1_macro']:.4f}, Acc={dpo_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"    DPO eval failed: {e}")
    free_memory()

    # ═══════════════════════════════════════
    # Step 3: Confidence-Weighted DPO
    # ═══════════════════════════════════════
    conf_dir = str(base / "confidence_dpo")
    print(f"\n{'='*60}")
    print(f"  [{model_key}] Step 3: Confidence-Weighted DPO")
    print(f"{'='*60}")

    if not checkpoint_exists(conf_dir):
        from src.models.confidence_dpo import train_confidence_dpo
        t0 = time.time()
        train_confidence_dpo(
            train_df=train_df, val_df=val_df,
            sft_model_path=sft_dir if checkpoint_exists(sft_dir) else None,
            model_name=hf_id, output_dir=conf_dir,
            beta=0.1, max_steps=max_steps, batch_size=2,
            confidence_alpha=1.0, agreement_alpha=1.0,
        )
        elapsed = time.time() - t0
        print(f"    Conf-DPO completed in {elapsed/60:.1f} min")
        free_memory()
    else:
        print(f"    Conf-DPO checkpoint exists, skipping")

    # Evaluate Conf-DPO
    print(f"    Evaluating Conf-DPO model...")
    try:
        conf_metrics, _, _ = evaluate_trained_model(hf_id, conf_dir, test_df)
        results_summary["confidence_dpo"] = conf_metrics
        save_result(f"training/{model_key}/confidence_dpo_eval", {
            "metrics": conf_metrics, "model": hf_id, "method": "confidence_dpo",
            "timestamp": datetime.now().isoformat()
        })
        print(f"    Conf-DPO → F1={conf_metrics['f1_macro']:.4f}, Acc={conf_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"    Conf-DPO eval failed: {e}")
    free_memory()

    # Summary
    print(f"\n{'='*60}")
    print(f"  [{model_key}] Training Summary")
    print(f"{'='*60}")
    for method, metrics in results_summary.items():
        print(f"    {method:20s}: F1={metrics['f1_macro']:.4f}  Acc={metrics['accuracy']:.4f}")

    save_result(f"training/{model_key}/summary", {
        "model_key": model_key, "hf_id": hf_id,
        "results": results_summary,
        "timestamp": datetime.now().isoformat()
    })

    return results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model key (e.g., Mistral-7B)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max DPO training steps")
    args = parser.parse_args()

    # Load data
    print("[DATA] Loading EXIST 2023 dataset...")
    loader = EXISTDataLoader("EXIST2023_training.json")
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, val_df, test_df = loader.create_train_val_test_split(df)
    print(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}

    all_results = {}
    for model_key, hf_id in models_to_run.items():
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_key} ({hf_id})")
        print(f"{'#'*60}")
        results = run_pipeline_for_model(
            model_key, hf_id, train_df, val_df, test_df,
            max_steps=args.max_steps,
        )
        all_results[model_key] = results
        free_memory()

    # Final comparison
    print(f"\n{'#'*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'#'*60}")
    print(f"{'Model':20s} {'Method':20s} {'F1-Macro':>10s} {'Accuracy':>10s}")
    print("-" * 65)
    for model_key, results in all_results.items():
        for method, metrics in results.items():
            print(f"{model_key:20s} {method:20s} {metrics['f1_macro']:10.4f} {metrics['accuracy']:10.4f}")


if __name__ == "__main__":
    main()
