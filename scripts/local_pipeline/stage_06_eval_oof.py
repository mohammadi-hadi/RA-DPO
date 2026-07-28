"""Stage 06 — evaluate every fine-tuned variant on the test set and fit OOF R(x).

For each variant with an adapter under results/local_pipeline/training/<v>/:
  1. Load base + LoRA adapter in fp16 on MPS.
  2. Run inference on 692 EN+ES test posts with structured prompt.
  3. Save per_instance JSON with identical schema to the gpt-4o pipeline.
  4. Fit 5-fold OOF α/β/γ (per-model) and compute coverage-accuracy.

Usage:
    python scripts/local_pipeline/stage_06_eval_oof.py --variants base sft std_dpo smart30_dpo ra_dpo
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import (
    load_config, config_hash, per_instance_path, per_instance_dir,
    training_dir, save_per_instance, write_checkpoint, shortname,
)
from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from src.pipeline.prompts import PromptBuilder
from src.utils.metrics import compute_metrics

REF_PATH = ROOT / "results" / "final_reliability_3factor" / "gpt-4o_base.json"
COV_LEVELS = [1.00, 0.90, 0.80, 0.60, 0.50]
VARIANT_PAIRS = {
    "base": None, "sft": 5536, "std_dpo": 5536, "smart10_dpo": 554,
    "smart30_dpo": 1661, "smart50_dpo": 2768, "random50_dpo": 2768,
    "random30_dpo": 1661,
    "flip10_dpo": 554, "randunan10_dpo": 554, "random10_dpo": 554,
    "strat10_dpo": 554, "noworst10_dpo": 554,
    "noisy1_random10_dpo": 554, "noisy1_strat10_dpo": 554,
    "hard10_llama_dpo": 554, "hardctl10_llama_dpo": 554, "hardall10_llama_dpo": 554,
    "hard10_qwen_dpo": 554, "hardctl10_qwen_dpo": 554, "hardall10_qwen_dpo": 554,
    "ambiguous_dpo": 665, "ra_dpo": 8984,
    "agree30_dpo": 1661, "agree30_tb2_dpo": 1661, "uncert30_dpo": 1661,
    "conf30_dpo": 1661, "wsft": 5536, "softlabel_sft": 5536,
}


def oof_rx(conf, agree, sig, correct, seed=42):
    X = np.column_stack([conf, agree, 1 - sig])
    r = np.zeros(len(correct))
    ws = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, correct):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), correct[tr])
        a = np.abs(lr.coef_[0]); a = a / a.sum()
        ws.append(a)
        r[te] = X[te] @ a
    return r, np.mean(ws, axis=0)


def acc_top(r, correct, cov):
    k = max(1, int(round(len(r) * cov)))
    return float(correct[np.argsort(-r)[:k]].mean())


def load_model(variant: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    cfg = load_config()
    model_id = cfg["model"]["id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to("mps")
    if variant != "base":
        adapter = training_dir() / variant
        model = PeftModel.from_pretrained(model, str(adapter))
        model.to("mps")
    model.eval()
    return tok, model


def yes_no_ids(tok):
    yes_ids, no_ids = [], []
    for t in [" YES", "YES", " Yes", "Yes", " yes", "yes"]:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids: yes_ids.append(ids[0])
    for t in [" NO", "NO", " No", "No", " no", "no"]:
        ids = tok(t, add_special_tokens=False).input_ids
        if ids: no_ids.append(ids[0])
    return list(dict.fromkeys(yes_ids)), list(dict.fromkeys(no_ids))


def evaluate_variant(variant: str):
    cfg = load_config()
    out_path = per_instance_path(variant)
    if out_path.exists():
        print(f"[{variant}] already evaluated → {out_path.name}")
        return

    tok, model = load_model(variant)
    yes_ids, no_ids = yes_no_ids(tok)

    loader = EXISTDataLoader(str(ROOT / cfg["data"]["training_json"]))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    _, _, test_df = loader.create_train_val_test_split(df)
    test_df = test_df.reset_index(drop=True)

    ref = json.load(open(REF_PATH))
    agreements = ref["per_instance"]["agreements"]
    sigmoid_scores = ref["per_instance"]["sigmoid_scores"]

    pb = PromptBuilder()
    preds, confs = [], []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{variant}"):
        lang = row["lang"]
        messages = [
            {"role": "system", "content": pb.get_system_prompt(cfg["prompt"]["strategy"], lang)},
            {"role": "user",   "content": pb.format_user_prompt(row["tweet"], lang, cfg["prompt"]["strategy"])},
        ]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("mps")
        mask = enc["attention_mask"].to("mps") if "attention_mask" in enc else None
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = float(sum(probs[i].item() for i in yes_ids))
        p_no  = float(sum(probs[i].item() for i in no_ids))
        total = p_yes + p_no
        if total <= 0:
            pred, conf = "NO", 0.5
        else:
            pyn, pnn = p_yes / total, p_no / total
            pred = "YES" if pyn > pnn else "NO"
            conf = max(pyn, pnn)
        preds.append(pred)
        confs.append(float(conf))

    true_labels = test_df["majority_label"].tolist()
    correct = np.asarray([p == t for p, t in zip(preds, true_labels)], dtype=int)
    sm = compute_metrics(true_labels, preds)
    sm["avg_confidence"] = float(np.mean(confs))

    # OOF R(x)
    conf_arr = np.asarray(confs, dtype=float)
    agree_arr = np.asarray(agreements, dtype=float)
    sig_arr = np.asarray(sigmoid_scores, dtype=float)
    r_oof, w_mean = oof_rx(conf_arr, agree_arr, sig_arr, correct)
    acc_cov = {f"acc@{int(c*100)}%": acc_top(r_oof, correct, c) for c in COV_LEVELS}

    payload = {
        "model": f"{shortname()} ({variant})",
        "model_id": cfg["model"]["id"],
        "training_pairs": VARIANT_PAIRS.get(variant),
        "standard_metrics": sm,
        "optimized_weights": {"alpha": float(w_mean[0]), "beta": float(w_mean[1]), "gamma": float(w_mean[2])},
        "accuracy_at_coverage": acc_cov,
        "per_instance": {
            "predictions": preds,
            "confidences": confs,
            "agreements": agreements,
            "sigmoid_scores": sigmoid_scores,
            "correct": [bool(c) for c in correct],
        },
        "n_samples": 692,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_strategy": cfg["prompt"]["strategy"],
        "config_hash": config_hash(),
    }
    save_per_instance(variant, payload)
    print(f"[{variant}] F1={sm['f1_macro']:.4f}  Acc={sm['accuracy']:.4f}  @50%={acc_cov['acc@50%']:.4f}")
    write_checkpoint(f"stage_06_{variant}", "ok", {"f1": sm["f1_macro"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANT_PAIRS.keys()))
    args = ap.parse_args()
    for v in args.variants:
        if v not in VARIANT_PAIRS:
            raise KeyError(f"unknown variant {v}")
        print(f"\n=== eval {v} ===")
        evaluate_variant(v)


if __name__ == "__main__":
    main()
