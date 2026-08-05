"""Stage 02 — baseline inference.

Run the base model on the 692-sample EN+ES test set with the structured
prompt. Extract prediction + per-instance confidence (exp of the first-answer-
token logprob). Reuse agreements + sigmoid_scores from the gpt-4o reference
(per RULES invariants I3, I4).

Writes: results/local_pipeline/per_instance/<shortname>_base_local.json
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import (
    load_config, config_hash, save_per_instance, write_checkpoint,
)
from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.pipeline.prompts import PromptBuilder
from ra_dpo.utils.metrics import compute_metrics

REF_PATH = ROOT / "results" / "final_reliability_3factor" / "gpt-4o_base.json"


def load_test():
    cfg = load_config()
    loader = EXISTDataLoader(str(ROOT / cfg["data"]["training_json"]))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    _, _, test_df = loader.create_train_val_test_split(df)
    return test_df.reset_index(drop=True)


def yes_no_token_ids(tokenizer):
    """Find the first-token id for ' YES' and ' NO' answer tokens."""
    yes_ids, no_ids = [], []
    for t in [" YES", "YES", " Yes", "Yes", " yes", "yes"]:
        toks = tokenizer(t, add_special_tokens=False).input_ids
        if toks:
            yes_ids.append(toks[0])
    for t in [" NO", "NO", " No", "No", " no", "no"]:
        toks = tokenizer(t, add_special_tokens=False).input_ids
        if toks:
            no_ids.append(toks[0])
    return list(dict.fromkeys(yes_ids)), list(dict.fromkeys(no_ids))


def main():
    cfg = load_config()
    model_id = cfg["model"]["id"]
    device = cfg["hardware"]["device"]

    print(f"Loading {model_id} on {device}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(device).eval()

    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes_ids={yes_ids[:3]}... no_ids={no_ids[:3]}...")

    test_df = load_test()
    assert len(test_df) == 692, f"test size {len(test_df)}"

    ref = json.load(open(REF_PATH))
    agreements = ref["per_instance"]["agreements"]
    sigmoid_scores = ref["per_instance"]["sigmoid_scores"]

    pb = PromptBuilder()
    predictions, confidences = [], []

    t0 = time.time()
    for i, row in tqdm(test_df.iterrows(), total=len(test_df),
                        desc=f"base/{cfg['model']['shortname']}"):
        lang = row["lang"]
        system = pb.get_system_prompt(cfg["prompt"]["strategy"], lang)
        user = pb.format_user_prompt(row["tweet"], lang, cfg["prompt"]["strategy"])

        # Chat template (Llama-3 / Mistral handle chat messages)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device) if "attention_mask" in enc else None

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1]  # last-token distribution
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = float(sum(probs[i].item() for i in yes_ids))
        p_no = float(sum(probs[i].item() for i in no_ids))
        total = p_yes + p_no
        if total <= 0:
            pred = "NO"
            conf = 0.5
        else:
            p_yes_n = p_yes / total
            p_no_n = p_no / total
            pred = "YES" if p_yes_n > p_no_n else "NO"
            conf = max(p_yes_n, p_no_n)
        predictions.append(pred)
        confidences.append(float(conf))

    print(f"inference done in {time.time() - t0:.1f}s")

    true_labels = test_df["majority_label"].tolist()
    correct = [p == t for p, t in zip(predictions, true_labels)]
    sm = compute_metrics(true_labels, predictions)
    sm["avg_confidence"] = float(np.mean(confidences))

    payload = {
        "model": f"{cfg['model']['shortname']} (base)",
        "model_id": model_id,
        "training_pairs": None,
        "standard_metrics": sm,
        "per_instance": {
            "predictions": predictions,
            "confidences": confidences,
            "agreements": agreements,
            "sigmoid_scores": sigmoid_scores,
            "correct": [bool(c) for c in correct],
        },
        "n_samples": 692,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_strategy": cfg["prompt"]["strategy"],
        "config_hash": config_hash(),
    }
    out_path = save_per_instance("base", payload)
    print(f"wrote {out_path}")
    print(f"F1={sm['f1_macro']:.4f}  Acc={sm['accuracy']:.4f}")

    write_checkpoint("stage_02_baseline", "ok",
                     {"f1": sm["f1_macro"], "accuracy": sm["accuracy"]})


if __name__ == "__main__":
    main()
