"""Zero-training baseline: tune the base model's decision threshold on val.

The student-need arms showed the low-budget selection gains are class
recalibration. This measures how much of that is available with NO training:
run the base model on the 692-row val split, pick the macro-F1-maximizing
threshold there, apply it to the test split's stored base predictions.

Backbone from LOCAL_PIPELINE_CONFIG, like the pipeline stages.

    LOCAL_PIPELINE_CONFIG=configs/local_pipeline.yaml \
        venv/bin/python scripts/arr_ablations/val_threshold_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import load_config  # noqa: E402
from scripts.local_pipeline.stage_02_baseline_inference import yes_no_token_ids  # noqa: E402
from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score  # noqa: E402
from src.pipeline.prompts import PromptBuilder  # noqa: E402

OUT = ROOT / "results" / "local_pipeline" / "train_pool_base"


def load_val():
    cfg = load_config()
    loader = EXISTDataLoader(str(ROOT / cfg["data"]["training_json"]))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    _, val_df, _ = loader.create_train_val_test_split(df)
    return val_df.reset_index(drop=True)


def sweep(gold, p_yes):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 193):
        f = f1_score(gold, np.where(p_yes >= t, "YES", "NO"), average="macro")
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def main():
    cfg = load_config()
    shortname = cfg["model"]["shortname"]
    out_path = OUT / f"{shortname}_val_threshold.json"
    if out_path.exists():
        print(f"SKIP: {out_path.relative_to(ROOT)} exists")
        print(json.dumps(json.load(open(out_path)), indent=2))
        return

    val = load_val()
    assert len(val) == 692, f"val size {len(val)}"

    print(f"Loading {cfg['model']['id']} on {cfg['hardware']['device']}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(
        cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(cfg["hardware"]["device"]).eval()
    yes_ids, no_ids = yes_no_token_ids(tok)

    pb = PromptBuilder()
    p_yes_val = []
    t0 = time.time()
    for _, row in tqdm(val.iterrows(), total=len(val), desc=f"val/{shortname}"):
        messages = [
            {"role": "system",
             "content": pb.get_system_prompt(cfg["prompt"]["strategy"], row["lang"])},
            {"role": "user",
             "content": pb.format_user_prompt(row["tweet"], row["lang"],
                                              cfg["prompt"]["strategy"])},
        ]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(model.device))
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        py = float(sum(probs[i].item() for i in yes_ids))
        pn = float(sum(probs[i].item() for i in no_ids))
        p_yes_val.append(py / (py + pn) if py + pn > 0 else 0.5)
    print(f"val inference {time.time() - t0:.0f}s")

    gold_val = val["majority_label"].to_numpy()
    t_star, f1_val = sweep(gold_val, np.array(p_yes_val))

    # apply to the stored test-split base predictions
    d = json.load(open(ROOT / "results" / "local_pipeline" / "per_instance" /
                       f"{shortname}_base_local.json"))
    pi = d["per_instance"]
    pred, corr = pi["predictions"], pi["correct"]
    gold_test = np.array([p if c else ("NO" if p == "YES" else "YES")
                          for p, c in zip(pred, corr)])
    conf = np.array(pi["confidences"])
    p_yes_test = np.where(np.array(pred) == "YES", conf, 1 - conf)
    f1_base = f1_score(gold_test, pred, average="macro")
    f1_tuned = f1_score(gold_test,
                        np.where(p_yes_test >= t_star, "YES", "NO"),
                        average="macro")

    res = {"shortname": shortname, "t_star_val": round(t_star, 4),
           "f1_val_at_t": round(f1_val, 4),
           "f1_test_base": round(float(f1_base), 4),
           "f1_test_val_tuned": round(float(f1_tuned), 4)}
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
