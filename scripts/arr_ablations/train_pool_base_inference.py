"""Base-model predictions over the 5,536-pair TRAIN pool (per backbone).

The revised training-side selection ranks pairs by student need: the base
model's own prediction and confidence on each training item. This runs the
same forward-pass inference as stage_02_baseline_inference (structured
prompt, chat template, p_YES/p_NO from the last-token distribution) over the
train split instead of the test split.

Backbone comes from LOCAL_PIPELINE_CONFIG, exactly like the pipeline stages.

Writes: results/local_pipeline/train_pool_base/<shortname>_train_base.json
    { meta: {...}, items: {id: {pred, conf, p_yes}} }

    LOCAL_PIPELINE_CONFIG=configs/local_pipeline.yaml \
        venv/bin/python scripts/arr_ablations/train_pool_base_inference.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.local_pipeline import load_config, config_hash  # noqa: E402
from scripts.local_pipeline.stage_02_baseline_inference import yes_no_token_ids  # noqa: E402
from src.pipeline.prompts import PromptBuilder  # noqa: E402
from build_component_ablation_subsets import load_train_df  # noqa: E402

OUT_DIR = ROOT / "results" / "local_pipeline" / "train_pool_base"


def main():
    cfg = load_config()
    model_id, device = cfg["model"]["id"], cfg["hardware"]["device"]
    shortname = cfg["model"]["shortname"]
    out_path = OUT_DIR / f"{shortname}_train_base.json"
    if out_path.exists():
        print(f"SKIP: {out_path.relative_to(ROOT)} exists")
        return

    df = load_train_df()
    assert len(df) == 5536, f"train size {len(df)}"

    print(f"Loading {model_id} on {device}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(
        cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    yes_ids, no_ids = yes_no_token_ids(tok)

    pb = PromptBuilder()
    items = {}
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"train-pool/{shortname}"):
        lang = row["lang"]
        messages = [
            {"role": "system",
             "content": pb.get_system_prompt(cfg["prompt"]["strategy"], lang)},
            {"role": "user",
             "content": pb.format_user_prompt(row["tweet"], lang,
                                              cfg["prompt"]["strategy"])},
        ]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device) if "attention_mask" in enc else None
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=att)
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        p_yes = float(sum(probs[i].item() for i in yes_ids))
        p_no = float(sum(probs[i].item() for i in no_ids))
        total = p_yes + p_no
        if total <= 0:
            pred, conf, p_yes_n = "NO", 0.5, 0.5
        else:
            p_yes_n = p_yes / total
            pred = "YES" if p_yes_n > 0.5 else "NO"
            conf = max(p_yes_n, 1 - p_yes_n)
        items[row["id"]] = {"pred": pred, "conf": round(conf, 6),
                            "p_yes": round(p_yes_n, 6)}

    wrong = sum(items[i]["pred"] != m
                for i, m in zip(df["id"], df["majority_label"]))
    print(f"done in {time.time() - t0:.0f}s   base wrong on train: "
          f"{wrong}/{len(df)} ({wrong / len(df):.1%})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"model_id": model_id, "shortname": shortname,
                        "prompt_strategy": cfg["prompt"]["strategy"],
                        "n": len(items), "n_wrong": int(wrong),
                        "config_hash": config_hash(),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
               "items": items}, open(out_path, "w"))
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
