"""Stage 05 — DPO fine-tunes for every variant.

Runs DPO with LoRA (TRL DPOTrainer) over each of:
  std_dpo, smart10_dpo, smart30_dpo, smart50_dpo, random50_dpo,
  ambiguous_dpo, ra_dpo.

Writes adapters to results/local_pipeline/training/<variant>/.

Usage:
    python scripts/local_pipeline/stage_05_dpo_variants.py               # all variants
    python scripts/local_pipeline/stage_05_dpo_variants.py --only smart30_dpo
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import load_config, training_dir, write_checkpoint

VARIANT_FILES = {
    "std_dpo":        ROOT / "results" / "openai_dpo_train.jsonl",
    "smart10_dpo":    ROOT / "results" / "smart_sampling" / "smart10_dpo.jsonl",
    "smart30_dpo":    ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl",
    "smart50_dpo":    ROOT / "results" / "smart_sampling" / "smart50_dpo.jsonl",
    "random50_dpo":   ROOT / "results" / "smart_sampling" / "random50_dpo.jsonl",
    # Size-matched random control for smart30_dpo (built by
    # scripts/arr_ablations/build_exist_random30.py).
    "random30_dpo":   ROOT / "results" / "smart_sampling" / "random30_dpo.jsonl",
    "flip10_dpo":     ROOT / "results" / "smart_sampling" / "flip10_dpo.jsonl",
    "randunan10_dpo": ROOT / "results" / "smart_sampling" / "randunan10_dpo.jsonl",
    "random10_dpo":   ROOT / "results" / "smart_sampling" / "random10_dpo.jsonl",
    "strat10_dpo":     ROOT / "results" / "smart_sampling" / "strat10_dpo.jsonl",
    "noworst10_dpo":   ROOT / "results" / "smart_sampling" / "noworst10_dpo.jsonl",
    # Single-annotator label-noise arms (same items as random10/strat10,
    # labels drawn from one annotator; built by
    # scripts/arr_ablations/build_noisy_label_arms.py).
    "noisy1_random10_dpo": ROOT / "results" / "smart_sampling" / "noisy1_random10_dpo.jsonl",
    "noisy1_strat10_dpo":  ROOT / "results" / "smart_sampling" / "noisy1_strat10_dpo.jsonl",
    # Student-need arms (per backbone; built by
    # scripts/arr_ablations/build_student_need_arms.py).
    "hard10_llama_dpo":    ROOT / "results" / "smart_sampling" / "hard10_llama_dpo.jsonl",
    "hardctl10_llama_dpo": ROOT / "results" / "smart_sampling" / "hardctl10_llama_dpo.jsonl",
    "hardall10_llama_dpo": ROOT / "results" / "smart_sampling" / "hardall10_llama_dpo.jsonl",
    "hard10_qwen_dpo":     ROOT / "results" / "smart_sampling" / "hard10_qwen_dpo.jsonl",
    "hardctl10_qwen_dpo":  ROOT / "results" / "smart_sampling" / "hardctl10_qwen_dpo.jsonl",
    "hardall10_qwen_dpo":  ROOT / "results" / "smart_sampling" / "hardall10_qwen_dpo.jsonl",
    "ambiguous_dpo":  ROOT / "results" / "smart_sampling" / "ambiguous_only_dpo.jsonl",
    "ra_dpo":         ROOT / "results" / "openai_ra_dpo_train.jsonl",
    # Component-ablation subsets (top-30% by one R(x) component each).
    "agree30_dpo":     ROOT / "results" / "smart_sampling" / "agree30_dpo.jsonl",
    "agree30_tb2_dpo": ROOT / "results" / "smart_sampling" / "agree30_tb2_dpo.jsonl",
    "uncert30_dpo":    ROOT / "results" / "smart_sampling" / "uncert30_dpo.jsonl",
    "conf30_dpo":      ROOT / "results" / "smart_sampling" / "conf30_dpo.jsonl",
}


def load_dpo_dataset(path, tok):
    """Convert OpenAI preference-pair JSONL → HF dataset with prompt / chosen / rejected."""
    from datasets import Dataset
    rows = []
    for line in open(path):
        d = json.loads(line)
        inp = d["input"]
        msgs = inp["messages"] if isinstance(inp, dict) else inp
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        pref = d["preferred_output"]
        rej = d["non_preferred_output"]
        pref = pref[0] if isinstance(pref, list) else pref
        rej = rej[0] if isinstance(rej, list) else rej
        rows.append({
            "prompt": prompt,
            "chosen": pref["content"],
            "rejected": rej["content"],
        })
    return Dataset.from_list(rows)


def train_one(variant: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig, get_peft_model

    cfg = load_config()
    path = VARIANT_FILES[variant]
    if not path.exists():
        raise FileNotFoundError(path)

    model_id = cfg["model"]["id"]
    out_dir = training_dir() / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to("mps")
    lora = LoraConfig(
        r=cfg["training"]["lora"]["r"],
        lora_alpha=cfg["training"]["lora"]["alpha"],
        lora_dropout=cfg["training"]["lora"]["dropout"],
        target_modules=cfg["training"]["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora)
    policy.print_trainable_parameters()

    dpo = cfg["training"]["dpo"]
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dpo["epochs"],
        beta=dpo["beta"],
        learning_rate=dpo["learning_rate"],
        per_device_train_batch_size=dpo["per_device_batch_size"],
        gradient_accumulation_steps=dpo["grad_accum"],
        max_prompt_length=dpo["max_prompt_length"],
        max_length=dpo["max_length"],
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=(cfg["model"].get("dtype", "fp16") == "fp16"),
        bf16=False,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        report_to="none",
    )

    ds = load_dpo_dataset(path, tok)
    print(f"[{variant}] dataset: {len(ds)} pairs from {path.name}")

    import inspect
    trainer_kwargs = {"model": policy, "ref_model": None, "args": args,
                      "train_dataset": ds}
    if "tokenizer" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tok
    elif "processing_class" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[{variant}] adapter saved → {out_dir}")
    write_checkpoint(f"stage_05_{variant}", "ok", {"adapter": str(out_dir)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these variant names")
    args = ap.parse_args()
    variants = list(VARIANT_FILES) if args.only is None else args.only
    for v in variants:
        if v not in VARIANT_FILES:
            raise KeyError(f"unknown variant {v}")
        print(f"\n=== training {v} ===")
        train_one(v)


if __name__ == "__main__":
    main()
