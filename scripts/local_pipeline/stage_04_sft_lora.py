"""Stage 04 — SFT with LoRA (fp16 on MPS).

Uses TRL's SFTTrainer + PEFT LoRA. Training data is
results/openai_sft_train.jsonl (5,535 lines). Writes adapters to
results/local_pipeline/training/sft/.

Note: this script expects MPS. It sets `use_mps_device=True` through
TrainingArguments and uses fp16 tensors.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import load_config, training_dir, write_checkpoint


def load_sft_dataset(tok):
    from datasets import Dataset
    rows = []
    path = ROOT / "results" / "openai_sft_train.jsonl"
    for line in open(path):
        d = json.loads(line)
        msgs = d["messages"]
        # Convert to a single "text" field via chat template
        full = tok.apply_chat_template(msgs, tokenize=False)
        rows.append({"text": full})
    return Dataset.from_list(rows)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model

    cfg = load_config()
    model_id = cfg["model"]["id"]
    out_dir = training_dir() / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to("mps")

    lora_cfg = LoraConfig(
        r=cfg["training"]["lora"]["r"],
        lora_alpha=cfg["training"]["lora"]["alpha"],
        lora_dropout=cfg["training"]["lora"]["dropout"],
        target_modules=cfg["training"]["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    sft = cfg["training"]["sft"]
    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=sft["per_device_batch_size"],
        gradient_accumulation_steps=sft["grad_accum"],
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=(cfg["model"].get("dtype", "fp16") == "fp16"),
        bf16=False,
        report_to="none",
        dataset_text_field="text",
    )
    # max_length vs max_seq_length renamed between TRL versions
    import inspect
    params = inspect.signature(SFTConfig).parameters
    if "max_length" in params:
        sft_kwargs["max_length"] = cfg["model"]["max_seq_len"]
    elif "max_seq_length" in params:
        sft_kwargs["max_seq_length"] = cfg["model"]["max_seq_len"]
    args = SFTConfig(**sft_kwargs)

    ds = load_sft_dataset(tok)
    print(f"SFT dataset: {len(ds)} rows")

    trainer_kwargs = {"model": model, "args": args, "train_dataset": ds}
    if "tokenizer" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tok
    elif "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"adapter saved → {out_dir}")

    write_checkpoint("stage_04_sft", "ok", {"adapter": str(out_dir)})


if __name__ == "__main__":
    main()
