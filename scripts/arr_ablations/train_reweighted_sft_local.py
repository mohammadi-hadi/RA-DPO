"""Reviewer-requested SFT baselines for the local pipeline (wsft, softlabel_sft).

Both variants train on the exact same 5,536 rows as stage_04
(results/openai_sft_train.jsonl) with identical LoRA config, hyperparameters,
and chat-template tokenization — the ONLY difference vs the plain `sft`
baseline is the loss:

  wsft          — causal-LM CE on the assistant answer tokens only, with each
                  example's per-token-mean loss multiplied by its annotator
                  agreement score in {1/2, 2/3, 5/6, 1}; the batch loss is the
                  weighted mean  sum(w_i * l_i) / sum(w_i).
  softlabel_sft — same answer-only masking, but the CE at the FIRST answer
                  token position targets a soft distribution
                  p(YES) = fraction of the 6 annotators voting YES,
                  p(NO) = 1 - p(YES)  (Wu et al. 2023 style soft labels).
                  Remaining answer tokens (the eos/eot token) use standard CE.

In both variants prompt positions are masked with -100 so the loss covers the
assistant answer tokens only.

Per-example metadata (agreement, yes-fraction) is joined from the
EXISTDataLoader seed-42 train split by exact tweet text (stripping the
"Post: " prefix and "\\n\\nClassification (YES or NO):" suffix from the user
message). The join must be 100% and duplicate tweets must carry identical
metadata, otherwise the script aborts.

Adapters are saved to results/local_pipeline/training/<shortname>/<variant>/
and a `stage_04_<variant>` checkpoint is written, so
scripts/local_pipeline/stage_06_eval_oof.py evaluates them unchanged.

Usage:
    python scripts/arr_ablations/train_reweighted_sft_local.py                    # both variants
    python scripts/arr_ablations/train_reweighted_sft_local.py --variant wsft
    python scripts/arr_ablations/train_reweighted_sft_local.py --smoke            # 16 rows, 2 steps
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import load_config, training_dir, write_checkpoint

SFT_PATH = ROOT / "results" / "openai_sft_train.jsonl"
PROMPT_PREFIX = "Post: "
PROMPT_SUFFIX = "\n\nClassification (YES or NO):"
VARIANTS = ["wsft", "softlabel_sft"]


# ---------------------------------------------------------------------------
# Data: join JSONL rows to the train split, tokenize with answer-only labels
# ---------------------------------------------------------------------------

def load_train_metadata(cfg) -> dict:
    """tweet text -> {agreement, yes_frac, majority} from the seed-42 train split."""
    from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score

    loader = EXISTDataLoader(str(ROOT / cfg["data"]["training_json"]))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    train_df, _, _ = loader.create_train_val_test_split(df)
    if len(train_df) != cfg["data"]["expected_n_train"]:
        raise RuntimeError(
            f"train split has {len(train_df)} rows, expected "
            f"{cfg['data']['expected_n_train']}")

    meta: dict[str, dict] = {}
    conflicts = 0
    for _, row in train_df.iterrows():
        labels = row["labels_task1"]
        entry = {
            "agreement": float(agreement_score(labels)),
            "yes_frac": sum(1 for lb in labels if lb == "YES") / len(labels),
            "majority": row["majority_label"],
        }
        prev = meta.get(row["tweet"])
        if prev is not None and prev != entry:
            conflicts += 1
        meta[row["tweet"]] = entry
    if conflicts:
        raise RuntimeError(
            f"{conflicts} duplicate tweet texts with conflicting metadata — "
            "cannot join by tweet text")
    return meta


def join_rows(cfg, smoke: bool) -> list[tuple[list, dict]]:
    """Read the SFT JSONL and attach (agreement, yes_frac) to every row."""
    rows = [json.loads(line) for line in open(SFT_PATH)]
    meta = load_train_metadata(cfg)

    joined, misses = [], 0
    for r in rows:
        user = r["messages"][1]["content"]
        if not (user.startswith(PROMPT_PREFIX) and user.endswith(PROMPT_SUFFIX)):
            raise RuntimeError(f"unexpected user-message format: {user[:60]!r}")
        tweet = user[len(PROMPT_PREFIX):-len(PROMPT_SUFFIX)]
        m = meta.get(tweet)
        if m is None:
            misses += 1
            continue
        answer = r["messages"][2]["content"]
        if answer not in ("YES", "NO"):
            raise RuntimeError(f"unexpected assistant answer: {answer!r}")
        if answer != m["majority"]:
            raise RuntimeError(
                f"assistant answer {answer!r} disagrees with majority vote "
                f"{m['majority']!r} for tweet {tweet[:60]!r}")
        joined.append((r["messages"], m))

    rate = 100.0 * len(joined) / len(rows)
    print(f"join: {len(joined)}/{len(rows)} JSONL rows matched to the train "
          f"split ({rate:.2f}%)")
    if misses:
        raise RuntimeError(f"{misses} JSONL rows did not join to the train split")

    if smoke:
        # Deterministic 16-row subset: per agreement level, up to 2 YES and
        # 2 NO examples (first occurrences), so both answers and all four
        # agreement levels appear.
        by_key: dict[tuple, list] = {}
        for ex in joined:
            key = (round(ex[1]["agreement"], 4), ex[0][2]["content"])
            by_key.setdefault(key, []).append(ex)
        picked = []
        for key in sorted(by_key):
            picked.extend(by_key[key][:2])
        joined = picked[:16]
        print(f"smoke: subsampled {len(joined)} examples "
              f"(agreement levels x YES/NO)")
    return joined


def build_examples(tok, cfg, smoke: bool):
    """Tokenize with the identical chat template used by stage_04 and build
    answer-only label masks. Returns (examples, yes_id, no_id)."""
    joined = join_rows(cfg, smoke)
    max_len = cfg["model"]["max_seq_len"]

    examples = []
    n_ans_counter: Counter = Counter()
    first_tok_by_answer: dict[str, int] = {}

    for msgs, m in joined:
        full_text = tok.apply_chat_template(msgs, tokenize=False)
        prompt_text = tok.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True)
        if not full_text.startswith(prompt_text):
            raise RuntimeError(
                "chat template: full text does not extend the generation prompt")
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        p_len = len(prompt_ids)
        if full_ids[:p_len] != prompt_ids:
            raise RuntimeError(
                "prompt tokens are not a prefix of the full tokenization — "
                "answer masking would be wrong")
        if len(full_ids) > max_len:
            raise RuntimeError(
                f"templated example has {len(full_ids)} tokens > "
                f"max_seq_len={max_len}; refusing to truncate the answer")
        n_ans = len(full_ids) - p_len
        if not 1 <= n_ans <= 3:
            raise RuntimeError(
                f"expected 1-3 answer tokens, got {n_ans} "
                f"({tok.convert_ids_to_tokens(full_ids[p_len:])})")
        n_ans_counter[n_ans] += 1

        answer = msgs[2]["content"]
        first_tok = full_ids[p_len]
        prev = first_tok_by_answer.setdefault(answer, first_tok)
        if prev != first_tok:
            raise RuntimeError(
                f"inconsistent first answer token for {answer!r}: "
                f"{prev} vs {first_tok}")

        examples.append({
            "input_ids": full_ids,
            "labels": [-100] * p_len + full_ids[p_len:],
            "first_ans_idx": p_len,
            "weight": m["agreement"],
            "yes_frac": m["yes_frac"],
            "answer": answer,
        })

    # Verify which token the templated answer actually starts with, and
    # compare against the bare-string first token id.
    yes_id = first_tok_by_answer.get("YES")
    no_id = first_tok_by_answer.get("NO")
    bare_yes = tok("YES", add_special_tokens=False).input_ids[0]
    bare_no = tok("NO", add_special_tokens=False).input_ids[0]
    if yes_id is None:
        print("note: no YES example in this subset; using bare-string YES id")
        yes_id = bare_yes
    if no_id is None:
        print("note: no NO example in this subset; using bare-string NO id")
        no_id = bare_no
    print(f"first answer token in template context: "
          f"YES={yes_id} ({tok.convert_ids_to_tokens([yes_id])[0]!r}), "
          f"NO={no_id} ({tok.convert_ids_to_tokens([no_id])[0]!r})")
    print(f"bare-string first token ids:            YES={bare_yes}, NO={bare_no}"
          + ("" if (yes_id, no_id) == (bare_yes, bare_no)
             else "  [differs — using the templated ids]"))
    if yes_id == no_id:
        raise RuntimeError("YES and NO share the same first token id")
    print(f"answer-token count distribution: {dict(sorted(n_ans_counter.items()))}")

    return examples, yes_id, no_id


class ListDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    """Right-pad input_ids / attention_mask; pad labels with -100; carry the
    per-example weight / yes_frac / first_ans_idx tensors."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        max_len = max(len(f["input_ids"]) for f in feats)
        input_ids, attention_mask, labels = [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention_mask.append([1] * n + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": torch.tensor([f["weight"] for f in feats],
                                   dtype=torch.float32),
            "yes_frac": torch.tensor([f["yes_frac"] for f in feats],
                                     dtype=torch.float32),
            "first_ans_idx": torch.tensor([f["first_ans_idx"] for f in feats],
                                          dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Trainer with the reweighted / soft-label loss
# ---------------------------------------------------------------------------

def make_trainer_class():
    """Deferred import so the module can be inspected without torch installed."""
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class ReweightedSFTTrainer(Trainer):
        def __init__(self, *args, variant: str, yes_id: int, no_id: int,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.variant = variant
            self.yes_id = yes_id
            self.no_id = no_id

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None, **kwargs):
            weight = inputs.pop("weight")
            yes_frac = inputs.pop("yes_frac")
            first_idx = inputs.pop("first_ans_idx")
            labels = inputs.pop("labels")

            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
            logits = outputs.logits.float()
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            bsz, seq, vocab = shift_logits.shape

            ce = F.cross_entropy(
                shift_logits.reshape(-1, vocab), shift_labels.reshape(-1),
                ignore_index=-100, reduction="none").view(bsz, seq)
            mask = (shift_labels != -100).float()

            # Position (in shifted coordinates) whose logits predict the
            # first answer token.
            j = first_idx - 1
            batch_idx = torch.arange(bsz, device=shift_logits.device)
            first_tok = shift_labels[batch_idx, j]
            valid = (first_tok == self.yes_id) | (first_tok == self.no_id)
            if not bool(valid.all()):
                raise RuntimeError(
                    f"first answer token ids {first_tok.tolist()} do not match "
                    f"YES={self.yes_id}/NO={self.no_id}")

            if self.variant == "softlabel_sft":
                logp = torch.log_softmax(shift_logits[batch_idx, j], dim=-1)
                soft_ce = -(yes_frac * logp[:, self.yes_id]
                            + (1.0 - yes_frac) * logp[:, self.no_id])
                pos_mask = torch.zeros_like(ce, dtype=torch.bool)
                pos_mask[batch_idx, j] = True
                ce = torch.where(pos_mask, soft_ce.unsqueeze(1).expand_as(ce),
                                 ce)
                per_example = (ce * mask).sum(1) / mask.sum(1)
                loss = per_example.mean()
            elif self.variant == "wsft":
                per_example = (ce * mask).sum(1) / mask.sum(1)
                loss = (weight * per_example).sum() / weight.sum()
            else:
                raise ValueError(f"unknown variant {self.variant!r}")

            return (loss, outputs) if return_outputs else loss

    return ReweightedSFTTrainer


# ---------------------------------------------------------------------------
# Training driver (mirrors stage_04_sft_lora.py)
# ---------------------------------------------------------------------------

def print_masking_evidence(tok, examples):
    print("\nsmoke batch per-example metadata:")
    for i, ex in enumerate(examples):
        n_ans = len(ex["input_ids"]) - ex["first_ans_idx"]
        print(f"  [{i:2d}] answer={ex['answer']:>3s}  "
              f"weight(agreement)={ex['weight']:.4f}  "
              f"yes_frac={ex['yes_frac']:.4f}  n_answer_tokens={n_ans}")

    ex0 = examples[0]
    p_len = ex0["first_ans_idx"]
    ans_ids = ex0["input_ids"][p_len:]
    n_masked = sum(1 for lb in ex0["labels"] if lb == -100)
    if n_masked != p_len:
        raise RuntimeError(f"masking broken: {n_masked} masked vs "
                           f"prompt length {p_len}")
    print("\nmasked-label proof (example 0):")
    print(f"  sequence length          : {len(ex0['input_ids'])} tokens")
    print(f"  prompt tokens masked -100: {n_masked}")
    print(f"  labels[{p_len - 3}:{len(ex0['labels'])}]      : "
          f"{ex0['labels'][p_len - 3:]}")
    print(f"  answer tokens (visible)  : {tok.convert_ids_to_tokens(ans_ids)}")
    print(f"  prompt tail (masked)     : "
          f"{tok.convert_ids_to_tokens(ex0['input_ids'][p_len - 6:p_len])}\n")


def train_variant(variant: str, smoke: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model

    cfg = load_config()
    model_id = cfg["model"]["id"]

    if smoke:
        out_dir = Path(tempfile.mkdtemp(prefix=f"smoke_{variant}_"))
        print(f"smoke mode: adapter goes to temp dir {out_dir}, "
              "no checkpoint will be written")
    else:
        out_dir = training_dir() / variant
        out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples, yes_id, no_id = build_examples(tok, cfg, smoke)
    print(f"[{variant}] dataset: {len(examples)} rows")
    if smoke:
        print_masking_evidence(tok, examples)

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
    args_kwargs = dict(
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
        remove_unused_columns=False,
    )
    if smoke:
        args_kwargs.update(max_steps=2, logging_steps=1, save_strategy="no")
    args = TrainingArguments(**args_kwargs)

    import inspect
    from transformers import Trainer
    trainer_cls = make_trainer_class()
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": ListDataset(examples),
        "data_collator": Collator(tok.pad_token_id),
        "variant": variant,
        "yes_id": yes_id,
        "no_id": no_id,
    }
    # tokenizer was renamed processing_class between transformers versions
    base_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in base_params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in base_params:
        trainer_kwargs["tokenizer"] = tok
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()

    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history
              if "loss" in h]
    if losses:
        print(f"[{variant}] logged losses (step, loss): {losses[:5]}"
              + (" ..." if len(losses) > 5 else ""))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[{variant}] adapter saved → {out_dir}")

    if not smoke:
        write_checkpoint(f"stage_04_{variant}", "ok", {"adapter": str(out_dir)})

    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return losses


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=VARIANTS, default=None,
                    help="run a single variant (default: both, sequentially)")
    ap.add_argument("--smoke", action="store_true",
                    help="16 examples, max_steps=2 — verifies loss/masking")
    args = ap.parse_args()

    variants = [args.variant] if args.variant else list(VARIANTS)
    for v in variants:
        print(f"\n=== training {v}{' (smoke)' if args.smoke else ''} ===")
        train_variant(v, smoke=args.smoke)


if __name__ == "__main__":
    main()
