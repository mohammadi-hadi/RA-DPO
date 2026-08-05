"""EDOS track — single driver for the RA-DPO pipeline on the second dataset.

Mirrors the EXIST local pipeline (scripts/local_pipeline/stage_02/04/05/06)
on SemEval-2023 Task 10 (EDOS; Kirk et al., 2023) without touching that
pipeline or its validator.

Variants: base, sft, std_dpo, smart30_dpo, random30_dpo, random50_dpo,
ra_dpo. random30 is the size-matched control against smart30 (same budget,
different selection rule); random50 matches the control reported on the
EXIST side.

Configs, one per backbone — pass --config to select:
    configs/edos_pipeline.yaml        Llama-3.2-3B-Instruct (default)
    configs/edos_pipeline_qwen.yaml   Qwen2.5-3B-Instruct
Per-instance files and unified/ output are keyed by model.shortname, so the
two backbones share results_root without colliding; give each its own
models_root, since adapter_dir() is not shortname-keyed.

Subcommands (in dependency order):

  build-pairs        Preference pairs from the official train split
                     (gold = chosen, opposite = rejected):
                       results/edos_pipeline/pairs/std_dpo.jsonl (14,000)
                       results/edos_pipeline/pairs/sft.jsonl     (14,000)
                     JSONL schema is verified record-for-record against
                     results/smart_sampling/smart30_dpo.jsonl.

  build-rx-subsets   Requires train sigmoid_scores (edos_token_scores.py).
                     Ranks train pairs by the EXIST train-side reliability
                     score, empirically recovered from the shipped artifacts:
                         R_train(x) = 0.8 * agreement + 0.2 * (1 - sigmoid)
                     (reproduces EXIST smart30/smart50 membership 1661/1661
                     and 2768/2768 exactly; matches jobs.json rx_stats within
                     1.5e-3; no model-confidence term on the train side).
                     Writes:
                       pairs/smart30_dpo.jsonl — top 30% (4,200), original
                         train row order preserved (threshold filter, exactly
                         like the EXIST smart30/smart50 files)
                       pairs/ra_dpo.jsonl — weight-proportional duplication:
                         2 copies iff agreement >= 5/6, else 1 copy, shuffled
                         (seed 42). Same rule recovered from EXIST's
                         openai_ra_dpo_train.jsonl: copies={1: 2088 items
                         with agreement <= 4/6, 2: 3448 items with agreement
                         >= 5/6}, 5,536 pairs -> 8,984 rows. For EDOS
                         (3 annotators) only unanimous items duplicate:
                         14,000 + 10,850 = 24,850 rows.
                       rx_stats.json — train R(x) stats (jobs.json mirror).

  train --variant {sft,std_dpo,smart30_dpo,ra_dpo}
                     LoRA SFT (stage_04 mirror) / DPO (stage_05 mirror) on
                     MPS. DPO variants start from the base model (identical
                     to the EXIST local pipeline). Adapters ->
                     models/edos_pipeline/<variant>/.

  infer --variant V --split {train,dev,test}
                     Structured-prompt YES/NO inference with confidence,
                     mirroring stage_06 exactly (yes/no first-token id sets,
                     last-token softmax, p_yes/p_no renormalization). Saves a
                     per-instance JSON with the same schema as the EXIST
                     pipelines (predictions / confidences / agreements /
                     sigmoid_scores / correct) to
                     results/edos_pipeline/per_instance/.

  eval-oof           5-fold OOF alpha/beta/gamma (StratifiedKFold, seed 42,
                     StandardScaler + LogisticRegression, |coef| normalized —
                     the exact stage_06 procedure) + coverage-accuracy on the
                     4,000-item test split for every evaluated variant.
                     Writes results/edos_pipeline/unified/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ra_dpo.data.edos_loader import EDOSDataLoader  # noqa: E402
import edos_token_scores as ets  # noqa: E402  (plug-in: sigmoid_scores)

CONFIG_PATH = ROOT / "configs" / "edos_pipeline.yaml"
EXIST_SCHEMA_REF = ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl"

# Identical to the EXIST training files (openai_sft_train.jsonl,
# openai_dpo_train.jsonl, smart_sampling/*.jsonl).
SYSTEM_PROMPT = (
    "You are an expert content moderator. Classify whether the social media "
    "post is sexist or not. Respond with ONLY YES or NO."
)

_CONFIG = None


def set_config_path(path) -> None:
    """Point the run at a different config (e.g. a second backbone).

    Every output that could collide between backbones is keyed by
    model.shortname, so two configs can share results_root safely; give each
    one its own models_root. config_hash() follows the active file, which is
    what keeps provenance on existing artifacts intact.
    """
    global CONFIG_PATH, _CONFIG
    CONFIG_PATH = Path(path).resolve()
    if not CONFIG_PATH.exists():
        raise SystemExit(f"config not found: {CONFIG_PATH}")
    _CONFIG = None


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        import yaml
        with open(CONFIG_PATH) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def config_hash() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:16]


def results_dir() -> Path:
    p = ROOT / load_config()["output_dirs"]["results_root"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def pairs_dir() -> Path:
    p = results_dir() / "pairs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def per_instance_dir() -> Path:
    p = results_dir() / "per_instance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def adapter_dir(variant: str) -> Path:
    return ROOT / load_config()["output_dirs"]["models_root"] / variant


def per_instance_path(variant: str, split: str) -> Path:
    short = load_config()["model"]["shortname"]
    return per_instance_dir() / f"{short}_{variant}_{split}_edos.json"


def load_split(split: str):
    loader = EDOSDataLoader(ROOT / "data" / "edos")
    return loader.get_split(split)


# --------------------------------------------------------------------------
# build-pairs
# --------------------------------------------------------------------------

def make_pair(text: str, gold_label: str) -> dict:
    """gold = chosen, opposite = rejected — same record layout as
    results/smart_sampling/smart30_dpo.jsonl."""
    chosen = gold_label.upper()
    rejected = "NO" if chosen == "YES" else "YES"
    return {
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Post: {text}\n\nClassification (YES or NO):",
                },
            ]
        },
        "preferred_output": [{"role": "assistant", "content": chosen}],
        "non_preferred_output": [{"role": "assistant", "content": rejected}],
    }


def make_sft_record(text: str, gold_label: str) -> dict:
    """Same layout as results/openai_sft_train.jsonl."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Post: {text}\n\nClassification (YES or NO):",
            },
            {"role": "assistant", "content": gold_label.upper()},
        ]
    }


def check_pair_schema_against_exist(record: dict) -> None:
    """Verify record structure against the EXIST smart30 reference file."""
    with open(EXIST_SCHEMA_REF) as f:
        ref = json.loads(f.readline())
    assert set(record) == set(ref), "top-level keys differ from EXIST schema"
    assert set(record["input"]) == set(ref["input"])
    for got, want in zip(record["input"]["messages"], ref["input"]["messages"]):
        assert set(got) == set(want) and got["role"] == want["role"]
    assert record["input"]["messages"][0]["content"] == \
        ref["input"]["messages"][0]["content"], "system prompt differs"
    for key in ("preferred_output", "non_preferred_output"):
        assert set(record[key][0]) == set(ref[key][0])
        assert record[key][0]["role"] == ref[key][0]["role"]


def cmd_build_pairs(args):
    cfg = load_config()
    if args.limit and not args.dry_run:
        raise SystemExit("--limit requires --dry-run "
                         "(prevents truncated real pair files)")
    train_df = load_split("train")
    if args.limit:
        train_df = train_df.head(args.limit)
    print(f"train split: {len(train_df)} items")

    records = [make_pair(r.text, r.gold_label) for r in train_df.itertuples()]
    check_pair_schema_against_exist(records[0])
    print("schema check vs results/smart_sampling/smart30_dpo.jsonl: OK")

    if args.dry_run:
        for rec in records[:5]:
            print(json.dumps(rec, ensure_ascii=False))
        print(f"(dry run — {len(records)} pairs built, nothing written)")
        return

    dpo_path = pairs_dir() / "std_dpo.jsonl"
    with open(dpo_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sft_path = pairs_dir() / "sft.jsonl"
    with open(sft_path, "w") as f:
        for r in train_df.itertuples():
            f.write(json.dumps(make_sft_record(r.text, r.gold_label),
                               ensure_ascii=False) + "\n")

    expected = cfg["data"]["training_pairs"]["std_dpo"]
    if not args.limit:
        assert len(records) == expected, \
            f"{len(records)} pairs != expected {expected}"
    print(f"std_dpo -> {dpo_path} ({len(records)} pairs)")
    print(f"sft     -> {sft_path} ({len(train_df)} records)")


# --------------------------------------------------------------------------
# build-rx-subsets
# --------------------------------------------------------------------------

def train_rx(train_df) -> np.ndarray:
    """EXIST train-side reliability score, replicated (see module docstring):
    R_train(x) = 0.8 * agreement + 0.2 * (1 - sigmoid_score)."""
    w = load_config()["rx"]["train_ranking"]
    agree = train_df["agreement_score"].to_numpy(dtype=float)
    sig = ets.load_sigmoid_scores(train_df["rewire_id"].tolist())
    return (w["agreement_weight"] * agree
            + w["uncertainty_weight"] * (1.0 - sig))


def cmd_build_rx_subsets(args):
    cfg = load_config()
    train_df = load_split("train")
    r = train_rx(train_df)
    train_df = train_df.assign(rx=r)

    stats = {
        "rx_stats": {
            "mean": float(np.mean(r)), "std": float(np.std(r)),
            "median": float(np.median(r)), "p70": float(np.quantile(r, 0.7)),
        },
        "train_ranking": cfg["rx"]["train_ranking"],
        "dataset_sizes": {"full": len(train_df)},
        "config_hash": config_hash(),
    }

    # ---- Smart-30%: top 30% by R(x); file preserves the original train row
    # order (threshold filter), exactly like the EXIST smart30/smart50 files.
    n30 = int(round(len(train_df) * 0.30))
    thresh = np.sort(r)[::-1][n30 - 1]
    smart30_df = train_df[train_df["rx"] >= thresh]
    if len(smart30_df) != n30:  # ties at the threshold: trim deterministically
        keep = np.argsort(-smart30_df["rx"].to_numpy(), kind="stable")[:n30]
        smart30_df = smart30_df.iloc[np.sort(keep)]
    smart30_path = pairs_dir() / "smart30_dpo.jsonl"
    with open(smart30_path, "w") as f:
        for row in smart30_df.itertuples():
            f.write(json.dumps(make_pair(row.text, row.gold_label),
                               ensure_ascii=False) + "\n")
    stats["dataset_sizes"]["smart30"] = len(smart30_df)
    print(f"smart30 -> {smart30_path} ({len(smart30_df)} pairs, "
          f"threshold R(x) >= {thresh:.4f})")

    # ---- RA-DPO: weight-proportional duplication (2 copies iff agreement
    # >= 5/6, else 1), shuffled with the pipeline seed — the rule recovered
    # from EXIST's openai_ra_dpo_train.jsonl (see module docstring).
    dup_mask = train_df["agreement_score"].to_numpy() >= 5.0 / 6.0
    records = [make_pair(row.text, row.gold_label)
               for row in train_df.itertuples()]
    ra_records = list(records)
    ra_records += [rec for rec, d in zip(records, dup_mask) if d]
    rng = np.random.default_rng(cfg["rx"]["seed"])
    rng.shuffle(ra_records)
    ra_path = pairs_dir() / "ra_dpo.jsonl"
    with open(ra_path, "w") as f:
        for rec in ra_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["dataset_sizes"]["ra"] = len(ra_records)
    print(f"ra_dpo  -> {ra_path} ({len(ra_records)} rows = "
          f"{len(train_df)} + {int(dup_mask.sum())} duplicated, "
          f"{len(ra_records) / len(train_df):.3f}x)")

    for name, key in [("smart30", "smart30_dpo"), ("ra", "ra_dpo")]:
        expected = cfg["data"]["training_pairs"][key]
        got = stats["dataset_sizes"][name]
        assert got == expected, f"{key}: {got} rows != expected {expected}"

    # ---- Random-N%: size-matched controls. Random-30 is the one that
    # isolates selection quality (same budget as Smart-30, different rule);
    # Random-50 matches the control reported on the EXIST side. Same pair
    # records, same seed source, only the membership differs.
    for frac in cfg["rx"].get("random_fractions", []):
        key = f"random{int(frac * 100)}_dpo"
        n_keep = int(round(len(train_df) * frac))
        rng_r = np.random.default_rng(cfg["rx"]["seed"])
        keep_idx = np.sort(rng_r.choice(len(train_df), size=n_keep,
                                        replace=False))
        rand_df = train_df.iloc[keep_idx]
        rand_path = pairs_dir() / f"{key}.jsonl"
        with open(rand_path, "w") as f:
            for row in rand_df.itertuples():
                f.write(json.dumps(make_pair(row.text, row.gold_label),
                                   ensure_ascii=False) + "\n")
        stats["dataset_sizes"][key] = len(rand_df)
        expected = cfg["data"]["training_pairs"].get(key)
        if expected is not None:
            assert len(rand_df) == expected, \
                f"{key}: {len(rand_df)} rows != expected {expected}"
        print(f"{key} -> {rand_path} ({len(rand_df)} pairs, "
              f"{frac:.0%} of {len(train_df)})")

    stats_path = results_dir() / "rx_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"rx stats -> {stats_path}: {stats['rx_stats']}")


# --------------------------------------------------------------------------
# train (stage_04 / stage_05 mirrors)
# --------------------------------------------------------------------------

def _lora_config():
    from peft import LoraConfig
    lora = load_config()["training"]["lora"]
    return LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    )


def _load_base(device="mps"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = load_config()
    model_id = cfg["model"]["id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype", "fp16"), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    return tok, model


def cmd_train(args):
    import inspect
    from peft import get_peft_model
    cfg = load_config()
    variant = args.variant
    out_dir = adapter_dir(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model = _load_base()
    model = get_peft_model(model, _lora_config())
    model.print_trainable_parameters()

    if variant == "sft":
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        rows = []
        for line in open(pairs_dir() / "sft.jsonl"):
            d = json.loads(line)
            rows.append({"text": tok.apply_chat_template(d["messages"],
                                                         tokenize=False)})
        ds = Dataset.from_list(rows)
        print(f"SFT dataset: {len(ds)} rows")
        sft = cfg["training"]["sft"]
        kw = dict(
            output_dir=str(out_dir), num_train_epochs=sft["epochs"],
            learning_rate=sft["learning_rate"],
            per_device_train_batch_size=sft["per_device_batch_size"],
            gradient_accumulation_steps=sft["grad_accum"],
            logging_steps=25, save_steps=500, save_total_limit=2,
            fp16=(cfg["model"].get("dtype", "fp16") == "fp16"), bf16=False,
            report_to="none", dataset_text_field="text",
        )
        params = inspect.signature(SFTConfig).parameters
        if "max_length" in params:
            kw["max_length"] = cfg["model"]["max_seq_len"]
        elif "max_seq_length" in params:
            kw["max_seq_length"] = cfg["model"]["max_seq_len"]
        targs = SFTConfig(**kw)
        tkw = {"model": model, "args": targs, "train_dataset": ds}
        sig_params = inspect.signature(SFTTrainer.__init__).parameters
        if "tokenizer" in sig_params:
            tkw["tokenizer"] = tok
        elif "processing_class" in sig_params:
            tkw["processing_class"] = tok
        trainer = SFTTrainer(**tkw)
    else:
        from datasets import Dataset
        from trl import DPOTrainer, DPOConfig
        path = pairs_dir() / f"{variant}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} — run build-pairs / build-rx-subsets first")
        rows = []
        for line in open(path):
            d = json.loads(line)
            inp = d["input"]
            msgs = inp["messages"] if isinstance(inp, dict) else inp
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
            pref = d["preferred_output"]
            rej = d["non_preferred_output"]
            pref = pref[0] if isinstance(pref, list) else pref
            rej = rej[0] if isinstance(rej, list) else rej
            rows.append({"prompt": prompt, "chosen": pref["content"],
                         "rejected": rej["content"]})
        ds = Dataset.from_list(rows)
        expected = cfg["data"]["training_pairs"].get(variant)
        assert expected is None or len(ds) == expected, \
            f"{variant}: {len(ds)} pairs != expected {expected}"
        print(f"[{variant}] dataset: {len(ds)} pairs from {path.name}")
        dpo = cfg["training"]["dpo"]
        targs = DPOConfig(
            output_dir=str(out_dir), num_train_epochs=dpo["epochs"],
            beta=dpo["beta"], learning_rate=dpo["learning_rate"],
            per_device_train_batch_size=dpo["per_device_batch_size"],
            gradient_accumulation_steps=dpo["grad_accum"],
            max_prompt_length=dpo["max_prompt_length"],
            max_length=dpo["max_length"],
            logging_steps=10, save_steps=500, save_total_limit=2,
            fp16=(cfg["model"].get("dtype", "fp16") == "fp16"), bf16=False,
            max_grad_norm=1.0, warmup_ratio=0.1, report_to="none",
        )
        tkw = {"model": model, "ref_model": None, "args": targs,
               "train_dataset": ds}
        sig_params = inspect.signature(DPOTrainer.__init__).parameters
        if "tokenizer" in sig_params:
            tkw["tokenizer"] = tok
        elif "processing_class" in sig_params:
            tkw["processing_class"] = tok
        trainer = DPOTrainer(**tkw)

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[{variant}] adapter saved -> {out_dir}")


# --------------------------------------------------------------------------
# infer (stage_02 / stage_06 inference mirror)
# --------------------------------------------------------------------------

def cmd_infer(args):
    import torch
    from tqdm import tqdm
    from ra_dpo.pipeline.prompts import PromptBuilder
    from ra_dpo.utils.metrics import compute_metrics

    cfg = load_config()
    variant, split = args.variant, args.split
    device = args.device
    out_path = per_instance_path(variant, split)
    if args.limit:  # smoke runs must never shadow full per-instance files
        out_path = out_path.with_name(
            out_path.stem + f"_limit{args.limit}.json")
    if out_path.exists() and not args.force:
        print(f"[{variant}/{split}] already evaluated -> {out_path.name} "
              f"(use --force to redo)")
        return

    df = load_split(split)
    if args.limit:
        df = df.head(args.limit)

    agreements = df["agreement_score"].astype(float).tolist()
    sigmoid_scores = ets.load_sigmoid_scores(df["rewire_id"].tolist()).tolist()

    tok, model = _load_base(device)
    if variant != "base":
        from peft import PeftModel
        adapter = adapter_dir(variant)
        if not adapter.exists():
            raise FileNotFoundError(f"no adapter at {adapter} — train first")
        model = PeftModel.from_pretrained(model, str(adapter)).to(device)
    model.eval()
    y_ids, n_ids = ets.yes_no_ids(tok)

    pb = PromptBuilder()
    strategy = cfg["prompt"]["strategy"]
    preds, confs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"{variant}/{split}"):
        messages = [
            {"role": "system",
             "content": pb.get_system_prompt(strategy, row["lang"])},
            {"role": "user",
             "content": pb.format_user_prompt(row["text"], row["lang"],
                                              strategy)},
        ]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(device)
        mask = (enc["attention_mask"].to(device)
                if "attention_mask" in enc else None)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = float(sum(probs[i].item() for i in y_ids))
        p_no = float(sum(probs[i].item() for i in n_ids))
        total = p_yes + p_no
        if total <= 0:
            pred, conf = "NO", 0.5
        else:
            pyn, pnn = p_yes / total, p_no / total
            pred = "YES" if pyn > pnn else "NO"
            conf = max(pyn, pnn)
        preds.append(pred)
        confs.append(float(conf))

    true_labels = df["gold_label"].tolist()
    correct = np.asarray([p == t for p, t in zip(preds, true_labels)],
                         dtype=int)
    sm = compute_metrics(true_labels, preds)
    sm["avg_confidence"] = float(np.mean(confs))

    payload = {
        "model": f"{cfg['model']['shortname']} ({variant})",
        "model_id": cfg["model"]["id"],
        "dataset": "EDOS (SemEval-2023 Task 10)",
        "split": split,
        "training_pairs": cfg["data"]["training_pairs"].get(variant),
        "standard_metrics": sm,
        "per_instance": {
            "rewire_ids": df["rewire_id"].tolist(),
            "predictions": preds,
            "confidences": confs,
            "agreements": agreements,
            "sigmoid_scores": sigmoid_scores,
            "correct": [bool(c) for c in correct],
        },
        "n_samples": len(df),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_strategy": strategy,
        "config_hash": config_hash(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{variant}/{split}] F1={sm['f1_macro']:.4f} "
          f"Acc={sm['accuracy']:.4f} -> {out_path}")


# --------------------------------------------------------------------------
# eval-oof (stage_06 OOF R(x) mirror)
# --------------------------------------------------------------------------

def oof_rx(conf, agree, sig, correct, n_folds, seed, regime="oracle"):
    """Exact stage_06 procedure, including its quirk of fitting the logistic
    regression on standardized features but scoring R(x) on raw features.

    regime="oracle"       R(x) = a*conf + b*agree + g*(1-sig)  [true agreement]
    regime="no_agreement" R(x) = a*conf + g*(1-sig)            [agreement dropped]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    if regime == "no_agreement":
        X = np.column_stack([conf, 1 - sig])
    else:
        X = np.column_stack([conf, agree, 1 - sig])
    r = np.zeros(len(correct))
    ws = []
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, correct):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(
            sc.transform(X[tr]), correct[tr])
        a = np.abs(lr.coef_[0])
        a = a / a.sum()
        ws.append(a)
        r[te] = X[te] @ a
    return r, np.mean(ws, axis=0)


def acc_top(r, correct, cov):
    k = max(1, int(round(len(r) * cov)))
    return float(correct[np.argsort(-r)[:k]].mean())


def cmd_eval_oof(args):
    import pandas as pd
    cfg = load_config()
    cov_levels = cfg["rx"]["coverage_levels"]
    rows_ft, rows_cov, rows_w, rows_cov_na = [], [], [], []
    rows_cov_pred = []

    pred_agree = None
    if getattr(args, "pred_agreement", None):
        pa = Path(args.pred_agreement)
        pred_agree = np.load(pa if pa.is_absolute() else ROOT / pa)
        assert len(pred_agree) == cfg["data"]["expected_n_test"], \
            (f"predicted agreement has {len(pred_agree)} values != "
             f"{cfg['data']['expected_n_test']} test items")
        print(f"deployable regime: predicted agreement from {pa}")
    summary = {"meta": {"dataset": "EDOS (SemEval-2023 Task 10)",
                        "base_model": cfg["model"]["id"],
                        "prompt_strategy": cfg["prompt"]["strategy"],
                        "n_test": cfg["data"]["expected_n_test"],
                        "n_folds": cfg["rx"]["n_folds"],
                        "config_hash": config_hash()},
               "models": {}}

    for variant in (args.variants or cfg["variants"]):
        path = per_instance_path(variant, "test")
        if not path.exists():
            print(f"[skip] {variant}: no {path.name}")
            continue
        d = json.load(open(path))
        pi = d["per_instance"]
        conf = np.asarray(pi["confidences"], dtype=float)
        agree = np.asarray(pi["agreements"], dtype=float)
        sig = np.asarray(pi["sigmoid_scores"], dtype=float)
        correct = np.asarray(pi["correct"], dtype=int)
        sm = d["standard_metrics"]

        r_oof, w = oof_rx(conf, agree, sig, correct,
                          cfg["rx"]["n_folds"], cfg["rx"]["seed"])
        assert abs(w.sum() - 1.0) < 1e-6, "alpha+beta+gamma != 1"
        acc_cov = {f"acc@{int(c * 100)}%": round(acc_top(r_oof, correct, c), 4)
                   for c in cov_levels}
        assert abs(acc_cov["acc@100%"] - sm["accuracy"]) < 1e-3, \
            "coverage@100% != accuracy"

        # No-agreement regime: the floor available without any annotator
        # signal, so the agreement term's contribution can be read off.
        r_na, w_na = oof_rx(conf, agree, sig, correct, cfg["rx"]["n_folds"],
                            cfg["rx"]["seed"], regime="no_agreement")
        assert abs(w_na.sum() - 1.0) < 1e-6, "alpha+gamma != 1"
        acc_cov_na = {f"acc@{int(c * 100)}%":
                      round(acc_top(r_na, correct, c), 4) for c in cov_levels}
        rows_cov_na.append({"variant": variant, **acc_cov_na})

        # Deployable regime: same three-term R(x), but beta's input is
        # predicted from text instead of read off the annotators.
        acc_cov_pred = None
        if pred_agree is not None:
            r_pd, w_pd = oof_rx(conf, pred_agree, sig, correct,
                                cfg["rx"]["n_folds"], cfg["rx"]["seed"])
            assert abs(w_pd.sum() - 1.0) < 1e-6, "alpha+beta+gamma != 1"
            acc_cov_pred = {f"acc@{int(c * 100)}%":
                            round(acc_top(r_pd, correct, c), 4)
                            for c in cov_levels}
            rows_cov_pred.append({"variant": variant, **acc_cov_pred})

        rows_ft.append({"variant": variant,
                        "pairs": d.get("training_pairs") or "—",
                        "f1_macro": round(sm["f1_macro"], 4),
                        "accuracy": round(sm["accuracy"], 4)})
        rows_cov.append({"variant": variant, **acc_cov})
        rows_w.append({"variant": variant,
                       "alpha (conf)": round(float(w[0]), 3),
                       "beta (agree)": round(float(w[1]), 3),
                       "gamma (tokens)": round(float(w[2]), 3)})
        summary["models"][variant] = {
            "pairs": d.get("training_pairs"),
            "f1_macro": sm["f1_macro"], "accuracy": sm["accuracy"],
            "weights_oof": {"alpha": float(w[0]), "beta": float(w[1]),
                            "gamma": float(w[2])},
            "acc_at_coverage": acc_cov,
            "no_agreement": {
                "weights_oof": {"alpha": float(w_na[0]),
                                "gamma": float(w_na[1])},
                "acc_at_coverage": acc_cov_na,
            },
        }
        if acc_cov_pred is not None:
            summary["models"][variant]["deployable"] = {
                "weights_oof": {"alpha": float(w_pd[0]), "beta": float(w_pd[1]),
                                "gamma": float(w_pd[2])},
                "acc_at_coverage": acc_cov_pred,
            }

    if not rows_ft:
        raise SystemExit("no evaluated variants found — run infer first")
    # Keyed by shortname so a second backbone cannot overwrite the first,
    # matching results/local_pipeline/unified/<backbone>/ on the EXIST side.
    out = results_dir() / "unified" / cfg["model"]["shortname"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_ft).to_csv(out / "fine_tuning.csv", index=False)
    pd.DataFrame(rows_cov).to_csv(out / "coverage_accuracy.csv", index=False)
    pd.DataFrame(rows_cov_na).to_csv(
        out / "coverage_accuracy_no_agreement.csv", index=False)
    if rows_cov_pred:
        pd.DataFrame(rows_cov_pred).to_csv(
            out / "coverage_accuracy_deployable.csv", index=False)
    pd.DataFrame(rows_w).to_csv(out / "weights.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== EDOS fine-tuning (structured prompt, test n="
          f"{summary['meta']['n_test']}) ===")
    print(pd.DataFrame(rows_ft).to_string(index=False))
    print("\n=== Coverage-accuracy (5-fold OOF R(x)) ===")
    print(pd.DataFrame(rows_cov).to_string(index=False))
    print("\n=== Learned weights (OOF mean) ===")
    print(pd.DataFrame(rows_w).to_string(index=False))
    print(f"\nSaved -> {out}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="config YAML to run against (default: "
                         "configs/edos_pipeline.yaml). Use a per-backbone "
                         "config rather than editing the default in place.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-pairs", help="std_dpo + sft JSONL from train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_build_pairs)

    p = sub.add_parser("build-rx-subsets",
                       help="smart30 + ra_dpo JSONL (needs token scores)")
    p.set_defaults(func=cmd_build_rx_subsets)

    TRAINABLE = ["sft", "std_dpo", "smart30_dpo", "ra_dpo",
                 "random30_dpo", "random50_dpo"]

    p = sub.add_parser("train", help="LoRA SFT / DPO for one variant")
    p.add_argument("--variant", required=True, choices=TRAINABLE)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("infer", help="structured-prompt YES/NO inference")
    p.add_argument("--variant", required=True,
                   choices=["base"] + TRAINABLE)
    p.add_argument("--split", required=True,
                   choices=["train", "dev", "test"])
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("eval-oof", help="OOF R(x) + coverage tables")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--pred-agreement", default=None,
                   help="npy of text-predicted agreement (one value per test "
                        "item, in test order). Adds the deployable regime "
                        "alongside the oracle and no-agreement ones.")
    p.set_defaults(func=cmd_eval_oof)

    args = ap.parse_args()
    if args.config:
        set_config_path(args.config)
    args.func(args)


if __name__ == "__main__":
    main()
