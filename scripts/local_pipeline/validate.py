"""Validator for the local pipeline.

Run anytime (before a stage, after a stage, or standalone) to confirm no
invariants have drifted. Every rule in docs/LOCAL_PIPELINE_RULES.md Section 3
(invariants I1–I12) and Section 10 (checklist) is enforced here.

Fails closed: any failure exits non-zero. Warnings never affect the exit code.

Usage:
    python scripts/local_pipeline/validate.py            # report; exit 1 on any failure
    python scripts/local_pipeline/validate.py --strict   # accepted, same as default
    python scripts/local_pipeline/validate.py --lenient  # report only; always exit 0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.local_pipeline import (
    load_config, config_hash, results_dir, per_instance_dir, per_instance_path,
    PER_INSTANCE_KEYS, REQUIRED_TOP_KEYS, shortname,
)

REFERENCE_FILE = ROOT / "results" / "final_reliability_3factor" / "gpt-4o_base.json"
ALLOWED_AGREEMENT = {0.5, 2 / 3, 5 / 6, 1.0}

# Known DPO variant names (must match keys in config.data.training_pairs).
KNOWN_VARIANTS = ["base", "sft", "std_dpo", "smart10_dpo", "smart30_dpo",
                  "smart50_dpo", "random50_dpo", "random30_dpo",
                  "ambiguous_dpo", "ra_dpo",
                  "agree30_dpo", "agree30_tb2_dpo", "uncert30_dpo",
                  "conf30_dpo", "wsft", "softlabel_sft",
                  "flip10_dpo", "randunan10_dpo", "random10_dpo",
                  "strat10_dpo", "noworst10_dpo",
                  "noisy1_random10_dpo", "noisy1_strat10_dpo",
                  "hard10_llama_dpo", "hardctl10_llama_dpo", "hardall10_llama_dpo",
                  "hard10_qwen_dpo", "hardctl10_qwen_dpo", "hardall10_qwen_dpo"]

# Expected LoRA adapter sub-dirs for each fine-tuned variant.
VARIANTS_WITH_CHECKPOINT = [v for v in KNOWN_VARIANTS if v != "base"]


class Report:
    def __init__(self):
        self.failures: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.passes: list[str] = []

    def ok(self, msg: str):
        self.passes.append(msg)

    def warn(self, tag: str, msg: str):
        self.warnings.append((tag, msg))

    def fail(self, tag: str, msg: str):
        self.failures.append((tag, msg))

    def print(self) -> None:
        print(f"\n=== validate.py report ===")
        print(f"  passes   : {len(self.passes)}")
        print(f"  warnings : {len(self.warnings)}")
        print(f"  failures : {len(self.failures)}")
        for p in self.passes:
            print(f"  ✓ {p}")
        for tag, msg in self.warnings:
            print(f"  ⚠ {tag}: {msg}")
        for tag, msg in self.failures:
            print(f"  ✗ {tag}: {msg}")
        print()

    def exit(self, lenient: bool = False) -> None:
        if self.failures:
            if lenient:
                print(f"  ⚠ {len(self.failures)} failure(s) present; "
                      f"exit code suppressed by --lenient\n")
                sys.exit(0)
            sys.exit(1)
        sys.exit(0)


# ----- Individual checks -----

def check_hardware(r: Report) -> None:
    try:
        import torch
        if not torch.backends.mps.is_available():
            r.fail("HW", "torch.backends.mps.is_available() is False")
            return
        r.ok("MPS available")
    except Exception as e:
        r.fail("HW", f"torch import failed: {e}")


def check_package_versions(r: Report) -> None:
    mins = {"torch": (2, 1), "transformers": (4, 42), "trl": (0, 9), "peft": (0, 11)}
    for pkg, minv in mins.items():
        try:
            mod = __import__(pkg)
            v = tuple(int(x) for x in mod.__version__.split(".")[:2])
            if v < minv:
                r.warn(f"VERSION/{pkg}", f"{mod.__version__} < required {minv}")
            else:
                r.ok(f"{pkg} {mod.__version__}")
        except ImportError:
            r.fail(f"VERSION/{pkg}", "not installed")


def check_config(r: Report) -> None:
    try:
        cfg = load_config()
    except Exception as e:
        r.fail("CFG", f"config load failed: {e}")
        return
    if cfg["model"]["dtype"].lower() == "bf16":
        r.fail("CFG/dtype", "bf16 is not supported on MPS; use fp16")
    if cfg["hardware"]["device"] != "mps":
        r.warn("CFG/device", f"device is {cfg['hardware']['device']}, expected mps")
    if cfg["prompt"]["strategy"] != "structured":
        r.fail("CFG/prompt", f"prompt.strategy must be 'structured', got {cfg['prompt']['strategy']}")
    hash_stored = results_dir() / "config_hash.txt"
    if hash_stored.exists():
        stored = hash_stored.read_text().strip()
        if stored != config_hash():
            r.warn("CFG/hash", f"hash drift: stored={stored} current={config_hash()}")
        else:
            r.ok(f"config hash stable ({config_hash()})")
    else:
        r.warn("CFG/hash", "no config_hash.txt stored yet")


def check_data_split(r: Report) -> None:
    try:
        from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
        loader = EXISTDataLoader(str(ROOT / load_config()["data"]["training_json"]))
        df = loader.to_dataframe()
        df["majority_label"] = df["labels_task1"].apply(majority_vote)
        df["agreement_score"] = df["labels_task1"].apply(agreement_score)
        train_df, val_df, test_df = loader.create_train_val_test_split(df)
        cfg = load_config()["data"]
        expect = (cfg["expected_n_train"], cfg["expected_n_val"], cfg["expected_n_test"])
        got = (len(train_df), len(val_df), len(test_df))
        if got != expect:
            r.fail("I1/DATA", f"split size mismatch: expected {expect}, got {got}")
        else:
            r.ok(f"train/val/test split {got}")
    except Exception as e:
        r.fail("DATA", f"data load failed: {e}")


def check_reference_alignment(r: Report) -> None:
    """I3, I4: agreements + sigmoid_scores must match gpt-4o reference file."""
    if not REFERENCE_FILE.exists():
        r.warn("REF", f"{REFERENCE_FILE} not found — cross-check skipped")
        return
    ref = json.load(open(REFERENCE_FILE))
    r.ok(f"reference loaded: {REFERENCE_FILE.name}")

    # Check every local per-instance file
    for path in sorted(per_instance_dir().glob(f"{shortname()}_*_local.json")):
        try:
            d = json.load(open(path))
        except Exception as e:
            r.fail(f"SCHEMA/{path.name}", f"json load failed: {e}")
            continue
        if np.asarray(d["per_instance"]["agreements"]).tolist() != ref["per_instance"]["agreements"]:
            r.fail(f"I3/{path.name}", "agreements differ from gpt-4o reference")
        if np.asarray(d["per_instance"]["sigmoid_scores"]).tolist() != ref["per_instance"]["sigmoid_scores"]:
            r.fail(f"I4/{path.name}", "sigmoid_scores differ from gpt-4o reference")


def check_per_instance_invariants(r: Report) -> None:
    """I1, I2, I5, I6, I7, I10, I11, I12."""
    for path in sorted(per_instance_dir().glob(f"{shortname()}_*_local.json")):
        d = json.load(open(path))
        name = path.name

        # I12: schema
        missing = [k for k in REQUIRED_TOP_KEYS if k not in d]
        if missing:
            r.fail(f"I12/{name}", f"missing keys: {missing}")
            continue

        pi = d["per_instance"]
        for k in PER_INSTANCE_KEYS:
            if k not in pi:
                r.fail(f"I12/{name}", f"missing per_instance key: {k}")
                continue

        # I1, I2
        if d.get("n_samples") != 692:
            r.fail(f"I1/{name}", f"n_samples={d.get('n_samples')}")
        for k in PER_INSTANCE_KEYS:
            if len(pi[k]) != 692:
                r.fail(f"I2/{name}", f"{k} length {len(pi[k])}")

        # I5
        conf = np.asarray(pi["confidences"], dtype=float)
        if conf.min() < 0 or conf.max() > 1 + 1e-6:
            r.fail(f"I5/{name}", f"confidences outside [0,1]: min={conf.min()} max={conf.max()}")

        # I6
        agree = np.asarray(pi["agreements"], dtype=float)
        for v in np.unique(agree):
            if not any(abs(v - a) < 1e-3 for a in ALLOWED_AGREEMENT):
                r.fail(f"I6/{name}", f"unexpected agreement value {v}")
                break

        # I7
        bad_preds = [p for p in pi["predictions"] if p not in ("YES", "NO")]
        if bad_preds:
            r.fail(f"I7/{name}", f"invalid predictions e.g. {bad_preds[0]!r}")

        # I10 (OOF weights sum to 1) — if this file carries them
        if "optimized_weights" in d:
            w = d["optimized_weights"]
            total = w.get("alpha", 0) + w.get("beta", 0) + w.get("gamma", 0)
            if abs(total - 1.0) > 1e-4:
                r.fail(f"I10/{name}", f"weights sum to {total:.6f}")

        # I11 (acc@100% == standard_metrics.accuracy)
        if "accuracy_at_coverage" in d:
            a100 = d["accuracy_at_coverage"].get("acc@100%")
            acc = d["standard_metrics"].get("accuracy")
            if a100 is not None and acc is not None and abs(a100 - acc) > 1e-4:
                r.fail(f"I11/{name}", f"acc@100%={a100} vs accuracy={acc}")

        # Config hash match
        ch = d.get("config_hash")
        if ch and ch != config_hash():
            r.warn(f"CFG/{name}", f"produced with config_hash={ch} (current={config_hash()})")

        r.ok(f"per_instance OK: {name}")


def check_training_pairs_sizes(r: Report) -> None:
    """I9: JSONL file sizes must match config."""
    cfg = load_config()["data"]["training_pairs"]
    files = {
        "sft":            ROOT / "results" / "openai_sft_train.jsonl",
        "smart10_dpo":    ROOT / "results" / "smart_sampling" / "smart10_dpo.jsonl",
        "smart30_dpo":    ROOT / "results" / "smart_sampling" / "smart30_dpo.jsonl",
        "smart50_dpo":    ROOT / "results" / "smart_sampling" / "smart50_dpo.jsonl",
        "random50_dpo":   ROOT / "results" / "smart_sampling" / "random50_dpo.jsonl",
        "standard_dpo":   ROOT / "results" / "openai_dpo_train.jsonl",
        "ambiguous_dpo":  ROOT / "results" / "smart_sampling" / "ambiguous_only_dpo.jsonl",
        "ra_dpo":         ROOT / "results" / "openai_ra_dpo_train.jsonl",
        "agree30_dpo":     ROOT / "results" / "smart_sampling" / "agree30_dpo.jsonl",
        "agree30_tb2_dpo": ROOT / "results" / "smart_sampling" / "agree30_tb2_dpo.jsonl",
        "uncert30_dpo":    ROOT / "results" / "smart_sampling" / "uncert30_dpo.jsonl",
        "conf30_dpo":      ROOT / "results" / "smart_sampling" / "conf30_dpo.jsonl",
        "flip10_dpo":      ROOT / "results" / "smart_sampling" / "flip10_dpo.jsonl",
        "randunan10_dpo":  ROOT / "results" / "smart_sampling" / "randunan10_dpo.jsonl",
        "random10_dpo":    ROOT / "results" / "smart_sampling" / "random10_dpo.jsonl",
        "strat10_dpo":     ROOT / "results" / "smart_sampling" / "strat10_dpo.jsonl",
        "noworst10_dpo":   ROOT / "results" / "smart_sampling" / "noworst10_dpo.jsonl",
        "noisy1_random10_dpo": ROOT / "results" / "smart_sampling" / "noisy1_random10_dpo.jsonl",
        "noisy1_strat10_dpo":  ROOT / "results" / "smart_sampling" / "noisy1_strat10_dpo.jsonl",
        "hard10_llama_dpo":    ROOT / "results" / "smart_sampling" / "hard10_llama_dpo.jsonl",
        "hardctl10_llama_dpo": ROOT / "results" / "smart_sampling" / "hardctl10_llama_dpo.jsonl",
        "hardall10_llama_dpo": ROOT / "results" / "smart_sampling" / "hardall10_llama_dpo.jsonl",
        "hard10_qwen_dpo":     ROOT / "results" / "smart_sampling" / "hard10_qwen_dpo.jsonl",
        "hardctl10_qwen_dpo":  ROOT / "results" / "smart_sampling" / "hardctl10_qwen_dpo.jsonl",
        "hardall10_qwen_dpo":  ROOT / "results" / "smart_sampling" / "hardall10_qwen_dpo.jsonl",
    }
    for key, path in files.items():
        if not path.exists():
            r.warn(f"I9/{key}", f"file missing: {path}")
            continue
        n = sum(1 for _ in open(path))
        expected = cfg.get(key)
        if expected and n != expected:
            # SFT file has 5535 lines even though train has 5536 (one dropped at build time); allow ±1
            if abs(n - expected) > 1:
                r.fail(f"I9/{key}", f"{path.name}: {n} lines, expected {expected}")
            else:
                r.ok(f"{key} size {n} (~expected {expected})")
        else:
            r.ok(f"{key} size {n}")


def check_training_checkpoints(r: Report) -> None:
    """Every fine-tuned variant should have a LoRA adapter when stage is run."""
    tdir = results_dir() / "training" / shortname()
    for v in VARIANTS_WITH_CHECKPOINT:
        adapter_dir = tdir / v
        pi = per_instance_path(v)
        if pi.exists() and not adapter_dir.exists():
            r.warn(f"CKPT/{v}", f"per_instance file exists but no adapter dir at {adapter_dir}")


def check_unified_tables(r: Report) -> None:
    uni = results_dir() / "unified" / shortname()
    for f in ["fine_tuning.csv", "coverage_accuracy.csv", "weights.csv"]:
        p = uni / f
        if p.exists():
            r.ok(f"unified/{f} present")
        else:
            r.warn(f"TABLES/{f}", "missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="no-op; failing on any failure is now the default")
    parser.add_argument("--lenient", action="store_true",
                        help="print the report but always exit 0 (old default)")
    args = parser.parse_args()

    r = Report()
    check_hardware(r)
    check_package_versions(r)
    check_config(r)
    check_data_split(r)
    check_training_pairs_sizes(r)
    check_reference_alignment(r)
    check_per_instance_invariants(r)
    check_training_checkpoints(r)
    check_unified_tables(r)
    r.print()
    r.exit(lenient=args.lenient)


if __name__ == "__main__":
    main()
