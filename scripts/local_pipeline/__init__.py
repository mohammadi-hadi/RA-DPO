"""Shared constants, IO, and invariant helpers for the local pipeline.

Import from here — do NOT duplicate constants in stage scripts.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Default is the canonical Llama config. LOCAL_PIPELINE_CONFIG selects a
# different backbone (see configs/local_pipeline_qwen.yaml) without editing
# the canonical file in place, which would change its config_hash and orphan
# the provenance on every existing artifact. Training dirs, per-instance
# files, and unified/ are all keyed by model.shortname, so the backbones
# cannot collide.
CONFIG_PATH = Path(os.environ.get(
    "LOCAL_PIPELINE_CONFIG", ROOT / "configs" / "local_pipeline.yaml"))
RULES_PATH = ROOT / "docs" / "LOCAL_PIPELINE_RULES.md"

_CONFIG_CACHE: dict[str, Any] | None = None


def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(CONFIG_PATH) as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def config_hash() -> str:
    """SHA-256 of the canonical config file. Used to detect drift."""
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:16]


def results_dir() -> Path:
    cfg = load_config()
    p = ROOT / cfg["output_dirs"]["results_root"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def per_instance_dir() -> Path:
    p = results_dir() / "per_instance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def unified_dir() -> Path:
    p = results_dir() / "unified" / shortname()
    p.mkdir(parents=True, exist_ok=True)
    return p


def training_dir() -> Path:
    p = results_dir() / "training" / shortname()
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = results_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def shortname() -> str:
    return load_config()["model"]["shortname"]


def per_instance_path(variant: str) -> Path:
    """Canonical per-instance file path for a given variant."""
    return per_instance_dir() / f"{shortname()}_{variant}_local.json"


PER_INSTANCE_KEYS = ["predictions", "confidences", "agreements",
                      "sigmoid_scores", "correct"]

REQUIRED_TOP_KEYS = ["model", "model_id", "training_pairs",
                     "standard_metrics", "per_instance", "n_samples",
                     "timestamp", "prompt_strategy", "config_hash"]


def save_per_instance(variant: str, payload: dict) -> Path:
    """Write a per-instance file, embedding the current config hash."""
    payload = dict(payload)
    payload.setdefault("config_hash", config_hash())
    payload.setdefault("prompt_strategy", load_config()["prompt"]["strategy"])
    for k in REQUIRED_TOP_KEYS:
        if k not in payload:
            raise ValueError(f"missing key in payload: {k}")
    out = per_instance_path(variant)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def write_checkpoint(stage: str, status: str, extras: dict | None = None) -> None:
    """Record stage completion to results/local_pipeline/checkpoints.json."""
    ckpt_path = results_dir() / "checkpoints.json"
    data = json.load(open(ckpt_path)) if ckpt_path.exists() else {}
    entry = {"status": status, "hash": config_hash()}
    if extras:
        entry.update(extras)
    data[stage] = entry
    json.dump(data, open(ckpt_path, "w"), indent=2)


def read_checkpoints() -> dict:
    ckpt_path = results_dir() / "checkpoints.json"
    return json.load(open(ckpt_path)) if ckpt_path.exists() else {}
