"""Regenerate gpt-4o confidences for the 5,536 train posts via the Batch API.

The paper's smart-sampling R(x) over the TRAIN split used per-post model
confidences that were never persisted. This script rebuilds them with the
exact request construction the paper used for test-set inference
(src/pipeline/openai_runner.py::_run_batch_api — structured prompt per
language, temperature 0.0, max_tokens 10, logprobs=True, top_logprobs=5)
and the p_yes/(p_yes+p_no) normalization (confidence = max of the two).

Subcommands:
  validate    Offline: build the full batch JSONL, verify every line parses,
              count 5,536 requests, print one EN + one ES request, assert
              prompt parity with openai_runner's construction, and unit-check
              the confidence-extraction function on synthetic payloads.
  submit      Build + upload the JSONL and create the batch. Saves the batch
              id to results/arr_ablations/train_confidence/batch_state.json.
  poll        Check batch status. On completion, download results, extract
              per-tweet confidences, write
              results/smart_sampling/train_confidences_gpt4o.json
              ({id_EXIST: confidence}) and run the R(x) sanity check against
              results/smart_sampling/jobs.json rx_stats.
  build-conf30
              Call scripts/arr_ablations/build_component_ablation_subsets.py
              with --confidences to write results/smart_sampling/conf30_dpo.jsonl
              (top 1,661 by confidence, tie-break id asc).

Usage:
    python scripts/arr_ablations/run_train_confidence_batch.py validate
    python scripts/arr_ablations/run_train_confidence_batch.py submit
    python scripts/arr_ablations/run_train_confidence_batch.py poll
    python scripts/arr_ablations/run_train_confidence_batch.py build-conf30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from src.pipeline.prompts import PromptBuilder, STRATEGY_MAX_TOKENS

MODEL = "gpt-4o"
STRATEGY = "structured"
N_TRAIN_EXPECTED = 5536

STATE_DIR = ROOT / "results" / "arr_ablations" / "train_confidence"
INPUT_JSONL = STATE_DIR / "train_batch_input.jsonl"
OUTPUT_JSONL = STATE_DIR / "train_batch_output.jsonl"
STATE_PATH = STATE_DIR / "batch_state.json"
CONF_OUT = ROOT / "results" / "smart_sampling" / "train_confidences_gpt4o.json"
JOBS_JSON = ROOT / "results" / "smart_sampling" / "jobs.json"
TOKEN_SCORES = ROOT / "results" / "token_scores" / "token_scores_cache.json"
SUBSET_BUILDER = ROOT / "scripts" / "arr_ablations" / "build_component_ablation_subsets.py"

# First-token variants counted as YES / NO when normalizing p_yes/(p_yes+p_no).
# SI variants cover the Spanish prompts ("SI o NO"); parse_prediction treats
# SI as YES in every language, so the same mapping is applied here.
YES_VARIANTS = {"YES", "SI", "SÍ"}
NO_VARIANTS = {"NO"}


def load_env_key() -> None:
    """Load OPENAI_API_KEY from the repo-root .env if not already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = val.strip().strip("'\"")
            return


def get_client():
    load_env_key()
    from openai import OpenAI
    return OpenAI()


# ---------------------------------------------------------------------------
# Request construction (mirrors openai_runner.OpenAIRunner._run_batch_api)
# ---------------------------------------------------------------------------

def load_train_df() -> pd.DataFrame:
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, _, _ = loader.create_train_val_test_split(df)
    return train_df.reset_index(drop=True)


def build_requests(train_df: pd.DataFrame) -> list:
    """One /v1/chat/completions request per train post, keyed by id_EXIST.

    Body fields replicate openai_runner._run_batch_api exactly:
    system + user from PromptBuilder (structured, per-language), max_tokens
    from STRATEGY_MAX_TOKENS, temperature 0.0, logprobs True, top_logprobs 5.
    """
    pb = PromptBuilder()
    max_tokens = pb.get_max_tokens(STRATEGY)
    requests = []
    for _, row in train_df.iterrows():
        lang = row["lang"]
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": pb.get_system_prompt(STRATEGY, lang)},
                {"role": "user",
                 "content": pb.format_user_prompt(row["tweet"], lang, STRATEGY)},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        requests.append({
            "custom_id": str(row["id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
    ids = [r["custom_id"] for r in requests]
    assert len(ids) == len(set(ids)), "duplicate custom_ids in batch"
    return requests


def write_jsonl(requests: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")


# ---------------------------------------------------------------------------
# Confidence extraction (p_yes / (p_yes + p_no), confidence = max)
# ---------------------------------------------------------------------------

def classify_token(token: str) -> str | None:
    t = token.strip().upper()
    if t in YES_VARIANTS:
        return "YES"
    if t in NO_VARIANTS:
        return "NO"
    return None


def extract_confidence(first_token_entry: dict) -> float:
    """Confidence from the FIRST completion token's top_logprobs.

    Sums probability mass over YES-variant and NO-variant tokens, normalizes
    p_yes/(p_yes+p_no), returns max(p_yes_n, p_no_n). Falls back to 0.5 when
    neither class appears (mirrors the pipeline's 0.5 default).
    """
    p_yes = 0.0
    p_no = 0.0
    for cand in first_token_entry.get("top_logprobs", []):
        cls = classify_token(cand.get("token", ""))
        if cls == "YES":
            p_yes += math.exp(cand["logprob"])
        elif cls == "NO":
            p_no += math.exp(cand["logprob"])
    total = p_yes + p_no
    if total <= 0:
        return 0.5
    return max(p_yes / total, p_no / total)


def parse_batch_line(line: str) -> tuple:
    """Return (custom_id, confidence or None) for one batch output line."""
    item = json.loads(line)
    cid = item["custom_id"]
    body = (item.get("response") or {}).get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return cid, None
    logprobs = choices[0].get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        return cid, None
    return cid, extract_confidence(content[0])


# ---------------------------------------------------------------------------
# R(x) sanity check against jobs.json rx_stats
# ---------------------------------------------------------------------------

def rx_sanity_check(confidences: dict) -> None:
    train_df = load_train_df()
    cache = json.load(open(TOKEN_SCORES))
    ids = train_df["id"].astype(str).tolist()
    missing = [i for i in ids if i not in confidences]
    if missing:
        print(f"WARNING: {len(missing)} train ids missing a confidence (0.5 used)")
    conf = np.array([confidences.get(i, 0.5) for i in ids])
    agree = train_df["agreement_score"].to_numpy(dtype=float)
    sig = np.array([cache[i]["sigmoid_score"] for i in ids], dtype=float)
    rx = (conf + agree + (1.0 - sig)) / 3.0

    ref = json.load(open(JOBS_JSON))["rx_stats"]
    got = {
        "mean": float(rx.mean()),
        "std": float(rx.std()),
        "median": float(np.median(rx)),
        "p70": float(np.percentile(rx, 70)),
    }
    print(f"\nConfidence summary: mean={conf.mean():.4f} std={conf.std():.4f} "
          f"min={conf.min():.4f} max={conf.max():.4f}")
    print("\nR(x) = (conf + agree + (1 - sigmoid)) / 3 over the train split")
    print(f"{'stat':<8}{'regenerated':>14}{'jobs.json':>14}{'abs diff':>12}")
    for k in ("mean", "std", "median", "p70"):
        print(f"{k:<8}{got[k]:>14.6f}{ref[k]:>14.6f}{abs(got[k]-ref[k]):>12.6f}")
    close = abs(got["mean"] - ref["mean"]) < 0.01 and abs(got["p70"] - ref["p70"]) < 0.01
    print("SANITY:", "OK (within 0.01 of the original run)" if close
          else "MISMATCH — inspect before building conf30")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_validate() -> None:
    train_df = load_train_df()
    assert len(train_df) == N_TRAIN_EXPECTED, \
        f"train split is {len(train_df)}, expected {N_TRAIN_EXPECTED}"
    requests = build_requests(train_df)
    write_jsonl(requests, INPUT_JSONL)

    # Every line must parse and carry the exact body the paper's runner sends
    pb = PromptBuilder()
    n = 0
    with open(INPUT_JSONL) as f:
        for line in f:
            req = json.loads(line)
            body = req["body"]
            assert req["method"] == "POST" and req["url"] == "/v1/chat/completions"
            assert body["model"] == MODEL
            assert body["max_tokens"] == STRATEGY_MAX_TOKENS[STRATEGY]
            assert body["temperature"] == 0.0
            assert body["logprobs"] is True and body["top_logprobs"] == 5
            assert [m["role"] for m in body["messages"]] == ["system", "user"]
            n += 1
    print(f"JSONL OK: {n} lines parse at {INPUT_JSONL}")
    assert n == N_TRAIN_EXPECTED

    # Prompt parity: rebuild messages independently for one EN + one ES row
    for lang in ("en", "es"):
        row = train_df[train_df["lang"] == lang].iloc[0]
        req = next(r for r in requests if r["custom_id"] == str(row["id"]))
        expect_sys = pb.get_system_prompt(STRATEGY, lang)
        expect_user = pb.format_user_prompt(row["tweet"], lang, STRATEGY)
        assert req["body"]["messages"][0]["content"] == expect_sys
        assert req["body"]["messages"][1]["content"] == expect_user
        print(f"\n--- sample request ({lang}, id={req['custom_id']}) ---")
        print(json.dumps(req, indent=2, ensure_ascii=False))
    print("\nPROMPT PARITY OK: messages match PromptBuilder(structured) per language")

    # Unit-check extract_confidence on synthetic payloads
    ln = math.log
    cases = [
        ("EN yes-dominant",
         {"token": "YES", "logprob": ln(0.90), "top_logprobs": [
             {"token": "YES", "logprob": ln(0.90)},
             {"token": " NO", "logprob": ln(0.06)},
             {"token": "no", "logprob": ln(0.02)},
             {"token": "Sure", "logprob": ln(0.01)},
         ]},
         0.90 / 0.98),
        ("ES si-vs-no",
         {"token": "SI", "logprob": ln(0.55), "top_logprobs": [
             {"token": "SI", "logprob": ln(0.55)},
             {"token": "NO", "logprob": ln(0.35)},
             {"token": "Sí", "logprob": ln(0.05)},
         ]},
         0.60 / 0.95),
        ("no YES/NO in top-k",
         {"token": "Hello", "logprob": ln(0.9), "top_logprobs": [
             {"token": "Hello", "logprob": ln(0.9)},
             {"token": "world", "logprob": ln(0.1)},
         ]},
         0.5),
    ]
    for name, payload, expected in cases:
        got = extract_confidence(payload)
        assert abs(got - expected) < 1e-9, f"{name}: got {got}, want {expected}"
        print(f"extract_confidence [{name}]: {got:.6f} == expected {expected:.6f}")

    # Offline pre-check: what mean confidence the stored rx_stats imply
    cache = json.load(open(TOKEN_SCORES))
    agree = train_df["agreement_score"].to_numpy(dtype=float)
    sig = np.array([cache[i]["sigmoid_score"] for i in train_df["id"]], dtype=float)
    ref = json.load(open(JOBS_JSON))["rx_stats"]
    implied = 3 * ref["mean"] - agree.mean() - (1 - sig).mean()
    print(f"\nImplied original mean confidence from rx_stats: {implied:.4f} "
          "(regenerated mean should land near this)")
    print("\nVALIDATION PASSED")


def cmd_submit() -> None:
    train_df = load_train_df()
    assert len(train_df) == N_TRAIN_EXPECTED
    requests = build_requests(train_df)
    write_jsonl(requests, INPUT_JSONL)
    sha = hashlib.sha256(INPUT_JSONL.read_bytes()).hexdigest()
    print(f"Built {len(requests)} requests -> {INPUT_JSONL} (sha256 {sha[:16]}...)")

    client = get_client()
    with open(INPUT_JSONL, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "n_requests": len(requests),
        "model": MODEL,
        "strategy": STRATEGY,
        "input_sha256": sha,
        "created_at": datetime.now().isoformat(),
        "status": batch.status,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Batch {batch.id} submitted (status={batch.status}); state -> {STATE_PATH}")


def cmd_poll() -> None:
    state = json.load(open(STATE_PATH))
    client = get_client()
    batch = client.batches.retrieve(state["batch_id"])
    counts = getattr(batch, "request_counts", None)
    print(f"Batch {batch.id}: status={batch.status}"
          + (f"  completed={counts.completed}/{counts.total} failed={counts.failed}"
             if counts else ""))
    state["status"] = batch.status
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    if batch.status != "completed":
        if batch.status in ("failed", "expired", "cancelled"):
            print("Batch did not complete — resubmit with `submit`.")
        return

    raw = client.files.content(batch.output_file_id).text
    OUTPUT_JSONL.write_text(raw)
    print(f"Raw output -> {OUTPUT_JSONL}")
    if getattr(batch, "error_file_id", None):
        err = client.files.content(batch.error_file_id).text
        (STATE_DIR / "train_batch_errors.jsonl").write_text(err)
        print(f"WARNING: error file present -> {STATE_DIR / 'train_batch_errors.jsonl'}")

    confidences = {}
    n_fallback = 0
    for line in raw.strip().split("\n"):
        cid, conf = parse_batch_line(line)
        if conf is None:
            conf = 0.5
            n_fallback += 1
        confidences[cid] = float(conf)
    print(f"Parsed {len(confidences)} results ({n_fallback} without usable logprobs -> 0.5)")

    train_ids = set(load_train_df()["id"].astype(str))
    missing = train_ids - set(confidences)
    if missing:
        print(f"WARNING: {len(missing)} train ids absent from batch output -> 0.5")
        for i in missing:
            confidences[i] = 0.5
    confidences = {k: confidences[k] for k in sorted(confidences)}

    with open(CONF_OUT, "w") as f:
        json.dump(confidences, f, indent=2)
    print(f"Confidences -> {CONF_OUT}")
    rx_sanity_check(confidences)


def cmd_build_conf30() -> None:
    assert CONF_OUT.exists(), f"{CONF_OUT} missing — run `poll` after the batch completes"
    cmd = [sys.executable, str(SUBSET_BUILDER), "--confidences", str(CONF_OUT)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    out = ROOT / "results" / "smart_sampling" / "conf30_dpo.jsonl"
    n = sum(1 for _ in open(out))
    assert n == 1661, f"{out}: {n} lines, expected 1661"
    print(f"conf30 OK: {out} ({n} pairs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",
                    choices=["validate", "submit", "poll", "build-conf30"])
    args = ap.parse_args()
    {"validate": cmd_validate,
     "submit": cmd_submit,
     "poll": cmd_poll,
     "build-conf30": cmd_build_conf30}[args.command]()


if __name__ == "__main__":
    main()
