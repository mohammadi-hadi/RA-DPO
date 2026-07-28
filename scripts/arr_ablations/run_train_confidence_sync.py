"""Synchronous fallback for the train-split confidence pass.

The Batch API rejected the full submission (org limit: 90,000 enqueued
tokens for gpt-4o), so this runner sends the exact same request bodies
from results/arr_ablations/train_confidence/train_batch_input.jsonl as
plain chat completions with bounded concurrency and retry-on-429.

Writes the same outputs the batch `poll` path would have produced:
  results/smart_sampling/train_confidences_gpt4o.json  {id_EXIST: conf}
plus the rx_stats sanity check against results/smart_sampling/jobs.json.

Resume-safe: raw responses are checkpointed to sync_responses.jsonl and
already-answered ids are skipped on restart.
"""
from __future__ import annotations
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.arr_ablations.run_train_confidence_batch import (  # noqa: E402
    load_env_key, get_client, extract_confidence, rx_sanity_check,
)

STATE_DIR = ROOT / "results" / "arr_ablations" / "train_confidence"
INPUT_PATH = STATE_DIR / "train_batch_input.jsonl"
RAW_PATH = STATE_DIR / "sync_responses.jsonl"
OUT_PATH = ROOT / "results" / "smart_sampling" / "train_confidences_gpt4o.json"

CONCURRENCY = 6
MAX_RETRIES = 8

_write_lock = threading.Lock()


def call_one(client, req: dict) -> tuple[str, float]:
    body = req["body"]
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**body)
            top = resp.choices[0].logprobs.content[0].top_logprobs
            entry = {"top_logprobs": [
                {"token": t.token, "logprob": t.logprob} for t in top]}
            return req["custom_id"], extract_confidence(entry)
        except Exception as e:  # noqa: BLE001 — retry transient API errors
            name = type(e).__name__
            if attempt == MAX_RETRIES - 1:
                raise
            if "RateLimit" in name or "APIConnection" in name or "Timeout" in name \
                    or "InternalServer" in name or "APIStatus" in name:
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60)
            else:
                raise
    raise RuntimeError("unreachable")


def main() -> None:
    load_env_key()
    client = get_client()

    requests = [json.loads(l) for l in open(INPUT_PATH)]
    done: dict[str, float] = {}
    if RAW_PATH.exists():
        for line in open(RAW_PATH):
            rec = json.loads(line)
            done[rec["custom_id"]] = rec["confidence"]
    todo = [r for r in requests if r["custom_id"] not in done]
    print(f"{len(requests)} total, {len(done)} cached, {len(todo)} to run "
          f"(concurrency {CONCURRENCY})")

    t0 = time.time()
    raw_f = open(RAW_PATH, "a")
    n_done = 0
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        futures = {ex.submit(call_one, client, r): r["custom_id"] for r in todo}
        for fut in as_completed(futures):
            cid, conf = fut.result()
            with _write_lock:
                raw_f.write(json.dumps({"custom_id": cid, "confidence": conf}) + "\n")
                raw_f.flush()
            done[cid] = conf
            n_done += 1
            if n_done % 250 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
                print(f"  {n_done}/{len(todo)}  ({rate:.1f}/s, ~{eta:.0f} min left)")
    raw_f.close()

    assert len(done) == len(requests), f"{len(done)} != {len(requests)}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(done, open(OUT_PATH, "w"), indent=1)
    print(f"wrote {OUT_PATH} ({len(done)} confidences)")
    rx_sanity_check(done)


if __name__ == "__main__":
    main()
