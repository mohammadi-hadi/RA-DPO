"""Master orchestrator for the local pipeline.

Runs every stage in order, invoking validate.py before AND after each.
Respects previous checkpoints — stages already in 'ok' state are skipped
unless --force is passed.

Usage:
    python scripts/local_pipeline/run_all.py                 # full run with gate
    python scripts/local_pipeline/run_all.py --only 02 04    # only these stages
    python scripts/local_pipeline/run_all.py --force         # ignore checkpoints
    python scripts/local_pipeline/run_all.py --dry-run       # report plan, run nothing
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = [
    ("01", "stage_01_preflight.py",      "preflight"),
    ("02", "stage_02_baseline_inference.py", "baseline inference"),
    ("04", "stage_04_sft_lora.py",       "SFT (LoRA)"),
    ("05", "stage_05_dpo_variants.py",   "DPO variants"),
    ("06", "stage_06_eval_oof.py",       "evaluate + OOF R(x)"),
    ("07", "stage_07_unified_tables.py", "unified tables"),
]
VALIDATOR = ROOT / "scripts" / "local_pipeline" / "validate.py"
CKPT = ROOT / "results" / "local_pipeline" / "checkpoints.json"


def load_ckpt() -> dict:
    return json.load(open(CKPT)) if CKPT.exists() else {}


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_validate(lenient: bool = False) -> int:
    """Validator exits non-zero on failure by default; --lenient suppresses that."""
    cmd = [sys.executable, str(VALIDATOR)]
    if lenient:
        cmd.append("--lenient")
    return run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these stage numbers (e.g. 02 04)")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoints and rerun")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="abort the run if the validator reports any failure "
                         "(warnings never abort)")
    args = ap.parse_args()

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] in args.only]

    print("=== local pipeline plan ===")
    ckpt = load_ckpt()
    for num, script, label in stages:
        key = f"stage_{num}_" + script.replace("stage_" + num + "_", "").replace(".py", "")
        done = any(k.startswith(f"stage_{num}_") and v.get("status") == "ok"
                   for k, v in ckpt.items())
        tag = "SKIP" if done and not args.force else "RUN"
        print(f"  [{tag}] stage {num} — {label}")

    if args.dry_run:
        return

    # Initial validator
    print("\n>>> initial validator")
    rc = run_validate()
    if rc != 0:
        if args.strict:
            print("validator failed — abort")
            sys.exit(rc)
        print("validator reported failures — continuing (pass --strict to abort)")

    for num, script, label in stages:
        ckpt = load_ckpt()
        done = any(k.startswith(f"stage_{num}_") and v.get("status") == "ok"
                   for k, v in ckpt.items())
        if done and not args.force:
            print(f"[{num}] already done — skip")
            continue
        t0 = time.time()
        rc = run([sys.executable, str(ROOT / "scripts" / "local_pipeline" / script)])
        print(f"[{num}] {label}  exit={rc}  wall={time.time() - t0:.1f}s")
        if rc != 0:
            print("stage failed — abort")
            sys.exit(rc)
        # Post-stage validator
        print(f">>> validator after stage {num}")
        rc_val = run_validate()
        if rc_val != 0:
            if args.strict:
                print("post-stage validator failed — abort")
                sys.exit(rc_val)
            print("post-stage validator reported failures — continuing "
                  "(pass --strict to abort)")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
