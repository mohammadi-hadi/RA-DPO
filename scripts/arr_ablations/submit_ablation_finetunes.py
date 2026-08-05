"""Submit + evaluate the component-ablation DPO fine-tunes (conf30/uncert30/agree30).

Each variant is a 1,661-pair top-30% subset ranked by ONE R(x) component
(results/smart_sampling/<variant>_dpo.jsonl, built by
scripts/arr_ablations/build_component_ablation_subsets.py). Jobs use the same
recipe as the paper's Smart-30% job: base model gpt-4o-2024-08-06, DPO method
with beta 0.1, remaining hyperparameters at API defaults.

Subcommands:
  submit --variant {conf30,uncert30,agree30} [--dry-run]
          Upload the JSONL and create the DPO fine-tune job (suffix
          sexism-<variant>). Job ids -> results/arr_ablations/ablation_ft_jobs.json.
          --dry-run prints the exact files.create / jobs.create kwargs
          without calling the API.
  status  List the state of every recorded job.
  inspect-ref
          Retrieve the paper's Smart-30% job (and Smart-10%) from the API and
          print their method/hyperparameters — run once before the first real
          submit to confirm parity.
  eval --variant <v> [--model-id ft:...]
          Evaluate the fine-tuned model on the 692-item test set exactly like
          scripts/evaluate_sft_4o.py (structured prompt, agreements +
          sigmoid_scores reused from gpt-4o_base.json), plus 5-fold OOF R(x)
          coverage-accuracy (method of scripts/analysis/fix_leakage_oof.py).
          Writes results/final_reliability_3factor/gpt-4o_<variant>.json.

Usage:
    python scripts/arr_ablations/submit_ablation_finetunes.py submit --variant conf30
    python scripts/arr_ablations/submit_ablation_finetunes.py status
    python scripts/arr_ablations/submit_ablation_finetunes.py eval --variant conf30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ra_dpo.pipeline.prompts import PromptBuilder
from ra_dpo.utils.metrics import compute_metrics

VARIANTS = ("conf30", "uncert30", "agree30")
BASE_MODEL = "gpt-4o-2024-08-06"
TRAINING_PAIRS = 1661  # top 30% of the 5,536 train pairs

DATA_DIR = ROOT / "results" / "smart_sampling"
JOBS_PATH = ROOT / "results" / "arr_ablations" / "ablation_ft_jobs.json"
OUT_DIR = ROOT / "results" / "final_reliability_3factor"

# Paper reference jobs (results/smart_sampling/jobs.json + unified_gpt4o/new_jobs.json)
REF_JOBS = {
    "smart30 (paper)": "ftjob-yZtTTO872XW5yIx2F6drVqyt",
    "smart10 (paper)": "ftjob-SfxnijkShoLr4Awr24cCAqhH",
}

COVERAGE_LEVELS = [1.00, 0.90, 0.80, 0.60, 0.50]
N_FOLDS = 5
SEED = 42


def load_env_key() -> None:
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


def read_jobs() -> dict:
    if JOBS_PATH.exists():
        return json.load(open(JOBS_PATH))
    return {}


def write_jobs(jobs: dict) -> None:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_PATH, "w") as f:
        json.dump(jobs, f, indent=2)


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def validate_training_file(path: Path) -> int:
    """Every line must be a well-formed DPO preference record."""
    n = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            assert set(rec) == {"input", "preferred_output", "non_preferred_output"}, \
                f"{path.name}: unexpected top-level keys {sorted(rec)}"
            msgs = rec["input"]["messages"]
            assert [m["role"] for m in msgs] == ["system", "user"]
            for key in ("preferred_output", "non_preferred_output"):
                assert rec[key][0]["role"] == "assistant"
                assert rec[key][0]["content"] in ("YES", "NO")
            n += 1
    assert n == TRAINING_PAIRS, f"{path.name}: {n} pairs, expected {TRAINING_PAIRS}"
    return n


def build_job_kwargs(training_file_id: str, variant: str, beta: float) -> dict:
    """Exact recipe of the paper's smart30 job (verified via inspect-ref):
    DPO with batch_size 2, beta 0.1, learning_rate_multiplier 1.0, n_epochs 2."""
    return {
        "training_file": training_file_id,
        "model": BASE_MODEL,
        "method": {
            "type": "dpo",
            "dpo": {"hyperparameters": {
                "batch_size": 2,
                "beta": beta,
                "learning_rate_multiplier": 1.0,
                "n_epochs": 2,
            }},
        },
        "suffix": f"sexism-{variant}",
    }


def cmd_submit(variant: str, beta: float, dry_run: bool) -> None:
    path = DATA_DIR / f"{variant}_dpo.jsonl"
    assert path.exists(), f"{path} missing — build it first"
    n = validate_training_file(path)
    print(f"{path.name}: {n} DPO pairs validated")

    kwargs = build_job_kwargs("<training_file_id>", variant, beta)
    if dry_run:
        print("\nDRY RUN — no API calls made.")
        print(f"client.files.create(file=open('{path}', 'rb'), purpose='fine-tune')")
        print("client.fine_tuning.jobs.create(**kwargs) with kwargs =")
        print(json.dumps(kwargs, indent=2))
        return

    jobs = read_jobs()
    if variant in jobs and jobs[variant].get("job_id"):
        print(f"NOTE: {variant} already has job {jobs[variant]['job_id']} — "
              "delete its entry from ablation_ft_jobs.json to resubmit.")
        return

    client = get_client()
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    kwargs["training_file"] = uploaded.id
    job = client.fine_tuning.jobs.create(**kwargs)
    jobs[variant] = {
        "file_id": uploaded.id,
        "job_id": job.id,
        "base_model": BASE_MODEL,
        "beta": beta,
        "suffix": f"sexism-{variant}",
        "n_pairs": n,
        "status": job.status,
        "fine_tuned_model": None,
        "created_at": datetime.now().isoformat(),
    }
    write_jobs(jobs)
    print(f"Job {job.id} created (status={job.status}); state -> {JOBS_PATH}")


# ---------------------------------------------------------------------------
# status / inspect-ref
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    jobs = read_jobs()
    if not jobs:
        print(f"No jobs recorded in {JOBS_PATH}")
        return
    client = get_client()
    for variant, meta in jobs.items():
        job = client.fine_tuning.jobs.retrieve(meta["job_id"])
        meta["status"] = job.status
        meta["fine_tuned_model"] = job.fine_tuned_model
        err = getattr(job, "error", None)
        err_msg = f"  error={err.message}" if err and getattr(err, "message", None) else ""
        print(f"{variant:<10} {meta['job_id']}  status={job.status}  "
              f"model={job.fine_tuned_model}{err_msg}")
    write_jobs(jobs)


def cmd_inspect_ref() -> None:
    client = get_client()
    for label, job_id in REF_JOBS.items():
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
        except Exception as e:
            print(f"{label}: retrieve failed: {e}")
            continue
        print(f"\n{label}  ({job_id})")
        print(f"  model:            {job.model}")
        print(f"  status:           {job.status}")
        print(f"  fine_tuned_model: {job.fine_tuned_model}")
        method = getattr(job, "method", None)
        if method is not None:
            print("  method:", json.dumps(method.model_dump()
                  if hasattr(method, "model_dump") else method, indent=4, default=str))
        hp = getattr(job, "hyperparameters", None)
        if hp is not None:
            print("  hyperparameters:", hp)


# ---------------------------------------------------------------------------
# eval (mirrors scripts/evaluate_sft_4o.py + OOF R(x) from fix_leakage_oof.py)
# ---------------------------------------------------------------------------

def oof_rx(conf: np.ndarray, agree: np.ndarray, sig: np.ndarray,
           correct: np.ndarray) -> tuple:
    """5-fold out-of-fold R(x); method identical to fix_leakage_oof.py."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X = np.column_stack([conf, agree, 1.0 - sig])
    y = correct.astype(int)
    oof_r = np.zeros(len(y))
    fold_weights = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr_idx])
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(scaler.transform(X[tr_idx]), y[tr_idx])
        abs_coefs = np.abs(lr.coef_[0])
        total = abs_coefs.sum()
        w = abs_coefs / total if total > 0 else np.array([1/3, 1/3, 1/3])
        fold_weights.append(w)
        oof_r[te_idx] = (w[0] * conf[te_idx] + w[1] * agree[te_idx]
                         + w[2] * (1.0 - sig[te_idx]))
    return oof_r, np.mean(fold_weights, axis=0)


def acc_at_coverage(r_scores: np.ndarray, correct: np.ndarray) -> dict:
    out = {}
    for cov in COVERAGE_LEVELS:
        k = max(1, int(round(len(r_scores) * cov)))
        top_idx = np.argsort(-r_scores)[:k]
        out[f"acc@{int(cov * 100)}%"] = float(correct[top_idx].mean())
    return out


def cmd_eval(variant: str, model_id: str | None) -> None:
    if model_id is None:
        jobs = read_jobs()
        meta = jobs.get(variant) or {}
        model_id = meta.get("fine_tuned_model")
        if not model_id:
            client = get_client()
            job = client.fine_tuning.jobs.retrieve(meta["job_id"])
            if job.status != "succeeded" or not job.fine_tuned_model:
                raise SystemExit(f"{variant}: job {meta.get('job_id')} status="
                                 f"{job.status} — not ready to evaluate")
            model_id = job.fine_tuned_model
            meta["fine_tuned_model"] = model_id
            meta["status"] = job.status
            write_jobs(jobs)
    print(f"Evaluating {variant}: {model_id}")

    load_env_key()
    from evaluate_sft_4o import load_test_and_ref  # scripts/evaluate_sft_4o.py
    from openai import OpenAI
    from tqdm import tqdm

    test_df, ref = load_test_and_ref()
    pb = PromptBuilder()
    client = OpenAI()

    predictions, confidences = [], []
    iter_df = test_df.reset_index(drop=True)
    for _, row in tqdm(iter_df.iterrows(), total=len(iter_df),
                       desc=f"eval {variant}"):
        lang = row["lang"]
        system = pb.get_system_prompt("structured", lang)
        user = pb.format_user_prompt(row["tweet"], lang, "structured")

        conf = 0.5
        pred_text = "NO"
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=10,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=3,
                )
                if r.choices[0].logprobs and r.choices[0].logprobs.content:
                    conf = min(float(np.exp(r.choices[0].logprobs.content[0].logprob)), 1.0)
                pred_text = r.choices[0].message.content or ""
                break
            except Exception as e:
                print(f"  retry {attempt + 1}/3: {e}")
                time.sleep(2 ** attempt)

        predictions.append(pb.parse_prediction(pred_text, lang))
        confidences.append(conf)

    # Reuse agreements + sigmoid_scores from gpt-4o_base.json (test-data
    # properties, identical ordering) — same as evaluate_sft_4o.py.
    sigmoid_scores = ref["per_instance"]["sigmoid_scores"]
    agreements = ref["per_instance"]["agreements"]
    assert len(sigmoid_scores) == len(predictions) == 692

    true_labels = iter_df["majority_label"].tolist()
    correct = np.array([p == t for p, t in zip(predictions, true_labels)])
    conf_a = np.asarray(confidences, dtype=float)
    agree_a = np.asarray(agreements, dtype=float)
    sig_a = np.asarray(sigmoid_scores, dtype=float)

    sm = compute_metrics(true_labels, predictions)
    sm["avg_confidence"] = float(conf_a.mean())

    # Full-fit weights + coverage curve (identical to evaluate_sft_4o.py)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([conf_a, agree_a, 1 - sig_a])
    y = correct.astype(int)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
    a = np.abs(lr.coef_[0]); a = a / a.sum()
    alpha, beta, gamma = [float(x) for x in a]
    r_scores = alpha * conf_a + beta * agree_a + gamma * (1 - sig_a)
    acc_at = acc_at_coverage(r_scores, correct)

    # Leakage-free OOF R(x) (method of scripts/analysis/fix_leakage_oof.py)
    oof_r, oof_w = oof_rx(conf_a, agree_a, sig_a, correct)
    acc_at_oof = acc_at_coverage(oof_r, correct)

    out = {
        "model": f"gpt-4o ({variant} DPO)",
        "model_id": model_id,
        "training_pairs": TRAINING_PAIRS,
        "standard_metrics": sm,
        "optimized_weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "accuracy_at_coverage": acc_at,
        "per_instance": {
            "predictions": predictions,
            "confidences": [float(c) for c in confidences],
            "agreements": agreements,
            "sigmoid_scores": sigmoid_scores,
            "correct": [bool(c) for c in correct],
        },
        "n_samples": len(predictions),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oof": {
            "weights_oof_mean": {"alpha": float(oof_w[0]),
                                 "beta": float(oof_w[1]),
                                 "gamma": float(oof_w[2])},
            "accuracy_at_coverage_oof": acc_at_oof,
            "n_folds": N_FOLDS,
            "seed": SEED,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"gpt-4o_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"F1: {sm['f1_macro']:.4f}   Acc: {sm['accuracy']:.4f}")
    print(f"Weights (full-fit): a={alpha:.3f} b={beta:.3f} g={gamma:.3f}")
    print(f"acc@coverage (full-fit): {acc_at}")
    print(f"acc@coverage (OOF):      {acc_at_oof}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--variant", required=True, choices=VARIANTS)
    p_submit.add_argument("--beta", type=float, default=0.1)
    p_submit.add_argument("--dry-run", action="store_true")

    sub.add_parser("status")
    sub.add_parser("inspect-ref")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--variant", required=True, choices=VARIANTS)
    p_eval.add_argument("--model-id", default=None,
                        help="Override the ft: model id (else read from job state)")

    args = ap.parse_args()
    if args.command == "submit":
        cmd_submit(args.variant, args.beta, args.dry_run)
    elif args.command == "status":
        cmd_status()
    elif args.command == "inspect-ref":
        cmd_inspect_ref()
    elif args.command == "eval":
        cmd_eval(args.variant, args.model_id)


if __name__ == "__main__":
    main()
