"""
Evaluate the freshly-trained gpt-4o SFT model on the 692-sample EN+ES test set
with the structured prompt. Reuses sigmoid_scores and agreements from existing
gpt-4o result files (these are test-data properties, not model properties).

Writes results to:
  results/final_reliability_3factor/gpt-4o_SFT.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.pipeline.prompts import PromptBuilder
from ra_dpo.utils.metrics import compute_metrics

from openai import OpenAI

OUT_DIR = ROOT / "results" / "final_reliability_3factor"
REF_FILE = OUT_DIR / "gpt-4o_base.json"  # source of sigmoid_scores + agreements alignment


def load_test_and_ref():
    """Load test set and the reference gpt-4o_base.json for sigmoid/agreement alignment."""
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    _, _, test_df = loader.create_train_val_test_split(df)
    ref = json.load(open(REF_FILE))
    assert len(test_df) == 692, f"expected 692 test samples, got {len(test_df)}"
    assert len(ref["per_instance"]["sigmoid_scores"]) == 692
    return test_df, ref


def run(model_id: str, out_label: str, training_pairs: int | None = None):
    test_df, ref = load_test_and_ref()
    pb = PromptBuilder()
    client = OpenAI()

    predictions = []
    confidences = []

    # Iterate in the same order the reference file was written (assumed same as the DataLoader's order)
    iter_df = test_df.reset_index(drop=True)
    for i, row in tqdm(iter_df.iterrows(), total=len(iter_df), desc=f"eval {out_label}"):
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
                print(f"  retry {attempt+1}/3: {e}")
                time.sleep(2 ** attempt)

        pred = pb.parse_prediction(pred_text, lang)
        predictions.append(pred)
        confidences.append(conf)

    # Align sigmoid_scores and agreements from the reference file (same ordering, same data)
    sigmoid_scores = ref["per_instance"]["sigmoid_scores"]
    agreements = ref["per_instance"]["agreements"]
    assert len(sigmoid_scores) == len(predictions) == 692

    true_labels = iter_df["majority_label"].tolist()
    correct = [p == t for p, t in zip(predictions, true_labels)]

    # Standard metrics
    sm = compute_metrics(true_labels, predictions)
    sm["avg_confidence"] = float(np.mean(confidences))

    # R(x) with weights fit on the full data (mirrors report's Table 1 method; used here only for comparability)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([confidences, agreements, 1 - np.asarray(sigmoid_scores)])
    y = np.asarray([int(c) for c in correct])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
    a = np.abs(lr.coef_[0]); a = a / a.sum()
    alpha, beta, gamma = [float(x) for x in a]
    r_scores = alpha * np.asarray(confidences) + beta * np.asarray(agreements) + gamma * (1 - np.asarray(sigmoid_scores))

    # Coverage-accuracy at matched levels
    acc_at = {}
    for cov in [1.00, 0.90, 0.80, 0.60, 0.50]:
        k = max(1, int(round(len(r_scores) * cov)))
        top_idx = np.argsort(-r_scores)[:k]
        acc_at[f"acc@{int(cov*100)}%"] = float(np.asarray(correct)[top_idx].mean())

    out = {
        "model": out_label,
        "model_id": model_id,
        "training_pairs": training_pairs,
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
    }
    # Filename derived from label so different runs don't overwrite each other.
    safe = (out_label.replace(" ", "_").replace("(", "").replace(")", "")
                     .replace("/", "_").replace("%", "pct"))
    out_path = OUT_DIR / f"{safe}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"F1: {sm['f1_macro']:.4f}   Acc: {sm['accuracy']:.4f}")
    print(f"Weights (leaky): α={alpha:.3f} β={beta:.3f} γ={gamma:.3f}")
    print(f"acc@coverage: {acc_at}")
    return out


if __name__ == "__main__":
    model_id = os.environ.get("SFT_MODEL_ID")
    if not model_id:
        print("set SFT_MODEL_ID env var to the fine_tuned_model id")
        sys.exit(1)
    run(model_id, "gpt-4o (SFT)", training_pairs=5535)
