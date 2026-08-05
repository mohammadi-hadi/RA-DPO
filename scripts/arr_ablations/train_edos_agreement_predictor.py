"""Train a text->agreement regressor on the EDOS train split.

This is what makes the *deployable* R(x) regime available on EDOS: at
deployment there are no annotator labels, so beta's input has to be predicted
from text alone. Mirrors the EXIST-side regressor
(scripts/analysis/train_agreement_predictor_fresh.py and
compare_agreement_predictors.py) and defaults to the same adopted backbone so
the two tracks stay method-identical.

One difference is inherent to the corpus rather than the method: EDOS has
three annotators, so agreement takes two values (2/3, 1.0). The target is
therefore binary and the correlation reported against it is point-biserial;
the *predicted* score stays continuous, which is all R(x) consumes.

Usage:
    python scripts/arr_ablations/train_edos_agreement_predictor.py
    python scripts/arr_ablations/train_edos_agreement_predictor.py \
        --model cardiffnlp/twitter-roberta-base --tag twitter-roberta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.edos_loader import EDOSDataLoader  # noqa: E402
from ra_dpo.models.agreement_predictor import AgreementPredictor  # noqa: E402

OUT_DIR = ROOT / "results" / "edos_pipeline" / "predicted_agreement"


def point_biserial(pred: np.ndarray, target: np.ndarray) -> dict:
    """Pearson r against a two-valued target is the point-biserial rho."""
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(pred, target)
    rho, p_s = spearmanr(pred, target)
    return {"pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(p_s),
            "mae": float(np.mean(np.abs(pred - target)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cardiffnlp/twitter-xlm-roberta-base",
                    help="HF backbone (default: the adopted EXIST one)")
    ap.add_argument("--tag", default="twitter-xlmr-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "edos_agreement_predictor" / args.tag
    model_dir.mkdir(parents=True, exist_ok=True)

    loader = EDOSDataLoader()
    # AgreementPredictor reads its text from a column named "tweet" (the EXIST
    # schema); EDOS calls the same field "text". Alias it rather than touching
    # the shared predictor, which the EXIST track also uses.
    def split(name):
        df = loader.get_split(name).copy()
        df["tweet"] = df["text"]
        return df

    train_df = split("train")
    val_df = split("dev")
    test_df = split("test")
    print(f"train={len(train_df)} dev={len(val_df)} test={len(test_df)}")

    levels = sorted(set(train_df["agreement_score"]))
    print(f"agreement levels on EDOS: {levels}  "
          f"(EXIST has four; the target here is binary)")

    predictor = AgreementPredictor(model_name=args.model)
    predictor.train(
        train_df=train_df, val_df=val_df,
        output_dir=str(model_dir),
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=0,
    )

    # predict() takes a list of strings, not a DataFrame.
    pred = np.asarray(predictor.predict(test_df["tweet"].tolist()), dtype=float)
    target = test_df["agreement_score"].to_numpy(dtype=float)
    assert len(pred) == len(target) == len(test_df)

    metrics = point_biserial(pred, target)
    metrics.update({"model": args.model, "tag": args.tag,
                    "n_test": int(len(test_df)),
                    "agreement_levels": [float(x) for x in levels],
                    "epochs": args.epochs})

    npy_path = OUT_DIR / f"pred_agreement_{args.tag}.npy"
    np.save(npy_path, pred)
    np.save(OUT_DIR / "true_agreement.npy", target)
    with open(OUT_DIR / f"metrics_{args.tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== EDOS agreement regressor ===")
    for k in ("pearson_r", "spearman_rho", "mae"):
        print(f"  {k}: {metrics[k]:.4f}")
    print(f"saved: {npy_path.relative_to(ROOT)}")
    print("feed it to eval-oof with --agreement predicted "
          f"--pred-agreement {npy_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
