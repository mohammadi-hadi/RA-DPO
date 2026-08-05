"""Quickly retrain the agreement predictor (mBERT regression) on the train split
and predict agreement for the test split. Saves predictions to a .npy for reuse."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_dpo.data.data_loader import EXISTDataLoader, majority_vote, agreement_score
from ra_dpo.models.agreement_predictor import AgreementPredictor

OUT_DIR = ROOT / "results" / "unified_gpt4o" / "predicted_agreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "agreement_predictor_fresh"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    loader = EXISTDataLoader(str(ROOT / "EXIST2023_training.json"))
    df = loader.to_dataframe()
    df["majority_label"] = df["labels_task1"].apply(majority_vote)
    df["agreement_score"] = df["labels_task1"].apply(agreement_score)
    train_df, val_df, test_df = loader.create_train_val_test_split(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    predictor = AgreementPredictor(model_name="bert-base-multilingual-cased")
    predictor.train(
        train_df=train_df, val_df=val_df,
        output_dir=str(MODEL_DIR),
        num_epochs=3, batch_size=32, learning_rate=3e-5,
        early_stopping_patience=0,  # skip callback
    )

    # Evaluate on test
    metrics = predictor.evaluate(test_df)
    print(f"\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Predict on test set (in the order of the saved per_instance arrays)
    test_df = test_df.reset_index(drop=True)
    preds = predictor.predict(test_df["tweet"].tolist())
    true = test_df["agreement_score"].to_numpy()
    print(f"\npred mean={preds.mean():.3f} std={preds.std():.3f}")
    print(f"true mean={true.mean():.3f} std={true.std():.3f}")
    print(f"Pearson r = {np.corrcoef(true, preds)[0, 1]:.3f}")

    np.save(OUT_DIR / "pred_agreement.npy", preds)
    np.save(OUT_DIR / "true_agreement.npy", true)
    print(f"\nSaved: {OUT_DIR / 'pred_agreement.npy'}")


if __name__ == "__main__":
    main()
