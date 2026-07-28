"""
EDOS (SemEval-2023 Task 10) Dataset Loading Utilities.

Loads the "Explainable Detection of Online Sexism" dataset (Kirk et al., 2023)
in a shape that mirrors ``EXISTDataLoader`` where sensible, so the RA-DPO
pipeline components (preference pairs, R(x), coverage-accuracy) can be reused
on a second dataset.

Files (both required, already downloaded to ``data/edos/``):
  - ``edos_labelled_individual_annotations.csv`` — 60,000 rows
    (20,000 items x 3 annotators). Columns: rewire_id, text, annotator,
    label_sexist, label_category, label_vector, split.
  - ``edos_labelled_aggregated.csv`` — 20,000 rows, one per item, with the
    adjudicated gold label. Same split column (train 14,000 / dev 2,000 /
    test 4,000).

Output DataFrame columns:
  rewire_id, text, lang ('en'), labels_task1 (list of 3 'YES'/'NO'),
  gold_label ('YES'/'NO'), majority_label ('YES'/'NO'),
  agreement_score (max label count / 3 -> {2/3, 1.0}), split.

IMPORTANT CAVEAT — gold vs majority vote:
  Unlike EXIST 2023 (where our pipeline's ground truth IS the majority vote
  of 6 annotators), the EDOS gold label is *expert-adjudicated*: items where
  the 3 crowd annotators disagreed were re-labelled by expert reviewers, and
  the experts sometimes overruled the crowd. Verified on the shipped CSVs:
    - majority_vote(3 annotators) != gold for 946 / 20,000 items (4.73%).
    - 926 of those are 2/3-split items (out of 4,444 total 2/3 items, i.e.
      the experts overruled the crowd majority in ~20.8% of disputed items).
    - 20 items are UNANIMOUS (3/3) yet the gold label still differs
      (expert override of the full crowd).
  The pipeline uses ``gold_label`` as ground truth (chosen response in
  preference pairs, correctness at evaluation) to match the official task,
  and ``agreement_score`` from the raw individual annotations as the
  R(x) agreement component.

Text canonicalisation: the two CSVs disagree on ``text`` for a handful of
items (16 observed); the *aggregated* file's text is used as canonical.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

# Official label strings -> pipeline label strings.
LABEL_MAP = {"sexist": "YES", "not sexist": "NO"}

DEFAULT_DATA_DIR = Path("data/edos")
INDIVIDUAL_CSV = "edos_labelled_individual_annotations.csv"
AGGREGATED_CSV = "edos_labelled_aggregated.csv"

EXPECTED_N_ITEMS = 20_000
EXPECTED_N_ANNOTATIONS = 60_000
EXPECTED_SPLITS = {"train": 14_000, "dev": 2_000, "test": 4_000}
N_ANNOTATORS = 3


def majority_vote(labels) -> str:
    """Majority label from a list of 'YES'/'NO' annotator labels.

    With 3 annotators a strict majority always exists (no ties).
    """
    if not labels:
        return "NO"
    return Counter(labels).most_common(1)[0][0]


def agreement_score(labels) -> float:
    """max(label counts) / n_labels — for 3 annotators this is 2/3 or 1.0."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


class EDOSDataLoader:
    """Loader for the EDOS dataset with multi-annotator label handling.

    Mirrors the parts of ``EXISTDataLoader`` the RA-DPO pipeline relies on
    (``to_dataframe``, ``get_statistics``); the train/val/test split is the
    OFFICIAL one from the ``split`` column, so there is no
    ``create_train_val_test_split`` — use :meth:`get_split` instead.
    """

    def __init__(self, data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.individual_path = self.data_dir / INDIVIDUAL_CSV
        self.aggregated_path = self.data_dir / AGGREGATED_CSV
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Join the two CSVs into one row per item (cached)."""
        if self._df is not None:
            return self._df

        ind = pd.read_csv(self.individual_path)
        agg = pd.read_csv(self.aggregated_path)

        if len(ind) != EXPECTED_N_ANNOTATIONS:
            raise ValueError(
                f"{self.individual_path}: {len(ind)} rows, "
                f"expected {EXPECTED_N_ANNOTATIONS}")
        if len(agg) != EXPECTED_N_ITEMS:
            raise ValueError(
                f"{self.aggregated_path}: {len(agg)} rows, "
                f"expected {EXPECTED_N_ITEMS}")

        per_item = ind.groupby("rewire_id")["label_sexist"].apply(list)
        bad_counts = per_item[per_item.str.len() != N_ANNOTATORS]
        if len(bad_counts):
            raise ValueError(
                f"{len(bad_counts)} items do not have exactly "
                f"{N_ANNOTATORS} annotations")

        df = agg[["rewire_id", "text", "label_sexist", "split"]].copy()
        df["labels_task1"] = df["rewire_id"].map(
            lambda rid: [LABEL_MAP[l] for l in per_item[rid]])
        df["gold_label"] = df["label_sexist"].map(LABEL_MAP)
        df["majority_label"] = df["labels_task1"].apply(majority_vote)
        df["agreement_score"] = df["labels_task1"].apply(agreement_score)
        df["lang"] = "en"
        df = df.drop(columns=["label_sexist"])
        df = df[["rewire_id", "text", "lang", "labels_task1", "gold_label",
                 "majority_label", "agreement_score", "split"]]

        split_sizes = df["split"].value_counts().to_dict()
        if split_sizes != EXPECTED_SPLITS:
            raise ValueError(
                f"split sizes {split_sizes} != expected {EXPECTED_SPLITS}")

        self._df = df.reset_index(drop=True)
        return self._df

    def get_split(self, split: str,
                  df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return one official split ('train' | 'dev' | 'test')."""
        if split not in EXPECTED_SPLITS:
            raise KeyError(f"unknown split {split!r}; "
                           f"expected one of {sorted(EXPECTED_SPLITS)}")
        if df is None:
            df = self.to_dataframe()
        return df[df["split"] == split].reset_index(drop=True)

    def get_statistics(self) -> Dict:
        """Dataset statistics incl. the majority-vs-gold mismatch rate."""
        df = self.to_dataframe()
        mismatch = df["majority_label"] != df["gold_label"]
        stats = {
            "total_instances": len(df),
            "split_sizes": df["split"].value_counts().to_dict(),
            "gold_label_distribution": df["gold_label"].value_counts().to_dict(),
            "agreement_distribution": {
                round(k, 4): int(v)
                for k, v in df["agreement_score"].value_counts().items()},
            "avg_agreement": float(df["agreement_score"].mean()),
            "majority_vs_gold_mismatch": int(mismatch.sum()),
            "majority_vs_gold_mismatch_pct": float(100 * mismatch.mean()),
            "mismatch_by_agreement": {
                round(k, 4): int(v)
                for k, v in df.loc[mismatch, "agreement_score"]
                              .value_counts().items()},
        }
        for split in EXPECTED_SPLITS:
            sub = df[df["split"] == split]
            stats[f"{split}_gold_yes_frac"] = float(
                (sub["gold_label"] == "YES").mean())
            stats[f"{split}_unanimous_frac"] = float(
                (sub["agreement_score"] == 1.0).mean())
        return stats


if __name__ == "__main__":
    loader = EDOSDataLoader()
    df = loader.to_dataframe()
    print(f"Loaded {len(df)} instances")
    print("\nStatistics:")
    for key, value in loader.get_statistics().items():
        print(f"  {key}: {value}")
