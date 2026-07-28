"""
Results manager with atomic saves, resume support, and report generation.

Every experiment result is saved immediately after completion as a JSON file.
An index.json tracks all completed experiments for fast resume checks.
"""

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ResultsManager:
    """
    Manages experiment results with incremental, atomic saves.

    Storage layout::

        {base_dir}/
            index.json
            openai/{model}/{strategy}/{scenario}/{lang}.json
            local/{model}/{strategy}/{scenario}/{lang}.json
            training/{model}/{method}.json
            efficiency/{model}/{fraction}/{sampling}.json
            tables/
            plots/
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._load_index()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index: Dict[str, Any] = json.load(f)
        else:
            self.index = {}

    def _save_index(self):
        """Save index with file locking for concurrent access."""
        lock_path = self.index_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Re-read to merge any concurrent writes
                if self.index_path.exists():
                    with open(self.index_path) as f:
                        disk_index = json.load(f)
                    disk_index.update(self.index)
                    self.index = disk_index
                tmp = self.index_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(self.index, f, indent=2)
                tmp.rename(self.index_path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def exists(self, key: str) -> bool:
        """Check if an experiment result already exists."""
        return key in self.index

    def save(self, key: str, result: Dict):
        """Save a result atomically and update the index."""
        parts = key.strip("/").split("/")
        file_path = self.base_dir.joinpath(*parts[:-1], f"{parts[-1]}.json")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = self._make_serializable(result)

        tmp = file_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(serializable, f, indent=2)
        tmp.rename(file_path)

        self.index[key] = {
            "file": str(file_path.relative_to(self.base_dir)),
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": self._extract_summary(result),
        }
        self._save_index()

    def load(self, key: str) -> Dict:
        """Load a specific result by key."""
        if key not in self.index:
            raise KeyError(f"No result for key: {key}")
        rel = self.index[key]["file"]
        with open(self.base_dir / rel) as f:
            return json.load(f)

    def load_all(self, prefix: str = "") -> Dict[str, Dict]:
        """Load all results matching a key prefix."""
        results = {}
        for key in self.index:
            if key.startswith(prefix):
                try:
                    results[key] = self.load(key)
                except Exception:
                    pass
        return results

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all experiment keys matching a prefix."""
        return [k for k in sorted(self.index) if k.startswith(prefix)]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, prefix: str = "") -> pd.DataFrame:
        """Flatten all results matching prefix into a comparison DataFrame."""
        rows = []
        for key in sorted(self.index):
            if not key.startswith(prefix):
                continue
            summary = self.index[key].get("metrics_summary", {})
            parts = key.strip("/").split("/")

            row = {"key": key}
            # Parse key structure
            if len(parts) >= 5:
                row["source"] = parts[0]       # openai / local
                row["model"] = parts[1]
                row["strategy"] = parts[2]
                row["scenario"] = parts[3]
                row["lang"] = parts[4]
            elif len(parts) >= 3:
                row["source"] = parts[0]
                row["model"] = parts[1]
                row["method"] = parts[2]

            row.update(summary)
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def generate_comparison_csv(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Generate a main comparison CSV from all inference results."""
        df = self.to_dataframe()
        if df.empty:
            return df

        out = self.base_dir / "tables"
        out.mkdir(exist_ok=True)
        path = output_path or str(out / "main_comparison.csv")
        df.to_csv(path, index=False)
        return df

    def generate_latex_table(
        self,
        df: pd.DataFrame,
        filename: str = "main_comparison.tex",
        caption: str = "Model comparison on EXIST 2023 Task 1",
        label: str = "tab:main-comparison",
    ) -> str:
        """Generate a LaTeX booktabs table from a DataFrame."""
        out_dir = self.base_dir / "tables"
        out_dir.mkdir(exist_ok=True)

        cols = [c for c in ["model", "strategy", "scenario", "lang",
                            "f1_macro", "accuracy", "avg_confidence"] if c in df.columns]
        sub = df[cols].copy()

        # Format floats
        for c in ["f1_macro", "accuracy", "avg_confidence"]:
            if c in sub.columns:
                sub[c] = sub[c].map(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)

        header = " & ".join(cols)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{" + "l" * len(cols) + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]
        for _, row in sub.iterrows():
            vals = " & ".join(str(row[c]) for c in cols)
            lines.append(vals + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        tex = "\n".join(lines)
        with open(out_dir / filename, "w") as f:
            f.write(tex)
        return tex

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {str(k): ResultsManager._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ResultsManager._make_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    @staticmethod
    def _extract_summary(result: Dict) -> Dict[str, Any]:
        """Extract key metrics for the index."""
        summary = {}
        metrics = result.get("metrics", result)
        for key in ["f1_macro", "accuracy", "f1_weighted", "precision_macro",
                     "recall_macro", "avg_confidence"]:
            if key in metrics:
                val = metrics[key]
                summary[key] = float(val) if isinstance(val, (int, float, np.floating)) else val
        if "n_samples" in result:
            summary["n_samples"] = result["n_samples"]
        return summary
