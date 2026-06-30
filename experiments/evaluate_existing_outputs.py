"""Evaluate existing ACOS prediction outputs if a suitable CSV is available.

Expected CSV columns:
    text, gold, prediction

The script evaluates predictions twice:
    1. raw parsed outputs
    2. post-processed outputs

Results are saved to reports/evaluation_summary.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "evaluation_summary.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_predictions


REQUIRED_COLUMNS = {"text", "gold", "prediction"}


def find_prediction_file() -> Optional[Path]:
    """Find the most suitable prediction CSV in the current project."""
    candidates = []
    for path in PROJECT_ROOT.rglob("*.csv"):
        if any(part.startswith(".") for part in path.parts):
            continue
        name = path.name.lower()
        if not any(key in name for key in ("prediction", "pred", "output", "result", "eval")):
            continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns.str.lower())
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(columns):
            candidates.append(path)

    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def load_prediction_csv(path: Path) -> pd.DataFrame:
    """Load and validate a prediction CSV with text, gold, and prediction columns."""
    df = pd.read_csv(path)
    rename = {column: column.lower() for column in df.columns}
    df = df.rename(columns=rename)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    return df[["text", "gold", "prediction"]].fillna("")


def metrics_to_markdown(metrics: dict) -> str:
    """Convert a metrics dictionary to a markdown table row."""
    return (
        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
        f"{metrics['f1']:.4f} | {metrics['correct_tuple_count']} | "
        f"{metrics['predicted_tuple_count']} | {metrics['gold_tuple_count']} |"
    )


def write_placeholder_report() -> None:
    """Create a template report when no prediction CSV is available yet."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Evaluation Summary\n\n"
        "**Status:** To be filled after running prediction generation.\n\n"
        "No existing prediction CSV with the required columns was found.\n\n"
        "## Required Input CSV Format\n\n"
        "| column | description |\n"
        "| --- | --- |\n"
        "| `text` | Original review text. |\n"
        "| `gold` | Gold ACOS output, e.g. `(aspect | category | opinion | sentiment)`. |\n"
        "| `prediction` | Model prediction output in the same ACOS format. |\n\n"
        "## How to Run\n\n"
        "Place a CSV file with the required columns anywhere in the project folder, "
        "preferably named `predictions.csv`, then run:\n\n"
        "```bash\n"
        "python experiments/evaluate_existing_outputs.py\n"
        "```\n\n"
        "The script will evaluate predictions with and without post-processing and "
        "update this report with real metrics.\n",
        encoding="utf-8",
    )


def write_metrics_report(csv_path: Path, row_count: int, raw_metrics: dict, post_metrics: dict) -> None:
    """Write a markdown report containing real evaluation metrics."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Evaluation Summary\n\n"
        f"**Prediction file:** `{csv_path}`\n\n"
        f"**Rows evaluated:** {row_count}\n\n"
        "## Exact Match Metrics\n\n"
        "| Setting | Precision | Recall | F1 | Correct Tuples | Predicted Tuples | Gold Tuples |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| Without post-processing | {metrics_to_markdown(raw_metrics)}\n"
        f"| With post-processing | {metrics_to_markdown(post_metrics)}\n\n"
        "Exact Match counts a prediction as correct only when aspect, category, "
        "opinion, and sentiment all match a gold tuple exactly.\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run evaluation or create a placeholder report if no prediction file exists."""
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_prediction_file()

    if csv_path is None:
        write_placeholder_report()
        print("No suitable prediction CSV found.")
        print(f"Created placeholder report: {REPORT_PATH}")
        return

    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")

    df = load_prediction_csv(csv_path)
    raw_metrics = evaluate_predictions(df["gold"], df["prediction"], use_postprocessing=False)
    post_metrics = evaluate_predictions(df["gold"], df["prediction"], use_postprocessing=True)
    write_metrics_report(csv_path, len(df), raw_metrics, post_metrics)

    print(f"Prediction file: {csv_path}")
    print(f"Rows evaluated: {len(df)}")
    print("Without post-processing:", raw_metrics)
    print("With post-processing:   ", post_metrics)
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
