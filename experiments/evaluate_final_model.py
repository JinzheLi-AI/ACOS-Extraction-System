"""Evaluate final model predictions with raw and post-processed Exact Match."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_predictions


def parse_args() -> argparse.Namespace:
    """Read command-line options for final model evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    """Load the required text,gold,prediction CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"text", "gold", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing column(s): {', '.join(sorted(missing))}")
    return df


def metric_lines(title: str, metrics: dict) -> list[str]:
    """Format one metric dictionary for markdown."""
    return [
        f"## {title}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| Correct tuple count | {metrics['correct_tuple_count']} |",
        f"| Predicted tuple count | {metrics['predicted_tuple_count']} |",
        f"| Gold tuple count | {metrics['gold_tuple_count']} |",
        "",
    ]


def write_report(report_path: Path, prediction_csv: Path, row_count: int, raw_metrics: dict, post_metrics: dict) -> None:
    """Save final evaluation metrics to markdown."""
    lines = [
        "# Final Model Evaluation Summary",
        "",
        f"**Prediction file:** `{prediction_csv}`",
        f"**Evaluated examples:** {row_count}",
        "",
    ]
    lines.extend(metric_lines("Without Post-processing", raw_metrics))
    lines.extend(metric_lines("With Post-processing", post_metrics))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Evaluate final predictions and write the summary report."""
    args = parse_args()
    project_root = args.project_root.resolve()
    prediction_csv = (args.prediction_csv or project_root / "data" / "predictions" / "predictions_final.csv").resolve()
    report_path = (args.report_path or project_root / "reports" / "final_evaluation_summary.md").resolve()

    df = load_predictions(prediction_csv)
    gold_outputs = df["gold"].fillna("").astype(str).tolist()
    pred_outputs = df["prediction"].fillna("").astype(str).tolist()

    raw_metrics = evaluate_predictions(gold_outputs, pred_outputs, use_postprocessing=False)
    post_metrics = evaluate_predictions(gold_outputs, pred_outputs, use_postprocessing=True)
    write_report(report_path, prediction_csv, len(df), raw_metrics, post_metrics)

    print(f"Prediction file: {prediction_csv}")
    print(f"Evaluated examples: {len(df)}")
    print(f"Raw metrics: {raw_metrics}")
    print(f"Post-processed metrics: {post_metrics}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
