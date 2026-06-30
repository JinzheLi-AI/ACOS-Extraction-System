"""Compare the old deployed ACOS model against the final trained model."""

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


OLD_RAW_METRICS = {
    "precision": 0.2458,
    "recall": 0.2154,
    "f1": 0.2296,
    "correct_tuple_count": 311,
    "predicted_tuple_count": 1265,
    "gold_tuple_count": 1444,
}

OLD_POST_METRICS = {
    "precision": 0.2504,
    "recall": 0.2147,
    "f1": 0.2312,
    "correct_tuple_count": 310,
    "predicted_tuple_count": 1238,
    "gold_tuple_count": 1444,
}


def parse_args() -> argparse.Namespace:
    """Read command-line options for old-vs-final comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def load_final_metrics(prediction_csv: Path) -> tuple[int, dict, dict]:
    """Compute raw and post-processed metrics from final predictions."""
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Final prediction CSV not found: {prediction_csv}")
    df = pd.read_csv(prediction_csv)
    required = {"text", "gold", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing column(s): {', '.join(sorted(missing))}")

    gold_outputs = df["gold"].fillna("").astype(str).tolist()
    pred_outputs = df["prediction"].fillna("").astype(str).tolist()
    raw_metrics = evaluate_predictions(gold_outputs, pred_outputs, use_postprocessing=False)
    post_metrics = evaluate_predictions(gold_outputs, pred_outputs, use_postprocessing=True)
    return len(df), raw_metrics, post_metrics


def comparison_row(label: str, old_metrics: dict, final_metrics: dict, key: str) -> str:
    """Format one comparison table row."""
    old_value = old_metrics[key]
    final_value = final_metrics[key]
    delta = final_value - old_value
    return f"| {label} | {old_value:.4f} | {final_value:.4f} | {delta:+.4f} |"


def count_row(label: str, old_metrics: dict, final_metrics: dict, key: str) -> str:
    """Format one tuple-count comparison row."""
    old_value = old_metrics[key]
    final_value = final_metrics[key]
    delta = final_value - old_value
    return f"| {label} | {old_value} | {final_value} | {delta:+d} |"


def write_report(
    report_path: Path,
    prediction_csv: Path,
    row_count: int,
    final_raw_metrics: dict,
    final_post_metrics: dict,
) -> None:
    """Save old-vs-final comparison as markdown."""
    post_f1_delta = final_post_metrics["f1"] - OLD_POST_METRICS["f1"]
    if post_f1_delta > 0:
        verdict = "The final model is better than the old deployed model by post-processed F1."
    elif post_f1_delta < 0:
        verdict = "The final model is worse than the old deployed model by post-processed F1."
    else:
        verdict = "The final model ties the old deployed model by post-processed F1."

    lines = [
        "# Old vs Final Model Comparison",
        "",
        f"**Final prediction file:** `{prediction_csv}`",
        f"**Evaluated examples:** {row_count}",
        "",
        "## Without Post-processing",
        "",
        "| Metric | Old Deployed Model | Final Model | Delta |",
        "| --- | ---: | ---: | ---: |",
        comparison_row("Precision", OLD_RAW_METRICS, final_raw_metrics, "precision"),
        comparison_row("Recall", OLD_RAW_METRICS, final_raw_metrics, "recall"),
        comparison_row("F1", OLD_RAW_METRICS, final_raw_metrics, "f1"),
        count_row("Correct tuples", OLD_RAW_METRICS, final_raw_metrics, "correct_tuple_count"),
        count_row("Predicted tuples", OLD_RAW_METRICS, final_raw_metrics, "predicted_tuple_count"),
        count_row("Gold tuples", OLD_RAW_METRICS, final_raw_metrics, "gold_tuple_count"),
        "",
        "## With Post-processing",
        "",
        "| Metric | Old Deployed Model | Final Model | Delta |",
        "| --- | ---: | ---: | ---: |",
        comparison_row("Precision", OLD_POST_METRICS, final_post_metrics, "precision"),
        comparison_row("Recall", OLD_POST_METRICS, final_post_metrics, "recall"),
        comparison_row("F1", OLD_POST_METRICS, final_post_metrics, "f1"),
        count_row("Correct tuples", OLD_POST_METRICS, final_post_metrics, "correct_tuple_count"),
        count_row("Predicted tuples", OLD_POST_METRICS, final_post_metrics, "predicted_tuple_count"),
        count_row("Gold tuples", OLD_POST_METRICS, final_post_metrics, "gold_tuple_count"),
        "",
        "## Verdict",
        "",
        verdict,
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Compare old deployed metrics with final model metrics."""
    args = parse_args()
    project_root = args.project_root.resolve()
    prediction_csv = (args.prediction_csv or project_root / "data" / "predictions" / "predictions_final.csv").resolve()
    report_path = (args.report_path or project_root / "reports" / "old_vs_final_comparison.md").resolve()

    row_count, final_raw_metrics, final_post_metrics = load_final_metrics(prediction_csv)
    write_report(report_path, prediction_csv, row_count, final_raw_metrics, final_post_metrics)

    print(f"Prediction file: {prediction_csv}")
    print(f"Evaluated examples: {row_count}")
    print(f"Final raw metrics: {final_raw_metrics}")
    print(f"Final post-processed metrics: {final_post_metrics}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
