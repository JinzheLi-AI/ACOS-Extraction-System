"""Run error analysis on final model predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from error_analysis import ERROR_TYPES, analyze_predictions, tuple_to_text


ERROR_EXPLANATIONS = {
    "exact_match": "The predicted tuple exactly matches a gold tuple across all four fields.",
    "missed_tuple": "A gold tuple was not matched by any predicted tuple.",
    "hallucinated_tuple": "The model predicted a tuple that does not correspond to any gold tuple.",
    "wrong_sentiment": "The aspect/category/opinion may partly match, but the sentiment label is wrong.",
    "wrong_category": "The model used the wrong ACOS category for a partially matched tuple.",
    "wrong_opinion": "The model generated an incorrect or different opinion phrase.",
    "wrong_aspect": "The model generated an incorrect aspect or missed an implicit aspect.",
    "partial_match": "At least one field matched, but the full quadruple was not exactly correct.",
    "malformed_output": "The model produced non-empty text that could not be parsed as an ACOS tuple.",
}


def parse_args() -> argparse.Namespace:
    """Read command-line options for final error analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    """Load prediction rows from a text,gold,prediction CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text", "gold", "prediction"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
        return [
            {
                "text": row.get("text", ""),
                "gold": row.get("gold", ""),
                "prediction": row.get("prediction", ""),
            }
            for row in reader
        ]


def escape_md(value: object) -> str:
    """Keep markdown tables readable for long text outputs."""
    return str(value or "").replace("\n", " ").replace("|", "\\|").strip()


def write_report(report_path: Path, prediction_csv: Path, result: dict) -> None:
    """Save final error analysis as markdown."""
    counts = result["error_counts"]
    examples = result["examples"]
    lines = [
        "# Final Model Error Analysis",
        "",
        f"**Prediction file:** `{prediction_csv}`",
        "",
        f"**Total evaluated examples:** {result['total_examples']}",
        f"**Total gold tuples:** {result['total_gold_tuples']}",
        f"**Total predicted tuples:** {result['total_predicted_tuples']}",
        f"**Total exact matched tuples:** {result['total_exact_matched_tuples']}",
        "",
        "## Error Type Counts",
        "",
        "| Error Type | Count | Explanation |",
        "| --- | ---: | --- |",
    ]

    for error_type in ERROR_TYPES:
        lines.append(
            f"| `{error_type}` | {counts.get(error_type, 0)} | {ERROR_EXPLANATIONS[error_type]} |"
        )

    lines.extend(
        [
            "",
            "## Representative Examples",
            "",
            "Examples are limited to a few cases per error type to keep the report readable.",
        ]
    )

    for error_type in ERROR_TYPES:
        cases = examples.get(error_type, [])
        if not cases:
            continue
        lines.extend(["", f"### {error_type}", ""])
        for idx, case in enumerate(cases, 1):
            matched = ", ".join(case.matched_fields) if case.matched_fields else "N/A"
            wrong = ", ".join(case.wrong_fields) if case.wrong_fields else "N/A"
            lines.extend(
                [
                    f"**Example {idx}**",
                    "",
                    f"- **Input text:** {escape_md(case.text)}",
                    f"- **Gold output:** `{escape_md(case.gold_output)}`",
                    f"- **Model prediction:** `{escape_md(case.prediction)}`",
                    f"- **Gold tuple:** `{escape_md(tuple_to_text(case.gold_tuple))}`",
                    f"- **Predicted tuple:** `{escape_md(tuple_to_text(case.predicted_tuple))}`",
                    f"- **Matched fields:** {escape_md(matched)}",
                    f"- **Wrong fields:** {escape_md(wrong)}",
                    "",
                ]
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run final prediction error analysis."""
    args = parse_args()
    project_root = args.project_root.resolve()
    prediction_csv = (args.prediction_csv or project_root / "data" / "predictions" / "predictions_final.csv").resolve()
    report_path = (args.report_path or project_root / "reports" / "final_error_analysis.md").resolve()

    rows = load_prediction_rows(prediction_csv)
    result = analyze_predictions(rows)
    write_report(report_path, prediction_csv, result)

    print(f"Prediction file: {prediction_csv}")
    print(f"Total evaluated examples: {result['total_examples']}")
    print(f"Total gold tuples: {result['total_gold_tuples']}")
    print(f"Total predicted tuples: {result['total_predicted_tuples']}")
    print(f"Total exact matched tuples: {result['total_exact_matched_tuples']}")
    print("Error type counts:")
    for error_type in ERROR_TYPES:
        print(f"  {error_type}: {result['error_counts'].get(error_type, 0)}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
