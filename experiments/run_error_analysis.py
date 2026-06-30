"""Run ACOS error analysis on data/predictions/predictions.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis.md"

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


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    """Load prediction rows from the required CSV format."""
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
    """Keep markdown tables readable for long model outputs."""
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    return text.strip()


def write_report(result: dict) -> None:
    """Write the error analysis markdown report."""
    counts = result["error_counts"]
    examples = result["examples"]

    lines = [
        "# ACOS Error Analysis",
        "",
        f"**Prediction file:** `{PREDICTIONS_PATH}`",
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
            f"| `{error_type}` | {counts.get(error_type, 0)} | "
            f"{ERROR_EXPLANATIONS[error_type]} |"
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

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load predictions, run error analysis, and write the markdown report."""
    rows = load_prediction_rows(PREDICTIONS_PATH)
    result = analyze_predictions(rows)
    write_report(result)

    print(f"Prediction file: {PREDICTIONS_PATH}")
    print(f"Total evaluated examples: {result['total_examples']}")
    print(f"Total gold tuples: {result['total_gold_tuples']}")
    print(f"Total predicted tuples: {result['total_predicted_tuples']}")
    print(f"Total exact matched tuples: {result['total_exact_matched_tuples']}")
    print("Error type counts:")
    for error_type in ERROR_TYPES:
        print(f"  {error_type}: {result['error_counts'].get(error_type, 0)}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
