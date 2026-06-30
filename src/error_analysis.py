"""Systematic error analysis for ACOS quadruple extraction.

The analysis is tuple-level. It compares parsed gold and predicted ACOS tuples
and groups errors into interpretable categories for reporting.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from .postprocessing import AcosTuple, parse_acos_output, postprocess_prediction
except ImportError:  # Allows direct script execution with src/ on sys.path.
    from postprocessing import AcosTuple, parse_acos_output, postprocess_prediction


FIELD_NAMES = ("aspect", "category", "opinion", "sentiment")
ERROR_TYPES = (
    "exact_match",
    "missed_tuple",
    "hallucinated_tuple",
    "wrong_sentiment",
    "wrong_category",
    "wrong_opinion",
    "wrong_aspect",
    "partial_match",
    "malformed_output",
)


@dataclass
class ErrorCase:
    """One representative error or match case for the markdown report."""

    error_type: str
    text: str
    gold_output: str
    prediction: str
    gold_tuple: AcosTuple | None = None
    predicted_tuple: AcosTuple | None = None
    matched_fields: Tuple[str, ...] = ()
    wrong_fields: Tuple[str, ...] = ()


def parse_output(output_text: object) -> List[AcosTuple]:
    """Parse one raw output string using the shared parser without deduping."""
    return parse_acos_output(output_text)


def is_malformed(raw_prediction: object, parsed_prediction: Sequence[AcosTuple]) -> bool:
    """Return True when the model produced text but no parseable ACOS tuple."""
    return bool(str(raw_prediction or "").strip()) and not parsed_prediction


def matched_field_names(gold_tuple: AcosTuple, pred_tuple: AcosTuple) -> Tuple[str, ...]:
    """Return names of fields that match between one gold and predicted tuple."""
    return tuple(name for name, gold, pred in zip(FIELD_NAMES, gold_tuple, pred_tuple) if gold == pred)


def wrong_field_names(gold_tuple: AcosTuple, pred_tuple: AcosTuple) -> Tuple[str, ...]:
    """Return names of fields that differ between one gold and predicted tuple."""
    return tuple(name for name, gold, pred in zip(FIELD_NAMES, gold_tuple, pred_tuple) if gold != pred)


def best_partial_pair(
    gold_tuples: Sequence[AcosTuple],
    pred_tuples: Sequence[AcosTuple],
) -> Tuple[int, int, Tuple[str, ...]] | None:
    """Find the unmatched gold/pred pair with the most shared fields."""
    best = None
    best_score = 0
    for gold_idx, gold_tuple in enumerate(gold_tuples):
        for pred_idx, pred_tuple in enumerate(pred_tuples):
            fields = matched_field_names(gold_tuple, pred_tuple)
            score = len(fields)
            if score > best_score:
                best = (gold_idx, pred_idx, fields)
                best_score = score
    return best


def classify_pair(gold_tuple: AcosTuple, pred_tuple: AcosTuple) -> List[str]:
    """Classify field-level errors for a non-exact paired tuple."""
    wrong = wrong_field_names(gold_tuple, pred_tuple)
    labels = [f"wrong_{field}" for field in wrong]
    if wrong and len(wrong) < 4:
        labels.append("partial_match")
    return labels


def analyze_example(text: str, gold_output: str, prediction: str) -> Tuple[Counter, List[ErrorCase], Dict[str, int]]:
    """Analyze one prediction row and return counts, cases, and tuple totals."""
    gold_tuples = parse_output(gold_output)
    pred_tuples = postprocess_prediction(prediction)
    counts = Counter({error_type: 0 for error_type in ERROR_TYPES})
    cases: List[ErrorCase] = []
    totals = {
        "gold_tuple_count": len(gold_tuples),
        "predicted_tuple_count": len(pred_tuples),
        "exact_tuple_count": 0,
    }

    if is_malformed(prediction, pred_tuples):
        counts["malformed_output"] += 1
        cases.append(
            ErrorCase(
                error_type="malformed_output",
                text=text,
                gold_output=gold_output,
                prediction=prediction,
            )
        )

    unmatched_gold = list(gold_tuples)
    unmatched_pred = list(pred_tuples)

    for gold_tuple in list(unmatched_gold):
        if gold_tuple in unmatched_pred:
            counts["exact_match"] += 1
            totals["exact_tuple_count"] += 1
            unmatched_gold.remove(gold_tuple)
            unmatched_pred.remove(gold_tuple)
            cases.append(
                ErrorCase(
                    error_type="exact_match",
                    text=text,
                    gold_output=gold_output,
                    prediction=prediction,
                    gold_tuple=gold_tuple,
                    predicted_tuple=gold_tuple,
                    matched_fields=FIELD_NAMES,
                )
            )

    while unmatched_gold and unmatched_pred:
        pair = best_partial_pair(unmatched_gold, unmatched_pred)
        if pair is None or len(pair[2]) == 0:
            break

        gold_idx, pred_idx, fields = pair
        gold_tuple = unmatched_gold.pop(gold_idx)
        pred_tuple = unmatched_pred.pop(pred_idx)
        wrong = wrong_field_names(gold_tuple, pred_tuple)
        labels = classify_pair(gold_tuple, pred_tuple)
        for label in labels:
            counts[label] += 1
            cases.append(
                ErrorCase(
                    error_type=label,
                    text=text,
                    gold_output=gold_output,
                    prediction=prediction,
                    gold_tuple=gold_tuple,
                    predicted_tuple=pred_tuple,
                    matched_fields=fields,
                    wrong_fields=wrong,
                )
            )

    for gold_tuple in unmatched_gold:
        counts["missed_tuple"] += 1
        cases.append(
            ErrorCase(
                error_type="missed_tuple",
                text=text,
                gold_output=gold_output,
                prediction=prediction,
                gold_tuple=gold_tuple,
            )
        )

    for pred_tuple in unmatched_pred:
        counts["hallucinated_tuple"] += 1
        cases.append(
            ErrorCase(
                error_type="hallucinated_tuple",
                text=text,
                gold_output=gold_output,
                prediction=prediction,
                predicted_tuple=pred_tuple,
            )
        )

    return counts, cases, totals


def analyze_predictions(rows: Iterable[Dict[str, object]], max_examples_per_type: int = 3) -> dict:
    """Analyze all prediction rows and collect counts plus representative cases."""
    total_counts = Counter({error_type: 0 for error_type in ERROR_TYPES})
    examples: Dict[str, List[ErrorCase]] = defaultdict(list)
    total_examples = 0
    total_gold = 0
    total_pred = 0
    total_exact = 0

    for row in rows:
        total_examples += 1
        text = str(row.get("text", ""))
        gold_output = str(row.get("gold", ""))
        prediction = str(row.get("prediction", ""))
        counts, cases, totals = analyze_example(text, gold_output, prediction)

        total_counts.update(counts)
        total_gold += totals["gold_tuple_count"]
        total_pred += totals["predicted_tuple_count"]
        total_exact += totals["exact_tuple_count"]

        for case in cases:
            bucket = examples[case.error_type]
            if len(bucket) < max_examples_per_type:
                bucket.append(case)

    return {
        "total_examples": total_examples,
        "total_gold_tuples": total_gold,
        "total_predicted_tuples": total_pred,
        "total_exact_matched_tuples": total_exact,
        "error_counts": dict(total_counts),
        "examples": dict(examples),
    }


def tuple_to_text(tuple_item: AcosTuple | None) -> str:
    """Format one ACOS tuple for markdown output."""
    if tuple_item is None:
        return "N/A"
    return f"({tuple_item[0]} | {tuple_item[1]} | {tuple_item[2]} | {tuple_item[3]})"
