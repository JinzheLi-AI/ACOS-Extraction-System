"""Exact Match evaluation for ACOS quadruple predictions."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, List, Sequence, Tuple

try:
    from .postprocessing import parse_acos_output, postprocess_prediction, split_acos_tuple_chunks
except ImportError:  # Allows running this file directly during quick checks.
    from postprocessing import parse_acos_output, postprocess_prediction, split_acos_tuple_chunks


RawTuple = Tuple[str, str, str, str]


def _safe_divide(numerator: int, denominator: int) -> float:
    """Avoid division-by-zero and return 0.0 when the denominator is empty."""
    return numerator / denominator if denominator else 0.0


def _raw_field(value: object) -> str:
    """Lightly clean a raw field without sentiment mapping or heavy normalization."""
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _parse_raw_output(output_text: object) -> List[RawTuple]:
    """Parse a raw ACOS output string with minimal cleaning only."""
    text = str(output_text or "").strip()
    if not text:
        return []

    tuples: List[RawTuple] = []
    for chunk in split_acos_tuple_chunks(text):
        chunk = chunk.strip()
        if chunk.startswith("(") and chunk.endswith(")"):
            chunk = chunk[1:-1].strip()
        fields = chunk.split("|") if "|" in chunk else chunk.split(",")
        if len(fields) != 4:
            continue
        tuples.append(tuple(_raw_field(field).strip(" .;,:|[]{}") for field in fields))
    return tuples


def _tuple_counter(tuples: Iterable[Sequence[object]]) -> Counter:
    """Convert tuple-like items into a comparable multiset."""
    comparable = Counter()
    for item in tuples:
        if item is None or len(item) != 4:
            continue
        comparable[tuple(_raw_field(field) for field in item)] += 1
    return comparable


def compute_exact_match_metrics(
    gold_tuples_list: Iterable[Iterable[Sequence[object]]],
    pred_tuples_list: Iterable[Iterable[Sequence[object]]],
) -> dict:
    """Compute quadruple-level Exact Match Precision, Recall, and F1.

    A predicted tuple is correct only when aspect, category, opinion, and
    sentiment all match a gold tuple exactly after normalization.
    """
    correct_tuple_count = 0
    predicted_tuple_count = 0
    gold_tuple_count = 0

    for gold_tuples, pred_tuples in zip(gold_tuples_list, pred_tuples_list):
        gold_counter = _tuple_counter(gold_tuples)
        pred_counter = _tuple_counter(pred_tuples)

        correct_tuple_count += sum((gold_counter & pred_counter).values())
        predicted_tuple_count += sum(pred_counter.values())
        gold_tuple_count += sum(gold_counter.values())

    precision = _safe_divide(correct_tuple_count, predicted_tuple_count)
    recall = _safe_divide(correct_tuple_count, gold_tuple_count)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct_tuple_count": correct_tuple_count,
        "predicted_tuple_count": predicted_tuple_count,
        "gold_tuple_count": gold_tuple_count,
    }


def evaluate_predictions(
    gold_outputs: Iterable[object],
    pred_outputs: Iterable[object],
    use_postprocessing: bool = True,
) -> dict:
    """Evaluate raw gold/prediction strings with optional post-processing.

    When use_postprocessing is True, predictions and gold outputs are parsed,
    normalized, sentiment-normalized, and deduplicated. When False, the same
    parser is used but without the final reusable post-processing wrapper.
    """
    gold_outputs = list(gold_outputs)
    pred_outputs = list(pred_outputs)
    if len(gold_outputs) != len(pred_outputs):
        raise ValueError(
            "gold_outputs and pred_outputs must have the same number of rows "
            f"({len(gold_outputs)} != {len(pred_outputs)})."
        )

    if use_postprocessing:
        gold_tuples = [parse_acos_output(output) for output in gold_outputs]
        pred_tuples = [postprocess_prediction(output) for output in pred_outputs]
    else:
        gold_tuples = [_parse_raw_output(output) for output in gold_outputs]
        pred_tuples = [_parse_raw_output(output) for output in pred_outputs]

    return compute_exact_match_metrics(gold_tuples, pred_tuples)
