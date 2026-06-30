"""Post-processing utilities for ACOS model outputs.

The FLAN-T5 model generates text, so the output can vary slightly in format.
This module converts common textual formats into stable ACOS tuples:

    (aspect | category | opinion | sentiment)
    (aspect, category, opinion, sentiment)
    aspect | category | opinion | sentiment
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


AcosTuple = Tuple[str, str, str, str]


_SENTIMENT_MAP = {
    "pos": "positive",
    "positive": "positive",
    "positif": "positive",
    "+": "positive",
    "neg": "negative",
    "negative": "negative",
    "negatif": "negative",
    "-": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "netral": "neutral",
    "0": "neutral",
}


def normalize_text(text: object) -> str:
    """Return a lowercase string with extra whitespace and wrappers removed."""
    if text is None:
        return ""
    value = str(text).strip()
    value = value.replace("\n", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip().strip('"').strip("'").strip()
    return value.lower()


def normalize_sentiment(sentiment: object) -> str:
    """Normalize sentiment labels to positive, negative, neutral, or original text."""
    value = normalize_text(sentiment)
    value = value.strip(" .;,:|()[]{}")
    return _SENTIMENT_MAP.get(value, value)


def _clean_field(field: object) -> str:
    """Clean one ACOS field while preserving meaningful words."""
    value = normalize_text(field)
    return value.strip(" .;,:|[]{}")


def _strip_outer_parentheses(chunk: str) -> str:
    """Remove one wrapping pair of parentheses from a complete tuple chunk."""
    value = chunk.strip()
    if value.startswith("(") and value.endswith(")"):
        return value[1:-1].strip()
    return value


def split_acos_tuple_chunks(output_text: object) -> List[str]:
    """Split an ACOS output string into complete tuple chunks.

    The project target format separates tuples with " ; ". Some opinion text
    can contain parentheses, so this function does not count "(" characters.
    If no tuple separator is found, it falls back to balanced-parentheses
    scanning and finally to a single unwrapped tuple.
    """
    text = str(output_text or "").strip()
    if not text:
        return []

    if " ; " in text or re.search(r"\)\s*;\s*\(", text):
        return [part.strip() for part in re.split(r"\s+;\s+|\s*;\s*(?=\()", text) if part.strip()]

    chunks: List[str] = []
    start = None
    depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start : idx + 1].strip())
                start = None

    if chunks:
        return chunks
    return [text]


def _split_tuple_fields(chunk: str) -> List[str]:
    """Split one possible tuple chunk using pipe first, then comma."""
    if "|" in chunk:
        return [part.strip() for part in chunk.split("|")]
    return [part.strip() for part in chunk.split(",")]


def parse_acos_output(output_text: object) -> List[AcosTuple]:
    """Parse raw model output into a list of normalized ACOS tuples.

    Malformed chunks are skipped safely. The function accepts parenthesized
    tuples, semicolon-separated tuples, and a single unwrapped tuple.
    """
    text = str(output_text or "").strip()
    if not text:
        return []

    candidates = split_acos_tuple_chunks(text)

    tuples: List[AcosTuple] = []
    for chunk in candidates:
        chunk = _strip_outer_parentheses(chunk)
        fields = _split_tuple_fields(chunk)
        if len(fields) != 4:
            continue

        aspect = _clean_field(fields[0]) or "implicit"
        category = _clean_field(fields[1])
        opinion = _clean_field(fields[2]) or "implicit"
        sentiment = normalize_sentiment(fields[3])

        if not category or not sentiment:
            continue
        tuples.append((aspect, category, opinion, sentiment))

    return tuples


def format_tuple(tuple_item: Sequence[object]) -> AcosTuple | None:
    """Normalize one tuple-like item into the canonical ACOS tuple format."""
    if tuple_item is None or len(tuple_item) != 4:
        return None

    aspect = _clean_field(tuple_item[0]) or "implicit"
    category = _clean_field(tuple_item[1])
    opinion = _clean_field(tuple_item[2]) or "implicit"
    sentiment = normalize_sentiment(tuple_item[3])

    if not category or not sentiment:
        return None
    return (aspect, category, opinion, sentiment)


def deduplicate_tuples(tuples: Iterable[Sequence[object]]) -> List[AcosTuple]:
    """Remove repeated tuples while preserving the original order."""
    seen = set()
    unique: List[AcosTuple] = []
    for item in tuples:
        formatted = format_tuple(item)
        if formatted is None or formatted in seen:
            continue
        seen.add(formatted)
        unique.append(formatted)
    return unique


def postprocess_prediction(output_text: object) -> List[AcosTuple]:
    """Parse, normalize, and deduplicate one raw ACOS prediction string."""
    return deduplicate_tuples(parse_acos_output(output_text))
