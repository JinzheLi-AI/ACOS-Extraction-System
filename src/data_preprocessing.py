"""Reusable data preparation utilities for OATS-ABSA ACOS datasets.

This module only parses and formats data. It does not train a model.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .postprocessing import parse_acos_output
except ImportError:  # Allows direct script execution with src/ on sys.path.
    from postprocessing import parse_acos_output


DOMAINS: Tuple[str, ...] = ("coursera", "hotels", "amazon")
SPLITS: Tuple[str, ...] = ("train", "dev", "test")


def normalize_null_value(value: object) -> str:
    """Convert OATS NULL fields into the project output word 'implicit'."""
    text = str(value).strip()
    return "implicit" if text.upper() == "NULL" else text


def parse_oats_line(line: str, domain: str, split: str) -> Optional[Dict[str, object]]:
    """Parse one OATS-ABSA line into text, domain, split, and ACOS tuples.

    OATS stores tuples in the order:
    aspect, category, sentiment, opinion.

    This project uses:
    aspect, category, opinion, sentiment.
    """
    line = line.strip()
    if not line or "####" not in line:
        return None

    text, raw_tuple_text = line.split("####", 1)
    try:
        raw_tuples = ast.literal_eval(raw_tuple_text.strip())
    except (ValueError, SyntaxError):
        return None

    if not isinstance(raw_tuples, list):
        return None

    tuples = []
    for item in raw_tuples:
        if not isinstance(item, list) or len(item) != 4:
            continue
        aspect, category, sentiment, opinion = item
        tuples.append(
            {
                "aspect": normalize_null_value(aspect),
                "category": str(category).strip(),
                "opinion": normalize_null_value(opinion),
                "sentiment": str(sentiment).strip().lower(),
            }
        )

    if not tuples:
        return None

    return {
        "text": text.strip(),
        "domain": domain,
        "split": split,
        "tuples": tuples,
    }


def format_acos_target(tuples: Sequence[Dict[str, str]]) -> str:
    """Format parsed ACOS tuples into the training target string."""
    return " ; ".join(
        f"({item['aspect']} | {item['category']} | {item['opinion']} | {item['sentiment']})"
        for item in tuples
    )


def count_formatted_acos_tuples(output_text: object) -> int:
    """Count formatted ACOS tuples using the shared parser, not parentheses."""
    return len(parse_acos_output(output_text))


def load_oats_split(data_dir: Path, domain: str, split: str) -> List[Dict[str, object]]:
    """Load one OATS domain/split file safely."""
    path = data_dir / f"{domain}_{split}.txt"
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_oats_line(line, domain, split)
            if record is not None:
                records.append(record)
    return records


def load_oats_records(
    data_dir: Path,
    domains: Iterable[str] = DOMAINS,
    splits: Iterable[str] = SPLITS,
) -> List[Dict[str, object]]:
    """Load multiple OATS files into one list of parsed records."""
    records: List[Dict[str, object]] = []
    for domain in domains:
        for split in splits:
            records.extend(load_oats_split(data_dir, domain, split))
    return records


def records_to_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    """Convert parsed records into CSV-ready rows."""
    rows = []
    for record in records:
        tuples = record["tuples"]
        rows.append(
            {
                "text": str(record["text"]),
                "gold": format_acos_target(tuples),  # type: ignore[arg-type]
                "domain": str(record["domain"]),
                "split": str(record["split"]),
            }
        )
    return rows
