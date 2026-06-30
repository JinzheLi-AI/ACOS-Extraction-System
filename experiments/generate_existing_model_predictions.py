"""Generate predictions from the existing fine-tuned FLAN-T5 ACOS model.

This script does not train or modify the model. It only:
1. loads the saved model from ../The Best Model/
2. loads the OATS-ABSA test split
3. generates ACOS predictions
4. saves reports/predictions.csv with columns: text,gold,prediction
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The clean project does not copy large model files yet. Reuse the current
# deployed model and raw dataset from the original parent project folder.
MODEL_DIR = PROJECT_ROOT.parent / "The Best Model"
DATA_DIR = PROJECT_ROOT.parent / "Sorted_Dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_CSV = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"

PROMPT_TEMPLATE = "Extract ACOS quadruples from this course review: {text}"
TEST_FILES = [
    ("coursera", "coursera_test.txt"),
    ("hotels", "hotels_test.txt"),
    ("amazon", "amazon_test.txt"),
]


def parse_oats_line(line: str, domain: str) -> Optional[Dict[str, object]]:
    """Parse one OATS-ABSA line into sentence text and gold ACOS quads."""
    line = line.strip()
    if not line or "####" not in line:
        return None

    sentence, quad_text = line.split("####", 1)
    sentence = sentence.strip()
    try:
        raw_quads = ast.literal_eval(quad_text.strip())
    except (ValueError, SyntaxError):
        return None

    quads = []
    for item in raw_quads:
        if len(item) != 4:
            continue
        # OATS order: aspect, category, sentiment, opinion
        aspect, category, sentiment, opinion = item
        quads.append(
            {
                "aspect": aspect,
                "category": category,
                "opinion": opinion,
                "sentiment": str(sentiment).lower(),
            }
        )

    if not quads:
        return None
    return {"text": sentence, "quads": quads, "domain": domain}


def load_oats_file(path: Path, domain: str) -> List[Dict[str, object]]:
    """Load one OATS-ABSA split file."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = parse_oats_line(line, domain)
            if record is not None:
                records.append(record)
    return records


def load_test_records(test_scope: str = "all") -> List[Dict[str, object]]:
    """Load the existing test set.

    The final notebook uses all-domain test data:
    coursera_test + hotels_test + amazon_test = 986 records.
    """
    selected = TEST_FILES if test_scope == "all" else [("coursera", "coursera_test.txt")]
    records: List[Dict[str, object]] = []
    for domain, filename in selected:
        records.extend(load_oats_file(DATA_DIR / filename, domain))
    return records


def quads_to_target(quads: List[Dict[str, str]]) -> str:
    """Convert gold ACOS dictionaries to the original training target format."""
    parts = []
    for quad in quads:
        aspect = quad["aspect"] if str(quad["aspect"]).upper() != "NULL" else "implicit"
        opinion = quad["opinion"] if str(quad["opinion"]).upper() != "NULL" else "implicit"
        parts.append(f"({aspect} | {quad['category']} | {opinion} | {quad['sentiment']})")
    return " ; ".join(parts)


def generate_batch(
    texts: List[str],
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    device: torch.device,
) -> List[str]:
    """Generate ACOS predictions for a batch of review texts."""
    prompts = [PROMPT_TEMPLATE.format(text=text) for text in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main() -> None:
    """Generate reports/predictions.csv from the saved fine-tuned model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scope", choices=["all", "coursera"], default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    records = load_test_records(args.test_scope)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No test records loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model folder: {MODEL_DIR}")
    print(f"Test scope: {args.test_scope}")
    print(f"Loaded test records: {len(records)}")
    print(f"Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    rows = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        predictions = generate_batch(
            [str(record["text"]) for record in batch],
            tokenizer,
            model,
            device,
        )
        for record, prediction in zip(batch, predictions):
            rows.append(
                {
                    "text": record["text"],
                    "gold": quads_to_target(record["quads"]),
                    "prediction": prediction,
                }
            )
        print(f"Generated {min(start + batch_size, len(records))}/{len(records)}")
        sys.stdout.flush()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["text", "gold", "prediction"]).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved predictions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
