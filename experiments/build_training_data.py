"""Build standardized processed OATS-ABSA CSV files.

This script does not train a model. It only converts raw OATS text files into
CSV files that are easier to reuse for future training and evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OLD_DATA_DIR = PROJECT_ROOT.parent / "Sorted_Dataset"
LOCAL_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "processed_data_summary.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import (
    DOMAINS,
    SPLITS,
    count_formatted_acos_tuples,
    load_oats_records,
    records_to_rows,
)


def choose_default_data_dir() -> Path:
    """Use local raw data if present; otherwise use the original project dataset."""
    local_candidate = LOCAL_RAW_DATA_DIR / "Sorted_Dataset"
    if local_candidate.exists():
        return local_candidate
    return DEFAULT_OLD_DATA_DIR


def write_summary(split_frames: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Write a short markdown summary for the processed data files."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Processed Data Summary",
        "",
        f"**Source data folder:** `{data_dir}`",
        "",
        "| Split | Examples | Gold ACOS Tuples |",
        "| --- | ---: | ---: |",
    ]

    for split in SPLITS:
        frame = split_frames.get(split, pd.DataFrame())
        tuple_count = int(frame["gold"].map(count_formatted_acos_tuples).sum()) if not frame.empty else 0
        lines.append(f"| {split} | {len(frame)} | {tuple_count} |")

    lines.extend(
        [
            "",
            "Recommended current composition:",
            "",
            "- Training candidate: train + dev OATS-ABSA splits.",
            "- Test set: all-domain OATS-ABSA test split.",
            "- EduRABSA: not included unless converted into ACOS quadruple labels.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Create processed train/dev/test CSV files from OATS-ABSA raw files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=choose_default_data_dir())
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_frames: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        records = load_oats_records(args.data_dir, domains=DOMAINS, splits=(split,))
        rows = records_to_rows(records)
        frame = pd.DataFrame(rows, columns=["text", "gold", "domain", "split"])
        split_frames[split] = frame
        output_path = PROCESSED_DIR / f"{split}.csv"
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(frame)} {split} examples to {output_path}")

    write_summary(split_frames, args.data_dir)
    print(f"Saved summary to {REPORT_PATH}")


if __name__ == "__main__":
    main()
