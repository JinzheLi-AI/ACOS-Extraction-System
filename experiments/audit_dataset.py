"""Audit available ACOS datasets before final retraining.

This script does not train any model. It scans Sorted_Dataset/, checks expected
OATS-ABSA files, counts examples and ACOS tuples, detects malformed lines, and
writes reports/dataset_audit.md.
"""

from __future__ import annotations

import ast
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer a local raw dataset copy if it is added later. For now, reuse the
# original project dataset without moving or duplicating it.
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Sorted_Dataset"
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else PROJECT_ROOT.parent / "Sorted_Dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "dataset_audit.md"

DOMAINS = ("coursera", "hotels", "amazon")
SPLITS = ("train", "dev", "test")
EXPECTED_OATS_FILES = [f"{domain}_{split}.txt" for domain in DOMAINS for split in SPLITS]
EXPECTED_EDURABSA_FILES = ["EduRABSA_train.csv", "EduRABSA_test.csv"]


@dataclass
class FileAudit:
    """Audit result for one dataset file."""

    filename: str
    domain: str
    split: str
    exists: bool
    examples: int = 0
    tuples: int = 0
    malformed_lines: int = 0
    empty_tuple_lines: int = 0


def parse_oats_line(line: str) -> Tuple[bool, int, bool]:
    """Parse one OATS line and return valid flag, tuple count, and empty flag."""
    line = line.strip()
    if not line:
        return False, 0, False
    if "####" not in line:
        return False, 0, False

    _, raw_quad_text = line.split("####", 1)
    try:
        quads = ast.literal_eval(raw_quad_text.strip())
    except (ValueError, SyntaxError):
        return False, 0, False

    if not isinstance(quads, list):
        return False, 0, False

    valid_tuple_count = sum(1 for item in quads if isinstance(item, list) and len(item) == 4)
    empty_tuple_line = len(quads) == 0
    malformed_tuple_line = bool(quads) and valid_tuple_count != len(quads)
    if malformed_tuple_line:
        return False, valid_tuple_count, empty_tuple_line
    return True, valid_tuple_count, empty_tuple_line


def audit_oats_file(path: Path, domain: str, split: str) -> FileAudit:
    """Audit one expected OATS-ABSA text file."""
    result = FileAudit(filename=path.name, domain=domain, split=split, exists=path.exists())
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            valid, tuple_count, empty_tuple_line = parse_oats_line(line)
            if not valid:
                result.malformed_lines += 1
                continue
            if empty_tuple_line:
                result.empty_tuple_lines += 1
                continue
            result.examples += 1
            result.tuples += tuple_count
    return result


def audit_all_oats_files() -> List[FileAudit]:
    """Audit every expected OATS-ABSA domain/split file."""
    audits = []
    for domain in DOMAINS:
        for split in SPLITS:
            filename = f"{domain}_{split}.txt"
            audits.append(audit_oats_file(DATA_DIR / filename, domain, split))
    return audits


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file without loading the full file into memory."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        row_count = sum(1 for _ in reader)
    return max(row_count - 1, 0)


def summarize(audits: List[FileAudit]) -> dict:
    """Build split/domain totals from file audits."""
    split_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"examples": 0, "tuples": 0})
    domain_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"examples": 0, "tuples": 0})
    for audit in audits:
        if not audit.exists:
            continue
        split_totals[audit.split]["examples"] += audit.examples
        split_totals[audit.split]["tuples"] += audit.tuples
        domain_totals[audit.domain]["examples"] += audit.examples
        domain_totals[audit.domain]["tuples"] += audit.tuples
    return {"split_totals": dict(split_totals), "domain_totals": dict(domain_totals)}


def markdown_table(headers: List[str], rows: List[List[object]]) -> List[str]:
    """Create a simple markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def write_report(audits: List[FileAudit]) -> None:
    """Write the dataset audit markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(audits)
    split_totals = summary["split_totals"]
    domain_totals = summary["domain_totals"]

    available = [audit.filename for audit in audits if audit.exists]
    missing = [audit.filename for audit in audits if not audit.exists]
    edurabsa_rows = {
        filename: count_csv_rows(DATA_DIR / filename) for filename in EXPECTED_EDURABSA_FILES
    }

    train_plus_dev_examples = (
        split_totals.get("train", {}).get("examples", 0)
        + split_totals.get("dev", {}).get("examples", 0)
    )
    train_plus_dev_tuples = (
        split_totals.get("train", {}).get("tuples", 0)
        + split_totals.get("dev", {}).get("tuples", 0)
    )
    test_examples = split_totals.get("test", {}).get("examples", 0)
    test_tuples = split_totals.get("test", {}).get("tuples", 0)

    lines = [
        "# Dataset Audit",
        "",
        f"**Dataset folder:** `{DATA_DIR}`",
        "",
        "## Available OATS-ABSA Files",
        "",
    ]
    lines.extend(f"- `{filename}`" for filename in available)
    lines.extend(["", "## Missing Expected OATS-ABSA Files", ""])
    lines.extend(f"- `{filename}`" for filename in missing) if missing else lines.append("- None")

    lines.extend(["", "## File-Level Counts", ""])
    lines.extend(
        markdown_table(
            [
                "File",
                "Domain",
                "Split",
                "Exists",
                "Examples",
                "ACOS Tuples",
                "Malformed Lines",
                "Empty Tuple Lines",
            ],
            [
                [
                    audit.filename,
                    audit.domain,
                    audit.split,
                    "yes" if audit.exists else "no",
                    audit.examples,
                    audit.tuples,
                    audit.malformed_lines,
                    audit.empty_tuple_lines,
                ]
                for audit in audits
            ],
        )
    )

    lines.extend(["", "## Examples and Tuples per Split", ""])
    lines.extend(
        markdown_table(
            ["Split", "Examples", "ACOS Tuples"],
            [
                [split, split_totals.get(split, {}).get("examples", 0), split_totals.get(split, {}).get("tuples", 0)]
                for split in SPLITS
            ],
        )
    )

    lines.extend(["", "## Examples and Tuples per Domain", ""])
    lines.extend(
        markdown_table(
            ["Domain", "Examples", "ACOS Tuples"],
            [
                [domain, domain_totals.get(domain, {}).get("examples", 0), domain_totals.get(domain, {}).get("tuples", 0)]
                for domain in DOMAINS
            ],
        )
    )

    lines.extend(["", "## EduRABSA Status", ""])
    lines.extend(
        markdown_table(
            ["File", "Exists", "Rows"],
            [
                [filename, "yes" if (DATA_DIR / filename).exists() else "no", edurabsa_rows[filename]]
                for filename in EXPECTED_EDURABSA_FILES
            ],
        )
    )
    lines.extend(
        [
            "",
            "**Recommendation:** EduRABSA is available and can be inspected, but it should not be included in current ACOS training unless it is converted into ACOS quadruple labels.",
            "",
            "## Recommended Final Train / Dev / Test Composition",
            "",
            f"- Training candidate: all OATS train + dev files across Coursera, Hotels, and Amazon = **{train_plus_dev_examples} examples** and **{train_plus_dev_tuples} ACOS tuples**.",
            f"- Test set: all OATS test files across Coursera, Hotels, and Amazon = **{test_examples} examples** and **{test_tuples} ACOS tuples**.",
            "- The current test set should be described as **all-domain test**, not Coursera-only, if coursera_test + hotels_test + amazon_test are used.",
            "- EduRABSA status for the current ACOS pipeline: **not used for training; inspected/planned for future ACOS annotation only**.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the dataset audit and print a terminal summary."""
    audits = audit_all_oats_files()
    write_report(audits)
    summary = summarize(audits)
    split_totals = summary["split_totals"]
    domain_totals = summary["domain_totals"]
    missing = [audit.filename for audit in audits if not audit.exists]

    print(f"Dataset folder: {DATA_DIR}")
    print("Missing expected OATS files:", ", ".join(missing) if missing else "None")
    print("\nSplit totals:")
    for split in SPLITS:
        totals = split_totals.get(split, {"examples": 0, "tuples": 0})
        print(f"  {split}: {totals['examples']} examples, {totals['tuples']} ACOS tuples")
    print("\nDomain totals:")
    for domain in DOMAINS:
        totals = domain_totals.get(domain, {"examples": 0, "tuples": 0})
        print(f"  {domain}: {totals['examples']} examples, {totals['tuples']} ACOS tuples")
    print("\nEduRABSA:")
    for filename in EXPECTED_EDURABSA_FILES:
        path = DATA_DIR / filename
        status = "exists" if path.exists() else "missing"
        print(f"  {filename}: {status}, rows={count_csv_rows(path)}")
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
