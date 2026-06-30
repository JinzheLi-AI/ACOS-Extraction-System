"""Generate predictions from the final fine-tuned FLAN-T5 ACOS model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE = "Extract ACOS quadruples from this course review: {text}"


def parse_args() -> argparse.Namespace:
    """Read command-line options for final prediction generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve model, test, and output paths from the project root."""
    project_root = args.project_root.resolve()
    model_dir = args.model_dir or project_root / "models" / "final_flan_t5_acos_model"
    test_file = args.test_file or project_root / "data" / "processed" / "test.csv"
    output_csv = args.output_csv or project_root / "data" / "predictions" / "predictions_final.csv"
    return model_dir.resolve(), test_file.resolve(), output_csv.resolve()


def load_tokenizer(model_dir: Path) -> AutoTokenizer:
    """Load tokenizer saved with the final model, with base tokenizer fallback."""
    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:
        print(f"Could not load tokenizer from {model_dir}: {exc}")
        print("Falling back to google/flan-t5-base tokenizer.")
        return AutoTokenizer.from_pretrained("google/flan-t5-base")


def generate_predictions(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> list[str]:
    """Generate ACOS output strings for test review texts."""
    predictions: list[str] = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(text=text) for text in batch_texts]
        encoded = tokenizer(
            prompts,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )

        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"Generated {min(start + batch_size, len(texts))}/{len(texts)} predictions")

    return predictions


def main() -> None:
    """Load final model, generate predictions, and save the required CSV."""
    args = parse_args()
    model_dir, test_file, output_csv = resolve_paths(args)

    if not model_dir.exists():
        raise FileNotFoundError(f"Final model folder not found: {model_dir}")
    if not test_file.exists():
        raise FileNotFoundError(f"Processed test file not found: {test_file}")

    test_df = pd.read_csv(test_file)
    required_columns = {"text", "gold"}
    missing = required_columns - set(test_df.columns)
    if missing:
        raise ValueError(f"Test file missing required column(s): {', '.join(sorted(missing))}")
    if args.limit is not None:
        test_df = test_df.head(args.limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model folder: {model_dir}")
    print(f"Test file: {test_file}")
    print(f"Output CSV: {output_csv}")
    print(f"Examples to predict: {len(test_df)}")
    print(f"Device: {device}")

    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        texts=test_df["text"].fillna("").astype(str).tolist(),
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "gold", "prediction"])
        writer.writeheader()
        for text, gold, prediction in zip(test_df["text"], test_df["gold"], predictions):
            writer.writerow({"text": text, "gold": gold, "prediction": prediction})

    print(f"Saved {len(predictions)} final predictions to {output_csv}")


if __name__ == "__main__":
    main()
