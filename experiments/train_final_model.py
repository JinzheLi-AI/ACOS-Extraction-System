"""Final FLAN-T5 ACOS training script.

This script prepares and runs the final supervised fine-tuning pipeline when
the user is ready. It does not run automatically unless this file is executed.
It uses train.csv for training, dev.csv for validation, and keeps test.csv only
for final evaluation after training.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "final_training_config.json"
REQUIRED_COLUMNS = {"text", "gold"}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from postprocessing import parse_acos_output


class AcosSeq2SeqDataset(Dataset):
    """Tokenized dataset for prompt-based ACOS generation."""

    def __init__(self, frame: pd.DataFrame, tokenizer: AutoTokenizer, config: Dict[str, Any]):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.prompt_template = config["prompt_template"]
        self.max_input_length = int(config["max_input_length"])
        self.max_target_length = int(config["max_target_length"])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        prompt = self.prompt_template.format(text=str(row["text"]))
        target = str(row["gold"])

        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return {key: torch.tensor(value) for key, value in model_inputs.items()}


def load_config(path: Path) -> Dict[str, Any]:
    """Load final training configuration from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(path_value: str) -> Path:
    """Resolve a config path relative to the clean project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_split(path: Path) -> pd.DataFrame:
    """Load a processed CSV split and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    frame = pd.read_csv(path).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
    return frame


def load_all_splits(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/dev/test splits from the configured processed CSV files."""
    return (
        load_split(resolve_project_path(config["train_file"])),
        load_split(resolve_project_path(config["dev_file"])),
        load_split(resolve_project_path(config["test_file"])),
    )


def count_gold_tuples(frame: pd.DataFrame) -> int:
    """Count ACOS tuples using the shared project parser."""
    return int(frame["gold"].map(lambda value: len(parse_acos_output(value))).sum())


def print_split_counts(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print example and ACOS tuple counts for each split."""
    print(f"Train examples: {len(train_df)} | ACOS tuples: {count_gold_tuples(train_df)}")
    print(f"Dev examples:   {len(dev_df)} | ACOS tuples: {count_gold_tuples(dev_df)}")
    print(f"Test examples:  {len(test_df)} | ACOS tuples: {count_gold_tuples(test_df)}")
    print("Test split is kept for final evaluation only.")


def print_sample(train_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Print one sample prompt and target for manual verification."""
    sample = train_df.iloc[0]
    prompt = config["prompt_template"].format(text=str(sample["text"]))
    print("\nSample prompt:")
    print(prompt)
    print("\nSample target:")
    print(str(sample["gold"]))


def ensure_output_dir_is_safe(output_dir: Path, overwrite: bool) -> None:
    """Prevent accidental overwriting of an existing final model folder."""
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists and is not empty: {output_dir}\n"
                "Use --overwrite only when you intentionally want to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def create_trainer(config: Dict[str, Any], overwrite: bool) -> Seq2SeqTrainer:
    """Create tokenizer, model, datasets, data collator, and Trainer."""
    output_dir = resolve_project_path(config["output_dir"])

    ensure_output_dir_is_safe(output_dir, overwrite=overwrite)

    train_df, dev_df, test_df = load_all_splits(config)
    print_split_counts(train_df, dev_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    train_dataset = AcosSeq2SeqDataset(train_df, tokenizer, config)
    dev_dataset = AcosSeq2SeqDataset(dev_df, tokenizer, config)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": overwrite,
        "do_train": True,
        "do_eval": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config["batch_size"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": False,
        "optim": "adamw_torch",
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
    }
    parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )


def run_dry_run(config: Dict[str, Any]) -> None:
    """Validate data/model wiring without training or saving a model."""
    train_df, dev_df, test_df = load_all_splits(config)
    print_split_counts(train_df, dev_df, test_df)
    print_sample(train_df, config)

    output_dir = resolve_project_path(config["output_dir"])
    print(f"\nFinal output folder later: {output_dir}")
    print(f"Output folder exists now: {output_dir.exists()}")
    print("Dry run will not create or write this folder.")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    sample_rows = train_df.head(2)
    prompts = [
        config["prompt_template"].format(text=str(text))
        for text in sample_rows["text"].tolist()
    ]
    targets = sample_rows["gold"].astype(str).tolist()

    inputs = tokenizer(
        prompts,
        max_length=int(config["max_input_length"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=int(config["max_target_length"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    print(f"Tokenized dry-run batch size: {inputs['input_ids'].shape[0]}")

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Dry-run forward loss: {outputs.loss.item():.4f}")
    print("Dry run complete. No training was started and no model was saved.")


def main() -> None:
    """Run final training only when this script is explicitly executed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        run_dry_run(config)
        return

    trainer = create_trainer(config, overwrite=args.overwrite)

    print("Starting final training...")
    trainer.train()

    output_dir = resolve_project_path(config["output_dir"])
    trainer.save_model(str(output_dir))
    trainer.tokenizer.save_pretrained(str(output_dir))
    print(f"Final model saved to: {output_dir}")
    print("Use data/processed/test.csv only after training for final model evaluation.")


if __name__ == "__main__":
    main()
