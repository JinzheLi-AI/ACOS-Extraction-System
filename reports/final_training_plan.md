# Final Training Plan

## Dataset Used

The final training preparation uses the standardized processed OATS-ABSA dataset files:

- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

EduRABSA is not included in this final training preparation because it has not yet been converted into ACOS quadruple labels.

## Assumptions

- The processed CSV files are already generated and verified.
- `train.csv` is used only for model training.
- `dev.csv` is used only for validation during training.
- `test.csv` is kept untouched until final evaluation.
- The current deployed model in `../The Best Model/` is not overwritten.

## Confirmed Dataset Counts

| Split | Examples | ACOS Tuples |
| --- | ---: | ---: |
| Train | 16,415 | 23,204 |
| Dev | 1,980 | 2,961 |
| Test | 986 | 1,444 |

The test split is kept only for final evaluation after training.

## Model Name

`google/flan-t5-base`

## Prompt Format

```text
Extract ACOS quadruples from this course review: {text}
```

## Target Format

```text
(aspect | category | opinion | sentiment)
```

For multiple tuples, the target uses:

```text
(aspect | category | opinion | sentiment) ; (aspect | category | opinion | sentiment)
```

## Optimizer

AdamW, implemented through the Transformers training argument:

```text
optim = adamw_torch
```

## Output Model Folder

The final retrained model will later be saved to:

```text
models/final_flan_t5_acos_model/
```

The script includes a safety check. If this folder already exists and is not empty, training stops unless `--overwrite` is explicitly provided.

Important: the old deployed model remains separate in:

```text
../The Best Model/
```

The final retrained model will not overwrite that deployed model. It will be saved separately under:

```text
models/final_flan_t5_acos_model/
```

## Commands

Run a safety dry-run first:

```bash
python experiments/train_final_model.py --dry-run
```

The dry-run loads the config and processed CSV files, prints dataset counts, prints one sample prompt and target, loads the tokenizer and base model, tokenizes a small batch, and runs one tiny forward check. It does not call `trainer.train()`, does not save a model, and does not create `models/final_flan_t5_acos_model/`.

Start real training only when ready:

```bash
python experiments/train_final_model.py
```

Use overwrite only if the final output folder already exists and you intentionally want to replace it:

```bash
python experiments/train_final_model.py --overwrite
```

## What Will Happen During Training

- Load `train.csv`, `dev.csv`, and `test.csv`.
- Use `train.csv` for supervised fine-tuning.
- Use `dev.csv` for validation during training.
- Keep `test.csv` untouched for final evaluation after training.
- Load `google/flan-t5-base`.
- Convert each input review into the configured prompt format.
- Train the model to generate ACOS targets.
- Save the final model and tokenizer to `models/final_flan_t5_acos_model/`.

## What Will Not Be Changed

- No existing deployed model files will be overwritten.
- The old Kaggle notebook will not be modified.
- The Streamlit app will not be rewritten.
- No LoRA training is added in this step.
- No final model is generated until training is intentionally started.
- No metric is manually created or faked.

## Verification Before Training

Before starting final training, confirm:

- `data/processed/train.csv` exists and contains 16,415 examples.
- `data/processed/dev.csv` exists and contains 1,980 examples.
- `data/processed/test.csv` exists and contains 986 examples.
- `config/final_training_config.json` uses `google/flan-t5-base`.
- `models/final_flan_t5_acos_model/` does not already contain a final model unless overwrite is intentional.

## Next Steps After Training

1. Generate final predictions from `models/final_flan_t5_acos_model/`.
2. Evaluate the final model on `data/processed/test.csv`.
3. Compare the old deployed model against the final retrained model.
4. Update the README, report, and app only after real final metrics are available.
