# Project Summary

## Project Goal

This project aims to build a university course evaluation system that extracts structured ACOS information from unstructured student feedback. Instead of returning only an overall sentiment label, the system extracts aspect, category, opinion, and sentiment so instructors or academic administrators can understand what students are commenting on and how they feel about it.

## Dataset

The current training and evaluation pipeline uses OATS-ABSA. The processed dataset contains Coursera, Hotels, and Amazon data and is evaluated on the all-domain OATS-ABSA test set.

| Split | Examples | ACOS Tuples |
| --- | ---: | ---: |
| Train | 16,415 | 23,204 |
| Dev | 1,980 | 2,961 |
| Test | 986 | 1,444 |

EduRABSA was inspected because it is closer to the education domain, but it was not used in the current FLAN-T5 training. It is planned for future expansion after ACOS annotation or conversion.

## Model

The model is based on `google/flan-t5-base`. ACOS extraction is formulated as a text-to-text generation task.

Prompt format:

```text
Extract ACOS quadruples from this course review: {text}
```

Target format:

```text
(aspect | category | opinion | sentiment)
```

The final retrained model is saved separately from the old deployed model:

- Old deployed model: `../The Best Model/`
- Final retrained model: `models/final_flan_t5_acos_model/`

## Training Process

The final training uses the processed OATS-ABSA train split for training, the dev split for validation, and the test split only for final evaluation. The optimizer is AdamW. The model learns to generate structured ACOS tuples from review text using the same prompt format used during inference.

## Final Evaluation

Final retrained model results on the all-domain OATS-ABSA test set:

| Evaluation Setting | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Without post-processing | 0.3333 | 0.2957 | 0.3134 |
| With post-processing | 0.3341 | 0.2957 | 0.3137 |

The final prediction file contains 986 examples and is stored at:

```text
data/predictions/predictions_final.csv
```

## Improvement Over Old Model

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Old deployed model with post-processing | 0.2504 | 0.2147 | 0.2312 |
| Final retrained model with post-processing | 0.3341 | 0.2957 | 0.3137 |

The final model improves post-processed Exact Match F1 by:

```text
+0.0825
```

## Error Analysis Findings

Final model error analysis shows:

- 427 exact matched tuples
- 220 missed tuples
- 54 hallucinated tuples
- 145 wrong sentiment errors
- 299 wrong category errors
- 508 wrong opinion errors
- 235 wrong aspect errors
- 797 partial matches
- 0 malformed outputs

The most important finding is that the final model produces valid tuple format, but still struggles with exact opinion spans and category selection. Many predictions are partially correct but fail Exact Match because one field is different.

## Limitations

- Exact Match is strict and may understate partially correct predictions.
- The final model is trained on OATS-ABSA, not EduRABSA.
- EduRABSA still needs ACOS-style annotation or conversion before training.
- The clean project folder does not currently include the Streamlit app source file.
- The model needs more domain-specific education feedback data for stronger university course evaluation performance.

## Future Work

- Update the Hugging Face Space app to load `models/final_flan_t5_acos_model/`.
- Add EduRABSA or local university course review data after ACOS annotation and validation.
- Improve category and opinion extraction.
- Add optional LoRA fine-tuning for a more efficient FYP2 experiment.
- Add more demo examples, including success cases and error cases.
- Evaluate multilingual English and Bahasa Melayu inputs more systematically.
