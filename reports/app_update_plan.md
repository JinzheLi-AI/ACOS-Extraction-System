# App Update Plan

The clean project folder currently does not contain a Streamlit app file such as:

- `app.py`
- `streamlit_app.py`
- `src/streamlit_app.py`

Therefore, no app code was modified in this polishing step.

## Later Hugging Face Space Changes

When updating the Hugging Face Space app, change only the app content needed for the final model.

### 1. Model Path

Update the model loading path to:

```text
models/final_flan_t5_acos_model/
```

Keep a comment in the app code showing the previous deployed model path:

```python
# Old deployed model path: ../The Best Model/
```

### 2. README Metadata

Update the Hugging Face Space README or model card text to say that the current final model uses:

```text
google/flan-t5-base
```

### 3. Dataset Wording

Use this wording:

```text
The current model is trained on the processed OATS-ABSA dataset and evaluated on the all-domain OATS-ABSA test set.
```

Keep the test-set wording consistent with the all-domain OATS-ABSA evaluation setup.

### 4. Optimizer Wording

Use:

```text
Optimizer: AdamW
```

Keep this optimizer wording consistent across the app and documentation.

### 5. Final Metrics

Use the final post-processed metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.3341 |
| Recall | 0.2957 |
| F1 | 0.3137 |

Also mention the improvement over the old deployed model:

```text
Old deployed model F1: 0.2312
Final retrained model F1: 0.3137
Improvement: +0.0825 F1
```

### 6. EduRABSA Wording

Use this wording:

```text
EduRABSA was inspected for future education-domain expansion, but it was not used in the current FLAN-T5 training.
```

### 7. Local App Test

After copying the app into this clean project folder, run:

```bash
streamlit run app.py
```

Then test:

- single review prediction
- CSV/Excel batch prediction
- model performance page
- download output
- English input
- Bahasa Melayu input, if the translation pipeline is included
