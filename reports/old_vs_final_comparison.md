# Old vs Final Model Comparison

**Final prediction file:** `C:\Users\Administrator\Documents\Codex\2026-05-31\files-mentioned-by-the-user-acos\course-acos-extraction\data\predictions\predictions_final.csv`
**Evaluated examples:** 986

## Without Post-processing

| Metric | Old Deployed Model | Final Model | Delta |
| --- | ---: | ---: | ---: |
| Precision | 0.2458 | 0.3333 | +0.0875 |
| Recall | 0.2154 | 0.2957 | +0.0803 |
| F1 | 0.2296 | 0.3134 | +0.0838 |
| Correct tuples | 311 | 427 | +116 |
| Predicted tuples | 1265 | 1281 | +16 |
| Gold tuples | 1444 | 1444 | +0 |

## With Post-processing

| Metric | Old Deployed Model | Final Model | Delta |
| --- | ---: | ---: | ---: |
| Precision | 0.2504 | 0.3341 | +0.0837 |
| Recall | 0.2147 | 0.2957 | +0.0810 |
| F1 | 0.2312 | 0.3137 | +0.0825 |
| Correct tuples | 310 | 427 | +117 |
| Predicted tuples | 1238 | 1278 | +40 |
| Gold tuples | 1444 | 1444 | +0 |

## Verdict

The final model is better than the old deployed model by post-processed F1.