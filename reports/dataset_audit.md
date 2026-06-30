# Dataset Audit

**Dataset folder:** `C:\Users\Administrator\OneDrive\Desktop\Semester 2 2025-2026\FYP1\Code\Sorted_Dataset`

## Available OATS-ABSA Files

- `coursera_train.txt`
- `coursera_dev.txt`
- `coursera_test.txt`
- `hotels_train.txt`
- `hotels_dev.txt`
- `hotels_test.txt`
- `amazon_train.txt`
- `amazon_dev.txt`
- `amazon_test.txt`

## Missing Expected OATS-ABSA Files

- None

## File-Level Counts

| File | Domain | Split | Exists | Examples | ACOS Tuples | Malformed Lines | Empty Tuple Lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coursera_train.txt | coursera | train | yes | 5304 | 6710 | 0 | 1884 |
| coursera_dev.txt | coursera | dev | yes | 683 | 929 | 0 | 215 |
| coursera_test.txt | coursera | test | yes | 173 | 236 | 0 | 726 |
| hotels_train.txt | hotels | train | yes | 5911 | 10180 | 0 | 1923 |
| hotels_dev.txt | hotels | dev | yes | 593 | 1052 | 0 | 386 |
| hotels_test.txt | hotels | test | yes | 130 | 242 | 0 | 850 |
| amazon_train.txt | amazon | train | yes | 5200 | 6314 | 0 | 2160 |
| amazon_dev.txt | amazon | dev | yes | 704 | 980 | 0 | 216 |
| amazon_test.txt | amazon | test | yes | 683 | 966 | 0 | 237 |

## Examples and Tuples per Split

| Split | Examples | ACOS Tuples |
| --- | --- | --- |
| train | 16415 | 23204 |
| dev | 1980 | 2961 |
| test | 986 | 1444 |

## Examples and Tuples per Domain

| Domain | Examples | ACOS Tuples |
| --- | --- | --- |
| coursera | 6160 | 7875 |
| hotels | 6634 | 11474 |
| amazon | 6587 | 8260 |

## EduRABSA Status

| File | Exists | Rows |
| --- | --- | --- |
| EduRABSA_train.csv | yes | 4000 |
| EduRABSA_test.csv | yes | 2500 |

**Recommendation:** EduRABSA is available and can be inspected, but it should not be included in current ACOS training unless it is converted into ACOS quadruple labels.

## Recommended Final Train / Dev / Test Composition

- Training candidate: all OATS train + dev files across Coursera, Hotels, and Amazon = **18395 examples** and **26165 ACOS tuples**.
- Test set: all OATS test files across Coursera, Hotels, and Amazon = **986 examples** and **1444 ACOS tuples**.
- The current test set should be described as **all-domain test**, not Coursera-only, if coursera_test + hotels_test + amazon_test are used.
- EduRABSA status for the current ACOS pipeline: **not used for training; inspected/planned for future ACOS annotation only**.