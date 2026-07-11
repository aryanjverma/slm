# APUSH FRQ Grader v4 Run Report — 2026-07-11

## Run

- Model: `apush-frq-grader-v4-assistant-only-r1`
- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Training rows: 226 reviewed synthetic cases
- Training: assistant-only QLoRA, 3 epochs, learning rate `2e-4`, 5% warmup
- Supervised assistant tokens: 41,985
- Seed: 13
- Evaluation: 53 held-out College Board-derived cases using the full v4 rubric prompt

## Results

| Metric | Result |
|---|---:|
| Structured-output validity | 98.11% (52/53) |
| Criterion exact match | 35.38% (75/212 criterion decisions) |
| Total-score exact match | 0.00% (0/53) |
| Total score within one point | 16.98% (9/53) |
| Total-score MAE | 3.0755 |
| Total-score QWK | 0.1158 |
| Evidence grounding | 88.68% (47/53) |
| Mean predicted total | 0.917 |

### Criterion exact-match rates

| Criterion | Exact match |
|---|---:|
| Thesis | 52.83% (28/53) |
| Contextualization | 39.62% (21/53) |
| Evidence | 16.98% (9/53) |
| Analysis/reasoning | 32.08% (17/53) |

## Interpretation

V4 largely learned the output contract and produced essay-grounded feedback, but it did not learn acceptable score calibration. The mean predicted total of 0.917, zero exact total matches, total MAE of 3.0755, and low QWK of 0.1158 indicate severe under-scoring and weak ordinal agreement. Evidence scoring is the largest criterion-level failure.

This run should be reported as an unsuccessful calibration experiment, not as a production-ready grader. Its useful gains are structured JSON validity and grounding; its main failure is converting grounded rubric analysis into correct criterion and total scores.

## Reproducibility and privacy

The source dataset is restricted to private, noncommercial assignment use with no redistribution. Share this aggregate report, training metadata, and aggregate summaries only. Do not publish training rows, held-out essays, or per-case result files.
