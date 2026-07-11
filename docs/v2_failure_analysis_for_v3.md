# V2 Failure Analysis

This report is generated from saved model outputs. It is diagnostic, not a verified golden evaluation.

## Overall

- Raw strict schema: 19/53 (35.9%)
- Parseable JSON: 44/53 (83.0%)
- Usable after representation-only normalization: 36/53 (67.9%)
- Normalized total MAE: 1.8889
- Normalized within-one: 38.9%
- Normalized QWK: 0.072
- Repetition: 13/53

## Failure categories

- json_schema_invalid: 25
- likely_max_token_truncation: 8
- malformed_or_non_json: 1
- valid_structured_json: 19

## Dominant normalized score vectors

- `(1, 1, 1, 1)`: 25
- `(0, 0, 0, 0)`: 10
- `(1, 1, 2, 1)`: 1

## Length, rubric, and reference-score slices

### Essay Length

- 250_399: n=11, raw=18.2%, parseable=100.0%, normalized=72.7%, MAE=1.5, truncated=0.0%, repetition=18.2%
- 400_599: n=6, raw=16.7%, parseable=50.0%, normalized=33.3%, MAE=1.0, truncated=50.0%, repetition=50.0%
- 600_plus: n=5, raw=20.0%, parseable=60.0%, normalized=40.0%, MAE=3.0, truncated=40.0%, repetition=20.0%
- under_250: n=31, raw=48.4%, parseable=87.1%, normalized=77.4%, MAE=2.0, truncated=9.7%, repetition=22.6%

### Reference Total

- 1: n=3, raw=33.3%, parseable=100.0%, normalized=33.3%, MAE=1.0, truncated=0.0%, repetition=33.3%
- 2: n=12, raw=33.3%, parseable=83.3%, normalized=75.0%, MAE=2.0, truncated=16.7%, repetition=33.3%
- 3: n=3, raw=33.3%, parseable=100.0%, normalized=66.7%, MAE=2.0, truncated=0.0%, repetition=0.0%
- 4: n=15, raw=40.0%, parseable=93.3%, normalized=80.0%, MAE=1.0, truncated=6.7%, repetition=20.0%
- 5: n=2, raw=0.0%, parseable=100.0%, normalized=100.0%, MAE=1.0, truncated=0.0%, repetition=0.0%
- 6: n=18, raw=38.9%, parseable=66.7%, normalized=55.6%, MAE=3.1, truncated=27.8%, repetition=27.8%

### Rubric Version

- 2023_leq: n=18, raw=50.0%, parseable=94.4%, normalized=94.4%, MAE=2.0, truncated=5.6%, repetition=16.7%
- 2024_2026_leq: n=35, raw=28.6%, parseable=77.1%, normalized=54.3%, MAE=1.7895, truncated=20.0%, repetition=28.6%

## Normalization actions

- computed_total: 36
- extracted_balanced_object: 13

## Provenance and extraction warnings

- missing_source_url: 53
- missing_file_sha256: 53
- missing_extraction_method: 53
- missing_extraction_confidence: 53
- commentary_text_present_in_essay: 28
