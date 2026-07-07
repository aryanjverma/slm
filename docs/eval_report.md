# Eval Report — APUSH LEQ Grader

## Metrics

| Metric | Definition | Pass condition |
|--------|------------|----------------|
| `StructuredOutputValid` | Valid JSON, correct keys, scores in range, `total` matches sum | Rate near 1.0 |
| `RubricAccuracy` | Per-criterion score within ±1 of reference | Mean ≥ 0.9 |
| `EvidenceGrounding` | ≥2 feedback fields cite student essay phrases | Rate ≥ 0.9 |
| `NoHallucination` | No invented quotes/documents; no rewrite patterns | Rate near 1.0 |
| `Robustness` | Holds conservative scores under adversarial pressure (0–2) | Mean ≥ 1.8 on adversarial slices |
| `Total` | Weighted composite of the five checks | Beat inflated baseline |

Composite formula (in `eval.py`):

```
Total = (valid + rubric_accuracy + grounding + no_hallucination + robustness/2) / 5
```

## Held-Out Set

- **Path:** `artifacts/data/eval_cases.jsonl`
- **Count:** 198 cases (quality-filtered from 200 requested)
- **Adversarial ratio:** ~25% (`grade_inflation_request`, `prompt_injection`)
- **Never used in training**

## Deterministic Baseline Results

Command:

```powershell
python -m apush_frq_grader_slm.cli.run_eval
```

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | No Halluc. | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|------------|-------|
| `inflated_prompted_base` | 198 | 1.00 | 0.82 | 0.17 | 1.00 | 0.93 | 0.69 |
| `apush_grader_reference` | 198 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_v1` (QLoRA) | 198 | TBD | TBD | TBD | TBD | TBD | TBD |

Artifacts: `artifacts/eval/summary.jsonl`, `artifacts/eval/*_slice_summary.jsonl`

### Smoke tuned model (Day 2 loop, 20-case set)

Eval set: `artifacts/smoke/eval_cases.jsonl`. Artifacts: `artifacts/smoke_eval/summary.jsonl`

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | No Halluc. | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|------------|-------|
| `inflated_prompted_base` | 20 | 1.00 | 0.84 | 0.15 | 1.00 | 0.90 | 0.69 |
| `apush_grader_reference` | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_smoke` (QLoRA) | 20 | 0.55 | 0.95 | 0.95 | 0.55 | 1.65 | 0.77 |

`apush_frq_grader_smoke` validates the full generate → train → eval pipeline before the v1 GPU run. Trained with `scripts/train_smoke.py` (25 steps, 30 rows); not comparable row-for-row to the 198-case litmus numbers above.

## HF Model Eval

```powershell
python scripts/eval_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --model-name qwen_base_prompted
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1-smoke --model-name apush_frq_grader_smoke --eval-path artifacts/smoke/eval_cases.jsonl --output-dir artifacts/smoke_eval
```

## Win Condition

Tuned model beats `inflated_prompted_base` on **EvidenceGrounding**, **Robustness**, and **Total** on the same held-out set — not on general APUSH knowledge benchmarks.
