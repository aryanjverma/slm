# Evaluation Report

The eval is built before model training and focuses on behavior, not broad math capability. Every model sees the same held-out arithmetic tutoring cases from `artifacts/data/eval_cases.jsonl`.

## Metrics

- `NoAnswerLeakRate`: response does not contain the final numeric answer.
- `HintCorrectness`: response points to the expected next step or mistake type.
- `StepCalibration`: response is short and stays to one next step.
- `Robustness`: holds the tutor contract under pressure.
- `LearningHelpfulness`: gives a useful next move for the student.

## Current Held-Out Results

| Model | Cases | No Answer Leak | Hint Correct | Calibrated | Robustness | Helpfulness | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prompted_base_leaky` | 200 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.40 |
| `socratic_tutor_reference` | 200 | 1.00 | 1.00 | 1.00 | 2.00 | 2.00 | 1.00 |

The `prompted_base_leaky` adapter models the failure the fine-tune is meant to remove: it is arithmetically helpful but violates the no-answer contract. The reference tutor shows the target behavior that the SFT dataset encodes.

## How To Evaluate A Real Model

After QLoRA training, evaluate the base model and the tuned adapter on the same held-out set:

```powershell
$env:PYTHONPATH='src'
python scripts/eval_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --model-name qwen_base
python scripts/eval_hf_model.py --model artifacts/models/arithmetic-tutor-v1 --model-name arithmetic_tutor_v1
```

The tuned model should beat the prompted base on `NoAnswerLeakRate`, `Robustness`, and `Total`.
