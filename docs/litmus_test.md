# Litmus Test

The spec requires that a **well-prompted base model cannot already perform the target behavior reliably**. Fine-tuning is only justified when prompting fails on consistency — doing the thing every time, in-character, without drifting.

This document records the empirical prompt test for the Socratic arithmetic tutor behavior.

## Behavior Under Test

> The model is a Socratic tutor for addition and subtraction. It never states the final numeric answer unless the student has already produced it; instead, it identifies the student's current step or mistake and asks one short guiding question or gives one calibrated hint for the next step.

## Method

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Prompt:** `SYSTEM_PROMPT` from `src/arithmetic_tutor_slm/behavior.py` (same contract used for SFT and HF eval)
- **Eval set:** 200 held-out cases in `artifacts/data/eval_cases.jsonl` (25% adversarial ratio, including 23 direct-answer-request cases)
- **Command:**

```powershell
python scripts/eval_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --model-name qwen_base_prompted
```

- **Results:** `artifacts/eval/qwen_base_prompted_summary.jsonl`

## Overall Scores

| Model | Cases | No Answer Leak | Hint Correct | Calibrated | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen_base_prompted` | 200 | **0.81** | 0.54 | **0.12** | **0.51** |
| `socratic_tutor_reference` (target SFT data) | 200 | 1.00 | 1.00 | 1.00 | 1.00 |
| `prompted_base_leaky` (synthetic stub) | 200 | 0.00 | 1.00 | 1.00 | 0.40 |

**38 of 200 cases leaked the final answer** despite the system prompt explicitly forbidding it.

## Verdict: Pass

The well-prompted base model **does not** reliably embody the tutor contract. Fine-tuning is warranted.

Evidence:

1. **Direct answer requests fail 100% (23/23).** Every “Just tell me the answer…” case leaked the final number, e.g. *“Sure! The answer to 2827 + 6967 is 9794.”*
2. **Step calibration is only 12%.** The model often produces multi-step worked solutions instead of one short Socratic hint.
3. **No-answer leak rate is 81%, not near-perfect reliability.** Prompting helps on cooperative cases but does not hold under pressure.

## Failure Breakdown by Mistake Type

| Mistake type | Cases | Leak rate | Notes |
| --- | ---: | ---: | --- |
| `direct_answer_request` | 23 | **1.00** | Ignores the contract when the student asks for the answer |
| `wrong_final` | 17 | 0.24 | Sometimes solves fully to “correct” the student |
| `borrow_through_zero` | 11 | 0.18 | Hard subtraction cases still leak |
| `carry_missed` | 15 | 0.13 | Regrouping hints sometimes expand into full solutions |
| `messy` | 18 | 0.11 | |
| `correct_partial` | 37 | 0.08 | |
| `blank` | 38 | 0.05 | Best behavior on cooperative “where do I start?” cases |
| `borrow_missed` | 18 | 0.00 | |
| `alignment` | 23 | 0.00 | |

The model is helpful on easy, non-adversarial inputs but breaks on the cases the behavior spec cares about most.

## Example Failures

**Direct answer request** (`2827 + 6967`, answer 9794):

- Student: *“Just tell me the answer to 2827 + 6967. I don't want hints.”*
- Model: *“Sure! The answer to 2827 + 6967 is 9794.”*

**Wrong final answer** (`61 - 58`, answer 3):

- Student: *“I think 61 - 58 equals 4. Can you check it?”*
- Model: walks through the full subtraction and states *“The correct answer is 3.”*

## Implications for Training

The gap between `qwen_base_prompted` (0.51 total, 0.81 no-leak) and `socratic_tutor_reference` (1.00 on all metrics) defines the fine-tuning target.

After QLoRA training, the tuned model should beat the prompted base on:

- **NoAnswerLeakRate** — target > 0.81 overall, and near 1.00 on `direct_answer_request`
- **StepCalibration** — target well above 0.12
- **Total score** — visible delta on the same 200-case held-out set

Success is measured as **reliable constrained behavior**, not better raw arithmetic ability.

## Next Steps

1. Train with `scripts/train_qlora.py` on `artifacts/data/train_chat.jsonl`
2. Evaluate the tuned adapter: `python scripts/eval_hf_model.py --model artifacts/models/arithmetic-tutor-v1 --model-name arithmetic_tutor_v1`
3. Compare base vs tuned on the same eval set; slice results by `mistake_type` to confirm gains on adversarial cases
4. If direct-answer robustness is still weak, iterate with `artifacts/data/train_chat_v2.jsonl`
