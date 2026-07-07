# Model Card — APUSH FRQ Grader v1

## Model Description

QLoRA fine-tuned adapter on `Qwen/Qwen2.5-0.5B-Instruct` that grades APUSH LEQs and returns structured JSON with per-criterion scores and evidence-grounded feedback.

## Intended Use

- **Primary:** Automated APUSH LEQ draft grading with explainable, rubric-aligned JSON output.
- **Out of scope:** High-stakes exam scoring without human review; DBQ/SAQ; essay rewriting.

## Behavior Contract

See [`docs/behavior_spec.md`](../docs/behavior_spec.md).

## Training

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | QLoRA (Unsloth) |
| Data | `artifacts/data/train_chat.jsonl` (~997 rows) |
| Script | `scripts/train_qlora.py` |
| Output | `artifacts/models/apush-frq-grader-v1` |

## Evaluation

### Litmus eval (198-case held-out set)

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|-------|
| `inflated_prompted_base` | 198 | 1.00 | 0.82 | 0.17 | 0.93 | 0.69 |
| `apush_grader_reference` | 198 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_v1` | 198 | TBD | TBD | TBD | TBD | TBD |

Eval set: `artifacts/data/eval_cases.jsonl` (198 cases).

### Smoke tuned model — `apush_frq_grader_smoke` (Day 2, 20-case set)

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | No Halluc. | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|------------|-------|
| `inflated_prompted_base` | 20 | 1.00 | 0.84 | 0.15 | 1.00 | 0.90 | 0.69 |
| `apush_grader_reference` | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_smoke` | 20 | 0.55 | 0.95 | 0.95 | 0.55 | 1.65 | 0.77 |

`apush_frq_grader_smoke` is the Day 2 proof-of-loop adapter, not the production v1 model. LoRA fine-tuned for 25 steps on 30 synthetic rows (`scripts/train_smoke.py`), output at `artifacts/models/apush-frq-grader-v1-smoke/`, evaluated on `artifacts/smoke/eval_cases.jsonl`. It confirms the pipeline works and beats the inflated baseline on grounding (0.95 vs 0.15) before the full ~997-row v1 GPU train.

## Limitations

- Trained on **synthetic** essays; real student writing may differ.
- LEQ rubric only (not DBQ document analysis or SAQ).
- Small model may drift to prose or malformed JSON without sufficient SFT steps.
- Does not replace a human AP reader for summative assessment.

## Ethical Considerations

- Conservative scoring under grade-inflation pressure is an explicit design goal.
- Feedback must cite student text, not invent sources — eval checks for hallucination patterns.
- Teachers should use output as formative feedback, not sole basis for grades.
