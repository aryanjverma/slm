# Model Card: Arithmetic Tutor SLM

## Intended Behavior

The model is intended to act as a Socratic tutor for addition and subtraction. It should never give the final answer before the student produces it. It should give one short next-step hint or guiding question.

## Base Model

Recommended default:

- `Qwen/Qwen2.5-0.5B-Instruct` for fast iteration.

Alternate small bases:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

## Training

Use `scripts/train_qlora.py` with `artifacts/data/train_chat.jsonl`. For v2, train on `artifacts/data/train_chat_v2.jsonl` or a concatenation of v1 and v2.

## Evaluation

Evaluate with `scripts/eval_hf_model.py` against `artifacts/data/eval_cases.jsonl`.

Primary metrics:

- No answer leak rate.
- Hint correctness.
- Step calibration.
- Robustness.
- Learning helpfulness.

## Limitations

This model is not intended to be a general math solver. It only targets addition/subtraction tutoring behavior. It may fail on word problems, multiplication/division, algebra, or long multi-turn tutoring unless those are added to the dataset and eval.
