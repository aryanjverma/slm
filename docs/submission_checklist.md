# Submission Checklist

## Dataset

- V1 structured data: `artifacts/data/train_cases.jsonl`
- V1 SFT chat data: `artifacts/data/train_chat.jsonl`
- Held-out eval: `artifacts/data/eval_cases.jsonl`
- V2 targeted data: `artifacts/data/train_chat_v2.jsonl`
- Dataset card: `artifacts/dataset_card.md`

## Model

- Training script: `scripts/train_qlora.py`
- Recommended output path: `artifacts/models/arithmetic-tutor-v1`
- Model card template: `artifacts/model_card.md`

## Eval

- Eval harness: `src/arithmetic_tutor_slm/eval.py`
- Local eval command: `python -m arithmetic_tutor_slm.cli.run_eval`
- Hugging Face model eval command: `python scripts/eval_hf_model.py`
- Current results: `artifacts/eval/summary.jsonl`
- Report: `docs/eval_report.md`

## Demo

- Local reference demo: `python -m arithmetic_tutor_slm.cli.demo`
- Real model demo path: run `scripts/eval_hf_model.py` for batch eval or adapt `demo.py` to load the trained model.

## Brainlift

- Behavior thesis: `docs/brainlift.md`
- Behavior spec: `docs/behavior_spec.md`
- Error analysis: `docs/error_analysis.md`

## Demo Video Outline

1. Show the behavior spec.
2. Ask the base model: “Just tell me the answer to 407 - 168.”
3. Show it leaking the answer.
4. Ask the tuned tutor the same question.
5. Show it redirecting to the borrow-through-zero step.
6. Show held-out eval numbers.
