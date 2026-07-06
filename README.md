# Arithmetic Tutor SLM

This project trains and evaluates a small learning model for one narrow behavior: Socratic tutoring for addition and subtraction. The model should help a student learn the next arithmetic step without leaking the final answer.

Behavior spec:

> The model is a Socratic tutor for addition and subtraction. It never states the final numeric answer unless the student has already produced it; instead, it identifies the student's current step or mistake and asks one short guiding question or gives one calibrated hint for the next step.

## What Is Included

- `src/arithmetic_tutor_slm/`: reusable data, filtering, eval, and demo code.
- `scripts/train_qlora.py`: Unsloth QLoRA training script for a small instruct base model.
- `scripts/eval_hf_model.py`: held-out eval for a base or tuned Hugging Face model.
- `scripts/make_v2_dataset.py`: targeted v2 data generation for known failure modes.
- `artifacts/data/`: generated train/eval JSONL data.
- `artifacts/eval/`: base-vs-reference behavior results.
- `docs/`: behavior spec, eval report, and project notes.

## Quick Start

```powershell
python -m pip install -e .
$env:PYTHONPATH='src'
python -m arithmetic_tutor_slm.cli.generate_dataset --train-count 1000 --eval-count 200 --output-dir artifacts/data
python -m arithmetic_tutor_slm.cli.run_eval --eval-path artifacts/data/eval_cases.jsonl --output-dir artifacts/eval
python -m arithmetic_tutor_slm.cli.demo
```

## Train With QLoRA

Run this on a GPU machine after installing training extras:

```powershell
python -m pip install -e ".[train]"
$env:PYTHONPATH='src'
python scripts/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --data artifacts/data/train_chat.jsonl --output artifacts/models/arithmetic-tutor-v1
python scripts/eval_hf_model.py --model artifacts/models/arithmetic-tutor-v1 --model-name arithmetic_tutor_v1
```

## Current Local Eval

The deterministic smoke/reference eval passes on the 200-case held-out set:

- `prompted_base_leaky`: no-answer leak rate `0.00`, total `0.40`.
- `socratic_tutor_reference`: no-answer leak rate `1.00`, total `1.00`.

The next project step is to run the QLoRA script on GPU and compare the real tuned model against the same held-out eval.