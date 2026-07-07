# Submission Checklist

## Core Deliverables

- [x] **Brainlift** — [`brainlift.md`](../brainlift.md) at repo root (APUSH LEQ grader focus, DOK 2–4)
- [x] **Behavior spec** — [`docs/behavior_spec.md`](behavior_spec.md)
- [x] **Litmus test** — [`docs/litmus_test.md`](litmus_test.md) with deterministic baseline numbers
- [x] **Eval harness** — `src/apush_frq_grader_slm/eval.py`
- [x] **Data pipeline** — `src/apush_frq_grader_slm/data.py`, `filters.py`
- [x] **Train/eval artifacts** — `artifacts/data/`, `artifacts/eval/`
- [x] **QLoRA script** — `scripts/train_qlora.py`
- [x] **Day 2 smoke loop** — `scripts/run_smoke_pipeline.py`, `scripts/train_smoke.py`, `artifacts/smoke/`, `artifacts/smoke_eval/`
- [ ] **Trained model** — `artifacts/models/apush-frq-grader-v1` (requires GPU run)

## Commands

```powershell
# Install
python -m pip install -e .

# Generate data
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200

# Deterministic eval
python -m apush_frq_grader_slm.cli.run_eval

# Day 2 smoke loop (30 train / 20 eval, CPU-friendly)
python scripts/run_smoke_pipeline.py

# Demo (paste LEQ prompt + essay → JSON grade)
python -m apush_frq_grader_slm.cli.demo

# Train (GPU)
python -m pip install -e ".[train]"
python scripts/train_qlora.py --data artifacts/data/train_chat.jsonl --output artifacts/models/apush-frq-grader-v1

# HF eval
python scripts/eval_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct --model-name qwen_base_prompted
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1
```

## Package

| Item | Value |
|------|-------|
| Package | `apush-frq-grader-slm` |
| Module | `src/apush_frq_grader_slm/` |
| CLI | `apush-grader-generate`, `apush-grader-eval`, `apush-grader-demo` |
| Model output | `artifacts/models/apush-frq-grader-v1` |

## Docs

- [x] `README.md`
- [x] `spec.md` (framework + structured JSON grader example)
- [x] `docs/eval_report.md`
- [x] `docs/error_analysis.md`
- [x] `artifacts/dataset_card.md`
- [x] `artifacts/model_card.md`

## Pre-Submit Verification

- [x] `python -m pytest tests/ -v` passes
- [x] `apush_grader_reference` scores 1.00 on held-out eval
- [x] `inflated_prompted_base` scores below reference (litmus gap documented)
- [x] Smoke loop runs end-to-end (`scripts/run_smoke_pipeline.py`)
- [ ] Tuned model beats baseline on grounding + robustness (after GPU train)
