# APUSH FRQ Grader SLM

Train and evaluate a small open model for one narrow behavior: **grade APUSH LEQs** against the College Board 6-point rubric and explain each score with evidence grounded in the student's essay.

Behavior spec:

> The model is an APUSH LEQ grader and explainer. Given a prompt and student essay, it returns one valid JSON object with per-criterion scores (thesis, contextualization, evidence, analysis/reasoning) and short explanations that quote or paraphrase evidence from the student's text. It never invents historical facts, documents, or quotes; never rewrites the essay; and never inflates scores under student pressure.

## What Is Included

- `src/apush_frq_grader_slm/`: data generation, rubric validation, filtering, eval, and demo code.
- `scripts/train_qlora.py`: Unsloth QLoRA training on `Qwen/Qwen2.5-0.5B-Instruct`.
- `scripts/eval_hf_model.py`: held-out eval for a base or tuned Hugging Face model.
- `scripts/make_v2_dataset.py`: v2 data oversampling adversarial failure slices.
- `artifacts/data/`: generated train/eval JSONL and chat SFT rows.
- `artifacts/eval/`: deterministic baseline vs reference eval results.
- `docs/`: behavior spec, litmus test, eval report, and submission notes.

## Quick Start

```powershell
python -m pip install -e .
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200 --output-dir artifacts/data
python -m apush_frq_grader_slm.cli.run_eval --eval-path artifacts/data/eval_cases.jsonl --output-dir artifacts/eval
python -m apush_frq_grader_slm.cli.demo
```

## Train With QLoRA

Run on a GPU machine after installing training extras:

```powershell
python -m pip install -e ".[train]"
python scripts/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --data artifacts/data/train_chat.jsonl --output artifacts/models/apush-frq-grader-v1
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1
```

## Current Deterministic Eval (198-case held-out set)

On the same eval harness used for the litmus test:

| Model | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
|-------|------------|-------------|-----------|------------|-------|
| `inflated_prompted_base` | 1.00 | 0.82 | 0.17 | 0.93 | 0.69 |
| `apush_grader_reference` (SFT target) | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |

The inflated baseline simulates a lenient prompted model: valid JSON but generic feedback and score inflation on weak essays. Fine-tuning should close the gap to the reference grader on grounding and adversarial slices.
