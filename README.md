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

## Data v2 Pipeline

The v2 path replaces duplicate short templates with persona-driven essays and independent
labels. It uses 60 original prompt families, keeps whole families out of train, and blocks final
artifacts until strict leakage, provenance, consensus, and human-review gates pass.

```powershell
python scripts/build_prompt_catalog.py --protected-prompts artifacts/data/eval_cb_cases.jsonl
python scripts/gen_realistic_tasks.py --limit 100
python scripts/generate_synthetic_candidates.py --limit 100
python scripts/audit_synthetic_candidates.py
python scripts/grade_synthetic_candidates.py --limit 100
python scripts/assemble_realistic_dataset.py
python scripts/review_synthetic_v2.py --create-template
# Complete artifacts/reviews/synthetic_v2.jsonl, then:
python scripts/review_synthetic_v2.py
python scripts/build_v2_artifacts.py --target-count 100
python scripts/run_v2_checkpoints.py                 # prepare 200/500/1200 subsets
# Add --execute on a GPU machine to train and evaluate every available checkpoint.
```

`generate_synthetic_candidates.py` and `grade_synthetic_candidates.py` require the `judge` extra
and `OPENAI_API_KEY`. Offline reader outputs can be resolved with
`scripts/resolve_synthetic_grades.py`. Official College Board artifacts have a separate written-
permission and manual-review gate; see `docs/data_permission_checkpoint.md`.

## V3 Failure-Driven Pipeline

V3 keeps v2 reproducible and adds a separate layered grader, immutable 200-row audited training
artifact, assistant-only training, checkpoint generation evaluation, and a locked official split.
The model emits four scores plus four feedback strings; the application computes `total` and never
clamps or otherwise changes a selected criterion score.

```powershell
python scripts/analyze_v2_for_v3.py
python scripts/build_v3_dataset.py artifacts/data/train_cases.jsonl artifacts/data/v2/train_realistic_v2.jsonl artifacts/data/v2/train_adversarial_v2.jsonl
python scripts/eval_v3.py --model Qwen/Qwen2.5-0.5B-Instruct --model-name Qwen2.5-0.5B-base
python scripts/benchmark_v3_dev.py --base-summary PATH_TO_BASE_SET1_SUMMARY
python scripts/train_v3.py --model Qwen/Qwen2.5-0.5B-Instruct --output artifacts/models/qwen-0.5b-v3 --dev-eval-command "python scripts/eval_v3.py --model {checkpoint} --model-name qwen-0.5b-v3 --output-dir artifacts/eval/v3/qwen-0.5b"
python scripts/train_v3.py --model Qwen/Qwen2.5-1.5B-Instruct --output artifacts/models/qwen-1.5b-v3 --dev-eval-command "python scripts/eval_v3.py --model {checkpoint} --model-name qwen-1.5b-v3 --output-dir artifacts/eval/v3/qwen-1.5b"
```

`eval_v3.py` always selects the 27 set1 rows unless `--final-evaluation` and an exact passing lock
manifest are both supplied. A successful set2 run writes a receipt that prevents a second run.
The local 53-row College Board-derived file still carries the provenance and extraction warnings
reported in `docs/v2_failure_analysis_for_v3.md`; saved v2 results are diagnostics, not golden data.

## Day 2 Smoke Test (50 cases, full loop)

Proves generate → train → eval on a tiny dataset before the real v1 run:

```powershell
python -m pip install -e ".[train]"
python scripts/run_smoke_pipeline.py
```

This writes 30 train / 20 eval rows to `artifacts/smoke/`, fine-tunes a LoRA adapter to
`artifacts/models/apush-frq-grader-v1-smoke/` (CPU-friendly via `scripts/train_smoke.py`),
and evaluates baselines plus the tuned model under `artifacts/smoke_eval/`.

To re-run eval only (adapter already trained):

```powershell
python scripts/run_smoke_pipeline.py --skip-generate --skip-train
```

## Train With QLoRA

Run on a GPU machine after installing training extras:

```powershell
python -m pip install -e ".[train]"
python scripts/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --data artifacts/data/train_chat.jsonl --output artifacts/models/apush-frq-grader-v1
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1 --eval-path artifacts/data/eval_cb_cases.jsonl --real-eval
```

## Current Deterministic Eval (198-case held-out set)

On the same eval harness used for the litmus test:

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|-------|
| `inflated_prompted_base` | 198 | 1.00 | 0.82 | 0.17 | 0.93 | 0.69 |
| `apush_grader_reference` (SFT target) | 198 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_v1` (QLoRA) | 198 | TBD | TBD | TBD | TBD | TBD |

The inflated baseline simulates a lenient prompted model: valid JSON but generic feedback and score inflation on weak essays. Fine-tuning should close the gap to the reference grader on grounding and adversarial slices.

### Smoke tuned model (Day 2 loop, 20-case held-out set)

| Model | Cases | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
|-------|-------|------------|-------------|-----------|------------|-------|
| `inflated_prompted_base` | 20 | 1.00 | 0.84 | 0.15 | 0.90 | 0.69 |
| `apush_grader_reference` (SFT target) | 20 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| `apush_frq_grader_smoke` (QLoRA) | 20 | 0.55 | 0.95 | 0.95 | 1.65 | 0.77 |

`apush_frq_grader_smoke` is the Day 2 proof-of-loop adapter: LoRA fine-tuned on 30 synthetic train rows for 25 steps (`scripts/train_smoke.py`), evaluated on `artifacts/smoke/eval_cases.jsonl`. It confirms generate → train → eval works end-to-end and already beats the inflated baseline on grounding (0.95 vs 0.15) and total (0.77 vs 0.69). JSON validity is still low (0.55) because 25 steps on 30 examples is intentionally minimal — the full v1 run on ~997 rows is the real target.
