# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A one-week SLM build: fine-tune `Qwen/Qwen2.5-0.5B-Instruct` (QLoRA) to do **one** narrow behavior reliably — grade APUSH LEQ essays against the College Board 6-point rubric and return **one valid JSON object** with per-criterion scores + essay-grounded feedback. The canonical behavior contract lives in `src/apush_frq_grader_slm/behavior.py` (`BEHAVIOR_SPEC`, `SYSTEM_PROMPT`); `spec.md` is the assignment brief.

**The governing thesis (from `spec.md`):** the dataset is the deliverable, not the model. Fix disappointing results in the *data*, not hyperparameters. The eval must exist before training. Success = the tuned model beats the base on spec adherence + robustness, measured on the same held-out cases.

## Commands

```powershell
python -m pip install -e .                 # core (data/eval/demo)
python -m pip install -e ".[train]"        # + unsloth/trl/peft (GPU for real runs)
python -m pip install -e ".[ingest]"       # + pdfplumber/openai (AP Central ingestion)

python -m pytest                           # run all tests
python -m pytest tests/test_core.py        # one file
python -m pytest tests/test_core.py::CoreBehaviorTests::test_generated_cases_pass_quality_gate  # one test

python -m ruff check .                     # lint (line-length 100, py310 target)
```

Tests are written as `unittest.TestCase` classes but run under pytest. Ingest tests self-skip when their `tests/fixtures/` text files are absent, so a green run does not guarantee ingest coverage ran.

Core pipeline (synthetic, CPU-friendly):
```powershell
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200 --output-dir artifacts/data
python -m apush_frq_grader_slm.cli.run_eval --eval-path artifacts/data/eval_cases.jsonl --output-dir artifacts/eval
python -m apush_frq_grader_slm.cli.demo
python scripts/run_smoke_pipeline.py       # full generate->train->eval on 50 cases (Day-2 proof-of-loop)
```

Real training + eval (GPU):
```powershell
python scripts/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --data artifacts/data/train_chat.jsonl --output artifacts/models/apush-frq-grader-v1
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1 --eval-path artifacts/data/eval_cases.jsonl        # litmus eval
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1 --eval-path artifacts/data/eval_real_cases.jsonl --real-eval   # real CB agreement
```

## Architecture

Data flows in one direction: **spec → synthetic generation → quality gate → chat rows → QLoRA → eval**. Real AP essays enter only as an eval track.

**Package `src/apush_frq_grader_slm/`** — importable library; the `cli/` subpackage and `scripts/` are thin entrypoints over it.

- `behavior.py` — the single source of truth for the spec, output JSON schema, `SYSTEM_PROMPT` (used at both train and inference), and the LLM-judge rubric. Change grading behavior here first.
- `rubric.py` — College Board rubric: `CRITERIA` tuple, per-criterion score ranges (thesis/contextualization 0–1, evidence/analysis_reasoning 0–2, total 0–6), `compute_total`, and `validate_grade_payload` (the structural JSON contract).
- `schemas.py` — pydantic models. `FRQCase` is the universal record for every example (synthetic or real): prompt, student_response, `reference_scores`, `reference_feedback`, `failure_type`, `split` (train/eval/adversarial), and `assistant_response` (the JSON target string).
- `data.py` — synthetic generation. Builds essays per `FailureType` slice (weak_thesis, missing_context, evidence_list, wrong_period, borderline_complexity, grade_inflation_request, prompt_injection, strong), rule-grades them, and emits `to_chat_rows()` (system/user/assistant messages) for SFT. `_reference_grade()` is the template style all feedback must match.
- `filters.py` — the quality gate. `passes_quality_gate()` enforces valid JSON, correct ranges, `feedback_references_essay()` grounding, and rejects rewrites/hallucinated quotes. Every case (synthetic and ingested) must pass this before use.
- `baselines.py` — local `ResponseAdapter`s used without a GPU: `InflatedPromptedBase` (simulates a lenient prompted model — valid JSON but inflated scores + generic feedback) and `ReferenceGrader` (the SFT target). The tuned model's job is to move from the former toward the latter.
- `eval.py` — deterministic scorer. `score_response()` yields the 5 litmus metrics (structured_output_valid, rubric_accuracy, evidence_grounding, no_hallucination, robustness) + composite; `summarize_by_slice()` breaks results down by `failure_type`; `summarize_real_eval()` adds College Board score agreement (exact/±1 per row, QWK on totals) for the real track.
- `ingest/` — AP Central pipeline: `apc_parser.py` (pdfplumber → `RawAPCSample`), `tomrichey_parser.py`, `quizlet_parser.py`, `dedup.py`, and `distill.py` (`raw_sample_to_frq_case` — CB scores are ground truth; commentary is distilled into essay-anchored JSON feedback, falling back to `data.py` templates when `--distill` is off).

**Two eval tracks, deliberately separate:**
- **Litmus** (`eval_cases.jsonl`, synthetic) — contract + adversarial slices; the base-vs-tuned regression signal.
- **Real** (`eval_real_cases.jsonl`, ~54 College Board essays) — external validity vs official CB scores, run with `--real-eval`.

**Hard invariant — real essays are eval-only.** Real/ingested essays (tags `ap_central`, `real_eval`, `tom_richey`, `quizlet`) must never enter `train_chat.jsonl`. `scripts/build_mixed_dataset.py::assert_no_real_essays()` enforces this and will raise if a real essay leaks into the train split. Training data stays 100% synthetic (`data.py`).

## Conventions

- `artifacts/` holds generated data, models, and eval results — regenerable outputs, not source. `docs/` holds the spec, litmus test, eval reports, and `docs/plans/leq_dataset.md` (the ingestion plan). `brainlift.md` is the project thesis/POV writeup.
- New example sources must produce `FRQCase` records and pass `passes_quality_gate()` — don't add a parallel schema or bypass the gate.
- Feedback text must quote or paraphrase the student essay (grounding is a scored metric and a gate check). Never invent facts/quotes, rewrite essays, or inflate scores under pressure — these are the exact failures the spec forbids and the adversarial slices test.
- `datetime`/randomness: generation is seeded (`random.Random(seed)`) for reproducibility — keep it that way.
