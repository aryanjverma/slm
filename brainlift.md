# Brainlift: APUSH LEQ Grader SLM

Structured context for training and evaluating a tiny open model that reliably grades APUSH Long Essay Questions (LEQs) with rubric-aligned JSON and evidence-grounded feedback.

## Owners

- Aryan

## Purpose

Train a small open model to reliably embody one falsifiable grading behavior — APUSH LEQ scoring against the College Board 6-point rubric with structured JSON output and essay-grounded explanations — and prove the behavior came from curated SFT data, not prompting alone.

**Behavior contract** (from [docs/behavior_spec.md](docs/behavior_spec.md)):

> The model is an APUSH LEQ grader and explainer. Given a prompt and student essay, it returns one valid JSON object with per-criterion scores (thesis, contextualization, evidence, analysis/reasoning) and short explanations that quote or paraphrase evidence from the student's text. It never invents historical facts, documents, or quotes; never rewrites the essay; and never inflates scores under student pressure.



### In Scope

- APUSH LEQ rubric (6-point College Board contract)
- Structured JSON output with per-criterion scores and feedback
- Synthetic SFT data generation and hard quality filtering
- QLoRA fine-tuning on `Qwen/Qwen2.5-0.5B-Instruct`
- Behavioral eval: inflated prompted baseline vs reference grader vs tuned model
- Failure slices: weak thesis, missing context, evidence lists, wrong period, borderline complexity, grade inflation, prompt injection



### Out of Scope

- DBQ and SAQ rubrics (v1)
- Pretraining from scratch
- Real student essay corpus (synthetic + rule-based reference grades for v1)
- LLM-as-judge eval (deterministic checks first)
- Essay rewriting or improvement



## DOK 4: Spiky Points of View

**Spiky POV 1:** 

**Elaboration:** 

**Spiky POV 2:** 

**Elaboration:** A 

## Experts



### Andrej Karpathy

- **Who:** AI researcher; former Director of AI at Tesla, founding member of OpenAI; creator of nanoGPT and llm.c.
- **Focus:** Data-centric ML, small language models, "Software 2.0" — training data is the real program.
- **Why Follow:** Grounds the thesis that the dataset is the deliverable. Fine-tuning makes curated rubric-aligned grading data runnable on cheap hardware.
- **Where:** [karpathy.ai](https://karpathy.ai/) · [GitHub](https://github.com/karpathy)



### Sebastian Raschka

- **Who:** ML researcher and author; practical LLM fine-tuning guides.
- **Focus:** SFT workflows, LoRA/QLoRA tradeoffs, eval-minded training.
- **Why Follow:** Actionable patterns for when to fine-tune vs prompt and how to structure train/eval splits for narrow behaviors.
- **Where:** [sebastianraschka.com](https://sebastianraschka.com/) · [Ahead-of-AI](https://magazine.sebastianraschka.com/)



### Daniel Han / Unsloth

- **Who:** Creator of Unsloth; fast, memory-efficient QLoRA tooling.
- **Focus:** 2–5× faster training, ~70% less VRAM on consumer GPUs.
- **Why Follow:** Matches [scripts/train_qlora.py](scripts/train_qlora.py) — one-week, one-GPU build is realistic.
- **Where:** [unsloth.ai](https://unsloth.ai/) · [Docs](https://docs.unsloth.ai/)



### Lewis Tunstall / Hugging Face TRL

- **Who:** HF ML engineer; TRL maintainer.
- **Focus:** SFT, DPO, preference tuning on open models.
- **Why Follow:** TRL/PEFT underpin the training stack; DPO is the stretch path for inflation resistance after v1 SFT.
- **Where:** [Hugging Face](https://huggingface.co/lewtun) · [TRL docs](https://huggingface.co/docs/trl)



### Digital APUSH (2025)

- **Who:** Empirical APUSH essay grading study comparing multiple LLMs to College Board readers.
- **Focus:** Where AI matches human readers on thesis vs where it misses higher-order rubric rows.
- **Why Follow:** Direct empirical baseline for which LEQ rows are "easy" (thesis) vs hard (context, analysis) — shapes failure slices and eval metrics.
- **Where:** [Digital APUSH 2025](https://apush.omeka.net/2025)



### Michael Lomuscio

- **Who:** Educator; documented repeat LLM grading variance.
- **Focus:** Same essay + rubric scored 50× with ChatGPT yields unusable classroom variance.
- **Why Follow:** Proves consistency — not just accuracy — is why fine-tuning beats prompting for grading.
- **Where:** [ChatGPT can't grade essays yet](https://fullstackeducator.substack.com/p/chatgpt-cant-grade-essays-yet)



### College Board AP Histories

- **Who:** Official AP program; LEQ rubric source of truth.
- **Focus:** Thesis, contextualization, evidence, analysis/reasoning row definitions.
- **Why Follow:** Behavior contract and `rubric.py` validation derive directly from official language.
- **Where:** [AP History LEQ rubric](https://apcentral.collegeboard.org/)



### Mizumoto & Eguchi / AES Community

- **Who:** Researchers in automated essay scoring and LLM-human alignment.
- **Focus:** Rubric design gaps between human-oriented and model-oriented scoring.
- **Why Follow:** Explains why College Board rubrics need translation into training data, not just prompt paste.
- **Where:** AES / LLM scoring alignment literature



## DOK 3: Insights

**Insight 1: Capability is more than rubric fidelity:** Digital APUSH shows LLMs can match thesis rows at 84–94% while missing contextualization and complexity. My litmus inflated baseline achieves JSON validity (1.00) but only 0.17 evidence grounding. The model can "grade" in format, but can't grade in substance.

**Insight 2: Helpfulness bias leads to grade inflation:** EduFrameTrap and Wondering About AI document pedagogical sycophancy and rubric wording sensitivity. My `grade_inflation_request` slice fails 100% on the inflated baseline (robustness 0.00); the same alignment failure as tutoring answer-leakage, transposed to scoring.

**Insight 3: Granularity is the hard part:** Thesis matching is easier than row-by-row calibration on analysis/complexity. Rubric-Conditioned LLM Grading (2025) confirms alignment degrades with rubric granularity; LEQ's four rows are the stress test.

**Insight 4: Adversarial slices are the real eval:** Average metrics hide contract failures. `prompt_injection` (29 cases) and `grade_inflation_request` (33 cases) are where the inflated baseline collapses; v2 data oversamples these slices per MADRAG's finding that exemplar-grounded calibration is required.

**Insight 5: Data encoding beats model scale for narrow behavior:** Beyond the Score (EMNLP 2025) fine-tunes Qwen-2.5 3B on AES benchmarks for practical human alignment; direct precedent for this QLoRA stack at 0.5B with a narrower LEQ-only contract and deterministic quality gates.

## DOK 2: Knowledge Tree



### Category 1: Behavior-First Fine-Tuning



#### Subcategory 1.1: Dataset as Deliverable

**Source:** [spec.md](spec.md) · Karpathy, "A Recipe for Training Neural Networks"

**DOK 1 — Facts:**

- ~80% of outcome quality in a one-week build comes from data generation; training is downstream.
- "Train" = QLoRA SFT on a small instruct model, not pretraining.
- Behavior must fail the prompt litmus test before fine-tuning is justified.
- The behavior spec is simultaneously data rubric, eval criterion, and thesis.

**DOK 2 — Summary:**

- The synthetic LEQ grading dataset is the real artifact; the model makes it runnable locally.
- Success = reliable constrained grading behavior, not frontier APUSH knowledge.

**Link:** [spec.md](spec.md) · [Karpathy recipe](https://karpathy.github.io/2019/04/25/recipe/)

#### Subcategory 1.2: Prompt vs Fine-Tune Litmus Test

**Source:** [docs/litmus_test.md](docs/litmus_test.md) · Lomuscio · Wondering About AI

**DOK 1 — Facts:**

- Eval set: 198 held-out LEQ cases, ~25% adversarial.
- Inflated prompted baseline: JSON valid 1.00, rubric accuracy 0.82, grounding **0.17**, total **0.69**.
- Reference SFT data (`apush_grader_reference`): **1.00** on all metrics.
- `grade_inflation_request`: robustness **0.00** (33/33 inflated to 6/6).
- `prompt_injection`: robustness **0.00** (29/29).

**DOK 2 — Summary:**

- Litmus passes: prompting-style lenient grading does not hold the contract; fine-tuning is warranted.
- Gap 0.69 → 1.00 is exactly what SFT must close.

**Link:** [docs/litmus_test.md](docs/litmus_test.md) · [artifacts/eval/summary.jsonl](artifacts/eval/summary.jsonl)

#### Subcategory 1.3: QLoRA / Unsloth Stack

**Source:** QLoRA (2023) · Unsloth · [scripts/train_qlora.py](scripts/train_qlora.py)

**DOK 1 — Facts:**

- QLoRA: 4-bit base + LoRA adapters on consumer GPU.
- Base: `Qwen/Qwen2.5-0.5B-Instruct`.
- Train data: `train_chat.jsonl` (v1); v2 oversamples adversarial slices.

**DOK 2 — Summary:**

- One narrow behavior, one GPU, one week — QLoRA is the right tradeoff.

**Link:** [QLoRA](https://arxiv.org/abs/2305.14314) · [Unsloth](https://docs.unsloth.ai/)

### Category 2: APUSH LEQ Grading



#### Subcategory 2.1: College Board Rubric Contract

**Source:** [docs/behavior_spec.md](docs/behavior_spec.md) · [src/apush_frq_grader_slm/rubric.py](src/apush_frq_grader_slm/rubric.py)

**DOK 1 — Facts:**

- Four rows: thesis (0–1), contextualization (0–1), evidence (0–2), analysis_reasoning (0–2); total 0–6.
- JSON schema enforced in `behavior.py` and validated in `filters.py`.
- Pass: grounded feedback per row; fail: inflation, hallucination, rewrite.

**DOK 2 — Summary:**

- Official rubric language → programmatic validation → SFT targets — one chain from spec to data.

**Link:** [behavior_spec](docs/behavior_spec.md) · [rubric.py](src/apush_frq_grader_slm/rubric.py)

#### Subcategory 2.2: Failure Slices and Edge Cases

**Source:** [docs/error_analysis.md](docs/error_analysis.md) · [artifacts/dataset_card.md](artifacts/dataset_card.md)

**DOK 1 — Facts:**

- Slices: `weak_thesis`, `missing_context`, `evidence_list`, `wrong_period`, `borderline_complexity`, `grade_inflation_request`, `prompt_injection`, `strong`.
- v2 oversamples inflation, injection, weak thesis, wrong period.
- Quality gate rejects ungrounded feedback and invalid JSON.

**DOK 2 — Summary:**

- Base models break on adversarial and calibration slices first — fix in data, not hyperparameters.

**Link:** [error_analysis](docs/error_analysis.md) · [dataset_card](artifacts/dataset_card.md)

#### Subcategory 2.3: Structured JSON Output

**Source:** [src/apush_frq_grader_slm/behavior.py](src/apush_frq_grader_slm/behavior.py) · Reflect-and-Revise (2025)

**DOK 1 — Facts:**

- Model must return only JSON — no prose wrapper.
- `StructuredOutputValid` is the first eval gate.
- Reflect-and-Revise: rubrics built for humans need iterative calibration for LLMs.

**DOK 2 — Summary:**

- JSON makes each rubric row machine-checkable; prose graders hide inflation behind fluency.

**Link:** [Reflect-and-Revise](https://arxiv.org/html/2510.09030v1)

### Category 3: Why AI Fails at Humanities Grading



#### Subcategory 3.1: Human–AI Scoring Gap

**Source:** Digital APUSH (2025) · Mizumoto & Eguchi · Beyond the Score EMNLP 2025

**DOK 1 — Facts:**

- LLMs align with humans on thesis better than on complexity/analysis rows.
- Beyond the Score: Qwen-2.5 3B fine-tuned on AES reaches practical human alignment.
- My baseline: rubric accuracy 0.82 but grounding 0.17 — format without fidelity.

**DOK 2 — Summary:**

- The gap is rubric-row fidelity and grounding, not historical knowledge volume.

**Link:** [Digital APUSH](https://apush.omeka.net/2025) · [Beyond the Score](https://doi.org/10.18653/v1/2025.emnlp-main.992)

#### Subcategory 3.2: Score Inflation and Sycophancy

**Source:** EduFrameTrap (2025) · Wondering About AI · Lomuscio

**DOK 1 — Facts:**

- EduFrameTrap: pedagogical sycophancy under social pressure.
- Wondering About AI: rubric wording shifts scores 2+ points; injection breaks models.
- Lomuscio: 50× repeat grading shows unacceptable variance for classroom use.
- My eval: 100% failure on inflation/injection slices (robustness 0.00).

**DOK 2 — Summary:**

- RLHF-aligned models want to please students; grading requires conservative, consistent refusal.

**Link:** [EduFrameTrap](https://arxiv.org/html/2605.14604) · [Wondering About AI](https://wonderingaboutai.substack.com/p/i-ran-over-1000-api-calls-to-find)

#### Subcategory 3.3: Hallucinated Feedback

**Source:** MADRAG NLP4DH 2026 · [src/apush_frq_grader_slm/filters.py](src/apush_frq_grader_slm/filters.py)

**DOK 1 — Facts:**

- MADRAG: standard LLM-as-judge is biased/unstable; exemplar-grounded calibration required.
- Eval checks for invented quotes and document references.
- Inflated baseline uses generic praise without essay anchors — caught by grounding metric.

**DOK 2 — Summary:**

- Grounded feedback is a safety requirement, not a UX nicety — fabricating "evidence" misleads students.

**Link:** [MADRAG](https://aclanthology.org/2026.nlp4dh-1.30.pdf)

#### Subcategory 3.4: Run-to-Run Inconsistency

**Source:** Lomuscio · Rubric-Conditioned LLM Grading (2025)

**DOK 1 — Facts:**

- Same essay + rubric can yield different scores across runs on frontier models.
- Rubric granularity increases variance — LEQ row-by-row grading is high-variance for prompts.
- SFT on fixed reference grades targets consistency; DPO stretch for inflation pairs.

**DOK 2 — Summary:**

- Teachers need reproducible scores; fine-tuning on curated data is a consistency intervention.

**Link:** [Rubric-Conditioned LLM Grading](https://arxiv.org/pdf/2601.08843)

### Category 4: Fine-Tuning for Graders



#### Subcategory 4.1: SFT on Rubric-Aligned Data

**Source:** Beyond the Score EMNLP 2025 · [src/apush_frq_grader_slm/data.py](src/apush_frq_grader_slm/data.py)

**DOK 1 — Facts:**

- ~997 train chat rows with reference JSON grades.
- Each row: system prompt + LEQ prompt/essay + grounded JSON assistant response.
- Quality filter: valid JSON, score ranges, essay-anchored feedback.

**DOK 2 — Summary:**

- Curated rubric-aligned SFT is the literature-validated path for AES on small models.

**Link:** [Beyond the Score](https://doi.org/10.18653/v1/2025.emnlp-main.992)

#### Subcategory 4.2: Small-Model AES Fine-Tuning

**Source:** Beyond the Score · QLoRA stack

**DOK 1 — Facts:**

- Qwen-2.5 3B AES fine-tuning precedent at EMNLP 2025.
- This project: Qwen-2.5 0.5B + QLoRA for narrower LEQ-only contract.
- Reference grader scores 1.00 — defines achievable ceiling on synthetic eval.

**DOK 2 — Summary:**

- Specialist 0.5B grader with hard data gates can beat prompted leniency on grounding and robustness.

**Link:** [train_qlora.py](scripts/train_qlora.py)

#### Subcategory 4.3: DPO Stretch Path

**Source:** EduFrameTrap · MADRAG

**DOK 1 — Facts:**

- Preference pairs: conservative grounded grade (chosen) vs inflated generic grade (rejected).
- v2 dataset front-loads adversarial slices for SFT; DPO if robustness still wobbles post-train.

**DOK 2 — Summary:**

- SFT establishes grading behavior; DPO sharpens inflation resistance on hardest slices.



#### Subcategory 4.4: Litmus Numbers (Deterministic Baseline)

**Source:** [artifacts/eval/summary.jsonl](artifacts/eval/summary.jsonl) · [docs/litmus_test.md](docs/litmus_test.md)

**DOK 1 — Facts:**


| Model                         | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
| ----------------------------- | ---------- | ----------- | --------- | ---------- | ----- |
| `inflated_prompted_base`      | 1.00       | 0.82        | 0.17      | 0.93       | 0.69  |
| `apush_grader_reference`      | 1.00       | 1.00        | 1.00      | 2.00       | 1.00  |
| `apush_frq_grader_v1` (QLoRA) | TBD        | TBD         | TBD       | TBD        | TBD   |


- Biggest inflated-base gaps: grounding (0.17), adversarial robustness (0.00 on inflation/injection).
- Win condition: tuned model beats 0.69 total with grounding > 0.9 and adversarial robustness → 2.0.

**DOK 2 — Summary:**

- Litmus passes with large grounding and adversarial gaps — fine-tuning target is clear and measurable.

**Link:** [litmus_test](docs/litmus_test.md) · [eval/](artifacts/eval/)