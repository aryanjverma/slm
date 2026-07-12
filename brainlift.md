# Brainlift: APUSH LEQ Grader SLM

## V5 Final Evidence Addendum (2026-07-12)

The final design separates scoring from feedback generation, computes totals deterministically,
and makes human approval plus content hashes prerequisites to loading weights. This prevents a
revoked review or stale assembly audit from silently becoming the training corpus.

The release claim is deliberately narrow. The 53-case College Board-derived evaluation is
development-informed because aggregate style characteristics shaped synthetic generation. V5 is
production-ready only if every locked calibration, agreement, validity, and grounding gate passes;
otherwise the aggregate result is published as non-production-ready without retuning. Public data
is a separate project-authored synthetic companion, never the private v5 corpus.

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

**Spiky POV 1:** AI can grade LEQs better than humans.

**Elaboration:** Right now, the way AI grades LEQs is not the way humans grade LEQs.

- AI grades LEQs by looking for patterns, buzzwords, and the correct transition phrases; oftentimes, an AI will assign a grade and then reverse-engineer the justification rather than look through the rubric first and then assign a score. Out-of-the-box LLMs reach QWK below 0.30 with human raters while over-scoring short, underdeveloped essays and penalizing longer ones for minor grammar errors; they reward fluency, not argument. Digital APUSH (2025) shows thesis row matching at 84–94% but misses contextualization and complexity. My litmus baseline hits the same wall: valid JSON and 0.82 rubric accuracy, but only 0.17 evidence grounding. The model can grade in format, but not in substance. It may give a student with a well-written essay that has little historical relevance a good score while giving a student with a badly-written essay that has an amazing historical argument a bad score. It also may underscore an argument that has good historical evidence that was not in their training dataset, or an untraditional argument that contradicts the consensus view. AI doesn't grade arguments; it grades formats and clichés.
- Humans grade LEQs based on their historical argument, seeing if the writer answers the question and gives good historical context, and then evaluating their evidence and complex reasoning. Since human graders have a strong historical breakdown and have experience reading different types of arguments with different groups of evidence, they can understand a variety of essays at a deeper level than AI can, and can diagnose a good argument accurately. But humans aren't immune to the same structural problems; the same essay scored 50 times can yield unusable variance, and rubric wording alone can shift scores by two or more points. Under student pressure, a tired grader may inflate; under time pressure, feedback may stay generic instead of citing the essay.

Given these observations, it may seem obvious that AI sucks at grading LEQs, and out-of-the-box, it does. My inflated baseline awards 6/6 on every grade-begging and prompt-injection case (33/33 and 29/29), which confirms that the failure is behavioral, not a permanent ceiling. The real gap between AI and human grading is not historical knowledge volume but mismatched signals: identifying a good argument, calibrating row-by-row on analysis and complexity, and tying every score to evidence from the student's text. That behavior already exists in my reference grader, which scores 1.00 across 198 eval cases. The question is encoding it into a fine-tuned model, not hoping prompts alone will hold. Beyond the Score (EMNLP 2025) shows a fine-tuned Qwen-2.5 3B reaching practical human alignment on essay scoring; this project applies the same logic at 0.5B on a narrower LEQ contract with hard quality gates and adversarial slices oversampled in v2 data.

If an AI is trained to systematically grade an LEQ the same way as a human (rubric first, then score, then essay-anchored feedback) it can exceed humans on the dimensions that matter for classroom trust: same essay, same grade every time; feedback that always quotes or paraphrases the student's text; refusal to inflate under pleading or injection. That is what "better" means here; not replacing AP readers on day one, not knowing more history, but grading more consistently, more fairly, and more faithfully to the rubric than a human grader under real classroom conditions.

**How we plan to do it:**

**First: collect data.** The dataset is the deliverable; ~80% of outcome quality comes from what we put in, not the training loop. We start with official AP released essays and scoring — sample LEQ responses, reader commentary, and scoring guidelines from AP Central — because out-of-the-box models already match thesis rows at 84–94% but miss contextualization and complexity. Real reader-scored examples teach row-by-row calibration on the hard rows, not buzzword pattern-matching. We supplement with graded practice tests from third parties (Barron's, Princeton Review, AMSCO) to expand argument diversity: varied evidence sets, untraditional arguments, and the kind of essays default LLMs underscored in testing. Where real corpora have gaps, we generate essays using the existing pipeline in `data.py` — specifically for slices released exams rarely cover: grade-begging, prompt injection, weak thesis, wrong period, and borderline complexity. Average accuracy hides contract failures; 33/33 inflation and 29/29 injection cases are where prompted baselines collapse, and you cannot get enough adversarial pressure from released exams alone. Every source normalizes to the same `FRQCase` schema (`prompt`, `student_response`, row scores, essay-anchored feedback, JSON assistant target). College Board commentary is human-oriented prose, not JSON — it must be translated into per-row scores and feedback that quotes the student's text, not pasted into a prompt. Every row passes the quality gate in `filters.py` (valid JSON, score ranges, grounded feedback, no inflation). We hold out ~20% for eval and keep ~25% adversarial, matching the current litmus design. The litmus already proved prompting hits 0.17 grounding; this corpus must encode rubric-first, essay-anchored grading that real essays and real scoring commentary provide, with synthetics filling the adversarial holes.

**Second: train the model.** Prompting cannot hold the contract — 0.82 rubric accuracy, 0.17 grounding, 0.00 adversarial robustness on inflation and injection — so fine-tuning is the intervention that makes curated data runnable and reproducible. We QLoRA SFT `Qwen/Qwen2.5-0.5B-Instruct` via `scripts/train_qlora.py` on mixed `train_chat.jsonl` rows: system prompt from the behavior spec, user message = LEQ prompt + essay, assistant = reference JSON grade. v1 trains on the full mixed corpus (~800–1000+ quality-filtered rows). v2 retrains with adversarial oversampling via `make_v2_dataset.py` once failure slices are diagnosed. If inflation resistance still wobbles post-SFT, DPO preference pairs (conservative grounded grade vs inflated generic grade) are the stretch path. Narrow AES behavior is data-limited, not parameter-limited — Beyond the Score (EMNLP 2025) reached practical human alignment fine-tuning Qwen-2.5 3B; 0.5B is sufficient for one LEQ contract.

**Third: evaluate and adjust.** Eval is built before we trust the model. We compare `inflated_prompted_base` → `apush_frq_grader_v1` (QLoRA) against the reference ceiling on JSON validity, rubric accuracy, evidence grounding, adversarial robustness, and total score, broken down by failure slice. The Day 2 smoke adapter (`apush_frq_grader_smoke`, 25 steps on 30 rows) already beats the inflated baseline on grounding (0.95) and total (0.77) on the 20-case smoke set, proving the loop works before v1. Win condition on the 198-case litmus: beat 0.69 total with grounding above 0.9 and adversarial robustness reaching 2.0 on inflation and injection — the gap the litmus already measured. We run held-out eval on real AP and third-party essays, not just synthetic cases, for external validity. When slices fail, we fix in data, not hyperparameters: diagnose worst rows (expect `borderline_complexity`, `missing_context`, and adversarial cases first — granularity is the hard part), oversample failing slices in v2, add targeted synthetic or distilled examples, retrain, and re-eval until the contract holds on both clean and adversarial inputs. Each iteration closes the gap from 0.69 toward reference behavior and proves the spiky claim came from curated SFT data, not prompting alone.

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



### Jerin George Mathew et al. (University of Alberta)

- **Who:** Mathew, Taher, Kundu, and Barbosa; University of Alberta NLP/education researchers.
- **Focus:** Out-of-the-box LLM essay scoring vs human raters on ASAP and DREsS; which essay signals models actually use.
- **Why Follow:** March 2026 evidence that LLMs grade on different signals than humans — inflate short/underdeveloped essays, penalize longer essays for minor grammar errors, and stay internally consistent with their own praise/criticism while still misaligning with human QWK.
- **Where:** [LLMs Do Not Grade Essays Like Humans (2026)](https://arxiv.org/html/2603.23714v1)



## DOK 3: Insights

**Insight 1: Grading capability is more than rubric fidelity:** Digital APUSH shows LLMs can match thesis rows at 84–94% while missing contextualization and complexity. Mathew et al. (2026) find out-of-the-box LLMs reach QWK < 0.30 with human raters on ASAP (vs human inter-rater QWK 0.72) and inflate short essays while deflating longer ones over minor language errors — different signals, not just lower accuracy. My litmus inflated baseline achieves JSON validity (1.00) but only 0.17 evidence grounding. The model can "grade" in format, but can't grade in substance.

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

**Source:** [docs/litmus_test.md](docs/litmus_test.md) · Mathew et al. (2026) · Lomuscio · Wondering About AI

**DOK 1 — Facts:**

- Mathew et al. (2026): out-of-the-box LLM grading is how most educators will use these tools, but QWK with human raters stays weak without task-specific training.
- Eval set: 198 held-out LEQ cases, ~25% adversarial.
- Inflated prompted baseline: JSON valid 1.00, rubric accuracy 0.82, grounding **0.17**, total **0.69**.
- Reference SFT data (`apush_grader_reference`): **1.00** on all metrics.
- `grade_inflation_request`: robustness **0.00** (33/33 inflated to 6/6).
- `prompt_injection`: robustness **0.00** (29/29).

**DOK 2 — Summary:**

- Litmus passes: prompting-style lenient grading does not hold the contract; fine-tuning is warranted.
- Gap 0.69 → 1.00 is exactly what SFT must close.

**Link:** [docs/litmus_test.md](docs/litmus_test.md) · [Mathew et al. (2026)](https://arxiv.org/html/2603.23714v1) · [artifacts/eval/summary.jsonl](artifacts/eval/summary.jsonl)

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

**Source:** Mathew et al. (2026) · Digital APUSH (2025) · Mizumoto & Eguchi · Beyond the Score EMNLP 2025

**DOK 1 — Facts:**

- Mathew et al. (2026): out-of-the-box GPT and Llama models on ASAP and DREsS show weak human–LLM agreement (QWK < 0.30 on ASAP; human inter-rater QWK 0.72).
- Same paper: LLMs assign **higher** scores to short/underdeveloped essays than humans; humans assign **higher** scores to longer, developed essays despite minor grammar/spelling errors that LLMs penalize.
- LLMs align with humans on thesis better than on complexity/analysis rows (Digital APUSH).
- Beyond the Score: Qwen-2.5 3B fine-tuned on AES reaches practical human alignment — the fix Mathew's out-of-the-box study explicitly excludes.
- My baseline: rubric accuracy 0.82 but grounding 0.17 — format without fidelity.

**DOK 2 — Summary:**

- The gap is mismatched grading signals and rubric-row fidelity, not historical knowledge volume. LLMs reward fluency and length heuristics; human raters weight argument development — fine-tuning must encode human-like row-level calibration.

**Link:** [Mathew et al. (2026)](https://arxiv.org/html/2603.23714v1) · [Digital APUSH](https://apush.omeka.net/2025) · [Beyond the Score](https://doi.org/10.18653/v1/2025.emnlp-main.992)

#### Subcategory 3.2: Score Inflation and Sycophancy

**Source:** Mathew et al. (2026) · EduFrameTrap (2025) · Wondering About AI · Lomuscio

**DOK 1 — Facts:**

- Mathew et al. (2026): LLMs systematically **over-score short or underdeveloped essays** relative to human raters — a structural inflation bias, not just adversarial pressure.
- LLM scores cohere with their own feedback (praise → higher, criticism → lower) but SHAP analysis shows models weight different rubric traits than humans (e.g., GPT-4 dominated by positive "ideas" mentions).
- EduFrameTrap: pedagogical sycophancy under social pressure.
- Wondering About AI: rubric wording shifts scores 2+ points; injection breaks models.
- Lomuscio: 50× repeat grading shows unacceptable variance for classroom use.
- My eval: 100% failure on inflation/injection slices (robustness 0.00); `weak_thesis` and `evidence_list` slices still get inflated row scores.

**DOK 2 — Summary:**

- Inflation is both structural (short/weak essays) and social (grade-begging). RLHF-aligned models want to please students; grading requires conservative, consistent refusal encoded in data.

**Link:** [Mathew et al. (2026)](https://arxiv.org/html/2603.23714v1) · [EduFrameTrap](https://arxiv.org/html/2605.14604) · [Wondering About AI](https://wonderingaboutai.substack.com/p/i-ran-over-1000-api-calls-to-find)

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

**Source:** Lomuscio · Mathew et al. (2026) · Rubric-Conditioned LLM Grading (2025)

**DOK 1 — Facts:**

- Same essay + rubric can yield different scores across runs on frontier models (Lomuscio).
- Mathew et al. (2026): newer GPT/Llama generations do not consistently improve human alignment; score distributions vary by model (GPT-3.5 skews low; others cluster mid-scale).
- Rubric granularity increases variance — LEQ row-by-row grading is high-variance for prompts.
- SFT on fixed reference grades targets consistency; DPO stretch for inflation pairs.

**DOK 2 — Summary:**

- Teachers need reproducible scores; fine-tuning on curated data is a consistency intervention — scale alone does not fix the human–LLM gap Mathew documents.

**Link:** [Mathew et al. (2026)](https://arxiv.org/html/2603.23714v1) · [Rubric-Conditioned LLM Grading](https://arxiv.org/pdf/2601.08843)

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


| Model                         | Cases | JSON Valid | Rubric Acc. | Grounding | Robustness | Total |
| ----------------------------- | ----- | ---------- | ----------- | --------- | ---------- | ----- |
| `inflated_prompted_base`      | 198   | 1.00       | 0.82        | 0.17      | 0.93       | 0.69  |
| `apush_grader_reference`      | 198   | 1.00       | 1.00        | 1.00      | 2.00       | 1.00  |
| `apush_frq_grader_v1` (QLoRA) | 198   | TBD        | TBD         | TBD       | TBD        | TBD   |
| `apush_frq_grader_smoke` (QLoRA) | 20 | 0.55       | 0.95        | 0.95      | 1.65       | 0.77  |


- Biggest inflated-base gaps: grounding (0.17), adversarial robustness (0.00 on inflation/injection).
- Win condition: tuned model beats 0.69 total with grounding > 0.9 and adversarial robustness → 2.0.
- `apush_frq_grader_smoke` (20-case smoke eval) is the Day 2 loop checkpoint — 25 training steps on 30 rows, not the v1 production run. Already beats inflated baseline on grounding (0.95 vs 0.17 on litmus; 0.95 vs 0.15 on smoke) and total (0.77 vs 0.69). JSON validity (0.55) expected to improve on full v1 train.

**DOK 2 — Summary:**

- Litmus passes with large grounding and adversarial gaps — fine-tuning target is clear and measurable.
- Smoke tuned model (`apush_frq_grader_smoke`) proves the gap is closeable on a 20-case held-out set before v1 GPU training.

**Link:** [litmus_test](docs/litmus_test.md) · [eval/](artifacts/eval/)
