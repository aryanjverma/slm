# Brainlift: Arithmetic Tutor SLM

Structured context for training and evaluating a tiny open model that reliably tutors addition and subtraction without leaking the final answer.

## Owners

- Aryan

## Purpose

Train a small open model to reliably embody one falsifiable tutoring behavior — Socratic scaffolding for addition and subtraction without answer leakage — and prove the behavior came from data, not prompting alone.

**Behavior contract** (from [docs/behavior_spec.md](docs/behavior_spec.md)):

> The model is a Socratic tutor for addition and subtraction. It never states the final numeric answer unless the student has already produced it; instead, it identifies the student's current step or mistake and asks one short guiding question or gives one calibrated hint for the next step.



### In Scope

- Writing and enforcing a falsifiable behavior spec
- Synthetic SFT data generation and hard quality filtering
- QLoRA fine-tuning on a small instruct base model (Qwen2.5-0.5B)
- Behavioral evaluation: base vs tuned on a held-out set
- Arithmetic tutoring edge cases: carrying, borrowing, borrow-through-zero, column alignment, direct-answer requests, wrong-final confirmation
- Building structured context for AI conversations about this project



### Out of Scope

- Pretraining from scratch
- Broad math capability or trivia benchmarks
- Multiplication, division, fractions, algebra, or general math tutoring
- Complex retrieval-augmented generation (RAG) pipelines



## DOK 4: Spiky Points of View

**Spiky POV 1:** 

**Elaboration:** 

**Spiky POV 2:** 

**Elaboration:** 

## Experts



### Andrej Karpathy

- **Who:** AI researcher; former Director of AI at Tesla, founding member of OpenAI; creator of nanoGPT and llm.c.
- **Focus:** Data-centric machine learning, small language models, and the idea that training data is the real program ("Software 2.0").
- **Why Follow:** Grounds the project thesis that the dataset is the deliverable, not the model weights. Fine-tuning is how you make curated data runnable.
- **Where:** [karpathy.ai](https://karpathy.ai/) · [X @karpathy](https://x.com/karpathy) · [GitHub](https://github.com/karpathy)



### Sebastian Raschka

- **Who:** ML researcher and author; practical guides on LLM fine-tuning and evaluation.
- **Focus:** Supervised fine-tuning workflows, instruction tuning, LoRA/QLoRA tradeoffs, and eval-minded training decisions.
- **Why Follow:** Translates research into actionable SFT patterns — when to fine-tune vs prompt, how to structure train/eval splits, and what to measure.
- **Where:** [sebastianraschka.com](https://sebastianraschka.com/) · [Ahead-of-AI Substack](https://magazine.sebastianraschka.com/) · [GitHub](https://github.com/rasbt)



### Daniel Han / Unsloth

- **Who:** Creator of Unsloth; open-source tooling for fast, memory-efficient LLM fine-tuning.
- **Focus:** 2–5× faster QLoRA training with ~70% less VRAM; clean notebooks for single-GPU SFT on small models.
- **Why Follow:** Matches this project's stack ([scripts/train_qlora.py](scripts/train_qlora.py)) and makes a one-week, one-GPU build realistic.
- **Where:** [unsloth.ai](https://unsloth.ai/) · [GitHub unslothai/unsloth](https://github.com/unslothai/unsloth) · [Docs](https://docs.unsloth.ai/)



### Michelene Chi

- **Who:** Regents Professor of Psychology; ASU learning sciences researcher.
- **Focus:** ICAP framework (Interactive, Constructive, Active, Passive); how different learning activities produce different cognitive engagement and outcomes.
- **Why Follow:** Explains why a tutor that gives one targeted hint (constructive/interactive) beats one that walks through the full solution (passive). Informs the one-step calibration requirement.
- **Where:** [Google Scholar](https://scholar.google.com/citations?user=9f64YIoAAAAJ) · [ICAP paper (2014)](https://doi.org/10.1177/0042095913505851)



### Lewis Tunstall / Hugging Face TRL

- **Who:** ML engineer at Hugging Face; maintainer of TRL (Transformer Reinforcement Learning) and author of *Natural Language Processing with Transformers*.
- **Focus:** SFT, DPO, and preference tuning tooling; instruction-tuning best practices on open models.
- **Why Follow:** TRL/PEFT patterns underpin the training pipeline; DPO is the natural stretch goal after v1 SFT if spec adherence still wobbles.
- **Where:** [Hugging Face](https://huggingface.co/lewtun) · [TRL docs](https://huggingface.co/docs/trl) · [GitHub huggingface/trl](https://github.com/huggingface/trl)



### Kenneth Koedinger

- **Who:** Professor of Human-Computer Interaction and Psychology, Carnegie Mellon; co-founder of the Cognitive Tutor approach.
- **Focus:** The assistance dilemma — balancing how much and when to help; cognitive tutors for math; learning sciences applied to ITS design.
- **Why Follow:** Formalizes the core design tension this project encodes: one calibrated next-step hint vs. giving away the solution. The behavior spec is an assistance-dilemma policy for column-level add/sub.
- **Where:** [Google Scholar](https://scholar.google.com/citations?user=ZWy_GG4AAAAJ) · [Assistance dilemma paper (2007)](https://doi.org/10.1007/s10648-007-9049-0)



### Sarah Macina

- **Who:** PhD researcher, ETH Zurich NLP group; lead author of MathDial.
- **Focus:** Dialogue tutoring datasets, teacher-move taxonomies, and the solving-vs-tutoring gap in LLMs.
- **Why Follow:** MathDial directly proves that models strong at math reasoning fail at equitable scaffolding — and that fine-tuning on curated tutoring dialogues fixes it. This project's litmus test is the same finding at arithmetic granularity.
- **Where:** [MathDial paper (2023)](https://arxiv.org/abs/2305.14536) · [GitHub eth-nlped/mathdial](https://github.com/eth-nlped/mathdial)



### Hamsa Bastani

- **Who:** Associate Professor of Operations, Information and Decisions, Wharton; co-author of the PNAS GPT tutoring field experiment.
- **Focus:** AI in education, causal inference on learning outcomes, and guardrail design for generative AI tutors.
- **Why Follow:** Provides the strongest causal evidence that unfettered LLM access harms learning and that tutor-designed guardrails (hints, not answers) are pedagogically necessary — not optional UX polish.
- **Where:** [PNAS GPT Tutor study (2025)](https://doi.org/10.1073/pnas.2422633122) · [Google Scholar](https://scholar.google.com/citations?user=7_vN8kUAAAAJ)



## DOK 3: Insights

Original conclusions and connections you generate after processing your Knowledge Tree. Group thematically; these bridge DOK 2 summaries to DOK 4 Spiky POVs.

**Insight 1 — Solving is not tutoring:** MathDial and my litmus test show the same split: a model can compute addition and subtraction correctly yet fail as a tutor. Benchmarking on GSM8K or raw arithmetic accuracy misses the product thesis entirely. The behavior contract is orthogonal to capability.

**Insight 2 — Helpfulness is misaligned with learning:** The PNAS field experiment and sycophancy research explain why prompts fail. RLHF rewards giving students what they ask for — including the final answer. Fine-tuning must make "redirect to the next step" the default policy, not a system instruction the model overrides when the student applies direct pressure.

**Insight 3 — Step calibration is the hard part, not no-answer rules:** My prompted base achieves 0.81 NoAnswerLeak but only 0.12 StepCalibration. The model often knows the answer and sometimes withholds it on easy cases, but defaults to multi-step worked solutions instead of one Socratic hint. This mirrors the assistance dilemma: the failure is granularity of help, not arithmetic knowledge.

**Insight 4 — Adversarial slices are the real eval:** SocraticLM's ablations show that Incorrect Answer Recognition and Successful Rejection Rate are the hardest teaching abilities — the same slices where my base model fails completely (`direct_answer_request` at 100% leak rate). Average metrics hide contract failures; v2 data oversampling of adversarial mistake types is the literature-validated fix, not hyperparameter tuning.

**Insight 5 — Data encoding beats model scale for narrow behavior:** Karpathy's "dataset is the deliverable" thesis, SocraticLM's finding that ~26K dialogues are needed to surpass GPT-4 on pedagogical quality, and my 1K filtered synthetic rows all point the same direction: for one falsifiable behavior on cheap hardware, curated data quality dominates model size.

## DOK 2: Knowledge Tree



### Category 1: Behavior-First Fine-Tuning



#### Subcategory 1.1: Dataset as Deliverable

**Source:** [spec.md](spec.md) · Karpathy, "A Recipe for Training Neural Networks"

**DOK 1 — Facts:**

- In a one-week build, ~80% of outcome quality comes from the data you generate; training is a downstream button-press.
- "Train" here means supervised fine-tuning (QLoRA) on a small open instruct model, not pretraining from scratch.
- The target behavior must fail the prompt litmus test: if a well-prompted base model already does it reliably, fine-tuning is pointless.
- The behavior spec is simultaneously the data-generation rubric, the eval criterion, and the thesis to defend.

**DOK 2 — Summary:**

- The synthetic dataset is the real artifact; the model is just that data made runnable on cheap local hardware.
- Success is measured as reliable constrained behavior, not raw capability — a 0.5B specialist that never leaks beats a frontier model on trivia.

**Link:** [spec.md](spec.md) · [karpathy.github.io — A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

#### Subcategory 1.2: Prompt vs Fine-Tune Litmus Test

**Source:** [docs/litmus_test.md](docs/litmus_test.md) · Macina et al., MathDial (2023) · Si et al., "Does Instruction Tuning Make LLMs More Consistent?" (2024)

**DOK 1 — Facts:**

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` with the same `SYSTEM_PROMPT` used for SFT and eval.
- Held-out eval: 200 cases, 25% adversarial (23 direct-answer-request cases).
- Prompted base scores: NoAnswerLeak 0.81, StepCalibration 0.12, Total 0.51.
- All 23 direct-answer-request cases leaked the final number despite the system prompt forbidding it.
- Reference SFT data (`socratic_tutor_reference`) scores 1.00 on all metrics — defining the training target gap.
- MathDial found GPT-3 class models are strong problem solvers but poor tutors that reveal solutions too early; fine-tuning on tutoring dialogues is required to shift behavior.
- Instruction tuning increases consistency under input perturbation, but prompting alone cannot guarantee constrained behavior under adversarial pressure.

**DOK 2 — Summary:**

- The litmus test passes: prompting does not reliably hold the tutor contract under pressure, so fine-tuning is warranted.
- The gap between 0.51 (prompted base) and 1.00 (reference data) is exactly what the SFT dataset must close.
- This replicates MathDial's solving-vs-tutoring finding locally: the base model can compute but cannot tutor reliably.

**Link:** [docs/litmus_test.md](docs/litmus_test.md) · [MathDial (2023)](https://arxiv.org/abs/2305.14536) · [Instruction tuning consistency (2024)](https://arxiv.org/html/2404.15206v2)

#### Subcategory 1.3: QLoRA / Unsloth Stack

**Source:** Dettmers et al., QLoRA (2023) · Unsloth docs · [scripts/train_qlora.py](scripts/train_qlora.py)

**DOK 1 — Facts:**

- QLoRA quantizes the frozen base model to 4-bit and trains low-rank adapter (LoRA) weights on top.
- Unsloth reports ~2× training speed and ~70% less VRAM vs standard HF + PEFT workflows.
- Default base model: `Qwen/Qwen2.5-0.5B-Instruct` — fits a 24 GB consumer GPU.
- Training data: `artifacts/data/train_chat.jsonl` (v1); v2 oversamples failure modes via `train_chat_v2.jsonl`.

**DOK 2 — Summary:**

- For one narrow behavior on one GPU in one week, a small instruct base + QLoRA adapters is the right tradeoff — full fine-tuning would waste compute and risk catastrophic forgetting.

**Link:** [QLoRA paper](https://arxiv.org/abs/2305.14314) · [Unsloth docs](https://docs.unsloth.ai/) · [scripts/train_qlora.py](scripts/train_qlora.py)

### Category 2: Socratic Arithmetic Tutoring



#### Subcategory 2.1: Behavior Contract

**Source:** [docs/behavior_spec.md](docs/behavior_spec.md) · [src/arithmetic_tutor_slm/behavior.py](src/arithmetic_tutor_slm/behavior.py)

**DOK 1 — Facts:**

- Pass: does not reveal the final answer; gives exactly one next-step hint or question; targets the student's current arithmetic state; redirects direct answer requests.
- Fail: states the answer directly; solves multiple steps ahead; gives generic encouragement without arithmetic guidance; confirms an incorrect final answer.
- In scope: multi-digit add/sub, carrying, borrowing, borrow-through-zero, alignment, blank starts, partial work, wrong finals, direct requests.
- Out of scope: multiplication, division, fractions, algebra, long chain-of-thought solutions.

**DOK 2 — Summary:**

- Pass/fail is behavioral, not arithmetic — a response that gives the correct answer but leaks it fails; a response with imperfect wording that stays to one calibrated step passes.

**Link:** [docs/behavior_spec.md](docs/behavior_spec.md) · [src/arithmetic_tutor_slm/behavior.py](src/arithmetic_tutor_slm/behavior.py)

#### Subcategory 2.2: Tutoring Pedagogy

**Source:** Chi & Wylie, ICAP Framework (2014) · VanLehn, "The Behavior of Tutoring Systems" (2006) · VanLehn (2011), Educational Psychologist

**DOK 1 — Facts:**

- ICAP ranks learning activities: Interactive > Constructive > Active > Passive; higher engagement modes produce better learning outcomes.
- Effective tutoring identifies the learner's current knowledge state and provides the minimum scaffold needed for the next step.
- Giving full worked solutions is a passive activity for the student — it short-circuits productive struggle and transfer.
- Socratic questioning forces the student to construct the next inference themselves.
- VanLehn's meta-review found intelligent tutoring systems can approach the effectiveness of human one-on-one tutoring when they provide step-level, state-sensitive feedback.

**DOK 2 — Summary:**

- The model's job is calibrated next-step guidance, not correctness demonstration — one hint at the right column or regrouping step, not a walkthrough.

**Link:** [ICAP Framework paper](https://doi.org/10.1177/0042095913505851) · [VanLehn, "The Behavior of Tutoring Systems" (2006)](https://doi.org/10.1007/s11257-006-900-4) · [VanLehn (2011) meta-review](https://doi.org/10.1080/00461520.2011.611651)

#### Subcategory 2.3: Domain Edge Cases

**Source:** [docs/error_analysis.md](docs/error_analysis.md) · [artifacts/dataset_card.md](artifacts/dataset_card.md)

**DOK 1 — Facts:**

- Primary failure mode: answer leakage when the student asks directly or when the model "helps" by solving fully.
- Secondary failures: generic hints missing the current column; borrow-through-zero subtraction; misaligned columns; wrong-final confirmation.
- v2 data (`train_chat_v2.jsonl`) oversamples direct-answer pressure, alignment, wrong-answer checks, and borrow-through-zero.
- Filtering rejects rows that leak the final answer, contain too many answer digits, are too long, or don't behave like a Socratic hint.

**DOK 2 — Summary:**

- Base models break on adversarial and regrouping cases first — v2 iteration targets those failure modes in data, not hyperparameters.

**Link:** [docs/error_analysis.md](docs/error_analysis.md) · [artifacts/dataset_card.md](artifacts/dataset_card.md)

#### Subcategory 2.4: Assistance Dilemma

**Source:** Koedinger & Aleven (2007) · LAK26 hint-button study

**DOK 1 — Facts:**

- The assistance dilemma: too little help causes frustration; too much help undermines learning by replacing student reasoning.
- Effective hints preserve learner agency — they nudge reasoning in a productive direction rather than replacing it.
- ITS hint systems use graduated scaffolding: conceptual prompts first, then increasingly specific guidance, with direct answers only as a last resort ("bottom-out" hints).
- Large-scale deployments show that unproductive hint use (clicking through to the answer) correlates with worse learning outcomes.

**DOK 2 — Summary:**

- The behavior spec operationalizes the assistance dilemma for column-level add/sub: one calibrated hint, never bottom-out. StepCalibration (0.12 on the prompted base) is the metric that captures this failure mode.

**Link:** [Assistance dilemma paper](https://doi.org/10.1007/s10648-007-9049-0) · [LAK26 hint-button study](https://dl.acm.org/doi/10.1145/3785022.3785040)

#### Subcategory 2.5: ITS Effectiveness for Math

**Source:** Kulik & Fletcher (2016) · Steenbergen-Hu & Cooper (2013) · Beal et al. (2010), AnimalWatch

**DOK 1 — Facts:**

- Meta-analysis of 50 ITS evaluations: median effect size of 0.66 standard deviations over conventional instruction (50th → 75th percentile).
- A separate meta-analysis of K–12 math ITS found positive effects on mathematical learning (Steenbergen-Hu & Cooper, 2013).
- AnimalWatch, an ITS for basic computation and fractions, showed significant pre→post gains in three controlled studies; students who simply retook tests without the ITS showed no improvement.
- Gains were strongest for students with the weakest initial math skills — the population most likely to use hint resources.
- Effect sizes depend on alignment between tutoring content and assessment objectives.

**DOK 2 — Summary:**

- Arithmetic tutoring systems work when they adapt hint granularity to student state. This project narrows that proven ITS design pattern to add/sub with a falsifiable no-answer contract.

**Link:** [Kulik & Fletcher meta-analysis](https://doi.org/10.3102/0034654315581420) · [AnimalWatch evaluation](https://www.ncolr.org/jiol/issues/pdf/9.1.4.pdf)

#### Subcategory 2.6: **Numeric & Arithmetic Tokenization**

**Source:** **Brainlift: Numeric & Arithmetic Tokenization**

**DOK 1 — Facts:**

- LLM arithmetic errors partly stem from subword tokenization of multi-digit numbers.
- 

**DOK 2 — Summary:**

- Tokenization explains some raw computation failures, but this project targets behavioral failure modes that persist even when the model can compute correctly — answer leakage and poor step calibration are alignment and tutoring-policy problems, not tokenization problems.

**Link:** [My Brainlift](https://docs.google.com/document/d/1AtpalNhFMeDVqix6pF5xi69xbhsiImp1vTYLbB8Kd6I/edit?tab=t.0)

### Category 3: Behavioral Evaluation



#### Subcategory 3.1: Metrics and Harness

**Source:** [docs/eval_report.md](docs/eval_report.md) · [src/arithmetic_tutor_slm/eval.py](src/arithmetic_tutor_slm/eval.py)

**DOK 1 — Facts:**

- Metrics: `NoAnswerLeakRate`, `HintCorrectness`, `StepCalibration`, `Robustness`, `LearningHelpfulness`.
- Held-out set: 200 cases in `artifacts/data/eval_cases.jsonl`, never seen during training.
- Eval harness built before training — without it, "we fine-tuned a model" is unfalsifiable.
- Required comparison: same base model + strong prompt vs QLoRA-tuned adapter on the same scenarios.
- Win condition: tuned model beats base on `NoAnswerLeakRate`, `Robustness`, and `Total` — not on general math ability.

**DOK 2 — Summary:**

- Evaluation measures whether data encoded the behavior contract — slice results by `mistake_type` to confirm gains on direct-answer and regrouping cases specifically.

**Link:** [docs/eval_report.md](docs/eval_report.md) · [src/arithmetic_tutor_slm/eval.py](src/arithmetic_tutor_slm/eval.py)

#### Subcategory 3.2: Data Generation Pipeline

**Source:** [src/arithmetic_tutor_slm/data.py](src/arithmetic_tutor_slm/data.py) · [src/arithmetic_tutor_slm/filters.py](src/arithmetic_tutor_slm/filters.py)

**DOK 1 — Facts:**

- Cases generated with deterministic arithmetic ground truth via `first_step()` and `solve()`.
- Each row includes: problem, student state, hidden final answer, expected next step, mistake type, difficulty, tags, and target assistant response.
- Mistake types include: blank, correct_partial, carry_missed, borrow_missed, borrow_through_zero, alignment, wrong_final, direct_answer_request, messy.
- The hidden final answer is used for filtering and evaluation but is never revealed in the trained assistant response.
- CLI: `python -m arithmetic_tutor_slm.cli.generate_dataset --train-count 1000 --eval-count 200`.

**DOK 2 — Summary:**

- The craft is in the generation logic and quality gate, not raw volume — a thousand filtered examples that never leak beat ten thousand unfiltered ones.

**Link:** [src/arithmetic_tutor_slm/data.py](src/arithmetic_tutor_slm/data.py) · [src/arithmetic_tutor_slm/filters.py](src/arithmetic_tutor_slm/filters.py)

#### Subcategory 3.3: Post-Training Results

**Source:** [TBD after QLoRA run — artifacts/eval/ or Hugging Face model card]

**DOK 1 — Facts:**

- Comparison table (same 200-case held-out set):


| Model                                   | No Answer Leak | Hint Correct | Calibrated | Robustness | Total |
| --------------------------------------- | -------------- | ------------ | ---------- | ---------- | ----- |
| `qwen_base_prompted`                    | 0.81           | 0.54         | 0.12       | 0.82       | 0.51  |
| `socratic_tutor_reference` (SFT target) | 1.00           | 1.00         | 1.00       | 2.00       | 1.00  |
| `arithmetic_tutor_v1` (QLoRA)           | TBD            | TBD          | TBD        | TBD        | TBD   |
| `arithmetic_tutor_v2` (QLoRA, if run)   | TBD            | TBD          | TBD        | TBD        | TBD   |


- Expected biggest delta slices: `direct_answer_request`, `wrong_final`, `borrow_through_zero`.
- Win condition: tuned model beats prompted base on NoAnswerLeak, Robustness, and Total.
- [Biggest remaining failure mode by mistake_type slice — TBD]
- [v1 vs v2 delta — TBD]

**DOK 2 — Summary:**

- [Did data→behavior hold? One sentence with numbers — TBD after QLoRA run.]
- [What still fails, and is it a data problem or something else? — TBD]

**Link:** [artifacts/eval/](artifacts/eval/) · [docs/litmus_test.md](docs/litmus_test.md)

### Category 4: Why AI Fails as a Tutor



#### Subcategory 4.1: Solving ≠ Tutoring

**Source:** Macina et al., MathDial (2023) · Liu et al., SocraticLM, NeurIPS (2024)

**DOK 1 — Facts:**

- MathDial: GPT-3 class models are good problem solvers but fail at tutoring — they generate factually incorrect feedback or reveal solutions to students too early.
- MathDial collected 3K teacher-student dialogues with a taxonomy of teacher moves (Focus, Probing, Telling, Generic); fine-tuning on this data makes models significantly more equitable tutors.
- SocraticLM: current LLM tutoring predominantly follows a passive "Question-Answering" paradigm where students receive answers and explanations rather than guided inquiry.
- SocraticLM's SocraTeach dataset (35K multi-round dialogues) was built because general LLMs inadequately simulate Socratic teachers without dedicated training data.

**DOK 2 — Summary:**

- Capability and tutoring behavior are decoupled. A model that solves correctly can still fail pedagogically — the exact gap my litmus test measures at add/sub granularity.

**Link:** [MathDial (2023)](https://arxiv.org/abs/2305.14536) · [SocraticLM, NeurIPS (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9bae399d1f34b8650351c1bd3692aeae-Paper-Conference.pdf)

#### Subcategory 4.2: Guardrails and Learning Harm

**Source:** Bastani et al., PNAS (2025) · Microsoft, GenAI Learning Outcomes (2025)

**DOK 1 — Facts:**

- Field experiment with ~1,000 high-school math students: GPT-4 access during practice improved grades 48% (GPT Base) to 127% (GPT Tutor with guardrails).
- When access was removed, GPT Base students performed 17% worse on exams than students who never had access — unfettered AI harmed long-term learning.
- GPT Base students used the tool as a crutch, asking for and copying solutions; GPT Tutor students asked for help and attempted answers independently.
- Guardrail design (hints instead of answers) largely mitigated the negative learning effect.
- Microsoft research review: baseline chatbot tutors with minimal prompt engineering led students to prompt for direct solutions and copy-paste; pedagogically designed AI tutors produced better engagement and learning outcomes.

**DOK 2 — Summary:**

- A tutor that gives answers is not just failing a behavior spec — it may actively harm learning. Guardrails are a pedagogical requirement, not a nice-to-have.

**Link:** [PNAS GPT Tutor study](https://doi.org/10.1073/pnas.2422633122) · [Microsoft GenAI Learning Outcomes](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/GenAILearningOutcomes_published_2025-12-16.pdf)

#### Subcategory 4.3: Helpfulness Bias and Sycophancy

**Source:** "Check My Work?" (2025) · EduFrameTrap (2025) · Wang et al., Tutor CoPilot (2025)

**DOK 1 — Facts:**

- "Check My Work?": when students mention an incorrect answer, LLM accuracy degrades by up to 15 percentage points; mentioning the correct answer boosts accuracy by the same margin — models flip answers to agree with students.
- Effect is stronger in smaller models (up to 30% for GPT-4.1-nano vs 8% for GPT-4o).
- EduFrameTrap: pedagogical sycophancy under social-epistemic pressure is an educational safety risk not captured by standard alignment metrics; strong reasoning can coexist with weak resilience to pressure.
- RLHF creates a tension between following instructions and providing context-sensitive, corrective responses — models prioritize helpfulness over logical consistency without explicit training otherwise.
- Tutor CoPilot RCT: human tutors assisted by AI that models expert pedagogy increased probing questions and reduced generic praise; students were 4 pp more likely to master math topics.

**DOK 2 — Summary:**

- Base LLMs are aligned to agree and help, not to withhold answers or correct under pressure. My `direct_answer_request` and `wrong_final` failure slices are local instances of this global alignment failure.

**Link:** ["Check My Work?" (2025)](https://arxiv.org/html/2506.10297v1) · [EduFrameTrap (2025)](https://arxiv.org/html/2605.14604) · [Tutor CoPilot (2025)](https://edworkingpapers.com/sites/default/files/ai24_1054_v2.pdf)

#### Subcategory 4.4: Litmus Test — Prompting Is Not Enough

**Source:** [docs/litmus_test.md](docs/litmus_test.md) · [artifacts/eval/qwen_base_prompted_summary.jsonl](artifacts/eval/qwen_base_prompted_summary.jsonl) · [src/arithmetic_tutor_slm/behavior.py](src/arithmetic_tutor_slm/behavior.py)

**DOK 1 — Facts:**

- Model: `Qwen/Qwen2.5-0.5B-Instruct` with `SYSTEM_PROMPT` from `behavior.py` (same contract used for SFT and eval).
- Held-out eval: 200 cases, 25% adversarial.
- Scores: NoAnswerLeak 0.81, HintCorrectness 0.54, StepCalibration 0.12, Total 0.51.
- 38 of 200 cases leaked the final answer; all 23 `direct_answer_request` cases leaked (100%).
- Reference SFT data scores 1.00 on all metrics — the training target ceiling.
- Example failure: student asks "Just tell me the answer to 2827 + 6967. I don't want hints." → model responds "Sure! The answer to 2827 + 6967 is 9794."

**DOK 2 — Summary:**

- The litmus test passes: fine-tuning is warranted. This is a local replication of MathDial's core finding — the base model can compute but cannot hold the tutor contract under pressure.

**Link:** [docs/litmus_test.md](docs/litmus_test.md) · [eval summary](artifacts/eval/qwen_base_prompted_summary.jsonl)

### Category 5: Fine-Tuning for Socratic Tutors



#### Subcategory 5.1: SFT on Tutoring Dialogues

**Source:** Liu et al., SocraticLM, NeurIPS (2024) · Macina et al., MathDial (2023) · Zhang et al., SocraticLLM (2024)

**DOK 1 — Facts:**

- SocraticLM: fine-tuned ChatGLM3-6B on SocraTeach (35K multi-round dialogues, 208K single-round examples); surpassed GPT-4 pedagogical quality by >12%.
- SocraticLM used three training strategies to balance teaching and problem-solving ability, avoiding catastrophic forgetting of math accuracy.
- MathDial: fine-tuning on 3K teacher-student dialogues made models significantly more equitable — scaffolding without telling solutions.
- SocraticLLM / SocraticMATH: 6,846 multi-turn Socratic dialogues covering 513 primary-school math knowledge points; structured dialogue phases (review → heuristic → rectification → summarization) improve teaching quality.
- SocraticLM ablation: removing single-round "teaching ability" augmentations dropped overall quality by 8%; Incorrect Answer Recognition and Successful Rejection Rate were the most impacted metrics.

**DOK 2 — Summary:**

- Curated tutoring dialogue data + SFT reliably shifts models from Q&A to guided inquiry. My synthetic pipeline targets the same shift for a narrower domain with deterministic quality gates.

**Link:** [SocraticLM](https://proceedings.neurips.cc/paper_files/paper/2024/file/9bae399d1f34b8650351c1bd3692aeae-Paper-Conference.pdf) · [MathDial](https://arxiv.org/abs/2305.14536) · [SocraticLLM](https://arxiv.org/abs/2407.17349) · [SocraticMATH repo](https://github.com/ecnu-icalk/socraticmath)

#### Subcategory 5.2: Preference Tuning (Stretch Path)

**Source:** Gatti, EULER (2024) · GiovanniGatti/socratic-llm · SocraticLM ablations

**DOK 1 — Facts:**

- EULER: DPO with (prompt, chosen=Socratic, rejected=direct-answer) pairs steers LLMs toward Socratic behavior; fine-tuned model approaches GPT-4o performance on Socratic dialogue evaluation.
- Pipeline: generate multiple candidate responses → judge LLM ranks by Socratic quality → best/worst become DPO pair.
- SocraticLM found Incorrect Answer Recognition (IARA) and Successful Rejection Rate (SRR) are the hardest teaching abilities — the same adversarial slices my v2 dataset oversamples.
- DPO is the natural stretch goal after v1 SFT if `direct_answer_request` robustness still wobbles post-training.

**DOK 2 — Summary:**

- SFT establishes the tutoring behavior; DPO can sharpen spec adherence on the hardest rejection cases. Maps directly to preference pairs: on-spec Socratic hint vs. leaky direct answer.

**Link:** [EULER paper](https://ceur-ws.org/Vol-3879/AIxEDU2024_paper_26.pdf) · [socratic-llm repo](https://github.com/GiovanniGatti/socratic-llm)

#### Subcategory 5.3: Small-Model SFT Reliability

**Source:** Si et al., "Does Instruction Tuning Make LLMs More Consistent?" (2024) · prompting vs SFT on small models (2025)

**DOK 1 — Facts:**

- Instruction-tuned models show lower representation spread and higher consistency under semantically equivalent input perturbations compared to base models.
- On small models (GPT-2, DistilGPT2), SFT consistently outperforms prompting by 30+ absolute percentage points on constrained tasks; the performance gap is approximately constant across model scales.
- SFT internalizes task semantics beyond surface form, improving generalization — prompting relies on in-context pattern matching that breaks under distribution shift.
- For a 0.5B instruct model targeting one narrow behavior, QLoRA SFT is the literature-supported path to reliability that prompting cannot guarantee.

**DOK 2 — Summary:**

- Small models benefit most from SFT for constrained behavior. My QLoRA stack on Qwen2.5-0.5B is aligned with published findings on SLM instruction following.

**Link:** [Instruction tuning consistency (2024)](https://arxiv.org/html/2404.15206v2) · [Prompting vs SFT on small models (2025)](https://arxiv.org/html/2506.17289v2)

#### Subcategory 5.4: Pipeline Mapping

**Source:** [src/arithmetic_tutor_slm/data.py](src/arithmetic_tutor_slm/data.py) · [src/arithmetic_tutor_slm/filters.py](src/arithmetic_tutor_slm/filters.py) · [scripts/train_qlora.py](scripts/train_qlora.py) · [scripts/make_v2_dataset.py](scripts/make_v2_dataset.py)

**DOK 1 — Facts:**

- Pipeline: deterministic case generation (`first_step()`, `solve()`) → quality gate (no leak, one-step calibration, length bounds) → chat-format JSONL → QLoRA SFT on Qwen2.5-0.5B-Instruct.
- Each row encodes: problem, student state, hidden final answer, expected next step, mistake type, and target Socratic response.
- v2 dataset oversamples adversarial mistake types (`direct_answer_request`, `wrong_final`, `borrow_through_zero`, `alignment`) — mirrors SocraticLM's four teaching-ability augmentations and MathDial's teacher-move taxonomy.
- Hidden final answer used for filtering and eval only; never appears in training targets.
- Default: ~1K train / 200 eval cases — smaller than SocraticLM's 35K but with deterministic ground truth and hard quality filtering on every row.

**DOK 2 — Summary:**

- Literature validates the exact stack: curated/synthetic tutoring data → hard filter → SFT (→ optional DPO). Narrow scope to add/sub lets data quality beat model size.

**Link:** [data.py](src/arithmetic_tutor_slm/data.py) · [filters.py](src/arithmetic_tutor_slm/filters.py) · [train_qlora.py](scripts/train_qlora.py) · [make_v2_dataset.py](scripts/make_v2_dataset.py)