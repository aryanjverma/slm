# APUSH LEQ Data Improvement Plan (v2)

This plan supersedes the dataset-composition recommendations in `docs/plans/leq_dataset.md`.
The existing ingestion and generation code is useful, but the current artifacts are not yet a
reliable basis for measuring or improving real grading accuracy.

## Goal

Build a high-quality `(prompt, student response) -> (grade, feedback)` dataset that teaches a
0.5B model to apply the current College Board APUSH LEQ rubric to realistic timed essays.
Success means better agreement with a manually verified, official College Board holdout while
preserving valid JSON, grounded feedback, and adversarial robustness.

## Current-state audit (2026-07-10)

| Artifact | Finding | Consequence |
|---|---|---|
| `train_cases.jsonl` | 1,000 rows but only 147 unique essays; 853 are exact duplicates | The model can memorize a small template grammar instead of learning grading |
| `train_cases.jsonl` | Median essay length is 46 words (range 32-77) | Training inputs do not resemble timed LEQs |
| `train_cases.jsonl` | 105 unique assistant targets and zero `synth_realistic` rows | Feedback phrasing and score patterns are highly repetitive |
| `train_realistic_cases.jsonl` | 12 good pilot rows, covering only 2 prompts | The realistic pipeline works, but has not produced a usable training slice |
| `synth_tasks.jsonl` | 120 planned rows across 10 prompts; only 14 were generated and 12 accepted | Generation stopped before broad prompt/score coverage |
| `eval_cb_cases.jsonl` | 53 official rows, but 28 contain scoring-commentary/header text in `student_response` | The reported real-eval metrics are confounded |
| `eval_cb_cases.jsonl` | Contamination affects 18/18 rows from 2023 and 10/17 from 2024; 2025's 18 rows pass this marker check | Parser behavior differs materially by PDF layout/year |
| `eval_real_cases.jsonl` | Mixes 53 College Board rows with 19 external teacher/Quizlet rows | It is not a pure official golden set |
| `v1_eval_results.md` | Real JSON validity 0.29 and QWK -0.0864 | This confirms distribution shift, but must be remeasured after cleaning the eval |

Two structural issues matter more than adding raw row count:

1. The current quality gate checks schema, lexical grounding, and leakage, but it does not prove
   that a generated essay genuinely deserves its assigned rubric profile.
2. The same agent writes the realistic essay, sees the target score, and supplies its feedback.
   Echoing the requested score is not independent label validation.

## Data policy

- Treat official College Board prompts, scoring guidelines, samples, and reader commentary as
  the calibration source and golden evaluation reference.
- Obtain written College Board permission before automated collection, transformation,
  model training, publication, or redistribution. Its educator terms prohibit reuse, scraping,
  and data mining without express permission; public availability is not a dataset license.
- Keep existing official artifacts restricted and out of training while permission is unresolved.
- Use official prompts directly or create close adaptations only when permission covers that use.
  Otherwise write original prompts from historical topics and generic LEQ reasoning patterns.
- Never use an essay in both generation context and evaluation. Split by prompt family and
  source document, not only by row ID.
- Keep College Board, teacher-scored, and synthetic evaluations as separate tracks.
- Record source URL, file hash, year, set, question, sample ID, rubric version, extraction method,
  generator configuration, grader configuration, and review status for every row.

The currently complete public sample corpus is 2023-2025 (54 responses). Tag 2023 separately
from 2024-2025 because the complexity-point language changed in 2024. The 2026 archive currently
has prompts but no released labels, and the May 2027 exam changes prompt format while retaining
the scoring criteria. Primary-source findings and usage constraints are recorded separately in
`docs/college-board-data-source-research.md`.

## Phase 0: repair the golden set first

Do this before generating or training more data; otherwise there is no trustworthy optimization
target.

1. Complete the permission checkpoint and document allowed storage, evaluation, training, and
   redistribution uses. If permission is not granted, do not build or distribute a derived
   College Board corpus; use a separately authorized human-created evaluation instead.
2. Version the rubric as `2023_leq` and `2024_2026_leq`, and encode the official evidence and
   analysis/reasoning rules rather than the current simplified quantity-only descriptions.
3. Fix `apc_parser.py` for the 2023 and 2024 layouts. Do not reconstruct golden essays from
   commentary quotes when full essay extraction fails.
4. Add hard contamination rejects for scoring-commentary headers, row labels, page headers,
   copyright footers, and commentary boilerplate inside `student_response`.
5. Preserve extraction metadata in `FRQCase` (or a required sidecar manifest), including
   `essay_source` and parser confidence. The current schema drops this information.
6. Manually compare every official row with its PDF. Require two checks: essay boundaries are
   exact, and all four criterion scores match the scoring commentary.
7. Publish a new immutable `eval_cb_golden_v2.jsonl`; do not silently overwrite the old frozen
   artifact. Keep external rows in `eval_external_cases.jsonl` only.
8. Re-run the v1 model separately by rubric version and use that result as the true baseline.

Acceptance gate:

- 100% of official rows have verified provenance and scores.
- 0 contamination-marker hits and 0 commentary-reconstructed essays.
- A manual audit log names the reviewer and disposition of every row.
- Official rubric versions and external metrics are reported separately.

## Phase 1: build a prompt bank with held-out families

Expand from 10 canonical prompts to at least 60 prompt families spanning APUSH periods 2-9 and
the three LEQ reasoning patterns: causation, comparison, and continuity/change over time.

For each prompt, store:

- exact source or parent prompt, source year, period, reasoning skill, date range, and topic;
- whether text is official, adapted, or newly written;
- a normalized `prompt_family_id` used for split and leakage control;
- a short list of historically valid evidence and common misconceptions for validation only.

Split prompt families before essay generation:

- 70% train families;
- 15% synthetic development families;
- 15% synthetic challenge families;
- all official golden prompts remain outside training and generation context.

Maintain a small forward-looking challenge set for the broader May 2027 prompt format, but do not
mix that format into the primary benchmark until representative scored samples exist.

Do not create near-paraphrases across splits. Use token similarity plus a manual topic/date-range
review to enforce family separation.

## Phase 2: generate realistic student essays

Generate 1,200 accepted essays, expecting 1,600-2,000 raw candidates after rejection. Replace the
template-heavy training artifact rather than appending more duplicates.

### Student persona prompt

The writer receives the LEQ, a hidden target rubric profile, and a sampled student persona:

- time budget: 20, 25, 30, 35, or 40 minutes;
- historical knowledge: weak, uneven, competent, or strong;
- planning style: no outline, rushed thesis, evidence-first, or organized argument;
- mechanics: clean, ordinary errors, run-ons, fragments, misspellings, or uneven paragraphs;
- misconception profile: none, vague chronology, wrong-period evidence, factual mix-up, or
  unsupported generalization;
- response length sampled by persona and time budget, not a fixed 400-850-word rule.

Suggested writer instruction:

> You are a high-school student taking the APUSH LEQ under a specified time limit. Write the
> essay you could realistically produce from the supplied knowledge profile. Prioritize the
> argument and historical evidence over spelling and grammar. Do not mention the rubric, target
> score, dataset, or these instructions. Do not imitate or quote any released student sample.

Vary score profiles across all totals 0-6, but sample criterion combinations that are valid under
the selected official rubric version. Keep the released-sample distribution as a calibration
reference, not as an estimate of the real exam population.

Composition target:

- 75% ordinary realistic essays across totals 0-6;
- 15% adversarial essays (leniency, prompt injection, irrelevant instructions);
- 10% diagnostic edge cases (borderline thesis/context, evidence named versus used, partially
  wrong evidence, and attempted complexity).

Template rows may remain only as a small adversarial curriculum slice; they should not dominate
ordinary grading examples.

## Phase 3: label independently from generation

Create a separate grader stage. The grader must not see the writer's target profile, persona,
seed, or self-assessment.

Suggested grader instruction:

> You are an experienced College Board APUSH reader grading a released-example candidate. Apply
> the supplied official LEQ scoring guideline exactly. Ignore spelling and grammar unless they
> prevent meaning. Score only what is present in the essay. For each rubric row, cite or
> paraphrase the essay's relevant language, explain why it earns or misses the point, and return
> exactly the required JSON. Do not add historical evidence that the student did not write.

Labeling protocol:

1. Two independent graders score each anonymous candidate using the same official rubric.
2. Accept automatically only when all four row scores agree and both outputs pass structural and
   grounding gates.
3. Send disagreements to a third adjudicator that sees both rationales but not the generation
   target.
4. Reject cases where adjudication remains uncertain or where the accepted score differs from
   the generation target by more than one total point; regenerate that slice instead of forcing
   the target label.
5. Human-review at least 10% of accepted rows, oversampling disagreement, totals 0/1/5/6, and the
   analysis-reasoning row.

The accepted label is the independent consensus/adjudicated score, not the pre-assigned target.
Store grader agreement and confidence so low-confidence examples can be excluded or downweighted.

## Phase 4: strengthen automated quality gates

Add these checks before a row reaches `train_chat.jsonl`:

- exact and near-duplicate detection across prompt, essay, and feedback;
- train/dev/golden n-gram leakage and prompt-family leakage;
- no target-score, rubric-instruction, or generator-prompt leakage in the student essay;
- criterion-specific feedback grounding with stored essay spans, not two-word overlap;
- evidence-period plausibility and named-entity checks against the prompt's date range;
- rubric dependency checks (for example, evidence credit must satisfy the official quantity and
  argument-use rules for that rubric version);
- minimum feedback specificity and maximum repeated-feedback/template rate;
- contamination and extraction-confidence checks for all ingested text;
- score/profile, length, prompt-family, period, reasoning-skill, and persona distribution reports.

Every rejected row should be written to a sidecar with machine-readable reasons. Generation
should replenish rejected cells until the target matrix is full.

## Phase 5: assemble versioned datasets

Produce immutable, manifested artifacts:

- `train_realistic_v2.jsonl`: independently graded realistic synthetic essays;
- `train_adversarial_v2.jsonl`: retained/generated robustness cases;
- `train_chat_v2.jsonl`: final SFT mix, with no real essays;
- `dev_synthetic_v2.jsonl`: held-out prompt families for iteration;
- `eval_challenge_v2.jsonl`: frozen synthetic edge cases;
- `eval_cb_golden_v2.jsonl`: manually verified official College Board rows;
- `eval_external_v2.jsonl`: teacher/other rows, reported separately;
- `dataset_manifest_v2.json`: hashes, counts, provenance, rubric version, generation settings,
  dedup statistics, rejection statistics, and split rules.

Recommended final training mix: 80-85% realistic ordinary/edge cases and 15-20% adversarial
cases. Shuffle deterministically after all split and leakage checks. The current 12 realistic
pilot rows may be retained only if they pass the independent regrade.

## Phase 6: evaluate data revisions before scaling training

Train in increments (approximately 200, 500, then 1,200 accepted realistic rows) and compare on
the same frozen tracks. Stop adding rows when marginal gains flatten and inspect the remaining
error slices.

Report:

- JSON/schema validity and total consistency;
- exact and within-one agreement for each row and total;
- total-score MAE and quadratic weighted kappa;
- feedback grounding and hallucination rate;
- results by criterion, total, essay length, APUSH period, reasoning skill, time budget, prompt
  family, and adversarial type.

Initial go/no-go targets for v2:

- at least 0.95 valid JSON on the clean official set;
- at least 0.70 total within-one agreement;
- positive QWK and a material improvement over the remeasured clean-v1 baseline;
- no regression on the existing adversarial litmus slices;
- no train/golden leakage and no unreviewed official rows.

## Repository work order

1. Repair `src/apush_frq_grader_slm/ingest/apc_parser.py` and add year-layout fixtures/tests.
2. Extend `FRQCase` provenance or add a mandatory manifest schema.
3. Rebuild and manually verify `eval_cb_golden_v2.jsonl`.
4. Add a versioned official/adapted prompt catalog and family splitter.
5. Refactor `synth_realistic.py` to sample personas and emit unlabeled candidates.
6. Add an independent grading/adjudication script and criterion-span output.
7. Upgrade quality, dedup, contamination, and distribution gates.
8. Generate a 100-row pilot, audit it manually, and revise prompts/gates.
9. Scale to 1,200 accepted rows and build the versioned SFT artifacts.
10. Retrain and evaluate at 200/500/1,200-row checkpoints.

The highest-value next action is not another full training run. It is repairing and verifying the
official eval set, because every later data decision depends on a trustworthy real-world metric.
