# Evaluation Suites and End-to-End Data Pipeline

## Purpose and current state

This document is the operational record for how the APUSH LEQ grader's data is generated,
screened, reviewed, assembled, trained, and evaluated. It covers both kinds of evaluation in the
repository:

1. **Data-generation evaluation**: tests applied to writer-agent essays and their labels before a
   row can enter the private training corpus.
2. **Model evaluation**: tests applied to a base model, checkpoint, adapter, or packaged v5 bundle
   to decide whether it implements the grading behavior.

The target behavior is deliberately narrow. Given an APUSH Long Essay Question and a student
response, the model returns one JSON object containing thesis, contextualization, evidence, and
analysis/reasoning scores; a deterministic 0–6 total; and essay-grounded feedback. It must resist
grade begging and prompt injection and must not invent evidence or rewrite the essay.

As of **2026-07-12**, the replacement v5 dataset is finalized and CPU checks pass. The repository
contains receipts for 60/60 final human-reviewed essays, 540 new training rows, 60 development
rows, and 75 v4 replay rows. GPU training, the 60-case base/v4/v5 development comparison, the
53-case development-informed golden evaluation, and publication have not run.

## At-a-glance pipeline

```text
CB-derived prompt/style seeds + adapted prompts + semantic fact cards
                              |
                              v
                 fixed 1,500-task campaign plan
                  30 shards x 50; seed = 51
                  score targets hidden from writers
                              |
                              v
           private, score-blind packets for writer agents
        full matched style essay + facts + capability cues
              fresh agent context required per essay
                              |
                              v
          30-row pilot -> hard gates -> human approval
                              |
                              v
             remaining 1,470 writer-agent essays
                              |
                              v
        deterministic hard gates before judging capacity
                              |
                              v
         blind authenticity/rubric/fact judging records
                              |
                              v
      normalization + dedup/copy + consensus + fact gates
               1,500 returned -> 1,308 accepted
                              |
                              v
     quota selection: 420 distribution + 180 boundary rows
                              |
                              v
       grouped 540 train / 60 dev split; 60-row review sample
                              |
                              v
          personal terminal review of all 60 sampled rows
              60 accepted, 0 corrected, 0 rejected
                              |
                              v
       hash-bound finalization + 75 high-agreement v4 replay
                              |
                              v
      separate scorer and feedback LoRA training/checkpoints
                              |
                              v
       60-case development comparison -> freeze selection
                              |
                              v
        one 53-case development-informed golden evaluation
                              |
                              v
              locked release gates -> publish or fail
```

## 1. Data-generation campaign

### 1.1 Inputs and planning

`scripts/plan_v5_tasks.py` creates a deterministic campaign from the v5 CB-derived seed profiles.
The frozen invariants in `dataset_v5.py` are:

| Item | Value |
|---|---:|
| Planned candidates | 1,500 |
| Shards | 30 |
| Rows per shard | 50 |
| Campaign seed | 51 |
| Planned `golden_matched` rows | 1,068 |
| Planned boundary rows | 432 |
| Final selected rows | 600 |
| Final `golden_matched` rows | 420 |
| Final boundary rows | 180 |
| New train / development | 540 / 60 |
| v4 replay rows | 75 |
| Final manual-review packet | 60 |

The 432 boundary tasks are 36 lower/upper contrast pairs for each of six rubric transitions:

- thesis 0→1;
- contextualization 0→1;
- evidence 0→1 and 1→2;
- analysis/reasoning 0→1 and 1→2.

Boundary instructions describe observable writing behavior without exposing a score. Examples are
“discusses the topic but never settles on one overall answer” versus “states one overall answer
and previews why it is defensible.” Distribution rows cycle through historical-knowledge,
argument-control, mechanics, organization, and time-pressure profiles. Prompts are adapted from
the seed family, and each task retains grouping metadata for leakage-safe splitting.

Command:

```powershell
python scripts/plan_v5_tasks.py
```

Primary artifacts:

- `artifacts/data/v5/planning/generation_tasks_v5.jsonl`
- `artifacts/data/v5/planning/generation_manifest_v5.json`

### 1.2 Why the first v5 corpus was discarded

The original deterministic-composer campaign is not part of the final corpus. The aggregate r1
audit scanned all 1,500 essays and found **1,500 contaminated essays (100%)**. The principal
categories were:

| Artifact category | Count | Rate |
|---|---:|---:|
| Planning/draft-process language | 1,486 | 99.07% |
| Memory/notes language | 1,253 | 83.53% |
| Physical test-condition language | 925 | 61.67% |
| Knowledge admissions | 169 | 11.27% |
| Timing-theater filler | 49 | 3.27% |

The feature-based authenticity logic had rewarded informal or timed-looking surface cues while
missing systematic generator artifacts such as mentions of notes, outlines, memory, pencils,
clocks, and manufactured “running out of time” filler. The selected 600-row r1 corpus was also
100% contaminated. `scripts/report_v5_r1_authenticity_failure.py` preserves this failure as an
aggregate report without retaining private essay text.

Production response:

- `scripts/generate_v5_essays.py` now exits with status 2 unless the explicit
  `--allow-legacy-composer` experimental escape hatch is supplied.
- The deterministic composer is forbidden for production replacements.
- New deterministic hard gates override any reader approval.
- Writer generation moved to independent cloud agents with a fresh context for every essay.

The authoritative aggregate record is
`artifacts/data/v5/planning/v5_r1_authenticity_failure.json`.

### 1.3 Writer-agent packets and isolation contract

`scripts/export_v5_generation_packets.py` produces private packets. Each packet contains:

- task ID and adapted LEQ prompt;
- the complete directly matched golden essay as `style_reference_essay`;
- its reference word count;
- semantic fact cards to paraphrase;
- student capability and timed-composition cues;
- content-level boundary behavior, when applicable;
- an essay-only generation contract and private-use notice.

It explicitly excludes scores, total, rubric labels, resolved grades, feedback, source case IDs,
seed IDs, authenticity/rubric reviews, fact checks, failure types, tags, and assistant targets.
The exporter asserts that forbidden keys did not leak and records that a fresh cloud-agent context
is required per essay. Writers return only:

```json
{"task_id": "v5-0000", "student_response": "...essay text..."}
```

The writer is instructed to return only an authentic timed student essay. Weakness must appear as
omission, vagueness, plausible mistakes, or an underdeveloped argument—not a meta admission such
as “I cannot recall.” Fact cards must be paraphrased. The style essay is for tone, length, and
mechanics, subject to both of these copy limits:

- no more than 20 contiguous normalized words;
- no more than one candidate sentence sharing an eight-word source sequence.

Pilot export:

```powershell
python scripts/export_v5_generation_packets.py `
  --fact-cards PATH_TO_PRIVATE_FACT_CARDS `
  --pilot-only
```

Remaining-production export, after pilot approval:

```powershell
python scripts/export_v5_generation_packets.py `
  --fact-cards PATH_TO_PRIVATE_FACT_CARDS `
  --exclude-pilot
```

The second command fails closed unless the pilot approval exists and its essay-file SHA-256 still
matches.

### 1.4 The 30-essay pilot suite

The pilot contains 24 boundary essays—two complete lower/upper pairs for each of the six boundary
types—and six distribution-matched essays spanning period and capability. It exercises the most
important writer behaviors before authorizing the remaining 1,470 generations.

Deterministic pilot gate:

```powershell
python scripts/validate_v5_pilot_hard_gates.py `
  --essays artifacts/data/v5/private/pilot_essays_v5.jsonl `
  --audit artifacts/data/v5/private/pilot_hard_gate_audit_v5.json
```

Human pilot review:

```powershell
python scripts/review_v5_pilot.py --reviewer "Aryan Verma"
```

The reviewer sees the task class, boundary and side, period, capability profile, prompt, and essay,
then accepts, rejects, skips, or quits. Decisions are saved so the review can resume. A rejected
essay must be regenerated. Approval is emitted only when all 30 are accepted or corrected and is
bound to the SHA-256 of `pilot_essays_v5.jsonl`.

Completed evidence:

- hard-gate audit: 30 returned, 30 accepted, 0 rejected, ready for human review;
- approval: 30 accepted by Aryan Verma at `2026-07-12T01:26:51Z`;
- approved pilot SHA-256:
  `d3ed7755fcdf05169db6bab007df4a42d9e0ebdc5dea459f768fdadb274cbb83`.

### 1.5 Production hard-gate suite

The remaining campaign returned 1,470/1,470 expected essays with no missing or extra task IDs.
`scripts/validate_v5_production_hard_gates.py` runs before full judging and rejected none in the
replacement run. The audit marks all 1,470 ready for judging.

The hard gate is a union of checks from `authenticity_gates_v5.py`:

1. **Meta/process artifacts**
   - memory or notes;
   - outlines, drafts, checklists, margins, rewriting, or narrating the writing process;
   - pens, pencils, erasers, clocks, seating, bluebooks, or test conditions;
   - explicit missing-knowledge admissions;
   - generator “mattering” stubs;
   - prompt, persona, fact-card, score, or style-reference leakage;
   - artificial time-pressure filler;
   - stock teacher/classmate/worksheet filler.
2. **Style copying**
   - longest contiguous normalized overlap must be ≤20 words;
   - at most one sentence may contain an eight-word style-reference overlap.
3. **Length matching**
   - ordinarily about ±20% of the matched golden response;
   - widened handling for extreme short references and ±25% for references above 420 words;
   - no nonempty accepted essay below the absolute short floor.
4. **Presence**
   - empty essays fail.

These checks are deterministic vetoes. A candidate does not survive merely because an
authenticity reader approves it.

### 1.6 Blind judging and label evaluation

`scripts/judge_v5_essays.py` consumes only prompt plus essay, optionally one shard at a time. It
does not pass style references, capability profiles, coverage classes, boundary metadata, or seed
IDs into the judge. `scripts/validate_v5_external_candidates.py` restores trusted planner metadata
after judging.

The repository supports externally returned blind review records. The included production runner,
`src/apush_frq_grader_slm/judge_v5.py`, is specifically a **deterministic feature-based local
judge**, not a set of independent human or model-agent calls. It creates logically separate reader
identities with deterministic task-specific random seeds:

- authenticity readers `auth-a` and `auth-b`, plus `auth-c` on disagreement;
- rubric readers `reader-a`, `reader-b`, and `reader-c`;
- historical fact checker `facts-a`.

This distinction matters when interpreting “independent readers”: the validator requires distinct
reader IDs and the heuristics use different thresholds, but the bundled judge implementation is
one code path. Truly independent external readers can produce the same candidate schema.

The label gate requires:

- at least two distinct authenticity reviews;
- a third authenticity review if the first two disagree;
- at least two passing authenticity votes, where a pass means both `student_like` and
  `timed_ap_consistent`;
- three distinct rubric readers with schema-valid four-row scores;
- adjudication whenever readers disagree or any confidence is below 0.85;
- a schema-valid resolved score object;
- schema-valid grounded feedback for all four criteria;
- a passing historical fact check.

The deterministic rubric readers infer thesis, context, evidence, reasoning, and complexity cues.
They resolve each criterion by majority, breaking ties toward the median, and generate grounded
feedback from the essay. The local fact checker rejects empty/nonsense/repetitive text and obvious
year anachronisms. This is a data-labeling system, not the final model eval, and human review remains
the last acceptance authority for the sampled rows.

Example:

```powershell
python scripts/judge_v5_essays.py `
  --essays PATH_TO_RETURNED_ESSAYS.jsonl `
  --tasks artifacts/data/v5/planning/generation_tasks_v5.jsonl `
  --output-dir artifacts/data/v5/private/judged
```

### 1.7 Candidate validation: leakage, overlap, consensus, and realism

`scripts/validate_v5_external_candidates.py` checks the complete returned task set, refuses
duplicate or unknown task IDs, and can require exactly 1,500 planned and returned rows. It
recomputes trusted metadata and distribution match rather than trusting writer output.

Its overlap layers are deliberately separate:

- an eight-gram source-copy index over the AMSCO knowledge base and semantic fact cards, with
  historical-name/date exemptions;
- exact and near-duplicate checks against golden essays, v4/private overlap corpora, and already
  accepted v5 peers;
- the dedicated style-reference quota described above.

It then reapplies all non-overlap hard gates, authenticity consensus, rubric consensus,
adjudication, resolved-grade schema, fact check, selection-class metadata, prompt-family metadata,
and distribution/style eligibility.

Replacement-campaign result:

| Result | Count |
|---|---:|
| Planned/returned | 1,500 / 1,500 |
| Accepted | 1,308 |
| Rejected | 192 |
| Accepted `golden_matched` | 916 |
| Accepted boundary | 392 |
| Authenticity-gate failures | 137 |
| Golden-distribution-match failures | 116 |

Reason counts overlap because one rejected row may fail more than one gate. The validator used
2,065 allowed historical phrases. Private rows are explicitly marked non-redistributable.

### 1.8 Selection, distribution evaluation, and leakage-safe split

`scripts/assemble_v5_dataset.py prepare-review` reruns peer deduplication and every non-overlap
candidate gate, then selects:

- 420 golden-distribution-matched rows, sampled to match golden score vectors;
- 180 boundary rows: exactly 15 complete lower/upper pairs for each of six boundary types.

The split groups related cases so a prompt family or contrast pair does not leak across train and
development. The result is 540 train and 60 development rows.

The style-distribution audit compares the 420 matched rows with the golden reference set. The final
audit passed all active measures:

| Measure | Candidate | Golden | Gate/result |
|---|---:|---:|---|
| Mean words | 243.0643 | 263.5849 | within 10%, pass |
| Median words | 233.0 | 180.0 | within assembly band, pass |
| First quartile | 109.0 | 102.0 | pass |
| Third quartile | 364.75 | 346.0 | pass |
| Mean sentence words | 19.1352 | 19.7729 | pass |
| Sentence-length std. | 6.2678 | 8.4390 | pass |
| Punctuation per 100 words | 3.5031 | 3.1525 | pass |
| Informal markers per 100 | 0.0866 | 0.0176 | pass |
| Spelling-error density per 100 | 0.0523 | 0.0488 | pass |

Paragraph count was recorded but skipped as a gate because the extracted golden essays lack
paragraph breaks.

Command:

```powershell
python scripts/assemble_v5_dataset.py prepare-review `
  --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
```

This writes a provisional 600-row corpus, a private manifest, and the 60-row review packet.

## 2. How the 60 essays were reviewed through the CLI

### 2.1 Why 60 rows

`manual_review_packet()` takes a deterministic 10% sample of the selected 600 rows. It stratifies
by selection class and resolved total, cycling through nonempty buckets until it has exactly 60.
The completed packet contains:

- 30 boundary and 30 golden-matched essays;
- totals: five 1s, five 2s, twenty-two 3s, six 4s, ten 5s, and twelve 6s.

This packet is a final sampled audit of essay suitability, score correctness, and feedback quality;
it is not the 60-case development set, even though both happen to contain 60 rows.

### 2.2 Starting and resuming the reviewer

Install the package so imports and Rich terminal rendering are available, then run:

```powershell
python -m pip install -e .
python scripts/review_v5_manual_packet.py --reviewer Aryan
```

Useful resume/filter controls:

```powershell
# Resume display at one-based packet row 31
python scripts/review_v5_manual_packet.py --reviewer Aryan --start 31

# Show only rows not yet accepted/corrected by this reviewer
python scripts/review_v5_manual_packet.py --reviewer Aryan --only-unverified

# Inspect one exact task
python scripts/review_v5_manual_packet.py --reviewer Aryan --task-id v5-0000

# Screen-reader-friendly output and a narrower terminal
python scripts/review_v5_manual_packet.py --reviewer Aryan --plain --width 90
```

Default inputs and outputs:

- packet: `artifacts/data/v5/private/manual_review_packet_v5.jsonl`;
- approval: `artifacts/data/v5/private/manual_review_approval_v5.json`;
- backups: `artifacts/data/v5/private/logs/manual_review_backups/<UTC timestamp>/`.

### 2.3 What appeared for each essay

For every packet row, the CLI rendered:

1. case position out of 60, task ID, current decision, and current reviewer;
2. boundary type, pair ID, and lower/upper side when applicable;
3. full adapted prompt;
4. full student essay;
5. resolved score and each blind reader's score for all four criteria, plus total;
6. resolved grounded feedback for all four criteria;
7. number of authenticity readers and counts passing student-like/timed-AP judgments;
8. historical fact-check pass/fail and checker ID.

The action menu was printed before the prompt:

| Key | Action | Effect |
|---|---|---|
| `a` | Accept | Confirm essay, scores, and feedback; optional note |
| `c` | Correct | Edit any score and optionally any feedback field |
| `r` | Reject | Mark the essay unsuitable; replacement note required |
| `s` | Skip | Leave unchanged and continue |
| `b` | Back | Return to the preceding displayed row |
| `h` | Help | Reprint detailed instructions |
| `q` | Quit | Save completed work and exit |

Score corrections are range checked: thesis/contextualization 0–1 and evidence/analysis 0–2.
Feedback corrections must contain all four validated fields. A score-only issue should be corrected,
not rejected; rejection is reserved for a fundamentally unsuitable essay that must be regenerated.

### 2.4 Persistence and safety properties

Before the first mutation, the CLI creates a timestamped private backup of the packet and any
existing approval. Every accepted, corrected, or rejected action is written immediately using a
temporary file, flush, `fsync`, and atomic `os.replace`. A quit or interruption therefore loses at
most the decision currently being entered, not completed rows.

The CLI distinguishes automated/preliminary decisions from personal verification. A row counts as
human verified only when:

- its decision is `accept` or `corrected`; and
- `manual_review.reviewed_by` exactly matches the active reviewer ID.

Preliminary corrections are preserved when the human accepts the row; the decision remains
`corrected`. Rejections and pending/skipped rows block approval.

### 2.5 Approval receipt and completed result

At exit, the tool prints counts for total, accept, corrected, reject, pending, and human-verified.
It writes approval only when the same reviewer has personally accepted or corrected all 60 and no
reject remains. The approval records reviewer, UTC timestamp, accept/corrected counts, and the
exact packet SHA-256. Any later packet edit invalidates it.

The completed receipt is:

| Field | Value |
|---|---|
| Reviewer | Aryan |
| Approved at | `2026-07-12T20:27:51Z` |
| Accepted | 60 |
| Corrected | 0 |
| Rejected/pending | 0 / 0 |
| Packet SHA-256 | `b5ae4d76339c5d26986f93d655eff2abafc0887f5d092293b7127679e1498321` |

The receipt proves that this exact packet was reviewed; it does not expose essay text.

### 2.6 Final assembly after review

After approval:

```powershell
python scripts/assemble_v5_dataset.py finalize `
  --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
```

Finalization verifies the approval and packet hash, applies any human corrections back to the
provisional corpus, reruns selection, and writes case and task-specific chat files. It also chooses
75 v4 replay cases round-robin by observed total from cases with agreement ≥0.85 or explicit human
review.

Final counts:

- `train_cases_v5.jsonl`: 540 new rows;
- `dev_cases_v5.jsonl`: 60 rows;
- `replay_cases_v4_for_v5.jsonl`: 75 rows;
- `train_cases_v5_with_replay.jsonl`: 615 rows;
- scorer and feedback chat rows: 615 train and 60 dev for each task;
- golden evaluation rows in training: 0.

The assembly audit binds the approval, review packet, combined training data, and every case/chat
artifact by SHA-256. The combined training hash is
`0de79600ad2a4479732c74a485986e03b7ca8ea1a4285268ee1765834a0c23ae`.

Both `train_v5.py` and the Colab notebook call a strict preflight. Training is denied if counts,
approval, packet hash, artifact hash, combined hash, or zero-golden-leakage checks fail.

## 3. Model evaluation suites

### 3.1 Suite map

| Suite | Dataset | Purpose | Status |
|---|---|---|---|
| Deterministic behavior/litmus | 198 synthetic held-out cases | Prove prompted leniency fails grounding/adversarial behavior | Complete |
| Day-2 smoke loop | 20 held-out smoke cases | Prove generate→train→eval mechanics | Complete |
| HF real-eval | Any `FRQCase` JSONL; usually 53 CB-derived cases | Score agreement, grounding, QWK, slices | Harness complete |
| v3 locked official eval | 27 set1 development rows; set2 final behind lock | Reproducible checkpoint acceptance and one-shot final eval | Harness complete |
| v5 CPU contract smoke | Generated in memory | Contracts for plan, packets, chat rows, two-pass inference, release fail-closed | Complete/pass |
| v5 checkpoint ranking | 60-case private dev summaries | Select separate scorer and feedback adapters | Pending GPU training |
| v5 packaged bundle eval | 60 private dev, then 53 CB-derived cases | Final two-pass calibration and release decision | Pending GPU training |

### 3.2 Deterministic 198-case behavior suite

Run:

```powershell
python -m apush_frq_grader_slm.cli.run_eval
# or installed entry point:
apush-grader-eval
```

Input: `artifacts/data/eval_cases.jsonl`, 198 quality-filtered held-out cases from a requested 200,
with roughly 25% adversarial grade-inflation and prompt-injection cases. The harness compares the
deliberately lenient `inflated_prompted_base` with the rule/reference target.

Per-case metrics:

- **StructuredOutputValid**: parsable JSON, correct structure, legal score ranges, consistent total;
- **RubricAccuracy**: fraction of four criterion scores within ±1 of reference;
- **EvidenceGrounding**: at least two feedback fields reference phrases in the student essay;
- **NoHallucination**: no rewrite markers or feedback patterns unsupported by the essay;
- **Robustness (0–2)**: conservative scoring, especially under grade begging or injection;
- **Total**:

  ```text
  (valid + rubric_accuracy + grounding + no_hallucination + robustness/2) / 5
  ```

It writes per-case results, failure-type slice summaries, and an aggregate summary under
`artifacts/eval/`.

Completed results:

| Model | Cases | JSON | Rubric | Grounding | No halluc. | Robustness | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `inflated_prompted_base` | 198 | 1.00 | 0.82 | 0.17 | 1.00 | 0.93 | 0.69 |
| `apush_grader_reference` | 198 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |

The baseline inflated all 33 grade-begging and all 29 prompt-injection cases to 6/6. This suite
established the need for fine-tuning: format compliance was already perfect, while grounding and
adversarial calibration were not.

### 3.3 Day-2 generate/train/eval smoke suite

```powershell
python -m pip install -e ".[train]"
python scripts/run_smoke_pipeline.py
```

This creates 30 training and 20 held-out rows, performs a deliberately tiny 25-step LoRA run, and
evaluates the same behavior contract. It is a systems proof, not a production benchmark.

| Model | Cases | JSON | Rubric | Grounding | No halluc. | Robustness | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| Inflated baseline | 20 | 1.00 | 0.84 | 0.15 | 1.00 | 0.90 | 0.69 |
| Reference | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| Smoke adapter | 20 | 0.55 | 0.95 | 0.95 | 0.55 | 1.65 | 0.77 |

It proved the loop and improved the intended behaviors despite expected JSON instability from the
minimal run.

### 3.4 Generic Hugging Face and real-essay suite

`scripts/eval_hf_model.py` evaluates a remote HF model, local merged model, or PEFT adapter. It
supports the legacy or v4 prompt contract, deterministic decoding by default, resumable per-case
results, per-dimension diagnostics, and real-eval agreement metrics.

```powershell
python scripts/eval_hf_model.py `
  --model PATH_OR_HF_ID `
  --model-name candidate-name `
  --eval-path artifacts/data/eval_cb_cases.jsonl `
  --real-eval `
  --output-dir artifacts/eval/candidate-name
```

Resume safety rejects unknown case IDs, duplicate saved IDs, and model-name mismatches. Each result
is flushed immediately. If summary generation fails after inference, the expensive predictions
remain saved and `scripts/summarize_from_results.py` can recompute reports without regeneration.

Additional real-eval metrics:

- exact criterion-row rate and within-one criterion-row rate;
- exact-total and within-one-total rates;
- total mean absolute error;
- quadratic weighted kappa (QWK) over 0–6 totals;
- exact rates for each criterion;
- metrics split by rubric version;
- dimensions including failure type, total, essay length, prompt family, period, reasoning skill,
  time budget, and knowledge profile;
- aggregate-only output-failure diagnostics such as valid JSON, schema-invalid JSON, likely token
  truncation, incomplete JSON, and malformed/non-JSON output.

Invalid totals are penalized with an absolute error of 6 and excluded from valid QWK pairs. QWK
clamps out-of-range numeric predictions to the 0–6 rubric scale so they remain disagreements rather
than indexing errors.

### 3.5 v3 locked official evaluation suite

The v3 suite was designed to prevent accidental test-set iteration. The CB-derived file is divided
into set1 and set2. Ordinary execution always selects the 27 set1 cases; set2 is inaccessible unless
`--final-evaluation` is explicit and a matching passing lock manifest is supplied.

```powershell
# Development/set1
python scripts/eval_v3.py `
  --model PATH_OR_HF_ID `
  --model-name candidate `
  --output-dir artifacts/eval/v3/candidate

# One permitted locked final/set2 run
python scripts/eval_v3.py `
  --model FROZEN_MODEL `
  --model-name frozen-candidate `
  --final-evaluation `
  --lock-manifest PASSING_SET1_LOCK.json `
  --output-dir artifacts/eval/v3/frozen-candidate
```

Run identity hashes model bytes or resolved model revision, data, split, decoding settings, prompt
version, and model name. Resuming fails on any identity mismatch. Generation uses a 320-token cap,
balanced-JSON stopping, and score-enum constraints. Records preserve raw output and normalized
layered output, tokens, finish reason, normalization actions, and repetition detection.

Acceptance requires all of:

- layered schema validity = 100%;
- zero truncations;
- total MAE ≤1.5;
- totals within one ≥60%;
- QWK ≥0.35.

Set2 requires the exact set1 model/data/decoding/prompt identity and passing acceptance. A final
receipt then prevents a second set2 run.

### 3.6 v5 CPU contract smoke

```powershell
python scripts/smoke_v5_pipeline.py
```

This does not train or load GPU weights. It verifies:

- exactly 1,500 planned tasks in 30×50 shards;
- score-blind packet construction and fact-card paraphrase instructions;
- separate scorer-only and feedback-only chat targets;
- two-pass inference order and deterministic total computation;
- release gates reject a failing summary and accept a passing boundary example;
- weighted scorer loss is finite when PyTorch is available.

It prints JSON with `"ok": true` on success and is also invoked from the test suite.

### 3.7 v5 training and checkpoint selection suite

V5 begins from the merged v4 adapter and trains two separate rank-16 LoRA adapters:

- **scorer**: predicts only four criterion scores;
- **feedback**: receives validated scores and predicts only four feedback fields.

The application, not either model, computes the total. The frozen configuration is scorer 4 epochs
at 1e-4 with score-token weight 4.0; feedback 2 epochs at 5e-5; batch 1; gradient accumulation 4;
warmup 0.03; maximum length 4096; seed 13. Tokenization fails instead of silently truncating an
overlength row.

After a ten-case GPU smoke, checkpoints are evaluated on the 60-case private development set. Do
not use the 53 golden cases for checkpoint ranking.

`scripts/rank_v5_checkpoints.py` uses task-specific lexicographic rankings:

- scorer: highest QWK, then lowest MAE, then evidence exact-match, then macro criterion exact-match;
- feedback: highest grounding, then structured validity, then lowest fallback-feedback rate.

The selected scorer and feedback adapters are packaged with the inherited base, prompt files, and
artifact hashes. Bundle loading verifies hashes by default.

### 3.8 v5 packaged two-pass development and golden suites

Run the packaged bundle with:

```powershell
python scripts/eval_v5.py `
  --bundle PATH_TO_BUNDLE `
  --eval-path PATH_TO_EVAL_JSONL `
  --model-name apush-frq-grader-v5 `
  --output-dir artifacts/eval/v5
```

The runner is resumable, rejects duplicate or mismatched saved results, verifies bundle hashes by
default, flushes each case, and writes an aggregate real-eval summary plus calibration diagnostics.
Diagnostics include:

- per-criterion confusion matrices;
- reference and predicted 0–6 total distributions;
- calibration by essay-length band and reference total;
- group count, means, MAE, within-one rate, and QWK where defined;
- 2,000-sample seeded bootstrap 95% intervals for QWK, MAE, and within-one rate;
- deterministic-total and feedback-fallback rates.

The required order is:

1. ten-case GPU smoke;
2. full 60-case base/v4/v5 development comparison;
3. freeze scorer, feedback, prompts, and decoding configuration;
4. evaluate the 53 CB-derived cases exactly once for the release decision.

The golden track must be marked with `--development-informed`. All 53 full golden essays were used
as private style references for writer agents, so this is **not an untouched generalization
holdout**. It still measures agreement on the reference cases, but claims must disclose this
contamination and no tuning may follow it.

### 3.9 Locked v5 release gate

```powershell
python scripts/check_v5_release.py `
  --summary PATH_TO_V5_REAL_SUMMARY.json `
  --output artifacts/eval/v5/release_decision.json
```

Every gate is required:

| Metric | Required value |
|---|---:|
| QWK | ≥0.40 |
| Total MAE | ≤1.50 |
| Totals within one | ≥60% |
| Structured output validity | ≥98% |
| Evidence grounding | ≥85% |
| Absolute predicted-mean bias from 4.0377 | ≤0.50 |
| Thesis exact match | strictly >52.83% |
| Contextualization exact match | strictly >39.62% |
| Evidence exact match | strictly >16.98% |
| Analysis/reasoning exact match | strictly >32.08% |

Missing or nonnumeric metrics fail. Merely tying a v4 criterion rate fails. If any gate fails, the
decision is `non_production_ready`; results are reported without relaxing thresholds or retuning on
golden cases.

## 4. Automated regression suite

Run all repository tests after finalization or pipeline changes:

```powershell
python -m pytest
```

Relevant groups include:

- `test_eval_v3.py`, `test_eval_resume.py`, and `test_eval_diagnostics.py`: locked splits,
  identity-safe resume, raw/layered records, invalid-output penalties, and aggregate diagnostics;
- `test_manual_review_v5.py`: human/preliminary distinction, score schema validation, rejection
  blocking, atomic writes, hash-bound approval, complete menus, and preserved corrections;
- `test_authenticity_gates_v5.py`, `test_dataset_v5.py`, `test_judge_v5.py`, and
  `test_fact_cards_v5.py`: packet privacy, copy/meta gates, planning, judge schema, consensus,
  selection, and knowledge inputs;
- `test_v5_model_pipeline.py`: strict training preflight, split targets, weighted loss, two-pass
  inference, fallback behavior, bundle hashes, diagnostics, and the executable CPU smoke;
- `test_rank_v5.py` and `test_release_v5.py`: checkpoint ranking and fail-closed release thresholds;
- `test_notebook_v5.py` and `test_notebook_official_eval.py`: Colab orchestration and official-eval
  policy.

The tests verify contracts and small synthetic fixtures; they do not substitute for human essay
review, a real GPU training run, or the final golden evaluation.

## 5. Artifact and privacy policy

Private artifacts include generated essays, full style-reference essays, writer packets, per-case
scores and feedback, judging records, manual-review packets, and per-case model predictions. They
must not be committed or published. Only aggregate audits, summaries, diagnostics without response
text, manifests, and release decisions may leave private storage.

The public companion dataset is a separate 1,000-row project-authored synthetic baseline. It is
not the 600-row private v5 corpus and must never be described as such.

Important receipts:

| Evidence | Path |
|---|---|
| Campaign plan | `artifacts/data/v5/planning/generation_manifest_v5.json` |
| Discarded r1 audit | `artifacts/data/v5/planning/v5_r1_authenticity_failure.json` |
| Pilot hard gates | `artifacts/data/v5/private/pilot_hard_gate_audit_v5.json` |
| Pilot approval | `artifacts/data/v5/private/pilot_approval_v5.json` |
| Replacement validation | `artifacts/data/v5/private/validation_audit_r2.json` |
| Production hard gates | `artifacts/data/v5/private/production_hard_gate_audit_r2.json` |
| Final review packet | `artifacts/data/v5/private/manual_review_packet_v5.jsonl` |
| Final review approval | `artifacts/data/v5/private/manual_review_approval_v5.json` |
| Final private manifest | `artifacts/data/v5/private/private_use_manifest_v5.json` |
| Hash-bound assembly audit | `artifacts/data/v5/private/assembly_audit_v5.json` |

## 6. Reproduction checklist

Data preparation and approval:

```powershell
python scripts/plan_v5_tasks.py
python scripts/report_v5_r1_authenticity_failure.py
python scripts/export_v5_generation_packets.py --fact-cards PATH --pilot-only
# Independent writer agents return 30 task_id + student_response rows.
python scripts/validate_v5_pilot_hard_gates.py --essays PATH --audit PATH
python scripts/review_v5_pilot.py --reviewer YOUR_NAME
python scripts/export_v5_generation_packets.py --fact-cards PATH --exclude-pilot
# Independent writer agents return the remaining 1,470 rows.
python scripts/validate_v5_production_hard_gates.py --essays PATH --audit PATH
python scripts/judge_v5_essays.py --essays PATH --tasks PATH
python scripts/validate_v5_external_candidates.py --tasks PATH --candidates PATH --output PATH --audit PATH
python scripts/assemble_v5_dataset.py prepare-review --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
python scripts/review_v5_manual_packet.py --reviewer YOUR_NAME
python scripts/assemble_v5_dataset.py finalize --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
python scripts/smoke_v5_pipeline.py
python -m pytest
```

GPU training and evaluation:

```powershell
# In Colab, open notebooks/colab_train_v5.ipynb and run its cells in order. It
# orchestrates the inherited-base merge, both LoRA tasks, development evaluation,
# checkpoint selection, bundle creation, frozen golden evaluation, and receipts.

# Standalone equivalents are available when running outside Colab.
python scripts/merge_v4_adapter.py --v4-adapter PATH --output artifacts/models/v5-inherited-base
python scripts/train_v5.py --task scorer --model PATH --data PRIVATE/train_cases_v5_with_replay.jsonl --eval-data PRIVATE/dev_cases_v5.jsonl --private-dir PRIVATE --golden-cases artifacts/data/eval_cb_cases.jsonl --output artifacts/models/v5-scorer
python scripts/train_v5.py --task feedback --model PATH --data PRIVATE/train_cases_v5_with_replay.jsonl --eval-data PRIVATE/dev_cases_v5.jsonl --private-dir PRIVATE --golden-cases artifacts/data/eval_cb_cases.jsonl --output artifacts/models/v5-feedback
python scripts/rank_v5_checkpoints.py --scorer-summaries PATHS --feedback-summaries PATHS
python scripts/package_v5_bundle.py --bundle PATH --inherited-base PATH --scorer PATH --feedback PATH
python scripts/eval_v5.py --bundle PATH --eval-path PATH --output-dir artifacts/eval/v5
python scripts/check_v5_release.py --summary PATH_TO_SUMMARY
```

Do not rerun or tune against the 53-case golden evaluation after seeing its results. A failed gate
is an evaluation result, not permission to convert the development-informed golden set into a
training loop.
