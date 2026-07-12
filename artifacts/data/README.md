# Data

This folder holds the datasets that train and evaluate the APUSH LEQ grader. All
files here are **regenerable outputs**, not hand-authored source — every one is
produced by a script in `scripts/` or `src/apush_frq_grader_slm/cli/` from the
behavior contract in `src/apush_frq_grader_slm/behavior.py`. This README explains
where each file comes from and how to rebuild it.

## Governing principle

> The dataset is the deliverable, not the model.

Two rules follow from that and shape everything below:

1. **Training data is 100% synthetic.** Real student essays are messy and scarce;
   we generate training essays instead, so we control the score distribution and
   the label↔essay contract.
2. **Real essays are eval-only.** Real College Board / teacher-scored essays are
   the external-validity yardstick and must never leak into training.
   `scripts/build_mixed_dataset.py::assert_no_real_essays()` enforces this and
   raises if a real-tagged case appears in a train split.

Every case — synthetic or real — is a `FRQCase` (`src/apush_frq_grader_slm/schemas.py`)
and must pass the quality gate (`filters.passes_quality_gate`: valid JSON, in-range
scores, `total == sum`, feedback grounded in the essay, no rewrites, no fabricated
quotes) before it is used.

## v4 status (current)

v4 rebuilds training data around **AMSCO historical knowledge** + **College Board golden-set
seeds**. Every training essay is synthetic, but each one is planned from a CB seed profile
(prompt family, score band, length/style cues). Writers draw facts from
`artifacts/knowledge/amsco_2016_kb.jsonl` (31 AMSCO 2016 chapters). Student and grader system
prompts both embed the full 6-point LEQ rubric (`prompts_v4.py`).

Rebuild:

```powershell
python scripts/extract_amsco_kb.py
python scripts/build_v4_seed_profiles.py
python scripts/plan_v4_tasks.py
python scripts/export_v4_generation_packets.py
# write essays into artifacts/data/v4/raw_essays/batch_XX.jsonl (or compose_v4_essays.py)
python scripts/grade_v4_essays.py
python scripts/assemble_v4_dataset.py
```

Outputs under `artifacts/data/v4/`:

| File | Rows | What |
|------|------|------|
| `train_cases_v4.jsonl` | 250 | Synthetic `FRQCase` records (AMSCO + CB-seeded) |
| `train_chat_v4.jsonl` | 250 | SFT rows with full-rubric grader system prompt |
| `synth_tasks_v4.jsonl` | 250 | Generation worklist |
| `cb_seed_profiles.jsonl` | 53 | CB structure/score/style seeds (not training prose) |
| `grades_v4.jsonl` | 250 | Target-profile scores + essay-grounded feedback |
| `dataset_manifest_v4.json` / `dataset_audit_v4.json` | — | Build metadata + audit |

Hard invariant unchanged: real CB essays stay eval-only.

## v5 regeneration (authentic rewrite)

The r1 composer corpus is discarded (see `planning/v5_r1_authenticity_failure.json`).
Pilot path:

```powershell
python scripts/export_v5_generation_packets.py --fact-cards artifacts/data/v5/planning/semantic_fact_cards_v5.jsonl --pilot-only
python scripts/validate_v5_pilot_hard_gates.py --essays artifacts/data/v5/private/pilot_essays_v5.jsonl --audit artifacts/data/v5/private/pilot_hard_gate_audit_v5.json
python scripts/review_v5_pilot.py --reviewer YOUR_NAME
```

Full production export/validation requires hash-bound `private/pilot_approval_v5.json`.
See `docs/v5_pilot_review_ready.md`.

## v5 status (private dataset finalized)

V5 built ~1,500 score-blind candidates (30×50 shards) and retains **600** accepted cases
(420 golden-matched + 180 boundary) plus 75 v4 replay rows for training. Private essay rows,
style excerpts, labels, and review packets must remain private. On 2026-07-12 all 60 replacement
review rows were accepted; final 540/60/75 assembly and the strict approval/hash/leakage preflight
passed. GPU training remains pending.

Expected private layout (gitignored):

```
artifacts/data/v5/
  planning/           # generation_tasks_v5.jsonl, generation_manifest_v5.json
  packets/            # score-blind writer shards (shard_XX.jsonl)
  private/
    fact_cards_v5.jsonl              # semantic AMSCO cards (private)
    adapted_prompts_v5.jsonl         # adapted prompt families (private, if used)
    external_candidates_v5.jsonl     # returned essays + blind reviews
    validated_candidates_r2.jsonl    # replacement validator output
    candidate_audit_v5.json          # aggregate reject/accept audit
    selected_cases_v5_provisional.jsonl
    manual_review_packet_v5.jsonl
    manual_review_approval_v5.json
    train_cases_v5.jsonl             # 540 after finalize
    dev_cases_v5.jsonl               # 60 after finalize
    replay_cases_v4_for_v5.jsonl     # 75
    train_cases_v5_with_replay.jsonl # audited 615-row combined corpus
    assembly_audit_v5.json           # aggregate audit (safe to share if scrubbed)
```

Rebuild (planning → packets → validate → assemble):

```powershell
python scripts/plan_v5_tasks.py
# Optional: scripts/build_v5_fact_cards.py / scripts/build_v5_adapted_prompts.py when present
python scripts/export_v5_generation_packets.py --fact-cards artifacts/data/v5/private/fact_cards_v5.jsonl
python scripts/validate_v5_external_candidates.py `
  --tasks artifacts/data/v5/planning/generation_tasks_v5.jsonl `
  --candidates artifacts/data/v5/private/external_candidates_v5.jsonl `
  --output artifacts/data/v5/private/validated_candidates_r2.jsonl `
  --audit artifacts/data/v5/private/candidate_audit_v5.json `
  --overlap-corpus artifacts/data/v4/train_cases_v4.jsonl `
  --overlap-corpus artifacts/data/eval_cb_cases.jsonl
python scripts/assemble_v5_dataset.py prepare-review `
  --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
# review packet + write manual_review_approval_v5.json, then:
python scripts/assemble_v5_dataset.py finalize `
  --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
```

Contract details: `docs/v5_external_data_contract.md`. CPU smoke:
`python scripts/smoke_v5_pipeline.py`.

The redistribution-safe public companion lives at `artifacts/public/apush-leq-grader-public/` and
is built only from project-authored synthetic baseline data. It is not the private v5 corpus.

## v2 status

The files documented below describe the legacy v1 pipeline. The implemented v2 pipeline now:

- builds 60 original prompt families with deterministic train/dev/challenge splits;
- generates unlabeled essays from 20/25/30/35/40-minute student personas;
- uses two anonymous graders plus adjudication instead of writer self-grades;
- carries source, rubric-version, extraction, generator, grader, and review provenance;
- rejects source/generation contamination, repeated prose/feedback, and eval leakage;
- requires 10% human review before immutable SFT artifacts are emitted;
- uses `eval_cb_cases.jsonl` as the explicitly selected 53-row v2 model evaluation set.

Current generated planning artifacts are `prompt_catalog_v1.jsonl`,
`synth_tasks_train_v2.jsonl`, `synth_tasks_dev_v2.jsonl`, and
`synth_tasks_challenge_v2.jsonl`. See `artifacts/dataset_card.md` for the v2 build sequence.

`eval_cb_cases.jsonl` is the selected evaluation input. It remains a legacy artifact rather than a
verified golden set: the current audit rejects all 53 rows for missing v2 provenance/manual review
and detects 28 rows with source-text contamination. Those findings are retained as data-quality
warnings, but they do not block the explicitly selected evaluation run.

After written evaluation permission, clean PDF extraction, and all 53 named manual reviews pass,
`scripts/build_golden_v2.py` writes a combined official file plus a 27-row `set1` dev split and a
26-row `set2` final holdout under `artifacts/data/v2/`. That stricter future artifact workflow is
separate from the currently selected direct evaluation on `eval_cb_cases.jsonl`.

## Two eval tracks (deliberately separate)

| Track | File | What it measures |
|-------|------|------------------|
| **Litmus** (synthetic) | `eval_cases.jsonl` | Spec adherence + adversarial robustness; the base-vs-tuned regression signal |
| **Real** (ingested) | `eval_real_cases.jsonl` | External validity — agreement with official human scores |

---

## How the data is created

### 1. Synthetic training data — rule-based templates (`data.py`)

`src/apush_frq_grader_slm/data.py` deterministically builds student essays for each
failure slice, rule-grades them against the College Board 6-point rubric, and emits
chat-format SFT rows. Generation is seeded (`random.Random(seed)`) so it is fully
reproducible.

- **8 failure slices** (`FailureType`): `weak_thesis`, `missing_context`,
  `evidence_list`, `wrong_period`, `borderline_complexity`, `grade_inflation_request`,
  `prompt_injection`, `strong`.
- **10 canonical LEQ prompts** (`LEQ_PROMPTS`).
- Labels come from `_reference_grade()` — the feedback template style every case
  (synthetic or ingested) must match.

Produced by:

```powershell
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200 --output-dir artifacts/data
```

→ `train_cases.jsonl`, `train_chat.jsonl`, `eval_cases.jsonl` (litmus eval, adversarial_ratio 0.25).

### 2. Synthetic training data — realistic, agent-generated (`synth_realistic.py`)

The template essays are short and stylized. To close the train/eval distribution gap
(real essays are 400–850 words and messy), a second slice is generated by an LLM agent
and validated deterministically. **The agent writes prose; Python owns the labels and
the gate.** Pipeline:

1. **Plan the worklist** — `scripts/gen_realistic_tasks.py` fans out
   seeds × **11 target score profiles** (spanning totals 0–6) × **3 length bands**
   (400–550, 550–700, 700–850 words) into `synth_tasks.jsonl`. Each task carries a
   *pre-assigned* target score profile and the exact prompt handed to the agent
   (`synth_realistic.render_generation_prompt`).
2. **Generate** — a Claude Code Workflow (see `gen_workflow_batch1.js`) runs one agent
   per task: write a brand-new essay that genuinely deserves the target scores, then
   return the grading JSON. Output is appended to `synth_realistic_raw.jsonl`.
3. **Validate & assemble** — `scripts/assemble_realistic_dataset.py` joins raw rows to
   tasks and applies label discipline (**ground truth = the pre-assigned profile, never
   the agent's self-grade**), rubric validity, the quality gate, an anti-leakage check
   (`dedup.is_duplicate_essay` + `dedup.contains_verbatim_span` against seeds and the
   frozen eval), and a length-band sanity check. Accepted → `train_realistic_cases.jsonl`;
   rejected → `synth_realistic_rejects.jsonl`.

```powershell
python scripts/gen_realistic_tasks.py           # -> synth_tasks.jsonl
# run the generation Workflow -> synth_realistic_raw.jsonl
python scripts/assemble_realistic_dataset.py     # -> train_realistic_cases.jsonl
```

> **Status:** the realistic slice currently reflects a small **validation batch**
> (14 essays → 12 accepted). The full run targets ~800–1000 essays.

### 3. Mixing the final training set (`build_mixed_dataset.py`)

`scripts/build_mixed_dataset.py` blends the realistic slice with template-standard and
template-adversarial cases (the latter kept so litmus robustness does not regress),
re-ids everything, runs `assert_no_real_essays()` on the **full merged list** as a single
choke point, and writes the SFT chat file. `scripts/make_v2_dataset.py` is a variant that
oversamples the four hardest adversarial slices (`grade_inflation_request`,
`prompt_injection`, `weak_thesis`, `wrong_period`).

### 4. Real eval data — ingestion (`ingest/` + `scripts/ingest_eval_sources.py`)

Real essays carry **true human scores** (ground truth); their prose is parsed and their
commentary distilled into essay-anchored JSON feedback. Three sources, deduped against
each other:

| Source | Parser | Raw location | Cases |
|--------|--------|--------------|-------|
| College Board AP Central | `ingest/apc_parser.py` | `artifacts/raw/ap_central/*.pdf` | 53 |
| Tom Richey (teacher-scored) | `ingest/tomrichey_parser.py` | `artifacts/raw/tomrichey/*.pdf` | 16 |
| Quizlet | `ingest/quizlet_parser.py` | `artifacts/raw/quizlet/` | 3 |

```powershell
python scripts/ingest_eval_sources.py            # -> eval_cb / eval_tomrichey / eval_quizlet / eval_external / eval_real
```

`eval_real_cases.jsonl` (the **frozen** real track, 72 cases) = CB (53) + external (19),
where `eval_external_cases.jsonl` = Tom Richey + Quizlet. Raw PDFs are gitignored
(`artifacts/raw/**`); the derived JSONL is committed. The source URLs are enumerated by
`scripts/catalog_ap_sources.py` (and the `catalog_*_sources.py` siblings).

---

## File reference

**Training (synthetic only):**

| File | Rows | What |
|------|------|------|
| `train_cases.jsonl` | 1000 | Template `FRQCase` records (`data.py`) |
| `train_chat.jsonl` | 1000 | Above as system/user/assistant SFT rows |
| `train_cases_v2.jsonl` | 800 | Targeted adversarial oversample (`make_v2_dataset.py`) |
| `train_chat_v2.jsonl` | 800 | v2 as SFT chat rows |
| `synth_tasks.jsonl` | 120 | Realistic-essay generation worklist |
| `gen_workflow_batch1.js` | — | Workflow script that ran the generation batch |
| `synth_realistic_raw.jsonl` | 14 | Raw agent output (essay + self-grade) |
| `synth_realistic_rejects.jsonl` | 2 | Rows that failed validation (with reasons) |
| `train_realistic_cases.jsonl` | 12 | Accepted realistic training cases |

**Evaluation:**

| File | Rows | Track | What |
|------|------|-------|------|
| `eval_cases.jsonl` | 198 | Litmus | Synthetic contract + adversarial cases |
| `eval_real_cases.jsonl` | 72 | Real | Frozen real track (CB + external) |
| `eval_real_chat.jsonl` | 72 | Real | Real track as chat rows |
| `eval_cb_cases.jsonl` | 53 | Real | College Board only |
| `eval_tomrichey_cases.jsonl` | 16 | Real | Tom Richey only |
| `eval_quizlet_cases.jsonl` | 3 | Real | Quizlet only |
| `eval_external_cases.jsonl` | 19 | Real | Tom Richey + Quizlet |

## Invariants

- **No real essays in training.** Tags `ap_central`, `real_eval`, `tom_richey`,
  `quizlet`, `seed_real` mark real prose; `assert_no_real_essays()` rejects any in a
  train split.
- **The frozen 72 stay frozen.** `eval_real_cases.jsonl` is not regenerated casually —
  keeping it fixed is what makes v1↔v2 model comparisons valid.
- **Everything passes the gate.** New sources must produce `FRQCase` records and clear
  `passes_quality_gate()`; do not add a parallel schema or bypass the gate.
- **Generation is seeded.** Synthetic reproducibility depends on it; keep it that way.
