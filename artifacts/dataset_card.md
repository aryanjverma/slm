# Dataset Card — APUSH LEQ Grader v2

## Summary

The v2 pipeline creates realistic synthetic APUSH LEQ essays, labels them with two anonymous
graders plus adjudication, and emits immutable manifested SFT artifacts. Official College Board
materials are restricted to a permission-gated, manually verified evaluation workflow and never
enter training.

The checked-in v1 artifacts remain legacy baselines. In particular,
`artifacts/data/eval_cb_cases.jsonl` is **not** the v2 golden set: 28 of its 53 rows contain parser
contamination and all rows still require provenance and human verification.

## v2 artifacts

| File | Purpose |
|---|---|
| `prompt_catalog_v1.jsonl` | 60 original prompt families with deterministic 42/9/9 splits |
| `synth_tasks_train_v2.jsonl` | Balanced 100-task implementation pilot |
| `synth_tasks_dev_v2.jsonl` | Held-out synthetic-development generation tasks |
| `synth_tasks_challenge_v2.jsonl` | Held-out and May-2027-format challenge tasks |
| `train_realistic_v2_unreviewed.jsonl` | Independently labeled candidates before human review |
| `train_realistic_v2_reviewed.jsonl` | Candidates after applying completed review records |
| `v2/train_realistic_v2.jsonl` | Final ordinary/edge training rows |
| `v2/train_adversarial_v2.jsonl` | Final prompt-injection/leniency rows |
| `v2/train_chat_v2.jsonl` | Combined SFT messages |
| `v2/dataset_manifest_v2.json` | Hashes, row counts, settings, distributions, and audit results |
| `eval_cb_golden_v2.jsonl` | Permission-gated and manually verified official evaluation rows |
| `eval_external_v2.jsonl` | External evaluation rows, always reported separately |

Files later in the pipeline do not exist until their gates pass. Builders refuse to overwrite
existing v2 artifacts unless `--force` is supplied deliberately.

## Training data

- Training prose is synthetic only.
- Ordinary rows must use `independent_consensus` or `adjudicated` labels.
- Two readers must agree on all four criteria, or a sufficiently confident third reader decides.
- The accepted score is the reader consensus, never the writer's hidden calibration target.
- Every criterion stores exact essay spans grounding its feedback.
- At least 10% of independently labeled rows must have completed human review.
- Exact/near duplicates, repeated feedback, prompt-family leakage, eval leakage, target leakage,
  and source contamination are rejected.
- The final mix targets 80–85% realistic rows and 15–20% adversarial rows.

## Prompt policy

`src/apush_frq_grader_slm/prompt_catalog.py` contains 60 project-authored prompt families across
APUSH periods 2–9 and causation, comparison, and continuity/change. Whole families are assigned
to train (70%), development (15%), or challenge (15%). Three challenge families model the May
2027 broad-prompt format. Protected holdout prompts are checked for token and topic/date leakage.

## Official evaluation policy

College Board public access is not a model-training or dataset license. The v2 builder requires:

1. a private written-permission record authorizing evaluation use;
2. full `pdf_text` extraction with parser confidence at least 0.90;
3. no commentary/header/footer contamination;
4. year-appropriate rubric version (`2023_leq` or `2024_2026_leq`);
5. a named manual review confirming essay boundaries, provenance, and all row scores.

Commentary-based essay reconstruction has been removed. `scripts/build_golden_v2.py` can create a
blank review template or run an audit without writing a derived golden corpus. The real v2 golden
artifact is emitted only when permission and every review gate pass.

See `docs/college-board-data-source-research.md` and `docs/data_permission_checkpoint.md`.

## Schema

Every `FRQCase` retains the original score/feedback fields plus:

- `provenance`: source type/ID/URL/hash, year/set/question/sample, rubric version, extraction
  method/confidence, prompt family, generator settings, and review status;
- `labeling`: method, reader IDs, agreement/confidence, adjudication status, criterion agreement,
  generation-target distance, human-review state, and grounded feedback spans.

## Build sequence

```powershell
python scripts/build_prompt_catalog.py --protected-prompts artifacts/data/eval_cb_cases.jsonl
python scripts/gen_realistic_tasks.py --limit 100
python scripts/generate_synthetic_candidates.py --limit 100
python scripts/audit_synthetic_candidates.py
python scripts/grade_synthetic_candidates.py --limit 100
python scripts/assemble_realistic_dataset.py
python scripts/review_synthetic_v2.py --create-template
# Complete the review JSONL, then:
python scripts/review_synthetic_v2.py
python scripts/build_v2_artifacts.py --target-count 100
python scripts/run_v2_checkpoints.py
```

Use `scripts/resolve_synthetic_grades.py` instead of the API grader when two reader-output JSONL
files were produced offline.

## Evaluation

Report synthetic litmus, synthetic development/challenge, official golden, and external tracks
separately. Real-eval summaries include JSON validity, row and total exact/within-one agreement,
total MAE, QWK, criterion exact rates, grounding, and separate `2023_leq` versus
`2024_2026_leq` summaries.

## License and permissions

Project-authored code and synthetic data follow the repository's license terms. College Board
content remains College Board property. Do not collect, transform, train on, publish, or
redistribute it without written permission covering that use. This is a conservative workflow
control, not legal advice.
