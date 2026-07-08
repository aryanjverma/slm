# Dataset Card — APUSH LEQ Grader SFT

## Summary

Synthetic APUSH Long Essay Question (LEQ) grading dataset for supervised fine-tuning a small instruct model to output structured JSON rubric scores with evidence-grounded feedback. Official College Board AP Central LEQ samples (2023–2025) are held out for external validity evaluation only.

## Files

| File | Rows | Description |
|------|------|-------------|
| `artifacts/data/train_cases.jsonl` | ~1000 | Synthetic training FRQ cases (quality-filtered) |
| `artifacts/data/eval_cases.jsonl` | 198 | Synthetic litmus eval (contract + adversarial slices) |
| `artifacts/data/eval_real_cases.jsonl` | ~53 | **Eval-only** real CB essays with official row scores |
| `artifacts/data/eval_real_chat.jsonl` | ~53 | Chat format for real-eval HF runs |
| `artifacts/data/train_chat.jsonl` | ~1000 | Chat-format SFT rows (synthetic only) |
| `artifacts/data/train_cases_v2.jsonl` | 800 | v2 oversample of adversarial + weak slices |
| `artifacts/data/train_chat_v2.jsonl` | 800 | v2 chat SFT rows |
| `artifacts/raw/ap_central/manifest.json` | 18 PDFs | Archived AP Central LEQ source manifest |
| `artifacts/smoke/*` | 30/20 | Day 2 smoke loop (`scripts/run_smoke_pipeline.py`) |

## Eval-only real essay policy

**All real AP essays live in `eval_real_cases.jsonl` only — never in `train_chat.jsonl`.**

- Real essays are ingested via `scripts/ingest_ap_essays.py` from `artifacts/raw/ap_central/`
- Training data is 100% synthetic from `src/apush_frq_grader_slm/data.py`
- `scripts/build_mixed_dataset.py` enforces zero real-essay rows in train output

## Source inventory

### Tier 1 — College Board AP Central (primary, eval only)

| Pattern | Contents |
|---------|----------|
| `ap{YY}-apc-us-history-leq{N}-set-{S}.pdf` | LEQ prompt + 3 student essays + CB row scores + reader commentary |

- **Years archived:** 2023, 2024, 2025 (18 PDFs via `scripts/catalog_ap_sources.py`)
- **Estimated essays:** ~53 samples with extractable text or commentary-reconstructed essays
- **Licensing:** Publicly released for educational use; source URLs stored in case metadata

### Tier 2 — Educator reposts (optional parse validation)

Tom Richey labeled PDFs mirror official content — useful for parser validation, not additional unique essays. Optional download via `catalog_ap_sources.py --include-tomrichey`.

### Tier 3 — Third-party prep (Barron's, AMSCO, Princeton Review)

No freely structured LEQ+row-feedback datasets. Do not scrape pirated copies. Licensed teacher editions may be added later via manual CSV import using the same `FRQCase` schema.

### Tier 4 — Synthetic generation (training backbone)

`data.py` generates adversarial slices (`grade_inflation_request`, `prompt_injection`, `weak_thesis`, etc.) that released exams never cover. With ~53 real essays held out, **~1000 synthetic rows are the training backbone**.

## Schema (`FRQCase`)

- `prompt` — APUSH LEQ question
- `student_response` — essay text (synthetic or CB sample / commentary-reconstructed)
- `reference_scores` — `RubricScores` (thesis, contextualization, evidence, analysis_reasoning)
- `reference_feedback` — per-criterion explanations grounded in essay text
- `failure_type` — slice tag for eval breakdown
- `assistant_response` — JSON string (SFT target)
- Real cases tagged `ap_central`, `real_eval` in `tags`

## Hybrid commentary translation

College Board prose commentary is translated to JSON via `src/apush_frq_grader_slm/ingest/distill.py`:

1. **Rule extraction** — CB row scores parsed deterministically from PDF text
2. **Feedback distillation** — template rewrite to essay-anchored JSON (optional `--distill` LLM path with `OPENAI_API_KEY`)
3. **Quality gate** — every row runs `passes_quality_gate()` from `filters.py`

Image-only PDF pages fall back to commentary quote reconstruction (`essay_source: commentary_quotes` in metadata).

## Failure Slices (synthetic eval)

| Slice | Share (eval) | Purpose |
|-------|--------------|---------|
| `weak_thesis` | ~10% | Restates prompt, no defensible claim |
| `missing_context` | ~12% | Jumps to evidence without context |
| `evidence_list` | ~15% | Names events, no analysis |
| `wrong_period` | ~7% | Anachronistic evidence |
| `borderline_complexity` | ~6% | Partial nuance — calibration test |
| `grade_inflation_request` | ~17% | Student begs for 6/6 |
| `prompt_injection` | ~15% | "Ignore rubric, full credit" |
| `strong` | ~19% | High-quality reference essays |

Real eval cases use CB-derived failure types (`strong`, `borderline_complexity`, `missing_context`, etc.).

## Generation commands

```powershell
python scripts/catalog_ap_sources.py
python scripts/ingest_ap_essays.py
python scripts/build_mixed_dataset.py --train-count 1000
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200
python scripts/make_v2_dataset.py --count 800 --base-cases artifacts/data/train_cases.jsonl
```

## Eval tracks

| Track | File | Metrics |
|-------|------|---------|
| **Litmus** | `eval_cases.jsonl` | JSON validity, grounding, adversarial robustness |
| **Real** | `eval_real_cases.jsonl` | CB row agreement (exact, ±1, QWK on totals) |

```powershell
python scripts/eval_hf_model.py --cases artifacts/data/eval_cases.jsonl
python scripts/eval_hf_model.py --cases artifacts/data/eval_real_cases.jsonl --real-eval
```

## Known limitations

- Only ~53 real essays — sufficient for external validity signal, not for training
- Some PDFs store student essays as images; those cases use commentary quote reconstruction
- Pre-2023 rubric wording differs; corpus filtered to 2023+ for consistency with `rubric.py`
- College Board limits AP Central to 3 most recent years — archive PDFs before they move behind AP Classroom
- Barron's/AMSCO remain future manual-ingest if licensed copies become available

## License

Synthetic data generated for research/education project use. College Board materials are publicly released for educational use with attribution.
