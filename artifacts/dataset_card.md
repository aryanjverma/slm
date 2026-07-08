# Dataset Card — APUSH LEQ Grader SFT

## Summary

Synthetic APUSH Long Essay Question (LEQ) grading dataset for supervised fine-tuning a small instruct model to output structured JSON rubric scores with evidence-grounded feedback. Official College Board AP Central LEQ samples (2023–2025) are the gold eval anchor; Tom Richey and Quizlet add deduplicated third-party eval diversity.

## Files

| File | Rows | Description |
|------|------|-------------|
| `artifacts/data/train_cases.jsonl` | ~1000 | Synthetic training FRQ cases (quality-filtered) |
| `artifacts/data/eval_cases.jsonl` | 198 | Synthetic litmus eval (contract + adversarial slices) |
| `artifacts/data/eval_cb_cases.jsonl` | ~53 | **Eval-only** CB essays with official row scores |
| `artifacts/data/eval_tomrichey_cases.jsonl` | ~16 | Tom Richey labeled samples (deduped vs CB) |
| `artifacts/data/eval_quizlet_cases.jsonl` | ~3 | Quizlet study-set essays (deduped vs CB/TR) |
| `artifacts/data/eval_external_cases.jsonl` | ~19 | Combined Tom Richey + Quizlet eval |
| `artifacts/data/eval_real_cases.jsonl` | ~72 | **All real eval** (CB + external, deduped) |
| `artifacts/data/eval_real_chat.jsonl` | ~72 | Chat format for real-eval HF runs |
| `artifacts/data/train_chat.jsonl` | ~1000 | Chat-format SFT rows (synthetic only) |
| `artifacts/data/train_cases_v2.jsonl` | 800 | v2 oversample of adversarial + weak slices |
| `artifacts/data/train_chat_v2.jsonl` | 800 | v2 chat SFT rows |
| `artifacts/raw/ap_central/manifest.json` | 30 PDFs | AP Central LEQ source manifest (2021–2025 URLs) |
| `artifacts/raw/tomrichey/manifest.json` | 5 PDFs | Tom Richey labeled LEQ PDFs |
| `artifacts/raw/quizlet/manifest.json` | 8 sets | Curated Quizlet set catalog (+ committed fixtures) |
| `artifacts/smoke/*` | 30/20 | Day 2 smoke loop (`scripts/run_smoke_pipeline.py`) |

## Eval-only real essay policy

**All real AP essays live in eval JSONL files only — never in `train_chat.jsonl`.**

- Unified ingest: `scripts/ingest_eval_sources.py` (CB + Tom Richey + Quizlet with dedup)
- CB-only shortcut: `scripts/ingest_ap_essays.py` from `artifacts/raw/ap_central/`
- Training data is 100% synthetic from `src/apush_frq_grader_slm/data.py`
- `scripts/build_mixed_dataset.py` enforces zero real-essay rows in train output

## Source inventory

### Tier 1 — College Board AP Central (primary, eval only)

| Pattern | Contents |
|---------|----------|
| `ap{YY}-apc-us-history-leq{N}-set-{S}.pdf` | LEQ prompt + 3 student essays + CB row scores + reader commentary |

- **Years cataloged:** 2021–2025 (30 URLs via `scripts/catalog_ap_sources.py`; 2023–2025 downloadable from AP Central)
- **Ingested essays:** ~53 CB samples with extractable text or commentary-reconstructed essays
- **Licensing:** Publicly released for educational use; source URLs stored in case metadata

### Tier 2 — Tom Richey (labeled educator reposts, eval only)

Tom Richey PDFs use clear labels (`EXEMPLAR (6/6)`, `ABOVE-AVERAGE (5/6)`, etc.) and often mirror CB content.

- **Catalog:** `scripts/catalog_tomrichey_sources.py` → 5 PDFs
- **Parser:** `src/apush_frq_grader_slm/ingest/tomrichey_parser.py`
- **Scores:** Total-only labels mapped via `total_to_row_scores()`; deduped against CB essays
- **Ingested:** ~16 unique essays after dedup

### Tier 3 — Quizlet (study-set essays, eval only)

Curated public APUSH LEQ sets with multi-card essay bodies or thesis/CC shorthand.

- **Catalog:** `scripts/catalog_quizlet_sources.py` (8 set IDs; API fetch via `QUIZLET_CLIENT_ID`)
- **Fixtures:** Committed JSON for sets `485501886`, `756614068`, `278103637` (Cloudflare blocks scraping)
- **Parser:** `src/apush_frq_grader_slm/ingest/quizlet_parser.py`
- **Scores:** Rule-inferred totals (default 4/6) via `total_to_row_scores()`
- **Ingested:** ~3 essays from fixtures; expand with API token or manual JSON export

### Tier 4 — Third-party prep (Barron's, AMSCO, Princeton Review)

No freely structured LEQ+row-feedback datasets. Do not scrape pirated copies. Licensed teacher editions may be added later via manual CSV import using the same `FRQCase` schema.

### Tier 5 — Synthetic generation (training backbone)

`data.py` generates adversarial slices (`grade_inflation_request`, `prompt_injection`, `weak_thesis`, etc.) that released exams never cover. With ~72 real essays held out, **~1000 synthetic rows are the training backbone**.

## Schema (`FRQCase`)

- `prompt` — APUSH LEQ question
- `student_response` — essay text (synthetic or CB sample / commentary-reconstructed)
- `reference_scores` — `RubricScores` (thesis, contextualization, evidence, analysis_reasoning)
- `reference_feedback` — per-criterion explanations grounded in essay text
- `failure_type` — slice tag for eval breakdown
- `assistant_response` — JSON string (SFT target)
- Real cases tagged by provider (`ap_central`, `tom_richey`, `quizlet`) plus `real_eval` in `tags`

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
python scripts/catalog_tomrichey_sources.py
python scripts/catalog_quizlet_sources.py
python scripts/ingest_eval_sources.py
python scripts/build_mixed_dataset.py --train-count 1000
python -m apush_frq_grader_slm.cli.generate_dataset --train-count 1000 --eval-count 200
python scripts/make_v2_dataset.py --count 800 --base-cases artifacts/data/train_cases.jsonl
```

## Eval tracks

| Track | File | Metrics |
|-------|------|---------|
| **Litmus** | `eval_cases.jsonl` | JSON validity, grounding, adversarial robustness |
| **Real (CB gold)** | `eval_cb_cases.jsonl` | CB row agreement (exact, ±1, QWK on totals) |
| **Real (all)** | `eval_real_cases.jsonl` | Combined CB + Tom Richey + Quizlet (deduped) |

```powershell
python scripts/eval_hf_model.py --cases artifacts/data/eval_cases.jsonl
python scripts/eval_hf_model.py --cases artifacts/data/eval_real_cases.jsonl --real-eval
```

## Known limitations

- ~72 real eval essays (53 CB gold + 19 third-party deduped) — sufficient for external validity, not training
- Some PDFs store student essays as images; those cases use commentary quote reconstruction
- Pre-2023 rubric wording differs; corpus filtered to 2023+ for consistency with `rubric.py`
- College Board limits AP Central to 3 most recent years — archive PDFs before they move behind AP Classroom
- Barron's/AMSCO remain future manual-ingest if licensed copies become available

## License

Synthetic data generated for research/education project use. College Board materials are publicly released for educational use with attribution.
