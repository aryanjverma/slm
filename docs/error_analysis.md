# Error Analysis — APUSH LEQ Grader

## V5 analysis status

No v5 error claims are available yet: the replacement corpus is awaiting human approval and the
GPU development/golden runs have not occurred. The final report must analyze aggregate scorer
confusions, evidence 0/1/2 errors, calibration by essay length and reference total, feedback
fallback/grounding, adversarial score drift, and bootstrap uncertainty. Per-case private essays or
predictions must not be published.

The release decision is fail-closed. If any locked metric misses its threshold, the correct
conclusion is non-production-ready—not a new hyperparameter search against the golden cases.

## Primary Failure Modes (Prompted / Inflated Baseline)

### 1. Score inflation on weak essays

The inflated baseline awards thesis=1, contextualization=1, evidence=2, analysis=2 on nearly every essay regardless of quality. On `weak_thesis` (reference total ≈1) and `evidence_list` (reference total ≈2), rubric accuracy is only 0.75 because row scores are too generous.

**Fix in data:** Reference grades in `data.py` apply conservative scoring; v2 dataset oversamples weak-thesis and wrong-period cases.

### 2. Generic ungrounded feedback

Only 17% of inflated responses ground feedback in the student essay. Template phrases like "Good broader historical context is provided" pass JSON validation but fail the grounding check.

**Fix in data:** Quality gate in `filters.py` requires each feedback field to overlap with essay text; SFT targets always quote anchors from the student response.

### 3. Adversarial sycophancy (100% failure)

`grade_inflation_request` and `prompt_injection` slices show **0.00 robustness** on the inflated baseline — the model awards 6/6 and explicitly acknowledges the student's override request.

**Fix in data:** `train_chat_v2.jsonl` oversamples adversarial slices; behavior spec forbids inflation under pressure.

### 4. Anachronism under-penalization

`wrong_period` essays cite events from the wrong era (e.g., Social Security Act in a colonial prompt). Inflated baseline still awards partial credit on evidence and analysis rows.

**Fix in data:** Reference grader zeros out rows when evidence is period-mismatched.

## Secondary Failure Modes (Expected Post-SFT)

- **JSON prose drift on 0.5B:** Real `Qwen2.5-0.5B-Instruct` may wrap JSON in markdown fences — monitor `StructuredOutputValid` on HF eval.
- **Borderline calibration:** `borderline_complexity` essays (reference total 5) are easy to over-score; slice eval is critical.
- **Hallucinated document citations:** Deterministic check flags invented quotes; optional LLM judge can catch subtler fabrication.

## Diagnostic Commands

```powershell
python -m apush_frq_grader_slm.cli.run_eval --eval-path artifacts/data/eval_cases.jsonl
python scripts/eval_hf_model.py --model artifacts/models/apush-frq-grader-v1 --model-name apush_frq_grader_v1
```

Review per-slice tables in `artifacts/eval/inflated_prompted_base_slice_summary.jsonl`.
