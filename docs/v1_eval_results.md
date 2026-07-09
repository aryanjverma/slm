# v1 Eval Results — 2026-07-09 (Colab, T4)

Base-vs-tuned on both tracks. **Litmus** = 198-case synthetic contract + adversarial
(the regression gate); **Real** = ~72 College Board essays scored against official CB
scores (external validity, eval-only — never trained on).

## Litmus (198-case synthetic contract + adversarial)

| model | n | json | rubric | ground | robust | total |
|-------|---|------|--------|--------|--------|-------|
| inflated_prompted_base | 198 | 1.00 | 0.82 | 0.17 | 0.93 | 0.69 |
| apush_frq_grader_v1 | 198 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |
| apush_grader_reference | 198 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |

## Real (College Board essays, eval-only)

| metric | value |
|--------|-------|
| cases | 72 |
| json_valid | 0.29 |
| row exact | 0.32 |
| row within-1 | 0.67 |
| total exact | 0.15 |
| total within-1 | 0.32 |
| QWK | -0.0864 |

## Interpretation

**Litmus: gate passed — but the win is on synthetic data.** The tuned model matches the
reference target exactly (grounding 0.17→1.00, robustness 0.93→2.00, rubric 0.82→1.00).
Since the litmus set is synthetic and rule-graded by the same logic that produced the
training targets, "tuned == reference" mostly proves the model faithfully learned the
synthetic grader. Combined with the training loss plateau at ~0.012, a perfect litmus
score is expected, not surprising.

**Real: fails external validity.** On genuine CB essays the model collapses:
`json_valid` drops **1.00 → 0.29** (71% of outputs are not valid/in-range grades), and
**QWK is negative (-0.086)** — agreement with official CB totals is no better than chance.
Exact/within-1 agreement is low (total exact 0.15, within-1 0.32). This is the classic
overfitting signature: the model memorized the synthetic distribution and does not
generalize to real, longer, messier essays.

**Per the project thesis, the fix is in the data, not the hyperparameters.** The training
set is 100% synthetic and does not resemble real CB essays (length, prose style, score
distribution), so the model never learned to emit a valid grade for out-of-distribution
inputs. v2 direction: make synthetic essays longer and messier, widen the score
distribution, and validate against this same real track.

**Open confound to rule out first:** the real run used `--max-new-tokens 320`. Some of the
`json_valid` failures may be *truncated* JSON rather than genuinely malformed grades. Check
the `notes`/response tails in `apush_frq_grader_v1_real_results.jsonl` (or re-run the real
track with a higher cap) before attributing the entire drop to generalization.
