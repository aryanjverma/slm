# V5 Submission Checklist

## Data and training

- [x] Human reviewer accepts/corrects all 60 replacement rows.
- [x] Approval receipt matches the packet SHA-256.
- [x] Finalization rebuilds 540 train, 60 development, and 75 replay rows from
  `validated_candidates_r2.jsonl`.
- [x] Direct-script preflight passes all counts, hashes, combined-corpus binding, and
  zero-golden-leakage checks.
- [x] CPU v5 contract smoke is an automated test.
- [ ] Ten-case GPU smoke passes.
- [ ] Full 60-case base/v4/v5 development comparison passes review.
- [ ] Configuration is frozen before golden evaluation.
- [ ] The 53-case development-informed golden evaluation is run exactly once.

## Release gates

- [ ] QWK ≥ 0.40 and MAE ≤ 1.50.
- [ ] Totals-within-one ≥ 60% and mean-total drift ≤ 0.50.
- [ ] Every criterion exact rate exceeds v4.
- [ ] Structured validity ≥ 98% and grounding ≥ 85%.
- [ ] Failed gates are disclosed as non-production-ready without golden retuning.

## Public deliverables

- [x] Redistribution-safe companion generated locally and clearly distinguished from private v5.
- [ ] Model bundle: `aryanjverma/apush-frq-grader-v5`.
- [ ] Companion dataset: `aryanjverma/apush-leq-grader-public`.
- [ ] Running Space: `aryanjverma/apush-frq-grader-v5-demo`.
- [ ] Aggregate evaluation report and release receipt linked from README.
- [x] Brainlift updated with v5 evidence policy.
- [x] 3–5 minute demo storyboard prepared.
- [ ] Demo video recorded and linked.
- [ ] README contains live model, dataset, Space, report, Brainlift, and video links.

Private essays, review packets, style references, labels, and per-case predictions must not appear
in any public repository, archive, report, demo asset, or video.
