# V5 Final Training Plan

## Summary

Create branch `v5` from the current clean `main`. Build a highly filtered 600-case dataset from approximately 1,500 cloud-generated candidates, prioritizing authentic CB-style writing and independently observed scores rather than score-forced essays.

V5 will inherit v4 by merging its adapter into Qwen2.5-0.5B, then training separate scoring and feedback LoRA adapters. The public output remains one JSON object, assembled through two internal passes.

## Data Production and Quality

- Generate 1,500 candidates in 30 independent 50-case shards.
- Give generators:
  - Short CB style excerpts, capped at the existing 400-character limit.
  - Adapted prompt-family questions rather than copies of official questions.
  - Semantic AMSCO fact cards rewritten as concepts, not source sentences.
  - Timed-student profiles controlling knowledge, organization, error density, and confidence.
  - No score targets or rubric-point instructions.
- Generate spelling, grammar, fragments, run-ons, repetition, uncertain phrasing, and uneven paragraphing naturally during composition. Do not mechanically corrupt otherwise polished essays.
- Require paraphrased historical evidence. Reject copied AMSCO or golden-set phrases using normalized eight-word overlap checks, with exemptions for names, dates, and unavoidable historical terms.
- Reject exact and near duplicates against v4, other v5 candidates, and the 53 golden essays.
- Use two blind authenticity judges. Require both to rate the essay student-like and consistent with timed AP writing; send disagreements to a third judge.
- Use three rubric readers who see only the prompt, essay, and corrected official rubric. They must not see personas, seed identities, style excerpts, or intended coverage categories.
- Adjudicate every criterion disagreement and every label below 0.85 confidence. Run a separate historical-fact checker before acceptance.
- Retain exactly 600 cases:
  - 420 matching the golden set's joint score, length, paragraph, reasoning-skill, and writing-error distributions.
  - 180 boundary cases: fifteen contrast pairs for each of thesis 0/1, context 0/1, evidence 0/1, evidence 1/2, analysis 0/1, and analysis 1/2.
- Split 540 training and 60 development cases by prompt family and style seed so related variants cannot cross splits.
- Add 75 high-agreement v4 replay cases to training only. Balance replay across totals and criteria rather than taking mostly high-scoring cases.
- Produce a 60-case stratified manual-review packet for the user. Final training remains blocked until corrections are applied and the packet is approved.
- Store private rows under the existing restricted-use policy; commit only permitted artifacts and aggregate audits.

## Model and Training

- Correct and shorten the v5 rubric, including removing the false "3-to-4 sentence" contextualization requirement.
- Merge `apush-frq-grader-v4-assistant-only-r1` into Qwen2.5-0.5B in FP16, save a hashed v5 base artifact, and train two fresh rank-16 QLoRA adapters:
  - **Scorer:** input is rubric, prompt, and essay; output contains only the four criterion scores.
  - **Feedback:** input additionally contains validated scores; output contains only four grounded feedback fields.
- Compute `total` deterministically from criterion scores. Clamp and validate all criterion ranges before the feedback pass.
- Weight numerical score tokens four times more than structural JSON tokens.
- Use a 4,096-token context window and reject any training row that loses the essay introduction, conclusion, or target.
- Scorer defaults: batch 1, gradient accumulation 4, learning rate `1e-4`, 3% warmup, up to four epochs, evaluation every 100 optimizer steps, and early stopping after two non-improving evaluations.
- Select the scorer checkpoint by development QWK, breaking ties with lower MAE, higher evidence exact match, then higher macro criterion accuracy.
- Feedback defaults: batch 1, gradient accumulation 4, learning rate `5e-5`, two epochs, selected by grounding rate and schema validity.
- Save resumable checkpoints, logs, dataset hashes, merged-base hash, adapter hashes, and complete training metadata.
- Add a Colab v5 notebook that performs private-data preflight, v4 merge, both training passes, smoke tests, evaluation, resume, and aggregate-only report export.

## Runtime Interface and Evaluation

- Package a v5 bundle containing `scorer/`, `feedback/`, tokenizer files, corrected prompts, and a manifest linking both adapters to the merged v4 base.
- Add a two-pass grading entry point that still returns:
  - `scores`
  - deterministic `total`
  - grounded `feedback`
- If feedback generation fails, retain the scores and return criterion-specific fallback feedback; never discard a valid score result.
- Before freezing configuration, compare the prompted base, v4, and v5 checkpoints on the private 60-case development set.
- Run the 53-case golden evaluation only after configuration is frozen. Report that it is development-informed because style excerpts were used during data creation.
- Produce criterion confusion matrices, predicted-versus-reference total distributions, calibration by essay length and reference total, evidence 0/1/2 errors, and bootstrap confidence intervals.
- Release gates:
  - QWK >= 0.40.
  - Total MAE <= 1.50.
  - At least 60% of totals within one point.
  - Every criterion exact-match rate exceeds v4.
  - Mean predicted total within 0.50 of the golden mean.
  - Structured validity >= 98%.
  - Feedback grounding >= 85%.
- If any gate fails, report v5 as non-production-ready and do not retune against the 53 golden answers.

## Tests and Acceptance

- Test score-only and score-conditioned message construction, weighted loss masking, deterministic totals, and two-adapter loading.
- Test candidate quotas, blind-label metadata removal, family-grouped splitting, overlap rejection, duplicate detection, label confidence, and manual-review approval.
- Test long and short essays, malformed generation, invalid score ranges, feedback failure, checkpoint resume, and a fresh Colab runtime.
- Run the complete existing suite plus v5 tests and a ten-case end-to-end smoke evaluation before the final training run.

## Locked Assumptions

- Branch name: `v5`.
- Accepted new cases: 600 from approximately 1,500 candidates.
- Distribution: 70% golden-matched and 30% boundary-focused.
- Review: triple-blind rubric grading, independent authenticity auditing, adjudication, and user review of 10%.
- Golden style excerpts are allowed, with strict copying controls and disclosure of evaluation contamination risk.
- V4 is inherited through merge-then-fresh adapters.
- A small v4 replay set is included.
- Runtime uses two internal passes while preserving the existing external JSON contract.
