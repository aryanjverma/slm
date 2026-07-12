# V5 Authentic Essay Regeneration Plan

## Summary

Discard the current active corpus and regenerate all 1,500 essays with independent cloud writer
agents. Each writer receives the complete golden essay directly matched to its prompt seed, but no
score, feedback, identity, or annotations.

The current corpus is unusable: 100% of candidates and selected training rows contain
writing-process, memory, notes, or physical-test artifacts. Preserve this failure in Git history and
an aggregate report, then replace the active v5 artifacts.

## Generation Changes

- Remove the deterministic composer from the production workflow; it must not generate replacement
  essays.
- Export private writer packets containing:
  - The adapted LEQ prompt.
  - The complete directly matched golden essay as a style reference.
  - Semantic fact cards.
  - Student capability and content-level boundary instructions.
  - An explicit essay-only contract.
- Require fresh cloud-agent context for every essay.
- Writers must never mention memory, notes, planning, drafts, outlines, checklists, margins,
  rewriting, pens, pencils, erasers, seating, clocks, bluebooks, test conditions, missing knowledge,
  or how the essay is being written.
- Weak knowledge must appear through omission, vagueness, plausible factual mistakes, or
  underdeveloped arguments—not admissions such as “I cannot recall.”
- Allow one short copied span from the style essay:
  - Maximum contiguous overlap: 20 normalized words.
  - No more than one candidate sentence containing an eight-word source overlap.
  - Reject larger or repeated copying.
- Keep scores, feedback, rubric decisions, source IDs, and evaluation annotations hidden from
  writers.

## Pilot and Full Production

- First generate a 30-essay pilot:
  - Two lower/upper pairs for each of the six rubric boundaries: 24 essays.
  - Six distribution-matched essays spanning periods and quality levels.
- Run deterministic hard gates, then have the user review all 30.
- Regenerate individual failures until all 30 are accepted. Full production remains blocked until a
  hash-bound pilot approval exists.
- Generate the remaining 1,470 essays only after pilot approval.
- Hard-reject candidates containing meta/process language, instruction leakage, excessive copying,
  or invalid length before spending judging capacity.
- Tighten realism matching:
  - Candidate length follows its matched golden essay within approximately ±20%, with reasonable
    handling for extreme short and long samples.
  - Aggregate mean remains within 10% of golden.
  - Median and quartiles remain within 15%.
  - Sentence variation, paragraphing, spelling-error density, and punctuation distribution match the
    golden set more closely than the failed permissive audit.

## Independent Judging and Assembly

- Authenticity readers see only the prompt and candidate essay, never the style reference or
  generation profile.
- Use two independent cloud authenticity readers and a third on disagreement.
- Use three independent rubric readers plus adjudication for disagreement or confidence below 0.85.
- Run a separate historical-fact checker.
- Deterministic hard gates override cloud-reader approval.
- Select 420 golden-distribution-matched and 180 scoring-boundary cases, then create the grouped
  540/60 train/dev split and add 75 v4 replay cases.
- Reset the current manual packet and approval. The user reviews all 60 replacement rows with the
  terminal reviewer.
- Any rejected manual-review row must be replaced. Training remains blocked until all 60 are
  personally accepted or corrected and hashes are regenerated.

## Interfaces and Reporting

- Keep the external writer result format as `task_id` plus `student_response`.
- Add full-reference fields only to private writer packets; never include them in final training rows.
- Add pilot approval and generation-attempt metadata so regeneration is resumable and auditable.
- Produce an aggregate `v5_r1_authenticity_failure` report documenting:
  - The 100% contamination rate.
  - Artifact categories and rates.
  - Why the previous authenticity judge failed.
  - No private essay text.
- Mark the final 53-case evaluation as contaminated/development-informed because all 53 full essays
  were used as writer references. Do not claim it is an independent generalization test.

## Tests and Acceptance

- Regression-test every prohibited artifact category from the first eight rejections.
- Verify writer packets include the full matched essay but exclude scores, feedback, and source
  annotations.
- Verify pilot approval blocks full generation until all 30 rows pass.
- Verify copying limits, independent-reader requirements, score quotas, grouped splits, manual
  approval, and artifact hashes.
- Require:
  - Zero meta/process violations among accepted candidates.
  - All 30 pilot essays manually accepted.
  - All 60 final review rows manually accepted or corrected.
  - 600 finalized new cases and 75 replay cases.
  - Complete test-suite success before training.

## Scope Constraint

This document is a plan only. Creating it does not authorize generation, regeneration, judging,
assembly, training, evaluation, or modification of private dataset artifacts.
