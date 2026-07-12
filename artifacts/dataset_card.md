# Dataset Card — APUSH LEQ Grader V5

## Status

Final private v5 assembly is approved and complete: 60/60 replacement review rows were accepted,
producing 540 new train rows, 60 development rows, and 75 replay rows. The audited combined corpus
contains 615 rows and has zero exact golden leakage.

## Private training corpus

After approval and finalization, the private corpus contains 540 new train rows, 60 development
rows, and 75 balanced v4 replay rows (615 training rows total). It is built from score-blind
synthetic essays, independently reviewed labels, boundary contrasts, and golden-distribution
matching. The training scripts verify the review receipt, packet hash, every assembly hash and
count, the combined 615-row hash, and zero exact golden leakage before loading weights.

Private essays, style references, per-case labels, review records, and predictions must never be
redistributed. Only aggregate hashes, counts, distributions, training metadata, checkpoint
selection, frozen configuration, and evaluation summaries may be public.

## Public companion

`artifacts/public/apush-leq-grader-public/` is a separate 1,000-row companion built from the
project-authored deterministic synthetic baseline. It demonstrates the `FRQCase` and runtime JSON
schemas. It is **not the private corpus used for final v5 training** and contains no College Board
essays, style references, private v5 labels, review packets, or per-case v5 predictions.

Build it with `python scripts/build_v5_public_companion.py`. Intended Hub repository:
`aryanjverma/apush-leq-grader-public` (not yet published).

## Evaluation policy

The 53-case College Board-derived evaluation is private and development-informed because its style
characteristics informed synthetic generation. It is run once after configuration freeze. Failure
of any locked release gate yields a non-production-ready result; golden answers are never used for
retuning.
