# Model Card — APUSH FRQ Grader V5

## Status

No final v5 model has been trained or published. The private dataset review and hash preflight now
pass; GPU/Colab training is the next stage. Intended repository: `aryanjverma/apush-frq-grader-v5`.

## Architecture and output

V5 merges the completed v4 adapter into `Qwen/Qwen2.5-0.5B-Instruct`, then trains separate QLoRA
scorer and score-conditioned feedback adapters. Runtime returns exactly one JSON object with four
criterion `scores`, a deterministic `total`, and four essay-grounded `feedback` strings.

Frozen settings: scorer 4 epochs at 1e-4 with score-token weight 4.0; feedback 2 epochs at 5e-5;
LoRA rank 16; batch 1; accumulation 4; warmup 0.03; max sequence length 4096; seed 13.

## Release criteria

QWK ≥ 0.40, MAE ≤ 1.50, totals-within-one ≥ 60%, every criterion exact rate above v4,
mean-total drift ≤ 0.50, structured validity ≥ 98%, and grounding ≥ 85%. Metrics remain unreported
until the one-time, 53-case development-informed evaluation produces an aggregate receipt.

## Intended use and limitations

Use for formative APUSH LEQ feedback with human oversight. It is not appropriate as the sole
high-stakes scorer, and it does not grade DBQs or SAQs. Its training essays are synthetic, its
evaluation is development-informed, and a 0.5B model can still be miscalibrated or produce brittle
feedback on unfamiliar writing.
