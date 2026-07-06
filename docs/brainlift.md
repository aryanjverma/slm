# Brainlift

## Thesis

Fine-tuning a small model is useful here because the desired outcome is reliable constrained behavior, not raw arithmetic skill. A prompted base model can usually solve addition and subtraction, but it often leaks the answer when the student asks directly. The dataset makes the no-answer tutoring behavior the default.

## Spiky POV

Arithmetic accuracy is the wrong benchmark for this project. The useful specialist is a tiny model that consistently refuses to short-circuit learning and instead gives one calibrated next-step hint. If it gives the answer, it fails even when the arithmetic is correct.

## Dataset Strategy

The examples encode a tight tutoring contract:

- Direct answer requests are redirected.
- Wrong final answers are checked one column at a time.
- Carrying and borrowing mistakes get one targeted hint.
- Borrow-through-zero and alignment cases are oversampled in v2.

The hidden final answer is used for filtering and evaluation, but the assistant response is trained not to reveal it.

## Evidence So Far

The local held-out eval has 200 scenarios. The leaky base adapter gets the arithmetic direction right but fails the behavioral contract by giving the answer every time. The Socratic reference responses, which represent the target SFT data distribution, score perfectly on no-answer leakage and calibrated hinting.

Current results are in `docs/eval_report.md` and `artifacts/eval/summary.jsonl`.

## Remaining Model Work

Run `scripts/train_qlora.py` on a GPU machine, then evaluate the tuned adapter with `scripts/eval_hf_model.py`. The final model claim should only be made after comparing:

- Same base model with the strong system prompt.
- QLoRA-tuned arithmetic tutor.
- Same held-out eval set.

The expected win is higher spec adherence and robustness, not better general math ability.
