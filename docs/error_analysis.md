# Error Analysis And V2 Iteration

## Primary Failure Mode

The behavior this project attacks is answer leakage. A helpful base model can solve arithmetic, but when asked directly it tends to reveal the final answer. That fails the learning objective.

## Secondary Failure Modes

- Generic hints that do not mention the student's current column.
- Subtraction with zeros where borrowing has to move left.
- Misaligned columns in multi-digit arithmetic.
- Student gives a wrong final answer and asks for confirmation.

## V2 Data Fix

`scripts/make_v2_dataset.py` oversamples the failure modes above and writes:

- `artifacts/data/train_cases_v2.jsonl`
- `artifacts/data/train_chat_v2.jsonl`

Use v2 by either training a second adapter directly on `train_chat_v2.jsonl` or concatenating v1 and v2. The second option is usually better because it preserves broad coverage while increasing pressure-test density.

## Recommended V2 Training Command

```powershell
$env:PYTHONPATH='src'
python scripts/train_qlora.py --data artifacts/data/train_chat_v2.jsonl --output artifacts/models/arithmetic-tutor-v2
python scripts/eval_hf_model.py --model artifacts/models/arithmetic-tutor-v2 --model-name arithmetic_tutor_v2
```

The v2 run should improve robustness on direct answer requests, wrong-answer checks, and borrow-through-zero cases.
