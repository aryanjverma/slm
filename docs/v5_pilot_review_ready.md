# V5 Pilot Review — READY FOR YOU

The regeneration infrastructure is in place, the r1 corpus is marked discarded, and
**all 30 pilot essays have passed deterministic hard gates**.

## Review now

```powershell
python scripts/review_v5_pilot.py --reviewer YOUR_NAME
```

Essays: `artifacts/data/v5/private/pilot_essays_v5.jsonl`  
Hard-gate audit: `artifacts/data/v5/private/pilot_hard_gate_audit_v5.json`  
(ready_for_human_review: true)

Accept or reject each essay. Rejected rows must be regenerated before
`pilot_approval_v5.json` can be written. Full 1,470 generation stays blocked until
that hash-bound approval exists.

## What changed

- Composer retired from production (`generate_v5_essays.py` exits unless
  `--allow-legacy-composer`).
- Writer packets include the **full matched golden essay**, fact cards, capability /
  boundary cues, and an essay-only contract (no scores/feedback/source IDs).
- Hard gates reject meta/process language, over-copying, and length mismatches.
- Aggregate `v5_r1_authenticity_failure` report documents 100% r1 contamination.
- Previous manual-review approval is voided pending replacement rows.
