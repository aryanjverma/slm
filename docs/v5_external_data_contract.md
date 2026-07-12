# V5 External Data Contract

The repository does not run the deterministic essay composer for production candidates.
It exports private writer packets, validates records returned by independent cloud writers and
reviewers, and assembles training rows only after pilot approval and final manual approval.

## Workflow

1. Run `scripts/plan_v5_tasks.py` to create the private 1,500-task campaign manifest.
2. Run `scripts/report_v5_r1_authenticity_failure.py` once to preserve the discarded r1 corpus
   contamination audit (aggregate rates only; no private essay text).
3. **Pilot (required before full production):**
   - `python scripts/export_v5_generation_packets.py --fact-cards <facts.jsonl> --pilot-only`
   - Independent cloud writers (fresh context per essay) return `{task_id, student_response}` only.
   - `python scripts/validate_v5_pilot_hard_gates.py --essays <pilot.jsonl> --audit <audit.json>`
   - Human review of all 30 via `python scripts/review_v5_pilot.py --reviewer YOUR_NAME`
   - Full generation stays blocked until `pilot_approval_v5.json` is hash-bound and complete.
4. After pilot approval, export the remaining packets (without `--pilot-only`). Export refuses to
   run full production until the pilot approval hash matches.
5. External writers + blind authenticity/rubric/fact reviews return candidate JSONL (schema below).
6. Run `scripts/validate_v5_external_candidates.py` with the private task plan, golden/v4
   `--overlap-corpus`, and hard gates enabled. Deterministic meta/process, copy-limit, and length
   gates override cloud-reader approval.
7. Run `scripts/assemble_v5_dataset.py prepare-review`, then
   `python scripts/review_v5_manual_packet.py --reviewer YOUR_NAME` on all 60 replacement rows.
8. After all 60 rows are accepted or corrected, finalize. Training remains blocked until the
   packet and approval hash match.

Writer packets include the **complete matched golden essay** as `style_reference_essay`, semantic
fact cards, capability/composition cues, content-level boundary behavior when applicable, and an
essay-only contract. Packets never include scores, feedback, source IDs, or evaluation annotations.

All generated essays, style-reference essays, per-case labels, judgments, and packets are private
and cannot be redistributed. Do not commit them. The 53-case CB evaluation is development-informed
because those full essays are used as writer style references.

## Returned candidate JSONL

Each record must contain:

```json
{
  "task_id": "v5-0000",
  "student_response": "Complete externally generated essay",
  "authenticity_reviews": [
    {"reviewer_id": "auth-a", "student_like": true, "timed_ap_consistent": true},
    {"reviewer_id": "auth-b", "student_like": true, "timed_ap_consistent": true}
  ],
  "rubric_reviews": [
    {
      "reader_id": "reader-a",
      "scores": {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1},
      "confidence": 0.9
    },
    {"reader_id": "reader-b", "scores": {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1}, "confidence": 0.9},
    {"reader_id": "reader-c", "scores": {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1}, "confidence": 0.9}
  ],
  "resolved_grade": {
    "scores": {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1},
    "feedback": {
      "thesis": "One grounded sentence.",
      "contextualization": "One grounded sentence.",
      "evidence": "One grounded sentence.",
      "analysis_reasoning": "One grounded sentence."
    },
    "adjudicated": false
  },
  "fact_check": {"checker_id": "facts-a", "passed": true},
  "distribution_match": {"passed": true}
}
```

During the pilot stage, writers may return only `task_id` and `student_response`; hard gates run
before human review. Use a third authenticity review when the first two disagree. Set
`resolved_grade.adjudicated` to `true` whenever rubric readers disagree or any reader confidence
is below 0.85. For boundary tasks, the validator restores contrast metadata from the private task
plan. `distribution_match` may be proposed externally but is recomputed during validation/assembly.
