# V5 External Data Contract

The repository does not generate essays or cloud judgments. It creates private work packets,
validates records returned by a separate tool, and assembles training rows only after manual
approval.

## Workflow

1. Run `scripts/plan_v5_tasks.py` to create the private 1,500-task campaign manifest.
2. Run `scripts/export_v5_generation_packets.py --fact-cards <private-facts.jsonl>` to create
   30 score-blind writer shards of 50 packets. Send only these packet files to essay writers.
3. Have the external system add independent reviews to every returned essay and emit one JSONL
   record per `task_id` using the schema below.
4. Run `scripts/validate_v5_external_candidates.py` with the private task plan and repeat
   `--overlap-corpus` for the golden and v4 essay files. The validator restores trusted planning
   metadata, rejects copied or incomplete records, and writes an audit.
5. Run `scripts/assemble_v5_dataset.py prepare-review` on the validated candidates. Review all
   60 packet rows and record `manual_review.decision` as `accept` or `corrected`. Corrections may
   contain only `scores` and/or `feedback`.
6. Create `manual_review_approval_v5.json` with `approved: true`, a reviewer, timestamp, and the
   SHA-256 of the completed packet. Run the assembler's `finalize` stage. Training remains blocked
   until the packet and approval hash match.

All generated essays, excerpts, per-case labels, judgments, and packets are private and cannot be
redistributed. Do not commit them.

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

Use a third authenticity review when the first two disagree. Set `resolved_grade.adjudicated` to
`true` whenever rubric readers disagree or any reader confidence is below 0.85. For boundary tasks,
the validator restores the contrast metadata from the private task plan; the external tool must not
invent or modify it. `distribution_match` may be proposed by the external tool, but assembly should
recompute it with `compute_distribution_match` / `annotate_distribution_match` from golden score-vector
membership and style tolerances rather than trusting the proposal blindly. It is required only for
distribution-matched tasks.
