"""Tests for anonymous two-reader grading, adjudication, and consensus labels."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.independent_grading import (
    GradeDecision,
    IndependentGrade,
    assemble_consensus_case,
    parse_grader_output,
    render_adjudication_prompt,
    render_grader_prompt,
    resolve_independent_grades,
)
from apush_frq_grader_slm.schemas import FailureType
from apush_frq_grader_slm.synth_realistic import GenTask, StudentPersona, SyntheticCandidate

ESSAY = (
    "The market revolution reshaped the early republic as canals and railroads linked distant "
    "farms to growing cities. Factory towns such as Lowell drew young women into wage labor, "
    "which slowly altered older family economies. Expanding suffrage and fierce debates over "
    "slavery deepened arguments about national identity, and these overlapping changes reveal "
    "a complex transformation across the period."
)
PROMPT = "Evaluate the extent of change in the early republic, 1800-1848."
SCORES_FOUR = {
    "thesis": 1,
    "contextualization": 1,
    "evidence": 1,
    "analysis_reasoning": 1,
}
SCORES_FIVE = {
    "thesis": 1,
    "contextualization": 1,
    "evidence": 2,
    "analysis_reasoning": 1,
}
FEEDBACK = {
    "thesis": "The claim says overlapping changes created a complex transformation.",
    "contextualization": (
        "Canals and railroads linking farms to cities frame the market revolution."
    ),
    "evidence": "Lowell factory towns and expanding suffrage provide specific evidence.",
    "analysis_reasoning": "Wage labor and debates over slavery are connected to national identity.",
}
SPAN = "canals and railroads linked distant farms to growing cities"


def _candidate() -> SyntheticCandidate:
    return SyntheticCandidate("task-1", PROMPT, ESSAY)


def _payload(scores=None, *, confidence: float = 0.9) -> dict:
    selected_scores = dict(scores or SCORES_FOUR)
    return {
        "scores": selected_scores,
        "total": sum(selected_scores.values()),
        "feedback": dict(FEEDBACK),
        "evidence_spans": {criterion: [SPAN] for criterion in SCORES_FOUR},
        "confidence": confidence,
    }


def _grade(grader_id: str, scores=None, *, confidence: float = 0.9) -> IndependentGrade:
    return parse_grader_output(_payload(scores, confidence=confidence), _candidate(), grader_id)


def _task(*, target_total: int = 5) -> GenTask:
    return GenTask(
        task_id="task-1",
        seed_id="prompt000",
        prompt=PROMPT,
        seed_scores=None,
        target_scores=dict(SCORES_FIVE),
        target_total=target_total,
        failure_type=FailureType.STRONG.value,
        length_band=(45, 70),
        seed_essay_excerpt="",
        persona=StudentPersona(30, "competent", "organized_argument", "ordinary_errors", "none"),
    )


class AnonymousPromptTests(unittest.TestCase):
    def test_grader_prompt_excludes_writer_target_and_persona(self) -> None:
        prompt = render_grader_prompt(_candidate())
        self.assertNotIn("HIDDEN CALIBRATION TARGET", prompt)
        self.assertNotIn("Target total", prompt)
        self.assertNotIn("organized argument", prompt)
        self.assertNotIn("30 minutes", prompt)
        self.assertIn(PROMPT, prompt)
        self.assertIn(ESSAY, prompt)

    def test_adjudicator_sees_reader_rationales_not_writer_controls(self) -> None:
        prompt = render_adjudication_prompt(_candidate(), _grade("a"), _grade("b", SCORES_FIVE))
        self.assertIn("READER A", prompt)
        self.assertIn("READER B", prompt)
        self.assertNotIn("generation target", prompt.lower())
        self.assertNotIn("student persona", prompt.lower())


class GradeValidationTests(unittest.TestCase):
    def test_rejects_ungrounded_criterion_span(self) -> None:
        payload = _payload()
        payload["evidence_spans"]["thesis"] = ["words absent from the candidate essay"]
        with self.assertRaisesRegex(ValueError, "ungrounded_evidence_span_thesis"):
            parse_grader_output(payload, _candidate(), "reader-a")

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid_confidence"):
            parse_grader_output(_payload(confidence=1.2), _candidate(), "reader-a")


class ConsensusTests(unittest.TestCase):
    def test_exact_row_agreement_accepts_consensus(self) -> None:
        decision = resolve_independent_grades("task-1", _grade("a", confidence=0.8), _grade("b"))
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.accepted_grade.scores.model_dump(), SCORES_FOUR)
        self.assertEqual(decision.consensus_metadata["resolution"], "direct_consensus")
        self.assertTrue(decision.consensus_metadata["all_criteria_agree"])

    def test_disagreement_requires_third_reader(self) -> None:
        decision = resolve_independent_grades("task-1", _grade("a"), _grade("b", SCORES_FIVE))
        self.assertEqual(decision.status, "rejected")
        self.assertIn("grader_disagreement_requires_adjudication", decision.reasons)

    def test_adjudicator_resolves_disagreement(self) -> None:
        decision = resolve_independent_grades(
            "task-1", _grade("a"), _grade("b", SCORES_FIVE), _grade("judge", SCORES_FIVE)
        )
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.accepted_grade.total, 5)
        self.assertEqual(decision.consensus_metadata["resolution"], "adjudicated")

    def test_uncertain_adjudication_is_rejected(self) -> None:
        decision = resolve_independent_grades(
            "task-1",
            _grade("a"),
            _grade("b", SCORES_FIVE),
            _grade("judge", SCORES_FIVE, confidence=0.3),
        )
        self.assertEqual(decision.status, "rejected")
        self.assertIn("low_adjudicator_confidence", decision.reasons)

    def test_decision_round_trip_preserves_consensus_metadata(self) -> None:
        decision = resolve_independent_grades("task-1", _grade("a"), _grade("b"))
        self.assertEqual(GradeDecision.from_row(decision.to_row()), decision)


class AssemblyTests(unittest.TestCase):
    def test_case_label_comes_from_consensus_not_target(self) -> None:
        decision = resolve_independent_grades("task-1", _grade("a"), _grade("b"))
        case, metadata, reasons = assemble_consensus_case(_candidate(), _task(), decision)
        self.assertEqual(reasons, [])
        self.assertIsNotNone(case)
        self.assertEqual(case.reference_scores.total, 4)
        self.assertEqual(case.reference_scores.model_dump(), SCORES_FOUR)
        self.assertEqual(metadata["generation_target_total"], 5)
        self.assertEqual(metadata["target_distance"], 1)

    def test_consensus_far_from_target_is_rejected_not_forced(self) -> None:
        decision = resolve_independent_grades("task-1", _grade("a"), _grade("b"))
        case, _, reasons = assemble_consensus_case(_candidate(), _task(target_total=6), decision)
        self.assertIsNone(case)
        self.assertIn("consensus_too_far_from_generation_target", reasons)


if __name__ == "__main__":
    unittest.main()
