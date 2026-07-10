"""Tests for deterministic realistic generation and unlabeled candidates."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores
from apush_frq_grader_slm.synth_realistic import (
    KNOWLEDGE_LEVELS,
    MECHANICS_PROFILES,
    MISCONCEPTION_PROFILES,
    PLANNING_STYLES,
    TIME_BUDGETS,
    GenTask,
    SyntheticCandidate,
    parse_candidate_row,
    parse_writer_response,
    plan_generation_tasks,
    render_generation_prompt,
    recalibrate_task,
    select_balanced_task_pilot,
    validate_generated_candidate,
)

CLEAN_ESSAY = (
    "The market revolution reshaped the early republic as canals and railroads linked distant "
    "farms to growing cities. Factory towns such as Lowell drew young women into wage labor, "
    "which slowly altered older family economies. Expanding suffrage and fierce debates over "
    "slavery deepened arguments about national identity, and these overlapping changes reveal "
    "a complex transformation across the period."
)


def _make_case(essay: str, prompt: str) -> FRQCase:
    return FRQCase(
        id="x",
        split="eval",
        prompt=prompt,
        student_response=essay,
        reference_scores=RubricScores(
            thesis=1, contextualization=1, evidence=1, analysis_reasoning=1
        ),
        reference_feedback=RubricFeedback(
            thesis="t", contextualization="c", evidence="e", analysis_reasoning="a"
        ),
        failure_type=FailureType.BORDERLINE_COMPLEXITY,
        difficulty="borderline",
        assistant_response="{}",
        tags=[],
    )


def _make_task(**overrides) -> GenTask:
    values = {
        "task_id": "train-real-000-p09-v00",
        "seed_id": "seed000",
        "prompt": "Evaluate the extent of change in the early republic, 1800-1848.",
        "seed_scores": None,
        "target_scores": {
            "thesis": 1,
            "contextualization": 1,
            "evidence": 2,
            "analysis_reasoning": 1,
        },
        "target_total": 5,
        "failure_type": FailureType.STRONG.value,
        "length_band": (45, 70),
        "seed_essay_excerpt": "",
    }
    values.update(overrides)
    return GenTask(**values)


class PersonaPlanningTests(unittest.TestCase):
    def test_planning_is_reproducible_and_covers_persona_dimensions(self) -> None:
        first = plan_generation_tasks([], ["Evaluate A."], variants_per_seed=120)
        second = plan_generation_tasks([], ["Evaluate A."], variants_per_seed=120)
        self.assertEqual([task.to_row() for task in first], [task.to_row() for task in second])
        self.assertEqual({task.persona.time_budget_minutes for task in first}, set(TIME_BUDGETS))
        self.assertEqual(
            {task.persona.historical_knowledge for task in first}, set(KNOWLEDGE_LEVELS)
        )
        self.assertEqual({task.persona.planning_style for task in first}, set(PLANNING_STYLES))
        self.assertEqual({task.persona.mechanics for task in first}, set(MECHANICS_PROFILES))
        self.assertEqual(
            {task.persona.misconception for task in first}, set(MISCONCEPTION_PROFILES)
        )

    def test_task_round_trip_preserves_persona(self) -> None:
        task = plan_generation_tasks([], ["Evaluate A."], variants_per_seed=1)[0]
        self.assertEqual(GenTask.from_row(task.to_row()), task)

    def test_each_prompt_gets_a_wide_score_range(self) -> None:
        prompts = ["Evaluate A.", "Evaluate B."]
        tasks = plan_generation_tasks([], prompts, variants_per_seed=24)
        for prompt in prompts:
            totals = {task.target_total for task in tasks if task.prompt == prompt}
            self.assertGreaterEqual(len(totals), 5)

    def test_balanced_pilot_covers_totals_and_prompt_families(self) -> None:
        prompts = [f"Evaluate historical change in region {index}." for index in range(20)]
        metadata = [
            {
                "prompt_family_id": f"family_{index}",
                "period": 4,
                "reasoning_skill": "causation",
            }
            for index in range(20)
        ]
        tasks = plan_generation_tasks(
            [], prompts, variants_per_seed=24, prompt_metadata=metadata
        )
        pilot = select_balanced_task_pilot(tasks, 100)
        self.assertEqual(len(pilot), 100)
        self.assertEqual({task.target_total for task in pilot}, set(range(7)))
        self.assertEqual(len({task.prompt_family_id for task in pilot}), 20)
        task_types = [task.task_type for task in pilot]
        self.assertEqual(task_types.count("ordinary"), 75)
        self.assertEqual(task_types.count("adversarial"), 15)
        self.assertEqual(task_types.count("diagnostic"), 10)


class WriterContractTests(unittest.TestCase):
    def test_writer_sees_persona_but_returns_only_essay(self) -> None:
        prompt = render_generation_prompt(_make_task())
        self.assertIn("STUDENT PERSONA", prompt)
        self.assertIn("HIDDEN CALIBRATION TARGET", prompt)
        self.assertIn('{"student_response":', prompt)
        self.assertNotIn("Then grade the essay", prompt)
        self.assertNotIn('"feedback":', prompt)

    def test_low_score_prompt_forbids_specific_evidence_and_argument(self) -> None:
        task = _make_task(
            target_scores={
                "thesis": 0,
                "contextualization": 0,
                "evidence": 0,
                "analysis_reasoning": 0,
            },
            target_total=0,
        )
        prompt = render_generation_prompt(recalibrate_task(task))
        self.assertIn("name no specific event", prompt)
        self.assertIn("do not explain causation", prompt)
        self.assertIn("Historical knowledge: weak", prompt)

    def test_candidate_discards_legacy_self_grade(self) -> None:
        task = _make_task()
        candidate = parse_candidate_row(
            {
                "task_id": task.task_id,
                "student_response": CLEAN_ESSAY,
                "scores": {"thesis": 0},
                "total": 0,
                "feedback": {"thesis": "self grade must be ignored"},
            },
            task,
        )
        self.assertIsInstance(candidate, SyntheticCandidate)
        self.assertEqual(
            candidate.to_row(),
            {"task_id": task.task_id, "student_response": CLEAN_ESSAY},
        )

    def test_writer_response_extracts_json_without_labels(self) -> None:
        task = _make_task()
        candidate = parse_writer_response(
            'prefix {"student_response": "A student essay about canals and factories."} suffix',
            task,
        )
        self.assertEqual(candidate.task_id, task.task_id)
        self.assertEqual(candidate.student_response, "A student essay about canals and factories.")


class CandidateValidationTests(unittest.TestCase):
    def test_clean_candidate_passes(self) -> None:
        task = _make_task()
        candidate = parse_candidate_row({"student_response": CLEAN_ESSAY}, task)
        self.assertEqual(validate_generated_candidate(candidate, task, []), (True, []))

    def test_rejects_leaked_prose(self) -> None:
        task = _make_task()
        source = _make_case(CLEAN_ESSAY, task.prompt)
        candidate = parse_candidate_row({"student_response": CLEAN_ESSAY}, task)
        ok, reasons = validate_generated_candidate(candidate, task, [source])
        self.assertFalse(ok)
        self.assertIn("leaked_prose", reasons)

    def test_rejects_writer_instruction_leakage(self) -> None:
        task = _make_task(length_band=(1, 100))
        candidate = SyntheticCandidate(task.task_id, task.prompt, "My target score is five.")
        ok, reasons = validate_generated_candidate(candidate, task, [])
        self.assertFalse(ok)
        self.assertIn("generator_instruction_leakage", reasons)


if __name__ == "__main__":
    unittest.main()
