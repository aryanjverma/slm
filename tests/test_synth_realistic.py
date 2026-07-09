"""Tests for realistic real-seeded generation: leakage, label discipline, gate."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from apush_frq_grader_slm.ingest.dedup import contains_verbatim_span, is_duplicate_essay
from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores
from apush_frq_grader_slm.synth_realistic import (
    GenTask,
    parse_agent_row,
    plan_generation_tasks,
    validate_generated_case,
)

CLEAN_ESSAY = (
    "The market revolution reshaped the early republic as canals and railroads linked distant "
    "farms to growing cities. Factory towns such as Lowell drew young women into wage labor, "
    "which slowly altered older family economies. Expanding suffrage and fierce debates over "
    "slavery deepened arguments about national identity, and these overlapping changes reveal "
    "a complex transformation across the period."
)

CLEAN_FEEDBACK = {
    "thesis": "The essay states a defensible claim that overlapping changes reveal a complex "
    "transformation across the period.",
    "contextualization": "It frames the market revolution with canals and railroads linking "
    "farms to cities.",
    "evidence": "It cites specific examples like Lowell factory towns and expanding suffrage.",
    "analysis_reasoning": "It connects wage labor and debates over slavery to national identity, "
    "showing causation over time.",
}

TARGET = {"thesis": 1, "contextualization": 1, "evidence": 2, "analysis_reasoning": 1}


def _make_case(essay: str, prompt: str, *, split: str = "eval", tags=None) -> FRQCase:
    return FRQCase(
        id="x",
        split=split,  # type: ignore[arg-type]
        prompt=prompt,
        student_response=essay,
        reference_scores=RubricScores(thesis=1, contextualization=1, evidence=1, analysis_reasoning=1),
        reference_feedback=RubricFeedback(
            thesis="t", contextualization="c", evidence="e", analysis_reasoning="a"
        ),
        failure_type=FailureType.BORDERLINE_COMPLEXITY,
        difficulty="borderline",
        assistant_response="{}",
        tags=tags or [],
    )


def _make_task(**overrides) -> GenTask:
    defaults = dict(
        task_id="train-real-000-p09-v00",
        seed_id="seed000",
        prompt="Evaluate the extent of change in the early republic, 1800-1848.",
        seed_scores=None,
        target_scores=dict(TARGET),
        target_total=5,
        failure_type=FailureType.STRONG.value,
        length_band=(45, 70),
        seed_essay_excerpt="",
    )
    defaults.update(overrides)
    return GenTask(**defaults)


def _load_build_mixed():
    spec = importlib.util.spec_from_file_location(
        "build_mixed_dataset",
        Path(__file__).resolve().parents[1] / "scripts" / "build_mixed_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_mixed_dataset"] = module
    spec.loader.exec_module(module)
    return module


class LeakageGuardTests(unittest.TestCase):
    def test_flags_full_duplicate_same_prompt(self) -> None:
        prompt = "Evaluate industrial capitalism, 1865-1898."
        source = _make_case(
            "Industrial capitalism after 1870 concentrated enormous wealth among railroad "
            "magnates and provoked bitter labor strikes across the entire nation for decades.",
            prompt,
        )
        self.assertTrue(is_duplicate_essay(source.student_response, [source], prompt=prompt))

    def test_flags_single_lifted_span(self) -> None:
        source = _make_case(
            "Industrial capitalism after 1870 concentrated enormous wealth among railroad "
            "magnates and provoked bitter labor strikes across the nation.",
            "Evaluate industrial capitalism, 1865-1898.",
        )
        # A different, longer essay that copies one 8-word span verbatim.
        candidate = (
            "My essay argues that industry concentrated enormous wealth among railroad magnates "
            "and provoked strikes, which shows meaningful change over time across the era."
        )
        self.assertFalse(
            is_duplicate_essay(candidate, [source], prompt="a totally different prompt")
        )
        self.assertTrue(contains_verbatim_span(candidate, [source]))

    def test_validate_rejects_leaked_prose(self) -> None:
        source = _make_case(CLEAN_ESSAY, "Evaluate the extent of change in the early republic.")
        task = _make_task()
        row = {"student_response": CLEAN_ESSAY, "scores": dict(TARGET), "total": 5,
               "feedback": dict(CLEAN_FEEDBACK)}
        case = parse_agent_row(row, task)
        ok, reasons = validate_generated_case(case, task, row, [source])
        self.assertFalse(ok)
        self.assertIn("leaked_prose", reasons)


class LabelDisciplineTests(unittest.TestCase):
    def test_rejects_score_mismatch(self) -> None:
        task = _make_task()
        row = {
            "student_response": CLEAN_ESSAY,
            "scores": {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1},
            "total": 4,
            "feedback": dict(CLEAN_FEEDBACK),
        }
        case = parse_agent_row(row, task)
        ok, reasons = validate_generated_case(case, task, row, [])
        self.assertFalse(ok)
        self.assertIn("score_mismatch", reasons)

    def test_label_is_target_not_agent_total(self) -> None:
        # Even if the agent botches the total, the rebuilt target obeys total == sum.
        task = _make_task()
        row = {"student_response": CLEAN_ESSAY, "scores": dict(TARGET), "total": 99,
               "feedback": dict(CLEAN_FEEDBACK)}
        case = parse_agent_row(row, task)
        self.assertEqual(case.reference_scores.total, 5)
        self.assertIn('"total": 5', case.assistant_response)


class QualityGateTests(unittest.TestCase):
    def test_clean_case_passes(self) -> None:
        task = _make_task()
        row = {"student_response": CLEAN_ESSAY, "scores": dict(TARGET), "total": 5,
               "feedback": dict(CLEAN_FEEDBACK)}
        case = parse_agent_row(row, task)
        ok, reasons = validate_generated_case(case, task, row, [])
        self.assertTrue(ok, msg=f"unexpected reasons: {reasons}")


class SeedDedupTests(unittest.TestCase):
    def test_seed_dedups_against_frozen_eval(self) -> None:
        frozen = _make_case(CLEAN_ESSAY, "Evaluate change in the early republic, 1800-1848.")
        # A candidate seed identical to a frozen eval essay must be flagged.
        self.assertTrue(
            is_duplicate_essay(CLEAN_ESSAY, [frozen], prompt=frozen.prompt)
        )
        # A genuinely different essay on a different prompt is not flagged.
        other = (
            "Cold War containment shaped American foreign policy after 1945 as leaders funded "
            "the Marshall Plan and confronted Soviet expansion in Europe and Asia repeatedly."
        )
        self.assertFalse(is_duplicate_essay(other, [frozen], prompt="Evaluate the Cold War."))


class NoRealEssayGuardTests(unittest.TestCase):
    def test_realistic_passes_and_seed_real_raises(self) -> None:
        module = _load_build_mixed()
        task = _make_task()
        row = {"student_response": CLEAN_ESSAY, "scores": dict(TARGET), "total": 5,
               "feedback": dict(CLEAN_FEEDBACK)}
        realistic = parse_agent_row(row, task)
        # Assembled realistic cases pass the guard.
        module.assert_no_real_essays([realistic])
        # A seed_real-tagged case in the train split must raise.
        leaked = _make_case(CLEAN_ESSAY, "p", split="train", tags=["seed_real"])
        with self.assertRaises(ValueError):
            module.assert_no_real_essays([leaked])


class PlanTests(unittest.TestCase):
    def test_plan_covers_all_totals_per_prompt(self) -> None:
        prompts = ["Evaluate A, 1800-1848.", "Evaluate B, 1865-1898."]
        tasks = plan_generation_tasks([], prompts, variants_per_seed=24)
        self.assertEqual(len({t.seed_id for t in tasks}), 2)
        for prompt in prompts:
            totals = {t.target_total for t in tasks if t.prompt == prompt}
            self.assertGreaterEqual(len(totals), 5)  # wide score spread breaks prompt->score


if __name__ == "__main__":
    unittest.main()
