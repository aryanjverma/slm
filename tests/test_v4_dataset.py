"""Unit tests for v4 prompts and dataset planning/assembly."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.dataset_v4 import (
    CBSeedProfile,
    assert_no_real_essays,
    assemble_v4_case,
    plan_v4_tasks,
    v4_chat_row,
)
from apush_frq_grader_slm.prompts_v4 import (
    GRADER_SYSTEM_PROMPT_V4,
    LEQ_RUBRIC_TEXT,
    STUDENT_SYSTEM_PROMPT_V4,
    V4_TRAIN_SYSTEM_PROMPT,
)
from apush_frq_grader_slm.schemas import FailureType, RubricFeedback, RubricScores


def _mock_seeds(count: int = 5) -> list[CBSeedProfile]:
    prompts = [
        (
            "Evaluate the extent to which the market revolution marked a turning point "
            "in the United States economy from 1800 to 1848."
        ),
        (
            "Evaluate the relative importance of different causes of the American "
            "Revolution in the period from 1754 to 1776."
        ),
        (
            "Evaluate the extent to which the Civil War changed American society "
            "in the period from 1861 to 1877."
        ),
        (
            "Evaluate the extent to which Progressive Era reformers were successful "
            "in improving American society from 1890 to 1920."
        ),
        (
            "Evaluate the extent to which the Cold War affected United States domestic "
            "policy from 1945 to 1980."
        ),
    ]
    seeds: list[CBSeedProfile] = []
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        seeds.append(
            CBSeedProfile(
                seed_id=f"cb{index:03d}",
                prompt=prompt,
                prompt_family_id=f"family{index:03d}",
                period=4 + (index % 5),
                reasoning_skill="causation",
                style_excerpt="Sample style excerpt about canals and factories.",
                amsco_chapter_ids=(f"ch{10 + index}",),
            )
        )
    return seeds


ESSAY = (
    "The market revolution reshaped the early republic as canals and railroads linked distant "
    "farms to growing cities. Factory towns such as Lowell drew young women into wage labor, "
    "which slowly altered older family economies. Expanding suffrage and fierce debates over "
    "slavery deepened arguments about national identity, and these overlapping changes reveal "
    "a complex transformation across the period from 1800 to 1848."
)


class PromptRubricTests(unittest.TestCase):
    def test_leq_rubric_appears_in_all_system_prompts(self) -> None:
        student = STUDENT_SYSTEM_PROMPT_V4.format(time_budget_minutes=30)
        for prompt in (student, GRADER_SYSTEM_PROMPT_V4, V4_TRAIN_SYSTEM_PROMPT):
            self.assertIn("Thesis / Claim", prompt)
            self.assertIn("Contextualization", prompt)
            self.assertIn("Evidence", prompt)
            self.assertIn("Analysis and Reasoning", prompt)
            self.assertIn(LEQ_RUBRIC_TEXT.strip().splitlines()[0], prompt)


class PlanV4TasksTests(unittest.TestCase):
    def test_plan_produces_unique_ids_and_all_totals(self) -> None:
        tasks = plan_v4_tasks(_mock_seeds(), [], target_count=250, seed=42)
        self.assertEqual(len(tasks), 250)
        ids = [task.task_id for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        totals = {task.target_total for task in tasks}
        self.assertEqual(totals, set(range(7)))
        for task in tasks:
            self.assertTrue(task.task_id.startswith(f"v4-{task.seed_id}-t{task.target_total}-n"))
            self.assertEqual(sum(task.target_scores.values()), task.target_total)
            self.assertIn(task.failure_type, {item.value for item in FailureType})


class AssembleV4Tests(unittest.TestCase):
    def test_assemble_rejects_real_tags(self) -> None:
        tasks = plan_v4_tasks(_mock_seeds(1), [], target_count=7, seed=1)
        case = assemble_v4_case(
            tasks[0],
            ESSAY,
            RubricScores(thesis=1, contextualization=1, evidence=2, analysis_reasoning=1),
            RubricFeedback(
                thesis="A defensible thesis appears when the essay argues the market revolution.",
                contextualization=(
                    "Broader context involving canals and railroads frames the argument."
                ),
                evidence="The essay uses Lowell and canals to support the claim.",
                analysis_reasoning=(
                    "The essay structures causation across overlapping changes in the period."
                ),
            ),
        )
        self.assertIn("synth_v4", case.tags)
        self.assertIn("amsco_kb", case.tags)
        self.assertEqual(case.provenance.source_type, "synthetic")
        self.assertEqual(case.provenance.generator_name, "v4_amsco_cb_seeded")
        assert_no_real_essays([case])

        poisoned = case.model_copy(deep=True)
        poisoned.tags = list(poisoned.tags) + ["ap_central"]
        with self.assertRaises(ValueError):
            assert_no_real_essays([poisoned])

    def test_chat_row_system_contains_thesis_claim(self) -> None:
        tasks = plan_v4_tasks(_mock_seeds(1), [], target_count=7, seed=2)
        case = assemble_v4_case(
            tasks[0],
            ESSAY,
            RubricScores(thesis=1, contextualization=0, evidence=1, analysis_reasoning=0),
            RubricFeedback(
                thesis="A defensible thesis appears when the essay argues the market revolution.",
                contextualization=(
                    "Before discussing Lowell, the essay does not establish broader context."
                ),
                evidence="The essay names Lowell and canals but develops them unevenly.",
                analysis_reasoning=(
                    "The essay lists changes without structuring clear historical reasoning."
                ),
            ),
        )
        row = v4_chat_row(case)
        system = row["messages"][0]["content"]
        self.assertEqual(row["messages"][0]["role"], "system")
        self.assertIn("Thesis / Claim", system)


if __name__ == "__main__":
    unittest.main()
