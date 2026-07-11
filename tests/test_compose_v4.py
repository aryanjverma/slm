"""Unit tests for the AMSCO v4 essay composer."""

from __future__ import annotations

import random
import unittest

from apush_frq_grader_slm.compose_v4 import STRONG_CLAIM_RE, compose_essay, rng_for_task


def _bundle() -> dict:
    return {
        "chapter_ids": ["amsco_ch09"],
        "key_facts": [
            "The cotton gin strengthened plantation slavery after 1793.",
            "Missouri Compromise debates exposed sectional fault lines in 1820.",
            "Nullification Crisis tested federal authority in the early 1830s.",
        ],
        "evidence_bank": [
            "cotton gin / King Cotton",
            "Missouri Compromise",
            "Nullification Crisis",
            "Nat Turner's rebellion",
        ],
        "misconceptions": [
            "Students may treat slavery as uniformly paternalistic across regions.",
        ],
        "context_hooks": [
            "In 1826, Americans celebrated fifty years of independence.",
            "Sectional loyalty grew as regional economies diverged.",
            "Atlantic markets linked Southern staples to Northern finance.",
        ],
    }


def _task(**overrides: object) -> dict:
    base: dict = {
        "task_id": "v4-test-compose-t0-n000",
        "prompt": (
            "Evaluate how sectional tensions shaped United States society "
            "from 1800 to 1848."
        ),
        "reasoning_skill": "causation",
        "target_scores": {
            "thesis": 0,
            "contextualization": 0,
            "evidence": 0,
            "analysis_reasoning": 0,
        },
        "length_band": [80, 200],
        "persona": {
            "time_budget_minutes": 25,
            "historical_knowledge": "competent",
            "planning_style": "no_outline",
            "mechanics": "clean",
            "misconception": "none",
        },
    }
    base.update(overrides)
    return base


class ComposeV4Tests(unittest.TestCase):
    def test_thesis_zero_lacks_strong_claim_pattern(self) -> None:
        task = _task(
            task_id="v4-test-thesis0",
            target_scores={
                "thesis": 0,
                "contextualization": 0,
                "evidence": 1,
                "analysis_reasoning": 0,
            },
        )
        essay = compose_essay(task, _bundle(), rng_for_task(task["task_id"]))
        self.assertFalse(
            STRONG_CLAIM_RE.search(essay),
            msg=f"thesis=0 essay unexpectedly matched strong claim: {essay[:240]!r}",
        )

    def test_evidence_two_contains_two_bank_strings(self) -> None:
        bundle = _bundle()
        task = _task(
            task_id="v4-test-evidence2",
            target_scores={
                "thesis": 1,
                "contextualization": 1,
                "evidence": 2,
                "analysis_reasoning": 1,
            },
            length_band=[200, 400],
        )
        essay = compose_essay(task, bundle, rng_for_task(task["task_id"]))
        hits = [item for item in bundle["evidence_bank"] if item in essay]
        self.assertGreaterEqual(
            len(hits),
            2,
            msg=f"expected ≥2 evidence_bank strings in essay; found {hits!r}\n{essay}",
        )

    def test_deterministic_for_same_rng_seed(self) -> None:
        task = _task(
            task_id="v4-test-det",
            target_scores={
                "thesis": 1,
                "contextualization": 1,
                "evidence": 2,
                "analysis_reasoning": 2,
            },
            length_band=[220, 360],
        )
        a = compose_essay(task, _bundle(), random.Random(99))
        b = compose_essay(task, _bundle(), random.Random(99))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
