"""Focused tests for the score-blind v5 essay composer."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.compose_v4 import rng_for_task
from apush_frq_grader_slm.compose_v5 import (
    GENERATOR_NAME,
    compose_essay,
    resolve_observable_behavior,
)
from apush_frq_grader_slm.fact_cards_v5 import has_eight_gram_overlap


def _packet(**overrides: object) -> dict:
    base: dict = {
        "task_id": "v5-test-0001",
        "prompt": (
            "Assess how far the expansion of Atlantic commerce remade "
            "British North American colonial social order from 1607 to 1776."
        ),
        "student_capability": {
            "historical_knowledge": "competent",
            "argument_control": "consistent",
        },
        "timed_composition_style": {
            "time_pressure": "normal",
            "mechanics": "minor_errors",
            "organization": "clear",
        },
        "style_reference": "",
        "semantic_fact_cards": [
            {
                "concept": (
                    "Transatlantic trade tied colonial ports to British credit and "
                    "consumer goods across the eighteenth century."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
            {
                "concept": (
                    "Plantation slavery expanded as Atlantic markets rewarded "
                    "staple crops in the southern colonies."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
            {
                "concept": (
                    "Northern shipbuilding and merchant networks grew alongside "
                    "the same Atlantic commercial circuit."
                ),
                "use": "paraphrase from memory; do not copy wording",
            },
        ],
    }
    base.update(overrides)
    return base


class ComposeV5Tests(unittest.TestCase):
    def test_limited_knowledge_shorter_and_weaker_than_strong(self) -> None:
        limited = _packet(
            task_id="v5-test-limited",
            student_capability={
                "historical_knowledge": "limited",
                "argument_control": "emerging",
            },
            timed_composition_style={
                "time_pressure": "severe",
                "mechanics": "frequent_natural_errors",
                "organization": "rough",
            },
        )
        strong = _packet(
            task_id="v5-test-strong",
            student_capability={
                "historical_knowledge": "strong",
                "argument_control": "nuanced",
            },
            timed_composition_style={
                "time_pressure": "normal",
                "mechanics": "minor_errors",
                "organization": "clear",
            },
        )
        limited_essay = compose_essay(limited, rng=rng_for_task("v5-test-limited"))
        strong_essay = compose_essay(strong, rng=rng_for_task("v5-test-strong"))
        self.assertLess(
            len(limited_essay.split()),
            len(strong_essay.split()),
            msg=(
                f"expected limited essay shorter than strong; "
                f"limited={len(limited_essay.split())} strong={len(strong_essay.split())}"
            ),
        )
        # Weak / emerging control should avoid a crisp overall claim framing.
        strong_markers = (
            "to a significant extent",
            "overall,",
            "i think",
            "mattered",
        )
        self.assertTrue(
            any(m in strong_essay.lower() for m in strong_markers),
            msg=f"strong essay lacked claim-like framing:\n{strong_essay[:300]}",
        )
        self.assertFalse(
            "to a significant extent" in limited_essay.lower()
            and "mainly becuase" in limited_essay.lower(),
            msg=f"limited essay unexpectedly used strong claim template:\n{limited_essay[:300]}",
        )

    def test_no_score_keys_leaked_in_packet_or_contract(self) -> None:
        packet = _packet()
        essay = compose_essay(packet, rng=rng_for_task(packet["task_id"]))
        self.assertIsInstance(essay, str)
        self.assertGreater(len(essay.split()), 80)
        # Composer must refuse packets that already carry score targets.
        dirty = dict(packet)
        dirty["target_scores"] = {"thesis": 1}
        with self.assertRaises(ValueError):
            compose_essay(dirty, rng=rng_for_task("v5-test-leak"))
        # Output contract for shard rows stays score-blind.
        row = {
            "task_id": packet["task_id"],
            "student_response": essay,
            "generator_name": GENERATOR_NAME,
            "shard_id": "v5-shard-00",
        }
        forbidden = {
            "target_scores",
            "target_total",
            "scores",
            "score",
            "rubric_text",
            "resolved_grade",
        }
        self.assertFalse(forbidden & set(row))
        self.assertEqual(row["generator_name"], "compose_v5_score_blind")

    def test_essay_references_paraphrased_concepts_when_cards_present(self) -> None:
        packet = _packet(
            task_id="v5-test-cards",
            student_capability={
                "historical_knowledge": "strong",
                "argument_control": "nuanced",
            },
        )
        essay = compose_essay(packet, rng=rng_for_task("v5-test-cards"))
        lowered = essay.lower()
        # Expect paraphrased anchors drawn from card concepts (not AMSCO copy).
        anchors = ("trade", "atlantic", "slavery", "plantation", "merchant", "ship", "colonial", "staple")
        hits = [a for a in anchors if a in lowered]
        self.assertGreaterEqual(
            len(hits),
            2,
            msg=f"expected paraphrased concept cues in essay; hits={hits}\n{essay}",
        )
        for card in packet["semantic_fact_cards"]:
            concept = card["concept"]
            self.assertFalse(
                has_eight_gram_overlap(essay, concept),
                msg=f"essay copied 8-gram from concept:\n{concept}\n---\n{essay}",
            )

    def test_style_reference_not_copied_as_long_verbatim_span(self) -> None:
        style = (
            "Although it furthered the British North American colonies ties to England, "
            "transatlantic trade succeeded in drastically changing the British North "
            "American colonial society to large extent by leading to the development of "
            "distinct regions throughout the colonies and leading to the dependence of "
            "Southern society on slave labor."
        )
        packet = _packet(
            task_id="v5-test-style",
            style_reference=style[:400],
            student_capability={
                "historical_knowledge": "competent",
                "argument_control": "consistent",
            },
        )
        essay = compose_essay(packet, rng=rng_for_task("v5-test-style"))
        self.assertFalse(
            has_eight_gram_overlap(essay, style),
            msg=f"essay retained 8-gram overlap with style_reference:\n{essay}",
        )
        # Long contiguous clone of the style opening should not appear.
        self.assertNotIn(
            "transatlantic trade succeeded in drastically changing",
            essay.lower(),
        )

    def test_boundary_behavior_restored_from_task(self) -> None:
        packet = _packet(
            task_id="v5-test-boundary",
            student_capability={
                "historical_knowledge": "uneven",
                "argument_control": "partial",
            },
        )
        task = {
            "task_id": "v5-test-boundary",
            "capability_profile": {
                "historical_knowledge": "uneven",
                "argument_control": "partial",
                "observable_writing_behavior": (
                    "discusses the topic but never settles on one overall answer"
                ),
            },
        }
        behavior = resolve_observable_behavior(packet, task)
        self.assertIn("never settles", behavior or "")
        essay = compose_essay(
            packet,
            observable_writing_behavior=behavior,
            rng=rng_for_task("v5-test-boundary"),
        )
        lowered = essay.lower()
        self.assertTrue(
            any(
                cue in lowered
                for cue in (
                    "hard to pick one answer",
                    "not sure there is one clear",
                    "different directions",
                    "lots of factors",
                )
            ),
            msg=f"boundary unsettle thesis not reflected:\n{essay}",
        )

    def test_deterministic_for_same_rng(self) -> None:
        packet = _packet(task_id="v5-test-det")
        a = compose_essay(packet, rng=rng_for_task("v5-test-det"))
        b = compose_essay(packet, rng=rng_for_task("v5-test-det"))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
