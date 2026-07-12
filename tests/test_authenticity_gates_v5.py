"""Tests for v5 authenticity hard gates and regeneration pilot controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apush_frq_grader_slm.authenticity_gates_v5 import (
    PROHIBITED_ARTIFACT_CATEGORIES,
    detect_artifact_categories,
    hard_gate_reasons,
    max_contiguous_overlap_words,
    meta_process_gate_reasons,
    style_copy_gate_reasons,
)
from apush_frq_grader_slm.dataset_v4 import CBSeedProfile
from apush_frq_grader_slm.dataset_v5 import (
    V5_PILOT_COUNT,
    V5GenerationTask,
    assert_pilot_approval,
    build_pilot_approval,
    candidate_gate_reasons,
    generator_packet,
    plan_v5_tasks,
    select_v5_pilot_tasks,
)


def _seed() -> CBSeedProfile:
    return CBSeedProfile(
        seed_id="style-a",
        prompt="Evaluate changes in the early republic.",
        prompt_family_id="family-a",
        period=4,
        reasoning_skill="causation",
        style_excerpt="i think canals changed markets but it also wasnt equal " * 20,
        amsco_chapter_ids=("ch10",),
        adapted_prompts=(
            "Assess how canals remade markets in the early republic.",
            "Judge whether canals remade markets in the early republic.",
            "Determine how far canals remade markets in the early republic.",
        ),
    )


def _task_with_style(**overrides: object) -> V5GenerationTask:
    base = dict(
        task_id="v5-0001",
        shard_id="v5-shard-00",
        prompt="Assess how canals remade markets in the early republic.",
        prompt_family_id="family-a",
        style_seed_id="style-a",
        style_excerpt="excerpt",
        period=4,
        reasoning_skill="causation",
        capability_profile={"historical_knowledge": "competent", "argument_control": "consistent"},
        composition_profile={
            "time_pressure": "normal",
            "mechanics": "minor_errors",
            "organization": "clear",
        },
        amsco_chapter_ids=("ch10",),
        style_reference_essay=(
            "Before the canals, farmers hauled grain slowly over bad roads. "
            "The Erie Canal then linked western farms to eastern ports and cut shipping costs. "
            "Towns along the route grew fast, and New York became a bigger commercial hub. "
            "Still, not every region gained equally because the South stayed more tied to rivers "
            "and staple crops than to northern canal networks."
        ),
        reference_word_count=70,
    )
    base.update(overrides)
    return V5GenerationTask(**base)  # type: ignore[arg-type]


class AuthenticityGateTests(unittest.TestCase):
    def test_all_first_eight_rejection_categories_are_detected(self) -> None:
        samples = {
            "memory_or_notes": "I pulled this from class notes about tariffs.",
            "planning_or_draft_process": "My outline for this essay is rough.",
            "physical_test_conditions": "The bluebook is almost full already.",
            "knowledge_admission": "I cannot recall the exact law name.",
            "generator_mattering_stub": "The Embargo around 1807 mattering for trade.",
            "prompt_or_instruction_leakage": "I was asked to write about canals.",
            "timing_theater_filler": "Hard to finish in time so I stop here.",
            "stock_classroom_filler": "Teachers say the market revolution mattered.",
        }
        self.assertEqual(set(samples), set(PROHIBITED_ARTIFACT_CATEGORIES))
        for category, text in samples.items():
            self.assertIn(category, detect_artifact_categories(text), category)
            self.assertTrue(meta_process_gate_reasons(text))

    def test_clean_student_essay_passes_meta_gate(self) -> None:
        essay = (
            "Canals changed markets after 1815 because western wheat could reach New York "
            "cheaper than before. The Erie Canal is the clearest example. Southern cotton "
            "still moved mostly by river, so the shift was uneven."
        )
        self.assertEqual(detect_artifact_categories(essay), [])
        self.assertEqual(meta_process_gate_reasons(essay), [])

    def test_style_copy_limits(self) -> None:
        style = (
            "Before the canals farmers hauled grain slowly over bad roads and the Erie Canal "
            "then linked western farms to eastern ports and cut shipping costs for many towns."
        )
        # 21+ contiguous words copied -> reject
        copied = style + " Also some local banks expanded credit."
        self.assertGreater(max_contiguous_overlap_words(copied, style), 20)
        self.assertIn(
            "style_copy_contiguous_overlap_exceeded",
            style_copy_gate_reasons(copied, style),
        )
        # One short borrowed sentence with an 8-gram is allowed.
        mild = (
            "Markets shifted after the war. linked western farms to eastern ports and cut "
            "shipping costs. Banks also grew in canal towns."
        )
        self.assertEqual(style_copy_gate_reasons(mild, style), [])

    def test_hard_gate_overrides_reader_leniency_signal(self) -> None:
        essay = "I dug this out of my notes about canals and markets."
        reasons = hard_gate_reasons(essay, style_reference_essay="", reference_word_count=80)
        self.assertTrue(any(r.startswith("meta_process:") for r in reasons))


class PilotAndPacketTests(unittest.TestCase):
    def test_writer_packet_includes_full_essay_excludes_scores(self) -> None:
        task = _task_with_style()
        packet = generator_packet(task, [{"concept": "Erie Canal cut freight costs."}])
        self.assertIn("style_reference_essay", packet)
        self.assertEqual(packet["style_reference_essay"], task.style_reference_essay)
        self.assertIn("essay_only_contract", packet)
        self.assertNotIn("style_reference", packet)  # truncated excerpt field removed
        forbidden = {
            "target_scores",
            "scores",
            "reference_scores",
            "reference_feedback",
            "source_case_id",
            "style_seed_id",
        }
        self.assertFalse(forbidden & set(packet))
        self.assertIn("paraphrase in your own words", packet["semantic_fact_cards"][0]["use"])

    def test_packet_requires_full_style_essay(self) -> None:
        task = _task_with_style(style_reference_essay="")
        with self.assertRaises(ValueError):
            generator_packet(task, [])

    def test_pilot_selection_shape(self) -> None:
        tasks = plan_v5_tasks([_seed()])
        pilot = select_v5_pilot_tasks(tasks, seed=51)
        self.assertEqual(len(pilot), V5_PILOT_COUNT)
        boundary = [t for t in pilot if t.coverage_class == "boundary"]
        golden = [t for t in pilot if t.coverage_class == "golden_matched"]
        self.assertEqual(len(boundary), 24)
        self.assertEqual(len(golden), 6)
        for boundary_type in {
            "thesis_0_1",
            "contextualization_0_1",
            "evidence_0_1",
            "evidence_1_2",
            "analysis_reasoning_0_1",
            "analysis_reasoning_1_2",
        }:
            pairs = {
                t.contrast_pair_id for t in boundary if t.boundary_type == boundary_type
            }
            self.assertEqual(len(pairs), 2)
            for pair_id in pairs:
                sides = {
                    t.contrast_side
                    for t in boundary
                    if t.contrast_pair_id == pair_id
                }
                self.assertEqual(sides, {"lower", "upper"})

    def test_pilot_approval_blocks_until_all_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            essays_path = Path(directory) / "pilot.jsonl"
            rows = [
                {
                    "task_id": f"v5-{i:04d}",
                    "student_response": f"Essay body for task {i} with enough words " * 8,
                }
                for i in range(30)
            ]
            essays_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            approval_path = Path(directory) / "approval.json"
            with self.assertRaises(PermissionError):
                assert_pilot_approval(essays_path, approval_path)
            decisions = {row["task_id"]: "accept" for row in rows}
            approval = build_pilot_approval(
                reviewer="tester",
                approved_at="2026-07-12T00:00:00Z",
                pilot_essays_path=essays_path,
                decisions=decisions,
            )
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            loaded = assert_pilot_approval(essays_path, approval_path)
            self.assertTrue(loaded["approved"])
            # Mutate essays after approval -> hash mismatch
            essays_path.write_text(
                essays_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaises(PermissionError):
                assert_pilot_approval(essays_path, approval_path)

    def test_candidate_gate_includes_hard_gates(self) -> None:
        row = {
            "task_id": "x",
            "student_response": "From my notes the canal changed markets after the war. " * 6,
            "authenticity_reviews": [
                {"reviewer_id": "a", "student_like": True, "timed_ap_consistent": True},
                {"reviewer_id": "b", "student_like": True, "timed_ap_consistent": True},
            ],
            "rubric_reviews": [
                {
                    "reader_id": f"r{i}",
                    "scores": {
                        "thesis": 1,
                        "contextualization": 1,
                        "evidence": 1,
                        "analysis_reasoning": 1,
                    },
                    "confidence": 0.9,
                }
                for i in range(3)
            ],
            "resolved_grade": {
                "scores": {
                    "thesis": 1,
                    "contextualization": 1,
                    "evidence": 1,
                    "analysis_reasoning": 1,
                },
                "feedback": {
                    "thesis": "claim",
                    "contextualization": "context",
                    "evidence": "evidence",
                    "analysis_reasoning": "analysis",
                },
                "adjudicated": False,
            },
            "fact_check": {"passed": True, "checker_id": "facts"},
            "distribution_match": {"passed": True},
            "selection_class": "golden_matched",
            "prompt_family_id": "f",
            "style_seed_id": "s",
        }
        reasons = candidate_gate_reasons(row, reference_word_count=80)
        self.assertTrue(any(r.startswith("meta_process:") for r in reasons))


if __name__ == "__main__":
    unittest.main()
