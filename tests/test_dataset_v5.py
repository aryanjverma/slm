"""Tests for the score-blind and approval-gated v5 data pipeline."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from apush_frq_grader_slm.dataset_v4 import CBSeedProfile
from apush_frq_grader_slm.dataset_v5 import (
    BOUNDARY_TYPES, assemble_v5_selection, assert_manual_approval,
    candidate_gate_reasons, candidate_to_case, generator_packet, manual_review_packet,
    normalize_external_candidate, overlap_reasons, plan_v5_tasks,
)


def _seed() -> CBSeedProfile:
    return CBSeedProfile(
        seed_id="style-a", prompt="Evaluate changes in the early republic.",
        prompt_family_id="family-a", period=4, reasoning_skill="causation",
        style_excerpt="i think canals changed markets but it also wasnt equal " * 20,
        amsco_chapter_ids=("ch10",),
    )


def _candidate(task_id: str, kind: str, group: str, *, boundary_type: str = "",
               pair: str = "", side: str = "") -> dict:
    scores = {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1}
    feedback = {
        "thesis": "The response states a defensible claim about historical change.",
        "contextualization": "The opening places the argument in broader historical context.",
        "evidence": "The response uses its historical examples to support the claim.",
        "analysis_reasoning": "The response organizes its explanation through causation.",
    }
    key = "".join(ch for ch in task_id if ch.isalnum())
    unique = " ".join(f"detail{key}{i}" for i in range(50))
    essay = f"Student {task_id} timed historical response {unique}."
    row = {
        "task_id": task_id, "prompt": "Evaluate historical change in the period.",
        "student_response": essay, "selection_class": kind,
        "prompt_family_id": f"family-{group}", "style_seed_id": f"style-{group}",
        "authenticity_reviews": [
            {"reviewer_id": "a", "student_like": True, "timed_ap_consistent": True},
            {"reviewer_id": "b", "student_like": True, "timed_ap_consistent": True},
        ],
        "rubric_reviews": [
            {"reader_id": f"r{i}", "scores": scores, "confidence": .9} for i in range(3)
        ],
        "resolved_grade": {"scores": scores, "feedback": feedback, "adjudicated": False},
        "fact_check": {"passed": True, "checker_id": "facts"},
        "distribution_match": {"passed": kind == "golden_matched"},
    }
    if kind == "boundary":
        row.update(boundary_type=boundary_type, contrast_pair_id=pair, contrast_side=side)
    return row


class V5DatasetTests(unittest.TestCase):
    def test_planner_makes_fixed_score_blind_shards(self) -> None:
        tasks = plan_v5_tasks([_seed()])
        self.assertEqual(len(tasks), 1500)
        self.assertEqual({sum(t.shard_id == shard for t in tasks) for shard in {t.shard_id for t in tasks}}, {50})
        packet = generator_packet(tasks[0], [{"concept": "Transportation reduced shipping costs."}])
        self.assertLessEqual(len(tasks[0].style_excerpt), 400)
        self.assertFalse({"target_scores", "target_total", "rubric_text"} & packet.keys())
        self.assertIn("paraphrase", packet["semantic_fact_cards"][0]["use"])
        self.assertEqual(sum(task.coverage_class == "boundary" for task in tasks), 432)
        pairs = {
            (task.boundary_type, task.contrast_pair_id)
            for task in tasks
            if task.coverage_class == "boundary"
        }
        self.assertEqual(len(pairs), 216)
        for boundary_type, pair_id in pairs:
            pair = [task for task in tasks if task.contrast_pair_id == pair_id]
            self.assertEqual({task.contrast_side for task in pair}, {"lower", "upper"})
            self.assertEqual(len({task.prompt for task in pair}), 1)

    def test_overlap_and_review_gates(self) -> None:
        source = "one two three four five six seven eight nine ten eleven twelve " * 8
        copied = "intro words here " + source + " a distinct ending"
        self.assertIn("verbatim_eight_word_overlap", overlap_reasons(copied, [source]))
        row = _candidate("x", "golden_matched", "x")
        row["authenticity_reviews"] = row["authenticity_reviews"][:1]
        self.assertIn("missing_two_authenticity_reviews", candidate_gate_reasons(row))
        row = _candidate("x", "golden_matched", "x")
        row["rubric_reviews"][0]["confidence"] = .7
        self.assertIn("rubric_adjudication_required", candidate_gate_reasons(row))

    def test_exact_selection_grouped_split_and_boundary_pairs(self) -> None:
        rows = []
        rows.extend(_candidate(f"g{i:03d}", "golden_matched", f"g{i:03d}") for i in range(420))
        for boundary_type in BOUNDARY_TYPES:
            for pair_index in range(15):
                pair = f"{boundary_type}-{pair_index:02d}"
                for side in ("lower", "upper"):
                    task_id = f"b-{pair}-{side}"
                    rows.append(_candidate(task_id, "boundary", task_id,
                                           boundary_type=boundary_type, pair=pair, side=side))
        train, dev = assemble_v5_selection(rows)
        self.assertEqual((len(train), len(dev)), (540, 60))
        self.assertEqual(sum(r["selection_class"] == "boundary" for r in train + dev), 180)
        train_groups = {(r["prompt_family_id"], r["style_seed_id"]) for r in train}
        dev_groups = {(r["prompt_family_id"], r["style_seed_id"]) for r in dev}
        self.assertTrue(train_groups.isdisjoint(dev_groups))

    def test_manual_review_approval_is_hash_bound(self) -> None:
        import tempfile
        rows = [_candidate(f"g{i}", "golden_matched", str(i)) for i in range(60)]
        packet = manual_review_packet(rows)
        for row in packet:
            row["manual_review"]["decision"] = "accept"
        with tempfile.TemporaryDirectory() as directory:
            packet_path = Path(directory) / "packet.jsonl"
            packet_path.write_text("".join(json.dumps(r) + "\n" for r in packet), encoding="utf-8")
            approval_path = Path(directory) / "approval.json"
            approval_path.write_text(json.dumps({
                "approved": True, "reviewer": "human", "approved_at": "2026-07-11T12:00:00Z",
                "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            self.assertTrue(assert_manual_approval(packet_path, approval_path)["approved"])
            packet_path.write_text(packet_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "changed"):
                assert_manual_approval(packet_path, approval_path)

    def test_approved_candidate_converts_to_training_schema(self) -> None:
        row = _candidate("case-one", "golden_matched", "one")
        row["manual_review"] = {
            "decision": "corrected",
            "corrections": {
                "scores": {
                    "thesis": 1,
                    "contextualization": 1,
                    "evidence": 2,
                    "analysis_reasoning": 1,
                }
            },
        }
        case = candidate_to_case(row, split="train")
        self.assertEqual(case.reference_scores.total, 5)
        self.assertEqual(case.split, "train")
        self.assertTrue(case.labeling.human_reviewed)
        self.assertEqual(case.provenance.generator_name, "external_v5_score_blind")

    def test_external_candidate_cannot_override_planner_metadata(self) -> None:
        task = plan_v5_tasks([_seed()])[0]
        row = _candidate(task.task_id, "golden_matched", "wrong")
        row.update(prompt="untrusted", selection_class="boundary", boundary_type="evidence_1_2")
        normalized = normalize_external_candidate(task, row)
        self.assertEqual(normalized["prompt"], task.prompt)
        self.assertEqual(normalized["selection_class"], task.coverage_class)
        self.assertEqual(normalized["boundary_type"], task.boundary_type)


if __name__ == "__main__":
    unittest.main()
