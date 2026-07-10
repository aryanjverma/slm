"""Tests for original prompt coverage, family splits, and leakage validation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.prompt_catalog import (
    DEFAULT_REVIEW_SIMILARITY_THRESHOLD,
    DEFAULT_SPLIT_SEED,
    ORIGINAL_PROMPT_FAMILIES,
    PromptCatalogEntry,
    PromptFormat,
    PromptSourceType,
    PromptSplit,
    ReasoningSkill,
    assign_family_splits,
    build_default_prompt_catalog,
    prompt_token_similarity,
    split_family_counts,
    validate_prompt_catalog,
)


def _entry(family_index: int, split: PromptSplit, **updates: object) -> PromptCatalogEntry:
    data = ORIGINAL_PROMPT_FAMILIES[family_index].model_dump(mode="python")
    data.update(updates)
    if "prompt_text" in updates and "source_prompt_text" not in updates:
        data["source_prompt_text"] = updates["prompt_text"]
    data["split"] = split
    return PromptCatalogEntry.model_validate(data)


def _load_build_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_prompt_catalog.py"
    spec = importlib.util.spec_from_file_location("build_prompt_catalog", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_prompt_catalog"] = module
    spec.loader.exec_module(module)
    return module


class PromptCatalogContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = build_default_prompt_catalog()

    def test_contains_sixty_unique_original_families(self) -> None:
        self.assertEqual(len(ORIGINAL_PROMPT_FAMILIES), 60)
        self.assertEqual(
            len({prompt.prompt_family_id for prompt in ORIGINAL_PROMPT_FAMILIES}), 60
        )
        self.assertTrue(
            all(prompt.source_type == PromptSourceType.ORIGINAL for prompt in self.entries)
        )
        self.assertTrue(all(prompt.parent_prompt is None for prompt in self.entries))
        self.assertTrue(
            all(prompt.source_prompt_text == prompt.prompt_text for prompt in self.entries)
        )

    def test_every_period_has_every_reasoning_skill(self) -> None:
        for period in range(2, 10):
            skills = {
                prompt.reasoning_skill for prompt in self.entries if prompt.period == period
            }
            self.assertEqual(skills, set(ReasoningSkill), f"period {period}")

    def test_all_required_metadata_is_populated(self) -> None:
        for prompt in self.entries:
            self.assertEqual(prompt.source_year, 2026)
            self.assertLessEqual(prompt.date_range.start_year, prompt.date_range.end_year)
            self.assertGreaterEqual(len(prompt.valid_evidence), 3)
            self.assertGreaterEqual(len(prompt.common_misconceptions), 2)
            self.assertTrue(prompt.topic.strip())
            self.assertTrue(prompt.catalog_version)
            self.assertEqual(prompt.split_seed, DEFAULT_SPLIT_SEED)

    def test_each_split_retains_period_and_skill_coverage(self) -> None:
        for split in PromptSplit:
            rows = [prompt for prompt in self.entries if prompt.split == split]
            self.assertEqual({prompt.period for prompt in rows}, set(range(2, 10)))
            self.assertEqual({prompt.reasoning_skill for prompt in rows}, set(ReasoningSkill))

    def test_small_2027_format_slice_is_challenge_only(self) -> None:
        forward = [prompt for prompt in self.entries if prompt.is_2027_challenge]
        self.assertEqual(len(forward), 3)
        self.assertEqual({prompt.reasoning_skill for prompt in forward}, set(ReasoningSkill))
        self.assertTrue(
            all(prompt.format_version == PromptFormat.MAY_2027_BROAD for prompt in forward)
        )
        self.assertTrue(
            all(prompt.split == PromptSplit.SYNTHETIC_CHALLENGE for prompt in forward)
        )


class FamilySplitTests(unittest.TestCase):
    def test_default_split_is_exact_70_15_15(self) -> None:
        entries = assign_family_splits(ORIGINAL_PROMPT_FAMILIES)
        counts = Counter(prompt.split for prompt in entries)
        self.assertEqual(
            counts,
            Counter(
                {
                    PromptSplit.TRAIN: 42,
                    PromptSplit.SYNTHETIC_DEV: 9,
                    PromptSplit.SYNTHETIC_CHALLENGE: 9,
                }
            ),
        )
        self.assertEqual(
            split_family_counts(60),
            {
                PromptSplit.TRAIN: 42,
                PromptSplit.SYNTHETIC_DEV: 9,
                PromptSplit.SYNTHETIC_CHALLENGE: 9,
            },
        )

    def test_split_assignment_is_input_order_independent(self) -> None:
        forward = assign_family_splits(ORIGINAL_PROMPT_FAMILIES)
        reverse = assign_family_splits(tuple(reversed(ORIGINAL_PROMPT_FAMILIES)))
        forward_map = {prompt.prompt_family_id: prompt.split for prompt in forward}
        reverse_map = {prompt.prompt_family_id: prompt.split for prompt in reverse}
        self.assertEqual(forward_map, reverse_map)

    def test_variants_in_one_family_receive_one_split(self) -> None:
        base = ORIGINAL_PROMPT_FAMILIES[0]
        variant = base.model_copy(
            update={
                "prompt_id": f"{base.prompt_id}_variant",
                "prompt_text": f"{base.prompt_text} Focus your response on regional variation.",
                "source_prompt_text": (
                    f"{base.prompt_text} Focus your response on regional variation."
                ),
            }
        )
        entries = assign_family_splits((*ORIGINAL_PROMPT_FAMILIES, variant))
        family_splits = {
            prompt.split
            for prompt in entries
            if prompt.prompt_family_id == base.prompt_family_id
        }
        self.assertEqual(len(family_splits), 1)


class CatalogValidationTests(unittest.TestCase):
    def test_default_catalog_passes_strict_validation(self) -> None:
        report = validate_prompt_catalog(assign_family_splits(ORIGINAL_PROMPT_FAMILIES))
        self.assertTrue(report.is_valid)
        self.assertLess(report.max_cross_split_similarity, DEFAULT_REVIEW_SIMILARITY_THRESHOLD)

    def test_detects_cross_split_near_paraphrase(self) -> None:
        left_text = (
            "Develop an argument explaining the major causes of industrial labor conflict "
            "in the United States from 1865 to 1898."
        )
        right_text = (
            "Develop an argument explaining the important causes of industrial labor conflict "
            "in the United States from 1865 to 1898."
        )
        entries = [
            _entry(0, PromptSplit.TRAIN, prompt_text=left_text),
            _entry(1, PromptSplit.SYNTHETIC_DEV, prompt_text=right_text),
        ]
        report = validate_prompt_catalog(entries, enforce_catalog_requirements=False)
        self.assertIn("cross_split_near_paraphrase", {issue.code for issue in report.issues})

    def test_moderate_similarity_requires_explicit_review(self) -> None:
        left_text = (
            "Develop an argument explaining causes involving trade, labor, migration, and "
            "politics in the United States from 1800 to 1848."
        )
        right_text = (
            "Develop an argument explaining causes involving trade, labor, banking, and "
            "religion in the United States from 1800 to 1848."
        )
        entries = [
            _entry(0, PromptSplit.TRAIN, prompt_text=left_text),
            _entry(1, PromptSplit.SYNTHETIC_DEV, prompt_text=right_text),
        ]
        similarity = prompt_token_similarity(left_text, right_text)
        self.assertGreater(similarity, 0.30)
        self.assertLess(similarity, 0.90)
        report = validate_prompt_catalog(
            entries,
            hard_similarity_threshold=0.90,
            review_similarity_threshold=0.30,
            enforce_catalog_requirements=False,
        )
        self.assertIn(
            "manual_similarity_review_required", {issue.code for issue in report.issues}
        )

        reviewed = validate_prompt_catalog(
            entries,
            hard_similarity_threshold=0.90,
            review_similarity_threshold=0.30,
            reviewed_pairs=[
                (entries[0].prompt_family_id, entries[1].prompt_family_id)
            ],
            enforce_catalog_requirements=False,
        )
        self.assertNotIn(
            "manual_similarity_review_required", {issue.code for issue in reviewed.issues}
        )

    def test_detects_topic_and_date_overlap_across_splits(self) -> None:
        left = _entry(0, PromptSplit.TRAIN)
        right = _entry(1, PromptSplit.SYNTHETIC_DEV, topic=left.topic)
        report = validate_prompt_catalog(
            [left, right],
            hard_similarity_threshold=0.99,
            review_similarity_threshold=0.98,
            enforce_catalog_requirements=False,
        )
        self.assertIn(
            "cross_split_topic_date_overlap", {issue.code for issue in report.issues}
        )

    def test_detects_protected_holdout_prompt_leakage(self) -> None:
        entry = _entry(0, PromptSplit.TRAIN)
        report = validate_prompt_catalog(
            [entry],
            protected_prompts=[entry.prompt_text],
            enforce_catalog_requirements=False,
        )
        self.assertIn("protected_prompt_leakage", {issue.code for issue in report.issues})

    def test_detects_family_split_leakage(self) -> None:
        left = _entry(0, PromptSplit.TRAIN)
        right = _entry(
            0,
            PromptSplit.SYNTHETIC_CHALLENGE,
            prompt_id=f"{left.prompt_id}_second",
            prompt_text=f"{left.prompt_text} Address variation among mainland colonies.",
        )
        report = validate_prompt_catalog([left, right], enforce_catalog_requirements=False)
        self.assertIn("family_split_leakage", {issue.code for issue in report.issues})


class BuildPromptCatalogScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_build_script()

    def test_build_script_writes_valid_deterministic_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt_catalog.jsonl"
            report = self.script.build_catalog_file(output)
            rows = read_jsonl(output)

        self.assertTrue(report.is_valid)
        self.assertEqual(len(rows), 60)
        self.assertEqual(
            [row["prompt_id"] for row in rows], sorted(row["prompt_id"] for row in rows)
        )
        self.assertEqual(
            Counter(row["split"] for row in rows),
            {"train": 42, "synthetic_dev": 9, "synthetic_challenge": 9},
        )
        self.assertTrue(all(row["source_type"] == "original" for row in rows))

    def test_check_only_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prompt_catalog.jsonl"
            report = self.script.build_catalog_file(output, check_only=True)
            self.assertTrue(report.is_valid)
            self.assertFalse(output.exists())

    def test_protected_jsonl_blocks_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected = root / "golden.jsonl"
            protected.write_text(
                json.dumps({"prompt": ORIGINAL_PROMPT_FAMILIES[0].prompt_text}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "protected_prompt_leakage"):
                self.script.build_catalog_file(
                    root / "catalog.jsonl", protected_prompt_paths=[protected]
                )


if __name__ == "__main__":
    unittest.main()
