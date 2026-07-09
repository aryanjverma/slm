"""Tests for AP Central LEQ PDF parsing and distillation."""

from __future__ import annotations

import unittest
from pathlib import Path

from apush_frq_grader_slm.filters import passes_quality_gate
from apush_frq_grader_slm.ingest.apc_parser import parse_apc_text
from apush_frq_grader_slm.ingest.distill import infer_failure_type, raw_sample_to_frq_case

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "apc" / "ap23_leq2_set1_full.txt"


class APCParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE_PATH.exists():
            cls.full_text = None
            return
        cls.full_text = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_parse_ap23_leq2_set1_samples(self) -> None:
        if self.full_text is None:
            self.skipTest("Missing tests/fixtures/apc/ap23_leq2_set1_full.txt")
        metadata = {"year": 2023, "leq_num": 2, "set": 1}
        samples = parse_apc_text(self.full_text, metadata=metadata)
        self.assertEqual(len(samples), 3)
        ids = {sample.sample_id for sample in samples}
        self.assertEqual(ids, {"2A", "2B", "2C"})

    def test_sample_2a_scores_and_prompt(self) -> None:
        if self.full_text is None:
            self.skipTest("Missing tests/fixtures/apc/ap23_leq2_set1_full.txt")
        samples = parse_apc_text(self.full_text, metadata={"year": 2023, "leq_num": 2, "set": 1})
        sample_2a = next(sample for sample in samples if sample.sample_id == "2A")
        self.assertIn("Evaluate the extent", sample_2a.prompt)
        self.assertEqual(sample_2a.scores["thesis"], 1)
        self.assertEqual(sample_2a.scores["contextualization"], 1)
        self.assertEqual(sample_2a.scores["evidence"], 2)
        self.assertEqual(sample_2a.scores["analysis_reasoning"], 2)
        self.assertEqual(sample_2a.total_score, 6)
        self.assertGreater(len(sample_2a.essay), 200)
        self.assertIn("transatlantic", sample_2a.essay.lower())

    def test_commentary_blocks_extracted(self) -> None:
        if self.full_text is None:
            self.skipTest("Missing tests/fixtures/apc/ap23_leq2_set1_full.txt")
        samples = parse_apc_text(self.full_text, metadata={"year": 2023, "leq_num": 2, "set": 1})
        sample_2a = next(sample for sample in samples if sample.sample_id == "2A")
        self.assertIn("thesis", sample_2a.commentary_by_row)
        self.assertIn("earned 1 point", sample_2a.commentary_by_row["thesis"].lower())

    def test_raw_sample_to_frq_case_passes_quality_gate(self) -> None:
        if self.full_text is None:
            self.skipTest("Missing tests/fixtures/apc/ap23_leq2_set1_full.txt")
        samples = parse_apc_text(self.full_text, metadata={"year": 2023, "leq_num": 2, "set": 1})
        for sample in samples:
            case = raw_sample_to_frq_case(sample)
            ok, reasons = passes_quality_gate(case)
            self.assertTrue(ok, msg=f"{sample.sample_id}: {reasons}")
            self.assertEqual(case.split, "eval")
            self.assertIn("ap_central", case.tags)

    def test_failure_type_inference(self) -> None:
        if self.full_text is None:
            self.skipTest("Missing tests/fixtures/apc/ap23_leq2_set1_full.txt")
        samples = parse_apc_text(self.full_text, metadata={"year": 2023, "leq_num": 2, "set": 1})
        by_id = {sample.sample_id: sample for sample in samples}
        self.assertEqual(infer_failure_type(by_id["2A"]).value, "strong")
        self.assertIn(
            infer_failure_type(by_id["2C"]).value,
            {"weak_thesis", "evidence_list"},
        )


class CatalogTests(unittest.TestCase):
    def test_enumerate_sources_count(self) -> None:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "catalog_ap_sources",
            Path(__file__).resolve().parents[1] / "scripts" / "catalog_ap_sources.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["catalog_ap_sources"] = module
        spec.loader.exec_module(module)
        ap_central = [entry for entry in module.enumerate_sources() if entry["source"] == "ap_central"]
        # 11 years (2015-2025) x LEQ {2,3,4} x sets {1,2}
        self.assertEqual(len(ap_central), 66)


class TomRicheyParserTests(unittest.TestCase):
    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tomrichey" / "tomrichey_2024_leq3_set1.txt"

    def test_parse_tomrichey_samples(self) -> None:
        if not self.FIXTURE_PATH.exists():
            self.skipTest("Missing Tom Richey text fixture")
        from apush_frq_grader_slm.ingest.tomrichey_parser import parse_tomrichey_text

        text = self.FIXTURE_PATH.read_text(encoding="utf-8")
        meta = {
            "year": 2024,
            "leq_num": 3,
            "set": 1,
            "prompt": (
                "Evaluate the extent to which economic development contributed to the growth "
                "of a distinct national culture in the United States in the period 1800-1848."
            ),
        }
        samples = parse_tomrichey_text(text, metadata=meta)
        self.assertEqual(len(samples), 3)
        by_id = {sample.sample_id: sample for sample in samples}
        self.assertEqual(by_id["A"].total_score, 6)
        self.assertEqual(by_id["C"].total_score, 2)
        self.assertEqual(by_id["A"].metadata["provider"], "tom_richey")

    def test_tomrichey_to_frq_case_tags(self) -> None:
        if not self.FIXTURE_PATH.exists():
            self.skipTest("Missing Tom Richey text fixture")
        from apush_frq_grader_slm.ingest.tomrichey_parser import parse_tomrichey_text

        text = self.FIXTURE_PATH.read_text(encoding="utf-8")
        samples = parse_tomrichey_text(text, metadata={"year": 2024, "leq_num": 3, "set": 1})
        case = raw_sample_to_frq_case(samples[0])
        self.assertIn("tom_richey", case.tags)
        self.assertIn("real_eval", case.tags)


class QuizletParserTests(unittest.TestCase):
    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "quizlet" / "485501886.json"

    def test_parse_multi_card_essay(self) -> None:
        if not self.FIXTURE_PATH.exists():
            self.skipTest("Missing Quizlet fixture")
        from apush_frq_grader_slm.ingest.quizlet_parser import load_quizlet_json

        samples = load_quizlet_json(self.FIXTURE_PATH)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("Spanish-American War", sample.prompt)
        self.assertIn("isolationism", sample.essay.lower())
        self.assertEqual(sample.metadata["provider"], "quizlet")

    def test_quizlet_to_frq_case(self) -> None:
        if not self.FIXTURE_PATH.exists():
            self.skipTest("Missing Quizlet fixture")
        from apush_frq_grader_slm.ingest.quizlet_parser import load_quizlet_json

        samples = load_quizlet_json(self.FIXTURE_PATH)
        case = raw_sample_to_frq_case(samples[0])
        ok, reasons = passes_quality_gate(case)
        self.assertTrue(ok, msg=str(reasons))
        self.assertIn("quizlet", case.tags)


class ScoringDedupTests(unittest.TestCase):
    def test_total_to_row_scores(self) -> None:
        from apush_frq_grader_slm.ingest.scoring import total_to_row_scores

        scores = total_to_row_scores(6)
        self.assertEqual(scores["thesis"], 1)
        self.assertEqual(scores["analysis_reasoning"], 2)

    def test_dedup_detects_overlap(self) -> None:
        from apush_frq_grader_slm.ingest.dedup import is_duplicate_essay
        from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores

        essay = (
            "The transatlantic trade network transformed colonial society through new goods "
            "and labor systems across the Atlantic world during the seventeenth century."
        )
        case = FRQCase(
            id="cb-1",
            split="eval",
            prompt="Evaluate the extent to which transatlantic trade changed colonial society.",
            student_response=essay,
            reference_scores=RubricScores(thesis=1, contextualization=1, evidence=2, analysis_reasoning=2),
            reference_feedback=RubricFeedback(
                thesis="x",
                contextualization="x",
                evidence="x",
                analysis_reasoning="x",
            ),
            failure_type=FailureType.STRONG,
            difficulty="strong",
            assistant_response="{}",
        )
        near_duplicate = essay
        self.assertTrue(is_duplicate_essay(near_duplicate, [case], prompt=case.prompt))
        slightly_edited = essay.replace("seventeenth", "17th")
        self.assertTrue(is_duplicate_essay(slightly_edited, [case], prompt=case.prompt))
        self.assertFalse(
            is_duplicate_essay(
                "Railroad expansion and industrial factory growth reshaped urban labor in the Gilded Age.",
                [case],
                prompt="Evaluate the extent to which railroads transformed the American economy.",
            )
        )


class RealEvalMetricsTests(unittest.TestCase):
    def test_score_agreement_exact_match(self) -> None:
        from apush_frq_grader_slm.eval import score_agreement
        from apush_frq_grader_slm.schemas import FRQCase, FailureType, RubricFeedback, RubricScores

        case = FRQCase(
            id="test",
            split="eval",
            prompt="Evaluate ...",
            student_response="The transatlantic trade changed colonial society significantly.",
            reference_scores=RubricScores(thesis=1, contextualization=1, evidence=2, analysis_reasoning=1),
            reference_feedback=RubricFeedback(
                thesis="The thesis argues 'transatlantic trade changed colonial society'.",
                contextualization="Context about transatlantic trade frames the argument.",
                evidence="Examples such as transatlantic trade support the topic.",
                analysis_reasoning="The essay links transatlantic trade to change and continuity.",
            ),
            failure_type=FailureType.STRONG,
            difficulty="strong",
            assistant_response='{"scores":{"thesis":1,"contextualization":1,"evidence":2,"analysis_reasoning":1},"total":5,"feedback":{"thesis":"x","contextualization":"x","evidence":"x","analysis_reasoning":"x"}}',
        )
        agreement = score_agreement(case, case.assistant_response)
        self.assertEqual(agreement["exact_rows"], 4)
        self.assertTrue(agreement["total_exact"])


if __name__ == "__main__":
    unittest.main()
