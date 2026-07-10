"""Focused regression tests for AP Central essay boundary extraction."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.ingest.apc_parser import (
    EssayContaminationError,
    EssayExtractionError,
    detect_essay_contamination,
    parse_apc_text,
    reject_contaminated_essay,
)


COMMENTARY = """AP® United States History 2024 Scoring Commentary
Long Essay Question 2
Note: Student samples are quoted verbatim and may contain spelling and grammatical errors.
Sample: 2A
Thesis Score: 1
Contextualization Score: 1
Evidence Score: 2
Analysis and Reasoning Score: 2
Total Score: 6
A. Thesis/Claim: 1
The response earned 1 point for its defensible claim.
B. Contextualization: 1
The response earned 1 point for relevant context.
C. Evidence: 2
The response earned 2 points for using evidence.
D. Analysis and Reasoning: 2
The response earned 2 points for its reasoning.
Sample: 2B
Thesis Score: 1
Contextualization Score: 1
Evidence Score: 1
Analysis and Reasoning Score: 1
Total Score: 4
A. Thesis/Claim: 1
The response earned 1 point.
Sample: 2C
Thesis Score: 1
Contextualization Score: 0
Evidence Score: 1
Analysis and Reasoning Score: 0
Total Score: 2
A. Thesis/Claim: 1
The response earned 1 point.
"""

AP24_LAYOUT_TEXT = """2024 AP United States History
Question 2: Long Essay Question, Colonial Encounters 6 points
Evaluate the extent to which encounters between distinct communities changed colonial society from 1607 to 1776.
Scoring Guidelines
Additional Notes:
These are rubric notes and are not part of a student response.
Sample 2A 1 of 2
Before the period, several communities established different political and economic traditions. The first response argues that trade changed relations among those communities.
Sample 2A 2 of 2
It supports that argument with two concrete examples and explains why one development mattered more than another.
Sample 2B 1 of 1
The second response presents a defensible claim, describes broader context, and names relevant evidence while offering a limited causal explanation.
Sample 2C 1 of 1
The third response makes a narrow claim and identifies one relevant example, but it does not develop a complete argument.
""" + COMMENTARY


class APCCleanExtractionTests(unittest.TestCase):
    def test_2024_sample_headers_join_multipage_essay(self) -> None:
        samples = parse_apc_text(
            AP24_LAYOUT_TEXT,
            metadata={"year": 2024, "leq_num": 2, "set": 1},
        )
        self.assertEqual([sample.sample_id for sample in samples], ["2A", "2B", "2C"])
        sample_2a = samples[0]
        self.assertIn("Before the period", sample_2a.essay)
        self.assertIn("It supports that argument", sample_2a.essay)
        self.assertNotIn("Sample 2A", sample_2a.essay)
        self.assertEqual(sample_2a.essay_source, "pdf_text")
        self.assertEqual(sample_2a.metadata["extraction_layout"], "sample_page_header")
        self.assertEqual(sample_2a.metadata["parser_confidence"], 0.99)

    def test_missing_pdf_essay_is_not_reconstructed_from_commentary(self) -> None:
        text = """Evaluate the extent to which trade changed colonial society from 1607 to 1776.
Scoring Guidelines only; no student response pages were extracted.
""" + COMMENTARY
        with self.assertRaisesRegex(
            EssayExtractionError,
            "commentary reconstruction is prohibited",
        ):
            parse_apc_text(text, metadata={"year": 2024, "leq_num": 2, "set": 1})

    def test_incomplete_multipage_essay_is_rejected(self) -> None:
        text = AP24_LAYOUT_TEXT.replace(
            "Sample 2A 2 of 2\n"
            "It supports that argument with two concrete examples and explains why one "
            "development mattered more than another.\n",
            "",
        )
        with self.assertRaisesRegex(EssayExtractionError, "sample 2A"):
            parse_apc_text(text, metadata={"year": 2024, "leq_num": 2, "set": 1})

    def test_contaminated_extraction_is_rejected_instead_of_cleaned(self) -> None:
        contaminated = AP24_LAYOUT_TEXT.replace(
            "The second response presents",
            "© 2024 College Board\nThe second response presents",
        )
        with self.assertRaises(EssayContaminationError) as context:
            parse_apc_text(
                contaminated,
                metadata={"year": 2024, "leq_num": 2, "set": 1},
            )
        self.assertEqual(context.exception.sample_id, "2B")
        self.assertIn("copyright_footer", context.exception.markers)

    def test_embedded_commentary_header_cannot_move_the_essay_boundary(self) -> None:
        contaminated = AP24_LAYOUT_TEXT.replace(
            "The second response presents",
            "Scoring Commentary\nThe second response presents",
        )
        with self.assertRaises(EssayContaminationError) as context:
            parse_apc_text(
                contaminated,
                metadata={"year": 2024, "leq_num": 2, "set": 1},
            )
        self.assertEqual(context.exception.sample_id, "2B")
        self.assertIn("scoring_commentary", context.exception.markers)

    def test_detection_covers_every_prohibited_document_text_class(self) -> None:
        cases = {
            "scoring_commentary": "AP United States History Scoring Commentary",
            "commentary_page_header": "Long Essay Question 2\n",
            "ap_page_header": "AP® United States History 2024",
            "sample_score_header": "Sample: 2A\nThesis Score: 1",
            "commentary_row_label": "A. Thesis/Claim: 1",
            "document_page_marker": "Page 1 of 2 2\nA\n",
            "copyright_footer": "© 2024 College Board",
            "commentary_boilerplate": "The response earned 1 point for evidence.",
        }
        for expected_marker, text in cases.items():
            with self.subTest(expected_marker=expected_marker):
                self.assertIn(expected_marker, detect_essay_contamination(text))
                with self.assertRaises(EssayContaminationError):
                    reject_contaminated_essay(text, sample_id="2A")


if __name__ == "__main__":
    unittest.main()
