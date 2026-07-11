"""Tests for v5 semantic fact cards, adapted prompts, exemptions, and distribution match."""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.adapted_prompts_v5 import (
    adapt_official_prompt,
    attach_adapted_prompts_to_seeds,
    build_adapted_prompt_family_row,
    year_tokens,
)
from apush_frq_grader_slm.dataset_v5 import (
    annotate_distribution_match,
    compute_distribution_match,
    overlap_reasons,
)
from apush_frq_grader_slm.fact_cards_v5 import (
    CONCEPT_MAX_CHARS,
    chapter_to_semantic_cards,
    default_allowed_overlap_phrases,
    has_eight_gram_overlap,
    kb_to_semantic_cards,
    rewrite_source_sentence,
)
from apush_frq_grader_slm.schemas import FRQCase, RubricFeedback, RubricScores


def _chapter() -> dict:
    return {
        "id": "amsco_ch01",
        "chapter": 1,
        "period": 1,
        "key_facts": [
            "In 1494, Spain and Portugal moved the pope's line a few degrees to the west "
            "and signed an agreement called the Treaty of Tordesillas.",
            "He persuaded the king to institute the New Laws of 1542.",
        ],
        "context_hooks": [
            "The original discovery, exploration, and settlement of North and South America "
            "occurred at least 10,000 years before Christopher Columbus was born.",
        ],
        "evidence_bank": [
            "Columbian Exchange",
            "Treaty of Tordesillas (1494)",
            "encomienda system",
        ],
    }


def _golden_case(
    case_id: str,
    *,
    scores: dict[str, int],
    essay: str,
) -> FRQCase:
    feedback = RubricFeedback(
        thesis="The response states a claim about change.",
        contextualization="The opening places the argument in context.",
        evidence="The response uses examples to support the claim.",
        analysis_reasoning="The response organizes explanation through causation.",
    )
    return FRQCase.model_validate(
        {
            "id": case_id,
            "split": "eval",
            "prompt": "Evaluate historical change in the period from 1800 to 1848.",
            "student_response": essay,
            "reference_scores": RubricScores.model_validate(scores),
            "reference_feedback": feedback,
            "failure_type": "strong",
            "difficulty": "borderline",
            "assistant_response": "{}",
            "tags": ["golden"],
        }
    )


class FactCardsV5Tests(unittest.TestCase):
    def test_fact_cards_are_paraphrases_without_eight_gram_overlap(self) -> None:
        chapter = _chapter()
        cards = chapter_to_semantic_cards(chapter)
        self.assertGreaterEqual(len(cards), 3)
        sources = list(chapter["key_facts"]) + list(chapter["context_hooks"])
        for card in cards:
            self.assertEqual(card["chapter_id"], "amsco_ch01")
            self.assertEqual(card["period"], 1)
            self.assertEqual(card["source_kind"], "semantic_rewrite")
            self.assertLessEqual(len(card["concept"]), CONCEPT_MAX_CHARS)
            for source in sources:
                self.assertFalse(
                    has_eight_gram_overlap(card["concept"], source),
                    msg=f"concept copies source: {card['concept']!r} / {source!r}",
                )
            # Direct rewrite helper must also break long source spans.
            for source in chapter["key_facts"]:
                rewritten = rewrite_source_sentence(source, period=1)
                self.assertFalse(has_eight_gram_overlap(rewritten, source))

    def test_kb_cards_cover_multiple_chapters(self) -> None:
        cards = kb_to_semantic_cards([_chapter(), {**_chapter(), "id": "amsco_ch02", "chapter": 2}])
        self.assertEqual({card["chapter_id"] for card in cards}, {"amsco_ch01", "amsco_ch02"})


class AdaptedPromptsV5Tests(unittest.TestCase):
    def test_adapted_prompts_differ_but_keep_year_tokens(self) -> None:
        official = (
            "Evaluate the extent to which the growth of transatlantic trade changed "
            "British North American colonial society from 1607 to 1776."
        )
        adapted = adapt_official_prompt(official, "extent", count=3)
        self.assertEqual(len(adapted), 3)
        years = year_tokens(official)
        self.assertEqual(years, ["1607", "1776"])
        for prompt in adapted:
            self.assertNotEqual(prompt.strip().lower(), official.strip().lower())
            for year in years:
                self.assertIn(year, prompt)

    def test_family_and_seed_wiring(self) -> None:
        family = {
            "prompt_family_id": "cb_demo",
            "prompt": (
                "Evaluate how sectional tensions shaped United States society "
                "from 1800 to 1848."
            ),
            "reasoning_skill": "causation",
            "period": 4,
        }
        family_row = build_adapted_prompt_family_row(family, count=2)
        self.assertEqual(len(family_row["adapted_prompts"]), 2)
        seeds = attach_adapted_prompts_to_seeds(
            [{"seed_id": "s1", "prompt_family_id": "cb_demo", "prompt": family["prompt"]}],
            [family_row],
        )
        self.assertEqual(seeds[0]["adapted_prompts"], family_row["adapted_prompts"])


class OverlapExemptionTests(unittest.TestCase):
    def test_allowed_phrases_suppress_name_date_false_positives(self) -> None:
        shared = (
            "treaty of tordesillas spain and portugal divided new world claims along 1494"
        )
        # Shared span is historical names/dates only; unique padding avoids near-dup Jaccard.
        source = (
            f"textbook note about {shared} with additional source only tokens "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        )
        essay = (
            f"student timed essay discussing {shared} then unique student material "
            "one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        )
        self.assertIn("verbatim_eight_word_overlap", overlap_reasons(essay, [source]))
        allowed = default_allowed_overlap_phrases(
            kb=[_chapter()],
            fact_cards=chapter_to_semantic_cards(_chapter()),
        )
        # Explicit long exemption covering the shared historical span.
        allowed = list(allowed) + [shared]
        self.assertEqual(overlap_reasons(essay, [source], allowed_phrases=allowed), [])


class DistributionMatchTests(unittest.TestCase):
    def test_annotate_distribution_match_uses_score_vector_membership(self) -> None:
        essay = (
            "i think markets changed a lot because canals linked farms to cities and "
            "also banks made credit easier but it wasnt equal for everyone in the west. "
            "factories grew too and wage work spread though many people still farmed. "
            "politics also shifted when more white men voted and parties argued about banks.\n\n"
            "overall the economy remade daily life from 1800 to 1848 in uneven ways."
        )
        scores = {
            "thesis": 1,
            "contextualization": 1,
            "evidence": 1,
            "analysis_reasoning": 1,
        }
        golden = [
            _golden_case("g1", scores=scores, essay=essay),
            _golden_case(
                "g2",
                scores={
                    "thesis": 1,
                    "contextualization": 0,
                    "evidence": 2,
                    "analysis_reasoning": 1,
                },
                essay=essay + " Extra sentence about tariffs and roads.",
            ),
        ]
        matching = {
            "task_id": "c1",
            "student_response": essay,
            "resolved_grade": {"scores": scores},
        }
        missing_vector = {
            "task_id": "c2",
            "student_response": essay,
            "resolved_grade": {
                "scores": {
                    "thesis": 0,
                    "contextualization": 0,
                    "evidence": 0,
                    "analysis_reasoning": 0,
                }
            },
        }
        match = compute_distribution_match(matching, golden)
        self.assertTrue(match["recomputed"])
        self.assertTrue(match["score_vector_in_golden"])
        self.assertTrue(match["passed"])

        miss = compute_distribution_match(missing_vector, golden)
        self.assertFalse(miss["score_vector_in_golden"])
        self.assertFalse(miss["passed"])

        annotated = annotate_distribution_match([matching, missing_vector], golden)
        self.assertTrue(annotated[0]["distribution_match"]["passed"])
        self.assertFalse(annotated[1]["distribution_match"]["passed"])
        # External proposal is overwritten by recomputation.
        trusted = dict(missing_vector)
        trusted["distribution_match"] = {"passed": True, "from_external": True}
        recomputed = annotate_distribution_match([trusted], golden)[0]
        self.assertTrue(recomputed["distribution_match"]["recomputed"])
        self.assertFalse(recomputed["distribution_match"]["passed"])


if __name__ == "__main__":
    unittest.main()
