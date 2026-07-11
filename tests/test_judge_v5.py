"""Tests for feature-based v5 blind judging."""

from __future__ import annotations

import inspect
import unittest

from apush_frq_grader_slm.filters import feedback_references_essay
from apush_frq_grader_slm.judge_v5 import (
    AUTH_REVIEWER_IDS,
    RUBRIC_READER_IDS,
    judge_essay,
)
from apush_frq_grader_slm.prompts_v5 import V5_RUBRIC_TEXT
from apush_frq_grader_slm.rubric import CRITERIA

PROMPT = (
    "Evaluate the extent to which the market revolution changed American society "
    "from 1800 to 1848."
)

STRONG_ESSAY = """
The market revolution reshaped American society to a great extent between 1800 and 1848
because canals, factories, and partisan politics pulled households into commercial networks.
Before this surge, many families still relied on local exchange after the Revolutionary era,
so broader Atlantic trade and early national banking set the stage for later change.

The Erie Canal linked western farms to eastern markets and lowered shipping costs, which
supported the claim that commercial ties remade daily life. Textile mills at Lowell hired
young women for wage labor, and this evidence shows how industrial work altered gender roles
and family economies. Expanding suffrage for white men and fierce debates over slavery also
intensified arguments about national identity.

These developments caused uneven outcomes: northern cities grew while southern plantation
slavery deepened. Although transportation improvements continued older patterns of westward
expansion, the scale of wage labor and sectional politics changed society in lasting ways.
However, Native dispossession and regional differences meant the transformation was complex
rather than uniform.
""".strip()

WEAK_ESSAY = "history stuff happened and things changed i guess."

EMPTY_ESSAY = ""


class JudgeV5Tests(unittest.TestCase):
    def test_strong_essay_scores_higher_than_weak_or_empty(self) -> None:
        strong = judge_essay(PROMPT, STRONG_ESSAY, task_id="v5-strong")
        weak = judge_essay(PROMPT, WEAK_ESSAY, task_id="v5-weak")
        empty = judge_essay(PROMPT, EMPTY_ESSAY, task_id="v5-empty")
        strong_total = sum(strong["resolved_grade"]["scores"][c] for c in CRITERIA)
        weak_total = sum(weak["resolved_grade"]["scores"][c] for c in CRITERIA)
        empty_total = sum(empty["resolved_grade"]["scores"][c] for c in CRITERIA)
        self.assertGreater(strong_total, weak_total)
        self.assertGreater(strong_total, empty_total)
        self.assertGreaterEqual(strong_total, 3)
        self.assertEqual(empty_total, 0)
        self.assertFalse(empty["fact_check"]["passed"])
        self.assertTrue(strong["fact_check"]["passed"])

    def test_disagreement_or_low_confidence_marks_adjudicated(self) -> None:
        # Short borderline prose produces low reader confidence -> adjudication.
        borderline = (
            "Markets changed some. Canals existed. People moved west. Slavery stayed. "
            "It was different than before the war maybe."
        )
        judged = judge_essay(PROMPT, borderline, task_id="v5-border")
        reviews = judged["rubric_reviews"]
        signatures = {tuple(r["scores"][c] for c in CRITERIA) for r in reviews}
        low_conf = any(float(r["confidence"]) < 0.85 for r in reviews)
        self.assertTrue(low_conf or len(signatures) > 1)
        self.assertTrue(judged["resolved_grade"]["adjudicated"])

        # Strong clear essays can avoid adjudication when readers agree at high confidence.
        strong = judge_essay(PROMPT, STRONG_ESSAY, task_id="v5-strong-adj")
        strong_sigs = {
            tuple(r["scores"][c] for c in CRITERIA) for r in strong["rubric_reviews"]
        }
        strong_low = any(float(r["confidence"]) < 0.85 for r in strong["rubric_reviews"])
        self.assertEqual(
            strong["resolved_grade"]["adjudicated"],
            bool(len(strong_sigs) > 1 or strong_low),
        )

        # Authenticity disagreement should append a third review.
        polished = (
            "In a carefully structured analysis, one may observe that commercial integration "
            "fundamentally reconstituted social relations throughout the early republic. "
            "Moreover, infrastructural modernization facilitated market participation among "
            "formerly isolated agrarian communities, thereby reconstituting political culture "
            "with remarkable consistency across regions and decades of development. "
            "Furthermore, the historiography of market formation demonstrates an elegant "
            "synthesis of demographic, political, and cultural explanatory frameworks."
        )
        auth = judge_essay(PROMPT, polished, task_id="v5-auth-polished")
        decisions = [
            bool(r["student_like"]) and bool(r["timed_ap_consistent"])
            for r in auth["authenticity_reviews"][:2]
        ]
        if decisions[0] != decisions[1]:
            self.assertEqual(len(auth["authenticity_reviews"]), 3)
            self.assertEqual(auth["authenticity_reviews"][2]["reviewer_id"], "auth-c")

    def test_feedback_references_essay(self) -> None:
        judged = judge_essay(PROMPT, STRONG_ESSAY, task_id="v5-fb")
        feedback = judged["resolved_grade"]["feedback"]
        for criterion in CRITERIA:
            self.assertTrue(
                feedback_references_essay(feedback[criterion], STRONG_ESSAY),
                msg=f"{criterion} feedback not grounded: {feedback[criterion]!r}",
            )

    def test_output_schema_keys_present(self) -> None:
        judged = judge_essay(PROMPT, STRONG_ESSAY, task_id="v5-0001")
        for key in (
            "task_id",
            "student_response",
            "authenticity_reviews",
            "rubric_reviews",
            "resolved_grade",
            "fact_check",
        ):
            self.assertIn(key, judged)
        self.assertNotIn("distribution_match", judged)  # optional; assembler recomputes
        self.assertEqual(judged["task_id"], "v5-0001")
        self.assertEqual(judged["student_response"], STRONG_ESSAY)
        self.assertGreaterEqual(len(judged["authenticity_reviews"]), 2)
        self.assertEqual(len(judged["rubric_reviews"]), 3)
        self.assertEqual(
            [r["reader_id"] for r in judged["rubric_reviews"]], list(RUBRIC_READER_IDS)
        )
        self.assertTrue(
            {r["reviewer_id"] for r in judged["authenticity_reviews"]}.issubset(
                set(AUTH_REVIEWER_IDS)
            )
        )
        scores = judged["resolved_grade"]["scores"]
        feedback = judged["resolved_grade"]["feedback"]
        for criterion in CRITERIA:
            self.assertIn(criterion, scores)
            self.assertIn(criterion, feedback)
        self.assertIn("adjudicated", judged["resolved_grade"])
        self.assertEqual(judged["fact_check"]["checker_id"], "facts-a")
        self.assertIn("passed", judged["fact_check"])
        # Blind contract: no planner / persona metadata on the judged row.
        for forbidden in (
            "capability_profile",
            "style_seed_id",
            "style_excerpt",
            "coverage_class",
            "boundary_type",
            "contrast_pair_id",
            "persona",
        ):
            self.assertNotIn(forbidden, judged)

    def test_readers_do_not_require_capability_profiles(self) -> None:
        signature = inspect.signature(judge_essay)
        self.assertNotIn("capability_profile", signature.parameters)
        self.assertNotIn("composition_profile", signature.parameters)
        self.assertNotIn("style_excerpt", signature.parameters)
        # Callable with only prompt, essay, and task_id.
        judged = judge_essay(PROMPT, STRONG_ESSAY, task_id="v5-no-persona")
        self.assertEqual(len(judged["rubric_reviews"]), 3)
        self.assertIn("thesis", V5_RUBRIC_TEXT.lower())

    def test_anachronistic_year_fails_fact_check(self) -> None:
        bad = (
            "The market revolution changed society because the internet in 1999 "
            "and smartphones in 2015 made canals irrelevant before 1848."
        )
        judged = judge_essay(PROMPT, bad, task_id="v5-anachron")
        self.assertFalse(judged["fact_check"]["passed"])

    def test_mattering_stub_fails_authenticity_majority(self) -> None:
        essay = """
i think markets remade daily life because canals and factories pulled households in.
Before that, local exchange was still common after the Revolution. The Erie Canal
linked farms to eastern buyers and this shows commercial ties growing. Lowell mills
hired young women for wages. Alger Hiss mattering in class notes somehow got mixed
into my outline even though the prompt is about 1800 to 1848. Slavery deepened in
the south while northern cities grew. Kinda uneven overall and hard under the clock.
""".strip()
        judged = judge_essay(PROMPT, essay, task_id="v5-auth-mattering")
        decisions = [
            bool(r["student_like"]) and bool(r["timed_ap_consistent"])
            for r in judged["authenticity_reviews"]
        ]
        self.assertTrue(all(not r["student_like"] for r in judged["authenticity_reviews"]))
        self.assertLessEqual(sum(decisions), len(decisions) // 2)

    def test_informal_student_essay_can_pass_authenticity(self) -> None:
        essay = """
i think the market revolution changed American society a lot between 1800 and 1848
because canals and factories pulled people into bigger markets. Before that, families
wasnt as tied into national trade after the Revolutionary period, so earlier local
exchange set the stage.

The Erie Canal helped western farms sell crops east and this shows how shipping costs
dropped. Lowell mills hired young women and kinda remade family economies. Expanding
suffrage for white men and fights over slavery also shifted politics.

These things caused uneven outcomes. Northern cities grew while southern plantation
slavery got worse. Although roads and canals continued older westward moves, the scale
of wage labor changed society. Not everything was neat under the clock but the claim
still holds.
""".strip()
        judged = judge_essay(PROMPT, essay, task_id="v5-auth-informal")
        decisions = [
            bool(r["student_like"]) and bool(r["timed_ap_consistent"])
            for r in judged["authenticity_reviews"]
        ]
        self.assertGreaterEqual(sum(decisions), 2)

    def test_repeated_prompt_clause_paste_fails_authenticity(self) -> None:
        clause = (
            "the market revolution changed American society from 1800 to 1848 "
            "in lasting commercial ways"
        )
        essay = (
            f"{clause.capitalize()}. Also, {clause}. Finally, {clause}. "
            "Canals and mills show some change but the repeated prompt wording "
            "makes this feel pasted rather than argued under timed conditions. "
            "i wasnt sure what else to add about Lowell or the Erie Canal here."
        )
        judged = judge_essay(PROMPT, essay, task_id="v5-auth-repeat")
        self.assertTrue(all(not r["student_like"] for r in judged["authenticity_reviews"]))
        decisions = [
            bool(r["student_like"]) and bool(r["timed_ap_consistent"])
            for r in judged["authenticity_reviews"]
        ]
        self.assertLessEqual(sum(decisions), len(decisions) // 2)


if __name__ == "__main__":
    unittest.main()
