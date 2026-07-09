"""Unit tests for the hallucinated-quote gate in filters.py.

These pin the boundary between *grounded quoting* (the agent quotes phrases that
are genuinely in the essay -- must PASS) and *fabricated quotes* (the agent
attributes text to the essay that is not there -- must be FLAGGED). The failing
cases here were distilled from real agent output that the exact-substring check
wrongly rejected as ``hallucinated_quote``.
"""

from __future__ import annotations

import unittest

from apush_frq_grader_slm.filters import contains_hallucination_pattern


class HallucinationCheckTests(unittest.TestCase):
    def test_grounded_quote_with_list_comma_passes(self) -> None:
        # English style puts the list comma INSIDE the closing quote, so the
        # quoted span is "money stuff," while the essay has "money stuff".
        essay = (
            "There were battles in different places. Important men got together. "
            "The essay mentions money stuff without naming anything specific."
        )
        feedback = (
            "The essay offers only vague generalities like 'battles in different places,' "
            "'important men got together,' and 'money stuff,' naming no actual events."
        )
        self.assertFalse(contains_hallucination_pattern(feedback, essay))

    def test_possessive_apostrophe_not_treated_as_quote(self) -> None:
        # A possessive apostrophe (colony's) must not open a spurious quote span
        # whose captured text ("s resentment of ...") is absent from the essay.
        essay = (
            "The colonists resented royal taxes for years. A rebellion soon followed "
            "and the Great Awakening spread new religious ideas across the region."
        )
        feedback = "It notes the colony's resentment of 'royal taxes' before the rebellion began."
        self.assertFalse(contains_hallucination_pattern(feedback, essay))

    def test_prose_between_two_short_quotes_not_captured(self) -> None:
        # Two short quotes (<12 chars each) with connective feedback prose between
        # them must not have that prose captured as a fabricated quote.
        essay = "The essay mentions the economy and that society changed in vague ways."
        feedback = "It refers to 'the economy' and concludes it 'changed society' with no detail."
        self.assertFalse(contains_hallucination_pattern(feedback, essay))

    def test_ellipsis_elided_quote_passes(self) -> None:
        # An elided quote joins two real essay spans with an ellipsis.
        essay = (
            "Spain had already built a huge empire in the Americas that was based on silver "
            "mined by forced Indigenous labor across the sixteenth century."
        )
        feedback = (
            "The essay observes that 'Spain had already built a huge empire... based on silver'."
        )
        self.assertFalse(contains_hallucination_pattern(feedback, essay))

    def test_exact_grounded_quote_still_passes(self) -> None:
        essay = "The market revolution linked distant farms to growing cities through canals."
        feedback = 'It cites how the revolution "linked distant farms to growing cities".'
        self.assertFalse(contains_hallucination_pattern(feedback, essay))

    def test_fabricated_double_quote_is_flagged(self) -> None:
        essay = "The essay discusses trade and religion in the colonies in only vague terms."
        feedback = 'The essay claims "the Stamp Act of 1765 provoked violent riots" as evidence.'
        self.assertTrue(contains_hallucination_pattern(feedback, essay))

    def test_fabricated_single_quote_is_flagged(self) -> None:
        essay = "The essay discusses trade and religion in the colonies in only vague terms."
        feedback = "It asserts 'the Boston Massacre killed five colonists' with no support."
        self.assertTrue(contains_hallucination_pattern(feedback, essay))

    def test_according_to_document_still_flagged(self) -> None:
        essay = "The essay talks generally about colonial society."
        feedback = "According to document A, the colonists rebelled against taxes."
        self.assertTrue(contains_hallucination_pattern(feedback, essay))


if __name__ == "__main__":
    unittest.main()
