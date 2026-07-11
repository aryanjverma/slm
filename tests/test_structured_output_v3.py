from __future__ import annotations

from apush_frq_grader_slm.structured_output_v3 import (
    first_complete_json_end,
    has_repetition,
    normalize_grade_output,
    pending_score_criterion,
    trim_after_first_json,
)


VALID = (
    '{"scores":{"thesis":1,"contextualization":0,"evidence":2,'
    '"analysis_reasoning":1},"feedback":{"thesis":"The claim is defensible.",'
    '"contextualization":"The essay gives no wider setting.",'
    '"evidence":"The canal example supports the claim.",'
    '"analysis_reasoning":"The essay explains change over time."}}'
)


def test_computes_total_without_changing_scores() -> None:
    result = normalize_grade_output(VALID)
    assert result.raw_valid
    assert result.layered_valid
    assert result.normalized_payload is not None
    assert result.normalized_payload["total"] == 4
    assert result.normalized_payload["scores"]["evidence"] == 2
    assert result.normalization_actions == ("computed_total",)


def test_repairs_representation_but_rejects_score_range() -> None:
    wrapped = "Result:\n```json\n" + VALID.replace('"thesis":1', '"thesis":"1"', 1) + "\n```"
    repaired = normalize_grade_output(wrapped)
    assert repaired.layered_valid
    assert "extracted_balanced_object" in repaired.normalization_actions
    assert "coerced_integral_string:thesis" in repaired.normalization_actions

    invalid = normalize_grade_output(VALID.replace('"evidence":2', '"evidence":3', 1))
    assert not invalid.layered_valid
    assert "out_of_range_score:evidence" in invalid.errors


def test_balanced_extraction_handles_braces_in_strings_and_stops_early() -> None:
    response = VALID.replace("The claim is defensible.", "The claim uses {change} defensibly.")
    response += " trailing material"
    end = first_complete_json_end(response)
    assert end is not None
    trimmed, changed = trim_after_first_json(response)
    assert changed
    assert trimmed.endswith("}")
    assert normalize_grade_output(trimmed).layered_valid


def test_repetition_detection() -> None:
    phrase = "the essay repeats the same historical claim without further analysis "
    assert has_repetition(phrase * 4)
    assert not has_repetition(VALID)


def test_multiple_grade_objects_are_ambiguous() -> None:
    result = normalize_grade_output(VALID + "\n" + VALID)
    assert not result.layered_valid
    assert result.errors == ("ambiguous_multiple_grade_objects",)


def test_score_enum_constraint_never_applies_to_feedback_fields() -> None:
    assert pending_score_criterion('{"scores":{"thesis":') == "thesis"
    assert pending_score_criterion('{"scores":{"thesis":1},"feedback":{"thesis":') is None
