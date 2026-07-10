from scripts.diagnose_eval_outputs import classify_response, diagnose_rows


VALID_RESPONSE = (
    '{"scores":{"thesis":1,"contextualization":1,"evidence":2,'
    '"analysis_reasoning":1},"total":5,"feedback":{"thesis":"a",'
    '"contextualization":"b","evidence":"c","analysis_reasoning":"d"}}'
)


def test_classifies_valid_and_schema_invalid_json() -> None:
    assert classify_response(
        VALID_RESPONSE,
        structured_output_valid=True,
        token_count=40,
        max_new_tokens=320,
    )[0] == "valid_structured_json"
    assert classify_response(
        '{"scores": {}}',
        structured_output_valid=False,
        token_count=5,
        max_new_tokens=320,
    )[0] == "json_schema_invalid"


def test_classifies_truncated_and_malformed_output() -> None:
    assert classify_response(
        '{"scores":{"thesis":1',
        structured_output_valid=False,
        token_count=320,
        max_new_tokens=320,
    )[0] == "likely_max_token_truncation"
    assert classify_response(
        '{"scores":{"thesis":1',
        structured_output_valid=False,
        token_count=20,
        max_new_tokens=320,
    )[0] == "incomplete_json_below_limit"
    assert classify_response(
        "The essay earns several points.",
        structured_output_valid=False,
        token_count=8,
        max_new_tokens=320,
    )[0] == "malformed_or_non_json"


def test_diagnostic_report_aggregates_without_responses() -> None:
    rows = [
        {"case_id": "valid", "response": VALID_RESPONSE, "structured_output_valid": True},
        {"case_id": "bad", "response": "not json", "structured_output_valid": False},
    ]
    report = diagnose_rows(rows, tokenizer=None, max_new_tokens=320, token_margin=2)
    assert report["total"] == 2
    assert report["categories"] == {
        "malformed_or_non_json": 1,
        "valid_structured_json": 1,
    }
    assert VALID_RESPONSE not in str(report)
    assert "not json" not in str(report)
