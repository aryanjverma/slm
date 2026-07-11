from __future__ import annotations

import json

import pytest

from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.eval_v3 import (
    build_run_identity,
    load_resumable_records,
    make_record,
    select_official_split,
)


RESPONSE = (
    '{"scores":{"thesis":1,"contextualization":1,"evidence":2,'
    '"analysis_reasoning":1},"feedback":{"thesis":"a","contextualization":"b",'
    '"evidence":"c","analysis_reasoning":"d"}}'
)


def _identity(model_hash: str = "model-a"):
    return build_run_identity(
        model_name="candidate",
        model_hash=model_hash,
        data_hash="data-a",
        split="set1",
        decoding_settings={"max_new_tokens": 320},
    )


def test_default_split_is_set1_and_set2_requires_explicit_flag() -> None:
    cases = generate_cases(count=2, split="eval", seed=1)
    cases[0].id = "ap_central_2025_leq2_set1_2A"
    cases[1].id = "ap_central_2025_leq2_set2_2A"
    assert [case.id for case in select_official_split(cases)] == [cases[0].id]
    assert [case.id for case in select_official_split(cases, final_evaluation=True)] == [
        cases[1].id
    ]


def test_resume_rejects_incompatible_run_identity(tmp_path) -> None:
    case = generate_cases(count=1, split="eval", seed=2)[0]
    record = make_record(
        case_id=case.id,
        identity=_identity(),
        raw_response=RESPONSE,
        prompt_tokens=100,
        completion_tokens=80,
        finish_reason="balanced_json",
        repetition_detected=False,
    )
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(record.model_dump(mode="json")) + "\n", encoding="utf-8")
    loaded = load_resumable_records(path, identity=_identity(), cases=[case])
    assert loaded[0].normalized_payload["total"] == 5
    with pytest.raises(RuntimeError, match="Incompatible"):
        load_resumable_records(path, identity=_identity("model-b"), cases=[case])


def test_record_preserves_raw_response_while_normalizing_balanced_object() -> None:
    raw = "grader prefix\n" + RESPONSE + "\ntrailing text"
    record = make_record(
        case_id="case",
        identity=_identity(),
        raw_response=raw,
        prompt_tokens=10,
        completion_tokens=20,
        finish_reason="balanced_json",
        repetition_detected=False,
    )
    assert record.raw_response == raw
    assert record.layered_schema_valid
    assert record.normalized_payload["total"] == 5
