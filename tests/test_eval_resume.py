from __future__ import annotations

import pytest

from apush_frq_grader_slm.baselines import ReferenceGrader
from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.eval import evaluate_adapter
from apush_frq_grader_slm.io import write_jsonl
from scripts.eval_hf_model import (
    hydrate_cb_provenance,
    load_existing_results,
    select_pending_cases,
)


def test_resume_loads_completed_ids_and_selects_only_missing_cases(tmp_path) -> None:
    cases = generate_cases(count=3, split="eval", seed=21)
    completed = evaluate_adapter(cases[:2], ReferenceGrader())
    results_path = tmp_path / "official_dev_real_results.jsonl"
    write_jsonl(results_path, completed)

    loaded = load_existing_results(results_path, cases, "apush_grader_reference")
    pending = select_pending_cases(cases, loaded)

    assert [result.case_id for result in loaded] == [case.id for case in cases[:2]]
    assert [case.id for case in pending] == [cases[2].id]


def test_resume_rejects_duplicate_saved_case_ids(tmp_path) -> None:
    cases = generate_cases(count=2, split="eval", seed=22)
    result = evaluate_adapter(cases[:1], ReferenceGrader())[0]
    results_path = tmp_path / "official_dev_real_results.jsonl"
    write_jsonl(results_path, [result, result])

    with pytest.raises(RuntimeError, match="duplicate case ID"):
        load_existing_results(results_path, cases, "apush_grader_reference")


def test_legacy_cb_id_hydrates_rubric_slice_metadata() -> None:
    case = generate_cases(count=1, split="eval", seed=23)[0]
    case.id = "ap_central_2023_leq2_set1_2A"
    hydrated = hydrate_cb_provenance(case)
    assert hydrated.provenance.source_type == "college_board"
    assert hydrated.provenance.year == 2023
    assert hydrated.provenance.set_number == 1
    assert hydrated.provenance.rubric_version == "2023_leq"
