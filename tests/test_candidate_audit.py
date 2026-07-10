from __future__ import annotations

from apush_frq_grader_slm.candidate_audit import audit_candidates
from apush_frq_grader_slm.schemas import FailureType
from apush_frq_grader_slm.synth_realistic import GenTask, StudentPersona


def _task(task_id: str) -> GenTask:
    return GenTask(
        task_id=task_id,
        seed_id="family_a",
        prompt="Evaluate historical change from 1800 to 1848.",
        seed_scores=None,
        target_scores={
            "thesis": 1,
            "contextualization": 1,
            "evidence": 2,
            "analysis_reasoning": 1,
        },
        target_total=5,
        failure_type=FailureType.STRONG.value,
        length_band=(12, 30),
        seed_essay_excerpt="",
        persona=StudentPersona.default(),
        prompt_family_id="family_a",
    )


def test_candidate_audit_rejects_duplicate_and_instruction_leakage() -> None:
    tasks = {task_id: _task(task_id) for task_id in ("one", "two", "three")}
    essay = (
        "Canals connected western farms to eastern cities while factories changed labor and "
        "helped create a larger national market during the period."
    )
    accepted, audit = audit_candidates(
        tasks,
        [
            {"task_id": "one", "student_response": essay},
            {"task_id": "two", "student_response": essay},
            {"task_id": "three", "student_response": "My target score is five points exactly."},
        ],
    )
    assert [candidate.task_id for candidate in accepted] == ["one"]
    assert len(audit.rejected) == 2
