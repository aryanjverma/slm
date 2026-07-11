from __future__ import annotations

from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.dataset_v3 import audit_v3_training_cases, v3_chat_row


def test_v3_target_omits_total() -> None:
    case = generate_cases(count=1, split="train", seed=4)[0]
    row = v3_chat_row(case)
    target = row["messages"][-1]["content"]
    assert '"total"' not in target
    assert '"scores"' in target
    assert '"feedback"' in target


def test_v3_audit_rejects_official_training_source() -> None:
    case = generate_cases(count=1, split="train", seed=5)[0]
    case.tags.append("ap_central")
    audit = audit_v3_training_cases([case])
    assert "official_source_tag" in audit.rejected[case.id]
    assert "non_synthetic_or_unknown_source" not in audit.rejected[case.id]
    assert "not_substantially_larger_than_v2" in audit.global_reasons
