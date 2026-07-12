from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from apush_frq_grader_slm.manual_review_v5 import (
    build_human_approval,
    load_review_packet,
    packet_sha256,
    review_status,
    set_review_decision,
    write_packet_atomic,
)
from scripts.review_v5_manual_packet import (
    ACTION_OPTIONS,
    accept_reviewed_row,
    render_action_menu,
)


SCORES = {"thesis": 1, "contextualization": 1, "evidence": 1, "analysis_reasoning": 1}
FEEDBACK = {
    "thesis": "The essay states a defensible claim about reform.",
    "contextualization": "The opening describes industrialization before reform.",
    "evidence": "The essay discusses settlement houses and labor unions.",
    "analysis_reasoning": "The response explains how industrialization caused reform.",
}


def _row(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "resolved_grade": {"scores": SCORES, "feedback": FEEDBACK},
        "manual_review": {"decision": "accept", "corrections": {}, "notes": "automated"},
    }


def test_human_review_is_distinct_from_existing_automated_acceptance() -> None:
    rows = [_row("one"), _row("two")]
    assert review_status(rows, "human")["accept"] == 2
    assert review_status(rows, "human")["human_verified"] == 0
    rows[0] = set_review_decision(
        rows[0], decision="accept", reviewer="human", reviewed_at="2026-07-12T00:00:00Z"
    )
    assert review_status(rows, "human")["human_verified"] == 1


def test_corrections_are_schema_validated_and_reject_blocks_approval(tmp_path: Path) -> None:
    row = set_review_decision(
        _row("one"),
        decision="corrected",
        reviewer="human",
        corrections={"scores": {**SCORES, "evidence": 2}},
    )
    assert row["manual_review"]["corrections"]["scores"]["evidence"] == 2
    with pytest.raises(ValueError, match="corrected row requires"):
        set_review_decision(_row("one"), decision="corrected", reviewer="human")
    with pytest.raises(ValueError):
        set_review_decision(
            _row("one"),
            decision="corrected",
            reviewer="human",
            corrections={"scores": {**SCORES, "evidence": 3}},
        )

    packet = tmp_path / "packet.jsonl"
    rejected = set_review_decision(
        _row("two"), decision="reject", reviewer="human", notes="Unrealistic essay"
    )
    write_packet_atomic(packet, [row, rejected])
    with pytest.raises(ValueError, match="has not accepted or corrected"):
        build_human_approval([row, rejected], packet_path=packet, reviewer="human")


def test_atomic_packet_and_hash_bound_human_approval(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    rows = [
        set_review_decision(
            _row(task_id),
            decision="accept",
            reviewer="human",
            reviewed_at="2026-07-12T00:00:00Z",
        )
        for task_id in ("one", "two")
    ]
    write_packet_atomic(packet, rows)
    loaded = load_review_packet(packet)
    assert [row["task_id"] for row in loaded] == ["one", "two"]
    approval = build_human_approval(loaded, packet_path=packet, reviewer="human")
    assert approval["packet_sha256"] == packet_sha256(packet)
    assert approval["accept_count"] == 2
    assert approval["corrected_count"] == 0
    assert not packet.with_name(packet.name + ".tmp").exists()
    assert json.loads(packet.read_text(encoding="utf-8").splitlines()[0])["task_id"] == "one"


def test_plain_action_menu_lists_every_option_before_input() -> None:
    console = Console(record=True, no_color=True, width=120)
    render_action_menu(console, plain=True)
    rendered = console.export_text()
    assert "INPUT OPTIONS" in rendered
    for key, label, description in ACTION_OPTIONS:
        assert f"{key} - {label}: {description}" in rendered


def test_accepting_preliminary_correction_preserves_it_for_human_reviewer() -> None:
    preliminary = set_review_decision(
        _row("one"),
        decision="corrected",
        reviewer="preliminary",
        corrections={"scores": {**SCORES, "evidence": 2}},
    )

    accepted = accept_reviewed_row(preliminary, reviewer="human")

    assert accepted["manual_review"]["decision"] == "corrected"
    assert accepted["manual_review"]["reviewed_by"] == "human"
    assert accepted["manual_review"]["corrections"]["scores"]["evidence"] == 2
