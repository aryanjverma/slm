"""State and persistence helpers for human review of the private v5 packet."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from apush_frq_grader_slm.schemas import RubricFeedback, RubricScores

FINAL_DECISIONS = {"accept", "corrected"}
KNOWN_DECISIONS = FINAL_DECISIONS | {"pending", "reject"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_review_packet(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Review packet is empty: {path}")
    task_ids = [str(row.get("task_id") or "") for row in rows]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Review packet must contain unique, non-empty task IDs")
    return rows


def review_status(rows: Sequence[Mapping[str, Any]], reviewer: str | None = None) -> dict[str, int]:
    counts = Counter(
        str((row.get("manual_review") or {}).get("decision") or "pending") for row in rows
    )
    human_verified = 0
    if reviewer:
        human_verified = sum(
            (row.get("manual_review") or {}).get("reviewed_by") == reviewer
            and (row.get("manual_review") or {}).get("decision") in FINAL_DECISIONS
            for row in rows
        )
    return {
        "total": len(rows),
        "accept": counts["accept"],
        "corrected": counts["corrected"],
        "reject": counts["reject"],
        "pending": counts["pending"],
        "human_verified": human_verified,
    }


def set_review_decision(
    row: Mapping[str, Any],
    *,
    decision: str,
    reviewer: str,
    notes: str = "",
    corrections: Mapping[str, Any] | None = None,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    if decision not in KNOWN_DECISIONS:
        raise ValueError(f"Unknown review decision: {decision}")
    if not reviewer.strip():
        raise ValueError("Reviewer name cannot be empty")
    normalized_corrections = dict(corrections or {})
    unknown = set(normalized_corrections) - {"scores", "feedback"}
    if unknown:
        raise ValueError(f"Unsupported correction fields: {sorted(unknown)}")
    if "scores" in normalized_corrections:
        normalized_corrections["scores"] = RubricScores.model_validate(
            normalized_corrections["scores"]
        ).model_dump()
    if "feedback" in normalized_corrections:
        normalized_corrections["feedback"] = RubricFeedback.model_validate(
            normalized_corrections["feedback"]
        ).model_dump()
    if decision == "corrected" and not normalized_corrections:
        raise ValueError("A corrected row requires score and/or feedback corrections")
    if decision != "corrected" and normalized_corrections:
        raise ValueError(f"A {decision} row cannot contain corrections")

    updated = copy.deepcopy(dict(row))
    updated["manual_review"] = {
        "decision": decision,
        "corrections": normalized_corrections,
        "notes": notes.strip(),
        "reviewed_by": reviewer.strip(),
        "reviewed_at": reviewed_at or utc_now(),
    }
    return updated


def write_packet_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_review_backup(packet: Path, approval: Path | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = packet.parent / "logs" / "manual_review_backups" / stamp
    suffix = 1
    while destination.exists():
        destination = destination.with_name(f"{stamp}-{suffix}")
        suffix += 1
    destination.mkdir(parents=True)
    shutil.copy2(packet, destination / packet.name)
    if approval and approval.exists():
        shutil.copy2(approval, destination / approval.name)
    return destination


def packet_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_human_approval(
    rows: Sequence[Mapping[str, Any]], *, packet_path: Path, reviewer: str
) -> dict[str, Any]:
    incomplete: list[str] = []
    for row in rows:
        review = row.get("manual_review") or {}
        if review.get("decision") not in FINAL_DECISIONS or review.get("reviewed_by") != reviewer:
            incomplete.append(str(row.get("task_id") or "unknown"))
    if incomplete:
        raise ValueError(
            f"Reviewer {reviewer!r} has not accepted or corrected {len(incomplete)} rows; "
            f"first incomplete IDs: {incomplete[:5]}"
        )
    status = review_status(rows, reviewer)
    return {
        "approved": True,
        "reviewer": reviewer,
        "approved_at": utc_now(),
        "packet_sha256": packet_sha256(packet_path),
        "notes": "Human terminal review completed for every packet row.",
        "accept_count": status["accept"],
        "corrected_count": status["corrected"],
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
