"""Reproducible v3 evaluation records, split locking, resume safety, and metrics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from apush_frq_grader_slm.eval import _quadratic_weighted_kappa
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.rubric import CRITERIA
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import normalize_grade_output

SET_PATTERN = re.compile(r"_set(?P<set>[12])_")


class V3RunIdentity(BaseModel):
    run_id: str
    model_name: str
    model_hash: str
    data_hash: str
    split: Literal["set1", "set2"]
    decoding_settings: dict[str, Any]
    prompt_version: str = "v3_scores_feedback_1"


class V3EvalRecord(BaseModel):
    case_id: str
    run_id: str
    model_name: str
    model_hash: str
    data_hash: str
    split: Literal["set1", "set2"]
    decoding_settings: dict[str, Any]
    prompt_version: str
    raw_response: str
    raw_payload: dict[str, Any] | None = None
    normalized_payload: dict[str, Any] | None = None
    raw_schema_valid: bool
    layered_schema_valid: bool
    normalization_actions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["balanced_json", "eos", "length", "error", "unknown"]
    repetition_detected: bool = False


class V3MetricSummary(BaseModel):
    count: int
    schema_valid_rate: float
    total_mae: float
    total_within_one_rate: float
    qwk: float | None
    truncation_count: int
    repetition_rate: float


class V3EvaluationSummary(BaseModel):
    run_id: str
    model_name: str
    split: str
    raw_model: V3MetricSummary
    layered_system: V3MetricSummary
    acceptance: dict[str, bool]


def sha256_path(path: Path) -> str:
    """Hash a file or directory deterministically, including relative file names."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_id: str) -> str:
    """Hash local model bytes or a remote model identifier plus resolved HF revision."""
    path = Path(model_id)
    if path.exists():
        return sha256_path(path)
    revision = "unresolved"
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        revision = str(getattr(config, "_commit_hash", None) or "unresolved")
    except (OSError, ValueError):
        pass
    material = json.dumps(
        {"model_id": model_id, "resolved_revision": revision}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def build_run_identity(
    *,
    model_name: str,
    model_hash: str,
    data_hash: str,
    split: Literal["set1", "set2"],
    decoding_settings: dict[str, Any],
    prompt_version: str = "v3_scores_feedback_1",
) -> V3RunIdentity:
    material = {
        "model_name": model_name,
        "model_hash": model_hash,
        "data_hash": data_hash,
        "split": split,
        "decoding_settings": decoding_settings,
        "prompt_version": prompt_version,
    }
    run_id = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return V3RunIdentity(run_id=run_id, **material)


def select_official_split(cases: list[FRQCase], *, final_evaluation: bool = False) -> list[FRQCase]:
    """Default to set1; set2 is inaccessible unless final evaluation is explicit."""
    desired = 2 if final_evaluation else 1
    selected: list[FRQCase] = []
    for case in cases:
        set_number = case.provenance.set_number
        if set_number is None:
            match = SET_PATTERN.search(case.id)
            set_number = int(match.group("set")) if match else None
        if set_number == desired:
            selected.append(case)
    if not selected:
        raise ValueError(f"No set{desired} cases found")
    return selected


def load_resumable_records(
    path: Path,
    *,
    identity: V3RunIdentity,
    cases: list[FRQCase],
) -> list[V3EvalRecord]:
    """Load only records with exactly matching provenance and decoding identity."""
    if not path.exists():
        return []
    allowed = {case.id for case in cases}
    by_id: dict[str, V3EvalRecord] = {}
    expected = identity.model_dump(mode="json")
    for row in read_jsonl(path):
        record = V3EvalRecord.model_validate(row)
        actual = {
            "run_id": record.run_id,
            "model_name": record.model_name,
            "model_hash": record.model_hash,
            "data_hash": record.data_hash,
            "split": record.split,
            "decoding_settings": record.decoding_settings,
            "prompt_version": record.prompt_version,
        }
        if actual != expected:
            raise RuntimeError(f"Incompatible saved v3 run record: {record.case_id}")
        if record.case_id not in allowed:
            raise RuntimeError(f"Saved v3 result has unknown case ID: {record.case_id}")
        if record.case_id in by_id:
            raise RuntimeError(f"Saved v3 results contain duplicate case ID: {record.case_id}")
        by_id[record.case_id] = record
    return [by_id[case.id] for case in cases if case.id in by_id]


def make_record(
    *,
    case_id: str,
    identity: V3RunIdentity,
    raw_response: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    repetition_detected: bool,
) -> V3EvalRecord:
    layered = normalize_grade_output(raw_response)
    return V3EvalRecord(
        case_id=case_id,
        run_id=identity.run_id,
        model_name=identity.model_name,
        model_hash=identity.model_hash,
        data_hash=identity.data_hash,
        split=identity.split,
        decoding_settings=identity.decoding_settings,
        prompt_version=identity.prompt_version,
        raw_response=raw_response,
        raw_payload=layered.raw_payload,
        normalized_payload=layered.normalized_payload,
        raw_schema_valid=layered.raw_valid,
        layered_schema_valid=layered.layered_valid,
        normalization_actions=list(layered.normalization_actions),
        errors=list(layered.errors),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        repetition_detected=repetition_detected,
    )


def summarize_v3(
    records: list[V3EvalRecord], cases: list[FRQCase], identity: V3RunIdentity
) -> V3EvaluationSummary:
    case_by_id = {case.id: case for case in cases}
    raw = _metric_summary(records, case_by_id, layered=False)
    layered = _metric_summary(records, case_by_id, layered=True)
    acceptance = {
        "schema_valid_100pct": layered.schema_valid_rate == 1.0,
        "zero_truncations": layered.truncation_count == 0,
        "mae_lte_1_5": layered.total_mae <= 1.5,
        "within_one_gte_60pct": layered.total_within_one_rate >= 0.60,
        "qwk_gte_0_35": layered.qwk is not None and layered.qwk >= 0.35,
    }
    acceptance["passed"] = all(acceptance.values())
    return V3EvaluationSummary(
        run_id=identity.run_id,
        model_name=identity.model_name,
        split=identity.split,
        raw_model=raw,
        layered_system=layered,
        acceptance=acceptance,
    )


def assert_final_lock(lock_path: Path, identity: V3RunIdentity, receipt_path: Path) -> None:
    """Require an exact frozen candidate and prevent a second set2 evaluation."""
    if identity.split != "set2":
        return
    if receipt_path.exists():
        raise RuntimeError(f"Locked set2 evaluation already recorded: {receipt_path}")
    if not lock_path.exists():
        raise RuntimeError("set2 requires --lock-manifest created from a passing set1 run")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = identity.model_dump(mode="json")
    expected["split"] = "set1"
    expected.pop("run_id")
    comparable = {key: lock.get(key) for key in expected}
    if comparable != expected or not lock.get("set1_acceptance_passed"):
        raise RuntimeError("Final-evaluation settings do not match the passing set1 lock")


def _metric_summary(
    records: list[V3EvalRecord], case_by_id: dict[str, FRQCase], *, layered: bool
) -> V3MetricSummary:
    ref_totals: list[int] = []
    predicted_totals: list[int] = []
    valid_count = 0
    absolute_error = 0
    within_one = 0
    for record in records:
        valid = record.layered_schema_valid if layered else record.raw_schema_valid
        payload = record.normalized_payload if layered else record.raw_payload
        if valid:
            valid_count += 1
        scores = payload.get("scores") if isinstance(payload, dict) else None
        predicted = _score_total(scores)
        reference = case_by_id[record.case_id].reference_scores.total
        ref_totals.append(reference)
        if predicted is None:
            absolute_error += 6
            predicted_totals.append(-1)
        else:
            error = abs(predicted - reference)
            absolute_error += error
            within_one += int(error <= 1)
            predicted_totals.append(predicted)
    count = len(records)
    qwk = _quadratic_weighted_kappa(ref_totals, predicted_totals) if count >= 5 else None
    return V3MetricSummary(
        count=count,
        schema_valid_rate=round(valid_count / count, 4) if count else 0,
        total_mae=round(absolute_error / count, 4) if count else 0,
        total_within_one_rate=round(within_one / count, 4) if count else 0,
        qwk=round(qwk, 4) if qwk is not None else None,
        truncation_count=sum(record.finish_reason == "length" for record in records),
        repetition_rate=(
            round(sum(record.repetition_detected for record in records) / count, 4)
            if count
            else 0
        ),
    )


def _score_total(scores: object) -> int | None:
    """Return a total for complete score objects; partial raw payloads are unscorable."""
    if not isinstance(scores, dict):
        return None
    values: list[int] = []
    for criterion in CRITERIA:
        value = scores.get(criterion)
        if isinstance(value, bool):
            return None
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return None
    return sum(values)
