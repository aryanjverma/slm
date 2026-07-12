"""V5 two-pass SFT rows, weighted tokenization, hashing, and bundle metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from apush_frq_grader_slm.prompts_v5 import (
    V5_FEEDBACK_SYSTEM_PROMPT,
    V5_RUBRIC_TEXT,
    V5_SCORER_SYSTEM_PROMPT,
    format_v5_feedback_user_message,
    format_v5_scorer_user_message,
    v5_feedback_target,
    v5_scorer_target,
)
from apush_frq_grader_slm.rubric import CRITERIA
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.training_v3 import _extract_input_ids

V5_MAX_SEQ_LENGTH = 4096
V5_SCORE_TOKEN_WEIGHT = 4.0
V5_BUNDLE_VERSION = 1
V5_PROMPT_FILES = {
    "scorer_system": "prompts/scorer_system.txt",
    "feedback_system": "prompts/feedback_system.txt",
    "rubric": "prompts/rubric.txt",
}

V5_PRIVATE_COUNTS = {
    "train_cases_v5.jsonl": 540,
    "dev_cases_v5.jsonl": 60,
    "replay_cases_v4_for_v5.jsonl": 75,
    "train_cases_v5_with_replay.jsonl": 615,
    "train_chat_v5_scorer.jsonl": 615,
    "train_chat_v5_feedback.jsonl": 615,
    "dev_chat_v5_scorer.jsonl": 60,
    "dev_chat_v5_feedback.jsonl": 60,
}


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v5_training_preflight(
    private_dir: Path,
    *,
    data_path: Path | None = None,
    golden_cases_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the private v5 corpus matches its human approval and audit."""
    from apush_frq_grader_slm.dataset_v5 import assert_manual_approval

    private_dir = private_dir.resolve()
    manifest_path = private_dir / "private_use_manifest_v5.json"
    audit_path = private_dir / "assembly_audit_v5.json"
    approval_path = private_dir / "manual_review_approval_v5.json"
    packet_path = private_dir / "manual_review_packet_v5.jsonl"
    required = [manifest_path, audit_path, approval_path, packet_path]
    required.extend(private_dir / name for name in V5_PRIVATE_COUNTS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PermissionError(f"v5 preflight missing required artifacts: {missing}")

    approval = assert_manual_approval(packet_path, approval_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_manifest = {"selected": 600, "train": 540, "dev": 60, "manual_review": 60}
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise PermissionError(f"v5 manifest {key}={manifest.get(key)!r}; expected {expected}")
    expected_audit = {
        "approved": True,
        "new_train": 540,
        "new_dev": 60,
        "v4_replay": 75,
        "training_rows_total": 615,
        "golden_eval_rows_in_training": 0,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise PermissionError(f"v5 audit {key}={audit.get(key)!r}; expected {expected}")
    if (
        audit.get("manual_review_packet_sha256") != approval.get("packet_sha256")
        or audit.get("manual_review_packet_sha256") != _file_sha256(packet_path)
    ):
        raise PermissionError("assembly audit is not bound to the approved manual-review packet")
    if audit.get("manual_review_approval_sha256") != _file_sha256(approval_path):
        raise PermissionError("assembly audit is not bound to the manual-review approval receipt")

    artifact_hashes = audit.get("artifacts") or {}
    for name, expected_count in V5_PRIVATE_COUNTS.items():
        path = private_dir / name
        rows = _jsonl_rows(path)
        if len(rows) != expected_count:
            raise PermissionError(f"v5 artifact {name} has {len(rows)} rows; expected {expected_count}")
        actual_hash = _file_sha256(path)
        if artifact_hashes.get(name) != actual_hash:
            raise PermissionError(f"v5 artifact hash mismatch: {name}")

    combined = private_dir / "train_cases_v5_with_replay.jsonl"
    combined_hash = _file_sha256(combined)
    if audit.get("combined_training_sha256") != combined_hash:
        raise PermissionError("615-row combined training hash does not match the assembly audit")
    if data_path is not None and _file_sha256(data_path.resolve()) != combined_hash:
        raise PermissionError("--data is not the audited 615-row combined v5 training corpus")

    if golden_cases_path is None or not golden_cases_path.is_file():
        raise PermissionError("golden cases are required to verify zero training leakage")
    golden_rows = _jsonl_rows(golden_cases_path)
    golden_ids = {str(row.get("id") or row.get("task_id") or "") for row in golden_rows}
    golden_text = {
        hashlib.sha256(str(row.get("student_response") or row.get("essay") or "").strip().encode()).hexdigest()
        for row in golden_rows
        if str(row.get("student_response") or row.get("essay") or "").strip()
    }
    for row in _jsonl_rows(combined):
        row_id = str(row.get("id") or row.get("task_id") or "")
        essay = str(row.get("student_response") or row.get("essay") or "").strip()
        essay_hash = hashlib.sha256(essay.encode()).hexdigest() if essay else ""
        if (row_id and row_id in golden_ids) or (essay_hash and essay_hash in golden_text):
            raise PermissionError("golden evaluation row detected in v5 training corpus")

    return {
        "approved": True,
        "review_packet_sha256": approval["packet_sha256"],
        "combined_training_sha256": combined_hash,
        "training_rows": 615,
        "development_rows": 60,
        "golden_eval_rows_in_training": 0,
    }


def build_v5_chat_row(case: FRQCase, task: Literal["scorer", "feedback"]) -> dict[str, Any]:
    scores = case.reference_scores.model_dump()
    if task == "scorer":
        system = V5_SCORER_SYSTEM_PROMPT
        user = format_v5_scorer_user_message(case.prompt, case.student_response)
        target = v5_scorer_target(scores)
    elif task == "feedback":
        system = V5_FEEDBACK_SYSTEM_PROMPT
        user = format_v5_feedback_user_message(case.prompt, case.student_response, scores)
        target = v5_feedback_target(case.reference_feedback)
    else:
        raise ValueError(f"Unknown v5 training task: {task}")
    return {
        "id": case.id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "metadata": {"task": task, "reference_total": case.reference_scores.total},
    }


def tokenize_v5_row(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    task: Literal["scorer", "feedback"],
    max_length: int = V5_MAX_SEQ_LENGTH,
    score_token_weight: float = V5_SCORE_TOKEN_WEIGHT,
) -> dict[str, list[int] | list[float]]:
    """Tokenize without truncation so no essay section or assistant target is lost."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Training row must end with one assistant message")
    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    )
    full_ids = _extract_input_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the generation-prompt prefix")
    if len(full_ids) > max_length:
        raise ValueError(
            f"V5 row has {len(full_ids)} tokens and exceeds the {max_length}-token context; "
            "reject or shorten it rather than truncating the essay"
        )
    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise ValueError("Assistant target tokenized to zero tokens")
    labels = [-100] * len(prompt_ids) + assistant_ids.copy()
    weights = [0.0] * len(prompt_ids) + [1.0] * len(assistant_ids)
    if task == "scorer":
        numeric_positions = _score_token_positions(assistant_ids, tokenizer)
        if len(numeric_positions) != len(CRITERIA):
            raise ValueError(
                "Expected exactly four standalone score tokens in the compact scorer target; "
                f"found {len(numeric_positions)}"
            )
        for index in numeric_positions:
            weights[len(prompt_ids) + index] = float(score_token_weight)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "loss_weights": weights,
    }


def _score_token_positions(token_ids: list[int], tokenizer: Any) -> list[int]:
    positions: list[int] = []
    for index, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        digit_count = sum(piece.count(digit) for digit in ("0", "1", "2"))
        if digit_count > 1:
            raise ValueError(f"One scorer token unexpectedly contains multiple score digits: {piece!r}")
        if digit_count == 1:
            positions.append(index)
    return positions


class WeightedAssistantCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        padded: dict[str, list[list[int]] | list[list[float]]] = {
            "input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []
        }
        for feature in features:
            padding = max_len - len(feature["input_ids"])
            padded["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            padded["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            padded["labels"].append(feature["labels"] + [-100] * padding)
            padded["loss_weights"].append(feature["loss_weights"] + [0.0] * padding)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(padded["loss_weights"], dtype=torch.float),
        }


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> Any:
    """Weighted next-token cross entropy, normalized by active token weight."""
    import torch.nn.functional as functional

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    shifted_weights = loss_weights[..., 1:].contiguous()
    token_loss = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    active_weights = shifted_weights * shifted_labels.ne(-100)
    return (token_loss * active_weights).sum() / active_weights.sum().clamp_min(1.0)


def sha256_tree(path: Path, *, exclude_names: Iterable[str] = ()) -> str:
    excluded = set(exclude_names)
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if item.name in excluded:
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_bundle_prompts(bundle_root: Path) -> dict[str, dict[str, str]]:
    """Write corrected v5 prompts into the bundle and return path/hash entries."""
    root = bundle_root.resolve()
    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "scorer_system": V5_SCORER_SYSTEM_PROMPT,
        "feedback_system": V5_FEEDBACK_SYSTEM_PROMPT,
        "rubric": V5_RUBRIC_TEXT,
    }
    entries: dict[str, dict[str, str]] = {}
    for key, relative in V5_PROMPT_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = contents[key]
        if not text.endswith("\n"):
            text = text + "\n"
        path.write_text(text, encoding="utf-8")
        entries[key] = _artifact_entry(root, path)
    return entries


def write_bundle_manifest(
    bundle_root: Path,
    *,
    inherited_base: Path | str,
    scorer_adapter: Path | str,
    feedback_adapter: Path | str,
    write_prompts: bool = True,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompts = write_bundle_prompts(root) if write_prompts else {}
    manifest = {
        "format": "apush-frq-grader-v5",
        "format_version": V5_BUNDLE_VERSION,
        "inherited_base": _artifact_entry(root, inherited_base),
        "scorer": _artifact_entry(root, scorer_adapter),
        "feedback": _artifact_entry(root, feedback_adapter),
        "prompts": prompts,
        "max_seq_length": V5_MAX_SEQ_LENGTH,
        "score_token_weight": V5_SCORE_TOKEN_WEIGHT,
        "two_pass": True,
        "total": "deterministic_sum",
    }
    (root / "v5_bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _artifact_entry(root: Path, value: Path | str) -> dict[str, str]:
    path = Path(value).resolve()
    try:
        displayed = path.relative_to(root).as_posix()
    except ValueError:
        displayed = str(path)
    return {"path": displayed, "sha256": sha256_tree(path)}


def resolve_manifest_path(bundle_root: Path, entry: Mapping[str, str]) -> Path:
    path = Path(entry["path"])
    return path if path.is_absolute() else bundle_root / path


def export_v5_chat_rows(
    cases: Iterable[FRQCase], task: Literal["scorer", "feedback"]
) -> list[dict[str, Any]]:
    """Build chat-template SFT rows for one v5 training task."""
    return [build_v5_chat_row(case, task) for case in cases]
