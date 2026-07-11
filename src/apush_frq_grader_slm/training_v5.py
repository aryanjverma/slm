"""V5 two-pass SFT rows, weighted tokenization, hashing, and bundle metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from apush_frq_grader_slm.prompts_v5 import (
    V5_FEEDBACK_SYSTEM_PROMPT,
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


def write_bundle_manifest(
    bundle_root: Path,
    *,
    inherited_base: Path | str,
    scorer_adapter: Path | str,
    feedback_adapter: Path | str,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "apush-frq-grader-v5",
        "format_version": V5_BUNDLE_VERSION,
        "inherited_base": _artifact_entry(root, inherited_base),
        "scorer": _artifact_entry(root, scorer_adapter),
        "feedback": _artifact_entry(root, feedback_adapter),
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
