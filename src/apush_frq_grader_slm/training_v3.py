"""Tokenization helpers that guarantee assistant-only supervised loss."""

from __future__ import annotations

from typing import Any


def tokenize_assistant_only(
    messages: list[dict[str, str]], tokenizer: Any, *, max_length: int = 3072
) -> dict[str, list[int]]:
    """Mask system/user tokens and preserve the complete assistant target when possible."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Training row must end with one assistant message")
    prompt_messages = messages[:-1]
    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _extract_input_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the generation-prompt prefix")
    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise ValueError("Assistant target tokenized to zero tokens")
    if len(assistant_ids) > max_length:
        raise ValueError("Assistant target alone exceeds context window")

    keep_prompt = max_length - len(assistant_ids)
    visible_prompt = prompt_ids[-keep_prompt:] if keep_prompt else []
    input_ids = visible_prompt + assistant_ids
    labels = [-100] * len(visible_prompt) + assistant_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _extract_input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict) or hasattr(tokenized, "keys"):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("Expected one tokenized chat row")
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def assert_assistant_only_example(example: dict[str, list[int]]) -> None:
    input_ids = example["input_ids"]
    labels = example["labels"]
    if len(input_ids) != len(labels) or len(input_ids) > 3072:
        raise ValueError("Tokenized training row violates context limits")
    visible = [index for index, value in enumerate(labels) if value != -100]
    if not visible:
        raise ValueError("No assistant labels remain visible")
    first = visible[0]
    if any(value != -100 for value in labels[:first]):
        raise ValueError("Non-assistant token contributes to loss")
    if labels[first:] != input_ids[first:]:
        raise ValueError("Assistant labels do not match visible target tokens")


class AssistantOnlyDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
