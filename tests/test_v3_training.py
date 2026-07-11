from __future__ import annotations

from apush_frq_grader_slm.training_v3 import (
    assert_assistant_only_example,
    tokenize_assistant_only,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text] if tokenize else text


def test_only_assistant_tokens_contribute_to_loss_and_target_survives_truncation() -> None:
    messages = [
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": '{"scores":{}}'},
    ]
    example = tokenize_assistant_only(messages, CharacterTokenizer(), max_length=80)
    assert_assistant_only_example(example)
    visible = [value for value in example["labels"] if value != -100]
    assert "scores" in "".join(chr(value) for value in visible)
    assert len(example["input_ids"]) == 80
