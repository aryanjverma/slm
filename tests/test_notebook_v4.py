from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_train_v4.ipynb")


def test_v4_notebook_uses_fresh_assistant_only_training_and_evaluation() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "scripts/train_v4.py" in source
    assert "tokenize_assistant_only" in source
    assert "RUN_ID = 'assistant-only-r1'" in source
    assert "WARMUP_RATIO = 0.05" in source
    assert "RUN_EVALUATION = True" in source
    assert "'--prompt-version', 'v4'" in source
    assert "FINAL_DIR / 'adapter_config.json'" in source
    assert "train_qlora.py" not in source


def test_v4_notebook_counts_token_ids_instead_of_mapping_keys() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "lengths = [len(example['input_ids'])" in source
    assert "supervised_lengths" in source
