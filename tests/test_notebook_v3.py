from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_train_eval_v3.ipynb")


def test_v3_notebook_is_separate_and_locks_set2() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Section 13 defaults to set1" in source
    assert "FINAL_EVALUATION = False" in source
    assert "--final-evaluation" in source
    assert "--lock-manifest" in source
    assert "Qwen/Qwen2.5-0.5B-Instruct" in source
    assert "Qwen/Qwen2.5-1.5B-Instruct" in source
