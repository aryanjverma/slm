from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_train_eval_v1.ipynb")


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_uses_selected_cb_evaluation_file() -> None:
    source = _source()
    assert 'CB_EVAL = Path("artifacts/data/eval_cb_cases.jsonl")' in source
    assert "run_hf_eval(\"challenge\"" not in source
    assert "run_hf_eval(\"external\"" not in source
    assert 'run_hf_eval("cb_eval", CB_EVAL)' in source
    assert "Expected 53 selected CB cases" in source
    assert "evaluation will still use all selected rows" in source
    assert "require_official_eval_ready" not in source


def test_notebook_evaluation_is_drive_backed_resumable_and_512_tokens() -> None:
    source = _source()
    assert 'drive.mount("/content/drive")' in source
    assert "slm_evaluation/apush-frq-grader-v2" in source
    assert '"--real-eval", "--resume"' in source
    assert '"--eval-output-root", str(EVAL_DIR / "checkpoints")' in source
    assert "MAX_NEW_TOKENS = 512" in source
    assert "evaluation_sha256" in source
