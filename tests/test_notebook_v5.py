from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_train_v5.ipynb")


def _notebook_source() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, source


def test_v5_notebook_cells_are_valid_python() -> None:
    notebook, _ = _notebook_source()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_v5_notebook_runs_the_locked_two_pass_pipeline() -> None:
    _, source = _notebook_source()

    assert "scripts/merge_v4_adapter.py" in source
    assert "scripts/train_v5.py" in source
    assert "train('scorer'" in source
    assert "train('feedback'" in source
    assert "--score-token-weight" in source
    assert "scripts/grade_v5.py" in source
    assert "scripts/eval_v5.py" in source
    assert "feedback_fallback_rate" in source
    assert "golden_evaluation_is_development_informed" in source


def test_v5_notebook_enforces_approval_freeze_and_release_gates() -> None:
    _, source = _notebook_source()

    assert "manual_review_approval_v5.json" in source
    assert "validate_v5_training_preflight" in source
    assert "sys.path.insert(0,src_path)" in source
    assert "--private-dir" in source
    assert "--golden-cases" in source
    assert "assert FROZEN_CONFIG.exists()" in source
    assert "scripts/check_v5_release.py" in source
    assert "Non-production-ready; do not retune on golden answers." in source


def test_v5_notebook_exports_aggregates_only() -> None:
    _, source = _notebook_source()

    assert "not any(name.endswith('.jsonl')" in source
    assert "review_packet" in source
    assert "Aggregate-only bundle" in source
