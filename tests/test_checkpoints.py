from __future__ import annotations

import json

from apush_frq_grader_slm.checkpoints import (
    build_checkpoint_plan,
    materialize_checkpoint_data,
)


def test_checkpoint_plan_skips_unavailable_counts_and_hashes_subsets(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(500)),
        encoding="utf-8",
    )
    runs = build_checkpoint_plan(
        counts=[200, 500, 1200],
        available_rows=500,
        checkpoint_root=tmp_path / "checkpoints",
        model="model-id",
        eval_paths=[("cb_eval", tmp_path / "eval.jsonl", True)],
        eval_output_root=tmp_path / "persistent_eval",
    )
    assert [run.count for run in runs] == [200, 500]
    assert runs[0].eval_commands[0][-1] == "--real-eval"
    assert str(tmp_path / "persistent_eval" / "200") in runs[0].eval_commands[0]
    manifest = materialize_checkpoint_data(source, runs)
    assert manifest["200"]["rows"] == 200
    assert manifest["500"]["rows"] == 500
    assert manifest["200"]["sha256"] != manifest["500"]["sha256"]
