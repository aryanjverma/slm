from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.eval_v5 import v5_diagnostics
from apush_frq_grader_slm.inference_v5 import (
    fallback_feedback,
    grade_two_pass,
    parse_and_normalize_scores,
)
from apush_frq_grader_slm.prompts_v5 import V5_RUBRIC_TEXT
from apush_frq_grader_slm.training_v5 import (
    build_v5_chat_row,
    resolve_manifest_path,
    tokenize_v5_row,
    validate_v5_training_preflight,
    write_bundle_manifest,
)


def test_complete_cpu_v5_smoke_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_v5_pipeline.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_v5_training_preflight_binds_approval_counts_hashes_and_leakage(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    packet = private / "manual_review_packet_v5.jsonl"
    reviewed = [
        {"task_id": f"review-{index}", "manual_review": {"decision": "accept"}}
        for index in range(60)
    ]
    _write_jsonl(packet, reviewed)
    packet_hash = hashlib.sha256(packet.read_bytes()).hexdigest()
    (private / "manual_review_approval_v5.json").write_text(
        json.dumps({
            "approved": True,
            "reviewer": "human",
            "approved_at": "2026-07-12T00:00:00Z",
            "packet_sha256": packet_hash,
        }),
        encoding="utf-8",
    )
    approval_hash = hashlib.sha256(
        (private / "manual_review_approval_v5.json").read_bytes()
    ).hexdigest()
    (private / "private_use_manifest_v5.json").write_text(
        json.dumps({"selected": 600, "train": 540, "dev": 60, "manual_review": 60}),
        encoding="utf-8",
    )
    train = [{"id": f"train-{index}", "student_response": f"private essay {index}"} for index in range(540)]
    dev = [{"id": f"dev-{index}"} for index in range(60)]
    replay = [{"id": f"replay-{index}", "student_response": f"replay essay {index}"} for index in range(75)]
    rows_by_name = {
        "train_cases_v5.jsonl": train,
        "dev_cases_v5.jsonl": dev,
        "replay_cases_v4_for_v5.jsonl": replay,
        "train_cases_v5_with_replay.jsonl": train + replay,
        "train_chat_v5_scorer.jsonl": [{"id": index} for index in range(615)],
        "train_chat_v5_feedback.jsonl": [{"id": index} for index in range(615)],
        "dev_chat_v5_scorer.jsonl": [{"id": index} for index in range(60)],
        "dev_chat_v5_feedback.jsonl": [{"id": index} for index in range(60)],
    }
    hashes = {}
    for name, rows in rows_by_name.items():
        path = private / name
        _write_jsonl(path, rows)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined_hash = hashes["train_cases_v5_with_replay.jsonl"]
    (private / "assembly_audit_v5.json").write_text(
        json.dumps({
            "approved": True,
            "new_train": 540,
            "new_dev": 60,
            "v4_replay": 75,
            "training_rows_total": 615,
            "golden_eval_rows_in_training": 0,
            "manual_review_packet_sha256": packet_hash,
            "manual_review_approval_sha256": approval_hash,
            "combined_training_sha256": combined_hash,
            "artifacts": hashes,
        }),
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    _write_jsonl(golden, [{"id": "golden-1", "student_response": "held out essay"}])

    result = validate_v5_training_preflight(
        private,
        data_path=private / "train_cases_v5_with_replay.jsonl",
        golden_cases_path=golden,
    )
    assert result["training_rows"] == 615
    with (private / "train_cases_v5_with_replay.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "tampered"}) + "\n")
    with pytest.raises(PermissionError, match="616 rows"):
        validate_v5_training_preflight(
            private,
            data_path=private / "train_cases_v5_with_replay.jsonl",
            golden_cases_path=golden,
        )


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text] if tokenize else text

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(chr(value) for value in token_ids)


def test_v5_prompts_correct_context_rule_and_split_targets() -> None:
    assert "No fixed sentence count" in V5_RUBRIC_TEXT
    assert "3-to-4 sentence" not in V5_RUBRIC_TEXT
    case = generate_cases(count=1, split="train", seed=51)[0]
    scorer = build_v5_chat_row(case, "scorer")
    feedback = build_v5_chat_row(case, "feedback")
    scorer_target = json.loads(scorer["messages"][-1]["content"])
    feedback_target = json.loads(feedback["messages"][-1]["content"])
    assert set(scorer_target) == {"scores"}
    assert set(feedback_target) == {"feedback"}
    assert "Validated scores" in feedback["messages"][1]["content"]


def test_scorer_tokenization_weights_four_scores_and_never_truncates() -> None:
    case = generate_cases(count=1, split="train", seed=52)[0]
    row = build_v5_chat_row(case, "scorer")
    example = tokenize_v5_row(row["messages"], CharacterTokenizer(), task="scorer")
    assert example["loss_weights"].count(4.0) == 4
    first_label = next(index for index, value in enumerate(example["labels"]) if value != -100)
    assert all(value == 0.0 for value in example["loss_weights"][:first_label])
    with pytest.raises(ValueError, match="rather than truncating"):
        tokenize_v5_row(
            row["messages"], CharacterTokenizer(), task="scorer", max_length=20
        )


def test_two_pass_grade_computes_total_and_conditions_feedback() -> None:
    calls = []

    def generate(adapter, messages, max_new_tokens):
        calls.append((adapter, messages, max_new_tokens))
        if adapter == "scorer":
            return 'prefix {"scores":{"thesis":1,"contextualization":1,"evidence":2,"analysis_reasoning":1}}'
        assert '"total":5' in messages[1]["content"]
        return json.dumps(
            {
                "feedback": {
                    "thesis": "The essay makes a defensible claim.",
                    "contextualization": "The opening supplies broader context.",
                    "evidence": "The essay uses two named examples.",
                    "analysis_reasoning": "The causal structure supports the argument.",
                }
            }
        )

    result = grade_two_pass("A prompt", "An essay", generate)
    assert result["total"] == 5
    assert [call[0] for call in calls] == ["scorer", "feedback"]
    assert set(result) == {"scores", "total", "feedback"}


def test_two_pass_keeps_scores_when_feedback_fails_and_clamps_integer_scores() -> None:
    def generate(adapter, messages, max_new_tokens):
        if adapter == "scorer":
            return '{"scores":{"thesis":7,"contextualization":-2,"evidence":8,"analysis_reasoning":"1"}}'
        return "not json"

    result = grade_two_pass("Prompt", "Essay", generate)
    assert result["scores"] == {
        "thesis": 1,
        "contextualization": 0,
        "evidence": 2,
        "analysis_reasoning": 1,
    }
    assert result["total"] == 4
    assert result["feedback"] == fallback_feedback(result["scores"])
    with pytest.raises(ValueError, match="Invalid scorer value"):
        parse_and_normalize_scores(
            '{"scores":{"thesis":true,"contextualization":0,"evidence":0,"analysis_reasoning":0}}'
        )


def test_bundle_manifest_hashes_prompts_and_resolves_artifacts(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    base = bundle / "base"
    scorer = bundle / "scorer"
    feedback = bundle / "feedback"
    for path, value in ((base, "base"), (scorer, "score"), (feedback, "feedback")):
        path.mkdir(parents=True)
        (path / "weights.safetensors").write_text(value, encoding="utf-8")
    manifest = write_bundle_manifest(
        bundle, inherited_base=base, scorer_adapter=scorer, feedback_adapter=feedback
    )
    assert manifest["two_pass"] is True
    assert resolve_manifest_path(bundle, manifest["scorer"]) == scorer
    assert len(manifest["scorer"]["sha256"]) == 64
    assert set(manifest["prompts"]) == {"scorer_system", "feedback_system", "rubric"}
    scorer_prompt = (bundle / "prompts" / "scorer_system.txt").read_text(encoding="utf-8")
    rubric = (bundle / "prompts" / "rubric.txt").read_text(encoding="utf-8")
    assert "No fixed sentence count" in rubric
    assert "Return exactly one JSON object" in scorer_prompt
    assert len(manifest["prompts"]["rubric"]["sha256"]) == 64


def test_v5_diagnostics_cover_confusion_distributions_calibration_and_ci() -> None:
    cases = generate_cases(count=8, split="eval", seed=54)
    predictions = []
    for index, case in enumerate(cases):
        scores = case.reference_scores.model_dump()
        if index == 0:
            scores["evidence"] = max(0, scores["evidence"] - 1)
        predictions.append({"scores": scores})
    diagnostics = v5_diagnostics(cases, predictions, bootstrap_samples=40, seed=9)
    assert diagnostics["count"] == 8
    assert set(diagnostics["criterion_confusion_matrices"]) == {
        "thesis", "contextualization", "evidence", "analysis_reasoning"
    }
    assert sum(diagnostics["total_distributions"]["predicted"].values()) == 8
    assert diagnostics["calibration_by_length"]
    assert diagnostics["calibration_by_reference_total"]
    assert set(diagnostics["bootstrap_confidence_intervals"]) == {
        "qwk", "total_mae", "within_one_rate"
    }


def test_weighted_loss_prefers_the_high_weight_token() -> None:
    torch = pytest.importorskip("torch")
    from apush_frq_grader_slm.training_v5 import weighted_causal_lm_loss

    logits = torch.tensor([[[0.0, 0.0], [4.0, -4.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    equal = weighted_causal_lm_loss(logits, labels, torch.tensor([[0.0, 1.0, 1.0]]))
    weighted = weighted_causal_lm_loss(logits, labels, torch.tensor([[0.0, 4.0, 1.0]]))
    assert weighted > equal
