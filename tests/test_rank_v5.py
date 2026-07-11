from __future__ import annotations

import json
from pathlib import Path

from apush_frq_grader_slm.checkpoint_rank_v5 import (
    build_feedback_ranking,
    build_scorer_ranking,
    build_v5_checkpoint_ranking,
    feedback_rank_key,
    normalize_ranking_candidate,
    scorer_rank_key,
)
from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.prompts_v5 import V5_FEEDBACK_SYSTEM_PROMPT, V5_SCORER_SYSTEM_PROMPT
from apush_frq_grader_slm.training_v5 import export_v5_chat_rows, write_bundle_manifest


def _scorer_summary(*, qwk, mae, evidence, thesis=0.5, context=0.4, analysis=0.3) -> dict:
    return {
        "model_name": "scorer",
        "qwk": qwk,
        "total_mae": mae,
        "criterion_exact_rates": {
            "thesis": thesis,
            "contextualization": context,
            "evidence": evidence,
            "analysis_reasoning": analysis,
        },
        "evidence_grounding_rate": 0.8,
        "structured_output_valid_rate": 0.9,
        "feedback_fallback_rate": 0.1,
    }


def _feedback_summary(*, grounding, validity, fallback) -> dict:
    return {
        "model_name": "feedback",
        "qwk": 0.2,
        "total_mae": 1.0,
        "criterion_exact_rates": {"evidence": 0.2},
        "evidence_grounding_rate": grounding,
        "structured_output_valid_rate": validity,
        "feedback_fallback_rate": fallback,
    }


def test_scorer_rank_key_prefers_qwk_then_mae_evidence_macro() -> None:
    worse_qwk = {"adapter": "a", "summary": _scorer_summary(qwk=0.2, mae=0.5, evidence=0.9)}
    better_qwk = {"adapter": "b", "summary": _scorer_summary(qwk=0.4, mae=1.5, evidence=0.1)}
    assert scorer_rank_key(better_qwk) > scorer_rank_key(worse_qwk)

    higher_mae = {"adapter": "c", "summary": _scorer_summary(qwk=0.4, mae=1.2, evidence=0.5)}
    lower_mae = {"adapter": "d", "summary": _scorer_summary(qwk=0.4, mae=0.8, evidence=0.5)}
    assert scorer_rank_key(lower_mae) > scorer_rank_key(higher_mae)

    low_evidence = {
        "adapter": "e",
        "summary": _scorer_summary(qwk=0.4, mae=0.8, evidence=0.2, thesis=0.9, context=0.9, analysis=0.9),
    }
    high_evidence = {
        "adapter": "f",
        "summary": _scorer_summary(qwk=0.4, mae=0.8, evidence=0.6, thesis=0.1, context=0.1, analysis=0.1),
    }
    assert scorer_rank_key(high_evidence) > scorer_rank_key(low_evidence)


def test_feedback_rank_key_prefers_grounding_validity_and_low_fallback() -> None:
    weak = {"adapter": "a", "summary": _feedback_summary(grounding=0.7, validity=0.99, fallback=0.0)}
    strong = {"adapter": "b", "summary": _feedback_summary(grounding=0.9, validity=0.8, fallback=0.5)}
    assert feedback_rank_key(strong) > feedback_rank_key(weak)

    high_fallback = {
        "adapter": "c",
        "summary": _feedback_summary(grounding=0.9, validity=0.95, fallback=0.4),
    }
    low_fallback = {
        "adapter": "d",
        "summary": _feedback_summary(grounding=0.9, validity=0.95, fallback=0.05),
    }
    assert feedback_rank_key(low_fallback) > feedback_rank_key(high_fallback)


def test_build_rankings_select_best_adapters() -> None:
    scorer = build_scorer_ranking(
        [
            {"adapter": "scorer-weak", "summary": _scorer_summary(qwk=0.1, mae=2.0, evidence=0.1)},
            {"adapter": "scorer-best", "summary": _scorer_summary(qwk=0.5, mae=1.0, evidence=0.4)},
        ]
    )
    feedback = build_feedback_ranking(
        [
            {"adapter": "feedback-weak", "summary": _feedback_summary(grounding=0.6, validity=0.9, fallback=0.2)},
            {"adapter": "feedback-best", "summary": _feedback_summary(grounding=0.95, validity=0.99, fallback=0.01)},
        ]
    )
    assert scorer["selected_adapter"] == "scorer-best"
    assert feedback["selected_adapter"] == "feedback-best"
    combined = build_v5_checkpoint_ranking(
        scorer_candidates=scorer["ranking"],
        feedback_candidates=feedback["ranking"],
    )
    assert combined["selected_scorer"] == "scorer-best"
    assert combined["selected_feedback"] == "feedback-best"


def test_normalize_accepts_raw_real_summary_json() -> None:
    raw = _scorer_summary(qwk=0.3, mae=1.1, evidence=0.2)
    raw["model_name"] = "scorer-03"
    item = normalize_ranking_candidate(raw, source="/tmp/scorer-03_real_summary.json")
    assert item["adapter"] == "scorer-03"
    assert item["summary"]["qwk"] == 0.3


def test_bundle_exports_prompt_files_with_hashes(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    for name in ("base", "scorer", "feedback"):
        path = bundle / name
        path.mkdir(parents=True)
        (path / "weights.bin").write_text(name, encoding="utf-8")
    manifest = write_bundle_manifest(
        bundle,
        inherited_base=bundle / "base",
        scorer_adapter=bundle / "scorer",
        feedback_adapter=bundle / "feedback",
    )
    scorer_path = bundle / "prompts" / "scorer_system.txt"
    feedback_path = bundle / "prompts" / "feedback_system.txt"
    assert scorer_path.read_text(encoding="utf-8").rstrip("\n") == V5_SCORER_SYSTEM_PROMPT
    assert feedback_path.read_text(encoding="utf-8").rstrip("\n") == V5_FEEDBACK_SYSTEM_PROMPT
    assert manifest["prompts"]["scorer_system"]["path"] == "prompts/scorer_system.txt"
    assert len(manifest["prompts"]["feedback_system"]["sha256"]) == 64


def test_export_v5_chat_rows_shapes() -> None:
    cases = generate_cases(count=2, split="train", seed=61)
    scorer_rows = export_v5_chat_rows(cases, "scorer")
    feedback_rows = export_v5_chat_rows(cases, "feedback")
    assert len(scorer_rows) == 2
    assert len(feedback_rows) == 2
    for row in scorer_rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        assert set(json.loads(row["messages"][-1]["content"])) == {"scores"}
        assert row["metadata"]["task"] == "scorer"
    for row in feedback_rows:
        assert set(json.loads(row["messages"][-1]["content"])) == {"feedback"}
        assert "Validated scores" in row["messages"][1]["content"]
