"""CPU-friendly v5 contract smoke: plan → packet → chat rows → two-pass → release gate.

Does not train adapters or require GPU. Weighted-loss check is skipped when torch is absent.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from apush_frq_grader_slm.data import generate_cases
from apush_frq_grader_slm.dataset_v4 import CBSeedProfile
from apush_frq_grader_slm.dataset_v5 import (
    V5_CANDIDATE_COUNT,
    V5_SHARD_COUNT,
    V5_SHARD_SIZE,
    generator_packet,
    plan_v5_tasks,
)
from apush_frq_grader_slm.inference_v5 import grade_two_pass
from apush_frq_grader_slm.release_v5 import evaluate_v5_release
from apush_frq_grader_slm.training_v5 import build_v5_chat_row


def _seed() -> CBSeedProfile:
    return CBSeedProfile(
        seed_id="smoke-style-a",
        prompt="Evaluate the extent to which transportation innovations changed the United States economy.",
        prompt_family_id="smoke-family-a",
        period=4,
        reasoning_skill="causation",
        style_excerpt="i think canals changed markets but it also wasnt equal for everyone ",
        amsco_chapter_ids=("ch10",),
        adapted_prompts=(
            "Evaluate how canals and early railroads reshaped regional markets in the early republic.",
        ),
    )


def _assert_plan(tasks: list) -> dict:
    assert len(tasks) == V5_CANDIDATE_COUNT, f"expected {V5_CANDIDATE_COUNT} tasks"
    shard_counts = Counter(task.shard_id for task in tasks)
    assert len(shard_counts) == V5_SHARD_COUNT, f"expected {V5_SHARD_COUNT} shards"
    assert set(shard_counts.values()) == {V5_SHARD_SIZE}, "expected 30 x 50 shards"
    return {"task_count": len(tasks), "shard_count": len(shard_counts), "shard_size": V5_SHARD_SIZE}


def _assert_score_blind_packet(task) -> dict:
    packet = generator_packet(
        task,
        [{"chapter_id": "ch10", "concept": "Improved waterways lowered freight costs between regions."}],
    )
    leaked = {"target_scores", "target_total", "rubric_text", "score"} & set(packet)
    assert not leaked, f"writer packet leaked scoring keys: {leaked}"
    assert packet["semantic_fact_cards"], "expected paraphrasable fact cards"
    assert "paraphrase" in packet["semantic_fact_cards"][0]["use"]
    return {"score_blind": True, "fact_card_count": len(packet["semantic_fact_cards"])}


def _assert_chat_rows() -> dict:
    case = generate_cases(count=1, split="train", seed=51)[0]
    scorer = build_v5_chat_row(case, "scorer")
    feedback = build_v5_chat_row(case, "feedback")
    scorer_target = json.loads(scorer["messages"][-1]["content"])
    feedback_target = json.loads(feedback["messages"][-1]["content"])
    assert set(scorer_target) == {"scores"}
    assert set(feedback_target) == {"feedback"}
    assert "Validated scores" in feedback["messages"][1]["content"]
    return {
        "case_id": case.id,
        "scorer_keys": sorted(scorer_target),
        "feedback_keys": sorted(feedback_target),
    }


def _assert_two_pass() -> dict:
    calls: list[str] = []

    def generate(adapter: str, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        del messages, max_new_tokens
        calls.append(adapter)
        if adapter == "scorer":
            return json.dumps(
                {
                    "scores": {
                        "thesis": 1,
                        "contextualization": 1,
                        "evidence": 2,
                        "analysis_reasoning": 1,
                    }
                }
            )
        return json.dumps(
            {
                "feedback": {
                    "thesis": "The essay states a defensible claim about economic change.",
                    "contextualization": "The opening places canals in a broader market context.",
                    "evidence": "The essay uses concrete transport examples to support the claim.",
                    "analysis_reasoning": "The causal structure links innovations to regional markets.",
                }
            }
        )

    result = grade_two_pass("Evaluate transport change.", "Student essay about canals.", generate)
    assert calls == ["scorer", "feedback"]
    assert result["total"] == 5
    assert set(result["scores"]) == {
        "thesis",
        "contextualization",
        "evidence",
        "analysis_reasoning",
    }
    assert set(result["feedback"]) == set(result["scores"])
    return {"total": result["total"], "passes": calls}


def _assert_release_gate() -> dict:
    failing = evaluate_v5_release(
        {
            "qwk": 0.10,
            "total_mae": 2.5,
            "total_within_one_rate": 0.40,
            "structured_output_valid_rate": 0.90,
            "evidence_grounding_rate": 0.70,
            "total_score_mean": 2.0,
            "criterion_exact_rates": {
                "thesis": 0.40,
                "contextualization": 0.30,
                "evidence": 0.10,
                "analysis_reasoning": 0.20,
            },
        }
    )
    assert failing["release_ready"] is False
    assert failing["decision"] == "non_production_ready"

    passing = evaluate_v5_release(
        {
            "qwk": 0.41,
            "total_mae": 1.49,
            "total_within_one_rate": 0.61,
            "structured_output_valid_rate": 0.99,
            "evidence_grounding_rate": 0.86,
            "total_score_mean": 4.1,
            "criterion_exact_rates": {
                "thesis": 0.54,
                "contextualization": 0.41,
                "evidence": 0.18,
                "analysis_reasoning": 0.33,
            },
        }
    )
    assert passing["release_ready"] is True
    return {
        "failing_decision": failing["decision"],
        "passing_decision": passing["decision"],
    }


def _optional_weighted_loss() -> dict:
    try:
        import torch

        from apush_frq_grader_slm.training_v5 import weighted_causal_lm_loss
    except ImportError:
        return {"skipped": True, "reason": "torch_unavailable"}

    vocab = 8
    logits = torch.zeros(1, 4, vocab)
    labels = torch.tensor([[-100, 1, 2, 3]])
    weights = torch.tensor([[0.0, 1.0, 4.0, 1.0]])
    for index, token_id in enumerate((1, 2, 3), start=1):
        logits[0, index - 1, token_id] = 10.0
    loss = weighted_causal_lm_loss(logits, labels, weights)
    assert torch.isfinite(loss)
    return {"skipped": False, "loss_finite": True}


def main() -> int:
    tasks = plan_v5_tasks([_seed()])
    summary = {
        "ok": True,
        "plan": _assert_plan(tasks),
        "packet": _assert_score_blind_packet(tasks[0]),
        "chat_rows": _assert_chat_rows(),
        "two_pass": _assert_two_pass(),
        "release_gate": _assert_release_gate(),
        "weighted_loss": _optional_weighted_loss(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — smoke entrypoint reports failure JSON
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        raise SystemExit(1) from exc
