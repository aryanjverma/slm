from scripts.rank_v3_checkpoints import build_ranking, select_candidate


def _candidate(name: str, qwk: float, mae: float, within: float) -> dict:
    return {
        "model_name": name,
        "layered_system": {
            "qwk": qwk,
            "total_mae": mae,
            "total_within_one_rate": within,
        },
    }


def test_prefers_smaller_model_only_when_effectively_tied() -> None:
    small = _candidate("Qwen2.5-0.5B-v3", 0.40, 1.20, 0.65)
    large = _candidate("Qwen2.5-1.5B-v3", 0.405, 1.18, 0.66)
    assert select_candidate([large, small]) is small

    clearly_better = _candidate("Qwen2.5-1.5B-v3", 0.45, 1.10, 0.70)
    assert select_candidate([clearly_better, small]) is clearly_better


def test_ranking_cleanly_locks_set2_when_nothing_passes() -> None:
    candidate = _candidate("Qwen2.5-0.5B-v3", 0.20, 2.0, 0.40)
    candidate["acceptance"] = {"passed": False}
    ranking = build_ranking([candidate])
    assert ranking["selected"] is None
    assert ranking["set2_locked"] is True
    assert ranking["reason"] == "no_set1_candidate_passed"
