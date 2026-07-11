"""Rank v5 scorer and feedback checkpoint evaluation summaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def scorer_rank_key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Rank by QWK, then lower MAE, evidence exact, then macro criterion accuracy."""
    summary = _summary(item)
    exact = summary.get("criterion_exact_rates") or {}
    if not isinstance(exact, Mapping):
        exact = {}
    evidence = float(exact.get("evidence") or 0.0)
    values = [float(exact[key]) for key in exact if _is_number(exact[key])]
    macro = sum(values) / len(values) if values else 0.0
    return (_qwk(summary), -_mae(summary), evidence, macro)


def feedback_rank_key(item: Mapping[str, Any]) -> tuple[float, float, float]:
    """Rank by grounding rate, schema validity, then lower feedback fallback rate."""
    summary = _summary(item)
    return (
        float(summary.get("evidence_grounding_rate") or 0.0),
        float(summary.get("structured_output_valid_rate") or 0.0),
        -float(summary.get("feedback_fallback_rate") or 0.0),
    )


def build_scorer_ranking(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted((dict(item) for item in candidates), key=scorer_rank_key, reverse=True)
    selected = ranked[0] if ranked else None
    return {
        "task": "scorer",
        "ranking": ranked,
        "selected": selected,
        "selected_adapter": _adapter_path(selected) if selected is not None else None,
    }


def build_feedback_ranking(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted((dict(item) for item in candidates), key=feedback_rank_key, reverse=True)
    selected = ranked[0] if ranked else None
    return {
        "task": "feedback",
        "ranking": ranked,
        "selected": selected,
        "selected_adapter": _adapter_path(selected) if selected is not None else None,
    }


def build_v5_checkpoint_ranking(
    *,
    scorer_candidates: Sequence[Mapping[str, Any]] = (),
    feedback_candidates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    scorer = build_scorer_ranking(scorer_candidates)
    feedback = build_feedback_ranking(feedback_candidates)
    return {
        "scorer": scorer["ranking"],
        "feedback": feedback["ranking"],
        "selected_scorer": scorer["selected_adapter"],
        "selected_feedback": feedback["selected_adapter"],
        "scorer_selected": scorer["selected"],
        "feedback_selected": feedback["selected"],
    }


def normalize_ranking_candidate(payload: Mapping[str, Any], *, source: str | None = None) -> dict[str, Any]:
    """Accept either a notebook-style candidate or a raw ``*_real_summary.json`` body."""
    if "summary" in payload and isinstance(payload["summary"], Mapping):
        summary = dict(payload["summary"])
        adapter = payload.get("adapter") or summary.get("adapter") or summary.get("model_name")
        item = {"adapter": adapter, "summary": summary}
        if "model_name" in payload:
            item["model_name"] = payload["model_name"]
        elif "model_name" in summary:
            item["model_name"] = summary["model_name"]
        return item
    summary = dict(payload)
    adapter = summary.get("adapter") or summary.get("model_name") or source
    return {
        "adapter": adapter,
        "model_name": summary.get("model_name"),
        "summary": summary,
    }


def _summary(item: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = item.get("summary")
    if isinstance(nested, Mapping):
        return nested
    return item


def _adapter_path(item: Mapping[str, Any] | None) -> str | None:
    if item is None:
        return None
    adapter = item.get("adapter")
    if adapter:
        return str(adapter)
    summary = _summary(item)
    for key in ("adapter", "model_name"):
        if summary.get(key):
            return str(summary[key])
    return None


def _qwk(summary: Mapping[str, Any]) -> float:
    value = summary.get("qwk")
    return float(value) if value is not None and _is_number(value) else -1.0


def _mae(summary: Mapping[str, Any]) -> float:
    value = summary.get("total_mae")
    return float(value) if value is not None and _is_number(value) else float("inf")


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True
