"""Rank set1 checkpoint summaries and optionally freeze the passing candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rank_key(item: dict) -> tuple[float, float, float, int]:
    metrics = item["layered_system"]
    qwk = metrics["qwk"] if metrics["qwk"] is not None else -1.0
    small_preference = int("0.5B" in item.get("model_name", ""))
    return (qwk, -metrics["total_mae"], metrics["total_within_one_rate"], small_preference)


def select_candidate(candidates: list[dict]) -> dict:
    """Rank by QWK/MAE/within-one and prefer 0.5B only for an effective tie."""
    ranked = sorted(candidates, key=rank_key, reverse=True)
    best = ranked[0]
    best_metrics = best["layered_system"]
    tied = []
    for item in ranked:
        metrics = item["layered_system"]
        if (
            abs(_qwk(metrics) - _qwk(best_metrics)) <= 0.01
            and abs(metrics["total_mae"] - best_metrics["total_mae"]) <= 0.05
            and abs(
                metrics["total_within_one_rate"]
                - best_metrics["total_within_one_rate"]
            )
            <= 0.02
        ):
            tied.append(item)
    return next((item for item in tied if "0.5B" in item.get("model_name", "")), best)


def _qwk(metrics: dict) -> float:
    return metrics["qwk"] if metrics["qwk"] is not None else -1.0


def build_ranking(candidates: list[dict]) -> dict:
    ranked = sorted(candidates, key=rank_key, reverse=True)
    passing = [item for item in candidates if item.get("acceptance", {}).get("passed")]
    winner = select_candidate(passing) if passing else None
    return {
        "ranking": ranked,
        "selected": winner,
        "set2_locked": winner is None,
        "reason": None if winner is not None else "no_set1_candidate_passed",
    }


def main() -> None:
    args = parse_args()
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    output = build_ranking(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    winner = output["selected"]
    if winner is None:
        if args.lock_manifest:
            args.lock_manifest.unlink(missing_ok=True)
        print("No checkpoint passes set1 acceptance; set2 remains locked")
        print(f"Ranking written to {args.output}")
        return
    if args.lock_manifest:
        identity = {key: winner[key] for key in ("model_name",) if key in winner}
        identity.update(winner.get("identity", {}))
        identity["set1_acceptance_passed"] = True
        args.lock_manifest.write_text(
            json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"Selected {winner['model_name']} ({winner['run_id']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/eval/v3/checkpoint_ranking.json")
    )
    parser.add_argument("--lock-manifest", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
