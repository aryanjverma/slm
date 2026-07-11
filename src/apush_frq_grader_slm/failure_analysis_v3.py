"""Reproducible v2 failure analysis used to motivate and verify v3."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.eval import _quadratic_weighted_kappa
from apush_frq_grader_slm.filters import parse_grade_json
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.rubric import CRITERIA
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import has_repetition, normalize_grade_output


def analyze_v2_failures(
    *,
    results_path: Path,
    cases_path: Path,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(results_path)
    cases = [FRQCase.model_validate(row) for row in read_jsonl(cases_path)]
    case_by_id = {case.id: case for case in cases}
    diagnostic_categories: dict[str, str] = {}
    supplied_category_counts: dict[str, int] = {}
    if diagnostics_path and diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        supplied_category_counts = diagnostics.get("categories", {})
        for category, ids in diagnostics.get("example_case_ids", {}).items():
            for case_id in ids:
                diagnostic_categories[case_id] = category

    raw_valid = sum(bool(row.get("structured_output_valid")) for row in rows)
    parseable = 0
    normalized_count = 0
    absolute_error = 0
    within_one = 0
    ref_totals: list[int] = []
    pred_totals: list[int] = []
    score_patterns: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    repetition_count = 0
    category_counts: Counter[str] = Counter()
    dimensions: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        case = case_by_id[str(row["case_id"])]
        response = str(row.get("response", ""))
        layered = normalize_grade_output(response)
        conventionally_parsed, _ = parse_grade_json(response)
        if conventionally_parsed is not None:
            parseable += 1
        category = diagnostic_categories.get(case.id, "unclassified")
        usable = layered.layered_valid and category != "malformed_or_non_json"
        if usable:
            normalized_count += 1
        actions.update(layered.normalization_actions)
        repetition_count += int(has_repetition(response))
        category_counts[category] += 1

        reference = case.reference_scores.total
        ref_totals.append(reference)
        payload = layered.normalized_payload if usable else None
        predicted = payload.get("total") if payload else None
        if isinstance(predicted, int):
            error = abs(predicted - reference)
            absolute_error += error
            within_one += int(error <= 1)
            pred_totals.append(predicted)
            scores = payload["scores"]
            score_patterns[str(tuple(scores[key] for key in CRITERIA))] += 1
        else:
            pred_totals.append(-1)

        metric = {
            "count": 1.0,
            "raw_valid": float(bool(row.get("structured_output_valid"))),
            "parseable": float(conventionally_parsed is not None),
            "normalized_valid": float(usable),
            "absolute_error": float(abs(predicted - reference))
            if isinstance(predicted, int)
            else -1.0,
            "truncated": float(category == "likely_max_token_truncation"),
            "repetition": float(has_repetition(response)),
        }
        words = len(case.student_response.split())
        length_band = (
            "under_250"
            if words < 250
            else "250_399"
            if words < 400
            else "400_599"
            if words < 600
            else "600_plus"
        )
        year_match = re.search(r"ap_central_(\d{4})_", case.id)
        year = int(year_match.group(1)) if year_match else case.provenance.year
        rubric = "2023_leq" if year is not None and year <= 2023 else "2024_2026_leq"
        for dimension, value in {
            "essay_length": length_band,
            "rubric_version": rubric,
            "reference_total": str(reference),
        }.items():
            dimensions[dimension][value].append(metric)

    count = len(rows)
    qwk = _quadratic_weighted_kappa(ref_totals, pred_totals) if count >= 5 else None
    provenance = _provenance_warnings(cases)
    return {
        "inputs": {
            "results_path": results_path.as_posix(),
            "cases_path": cases_path.as_posix(),
            "rows": count,
        },
        "overall": {
            "raw_schema_valid": raw_valid,
            "raw_schema_valid_rate": _rate(raw_valid, count),
            "parseable_json": parseable,
            "parseable_json_rate": _rate(parseable, count),
            "normalized_usable": normalized_count,
            "normalized_usable_rate": _rate(normalized_count, count),
            "normalized_total_mae": (
                round(absolute_error / normalized_count, 4) if normalized_count else 0
            ),
            "normalized_within_one_rate": _rate(within_one, normalized_count),
            "normalized_qwk": round(qwk, 4) if qwk is not None else None,
            "repetition_count": repetition_count,
            "repetition_rate": _rate(repetition_count, count),
        },
        "diagnostic_categories": dict(
            sorted(
                supplied_category_counts.items()
                if supplied_category_counts
                else category_counts.items()
            )
        ),
        "score_distribution": dict(score_patterns.most_common()),
        "normalization_actions": dict(sorted(actions.items())),
        "dimensions": {
            dimension: {
                value: _aggregate_metrics(bucket)
                for value, bucket in sorted(values.items())
            }
            for dimension, values in sorted(dimensions.items())
        },
        "provenance_and_extraction_warnings": provenance,
    }


def render_failure_report(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# V2 Failure Analysis",
        "",
        "This report is generated from saved model outputs. It is diagnostic, not a verified "
        "golden evaluation.",
        "",
        "## Overall",
        "",
        f"- Raw strict schema: {overall['raw_schema_valid']}/{report['inputs']['rows']} "
        f"({overall['raw_schema_valid_rate']:.1%})",
        f"- Parseable JSON: {overall['parseable_json']}/{report['inputs']['rows']} "
        f"({overall['parseable_json_rate']:.1%})",
        f"- Usable after representation-only normalization: "
        f"{overall['normalized_usable']}/{report['inputs']['rows']} "
        f"({overall['normalized_usable_rate']:.1%})",
        f"- Normalized total MAE: {overall['normalized_total_mae']}",
        f"- Normalized within-one: {overall['normalized_within_one_rate']:.1%}",
        f"- Normalized QWK: {overall['normalized_qwk']}",
        f"- Repetition: {overall['repetition_count']}/{report['inputs']['rows']}",
        "",
        "## Failure categories",
        "",
    ]
    lines.extend(
        f"- {name}: {value}" for name, value in report["diagnostic_categories"].items()
    )
    lines.extend(["", "## Dominant normalized score vectors", ""])
    lines.extend(f"- `{name}`: {value}" for name, value in report["score_distribution"].items())
    lines.extend(["", "## Length, rubric, and reference-score slices", ""])
    for dimension, values in report["dimensions"].items():
        lines.append(f"### {dimension.replace('_', ' ').title()}")
        lines.append("")
        for value, metrics in values.items():
            lines.append(
                f"- {value}: n={metrics['count']}, raw={metrics['raw_schema_valid_rate']:.1%}, "
                f"parseable={metrics['parseable_rate']:.1%}, "
                f"normalized={metrics['normalized_valid_rate']:.1%}, "
                f"MAE={metrics['normalized_total_mae']}, "
                f"truncated={metrics['truncation_rate']:.1%}, "
                f"repetition={metrics['repetition_rate']:.1%}"
            )
        lines.append("")
    lines.extend(["## Normalization actions", ""])
    lines.extend(f"- {name}: {value}" for name, value in report["normalization_actions"].items())
    lines.extend(["", "## Provenance and extraction warnings", ""])
    lines.extend(
        f"- {name}: {value}"
        for name, value in report["provenance_and_extraction_warnings"].items()
    )
    lines.append("")
    return "\n".join(lines)


def _aggregate_metrics(bucket: list[dict[str, float]]) -> dict[str, float]:
    count = len(bucket)
    return {
        "count": count,
        "raw_schema_valid_rate": round(sum(item["raw_valid"] for item in bucket) / count, 4),
        "parseable_rate": round(sum(item["parseable"] for item in bucket) / count, 4),
        "normalized_valid_rate": round(
            sum(item["normalized_valid"] for item in bucket) / count, 4
        ),
        "normalized_total_mae": _valid_mae(bucket),
        "truncation_rate": round(sum(item["truncated"] for item in bucket) / count, 4),
        "repetition_rate": round(sum(item["repetition"] for item in bucket) / count, 4),
    }


def _provenance_warnings(cases: list[FRQCase]) -> dict[str, int]:
    return {
        "missing_source_url": sum(not case.provenance.source_url for case in cases),
        "missing_file_sha256": sum(not case.provenance.file_sha256 for case in cases),
        "missing_extraction_method": sum(not case.provenance.extraction_method for case in cases),
        "missing_extraction_confidence": sum(
            case.provenance.extraction_confidence is None for case in cases
        ),
        "commentary_text_present_in_essay": sum(
            "scoring commentary" in case.student_response.lower() for case in cases
        ),
    }


def _valid_mae(bucket: list[dict[str, float]]) -> float:
    errors = [item["absolute_error"] for item in bucket if item["absolute_error"] >= 0]
    return round(sum(errors) / len(errors), 4) if errors else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
