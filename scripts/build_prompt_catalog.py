"""Build the original APUSH LEQ prompt catalog and reject split leakage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from apush_frq_grader_slm.io import write_jsonl
from apush_frq_grader_slm.prompt_catalog import (
    DEFAULT_HARD_SIMILARITY_THRESHOLD,
    DEFAULT_REVIEW_SIMILARITY_THRESHOLD,
    DEFAULT_SPLIT_SEED,
    ORIGINAL_PROMPT_FAMILIES,
    CatalogValidationReport,
    PromptSplit,
    assign_family_splits,
    validate_prompt_catalog,
)


def _prompts_from_json(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        prompts: list[str] = []
        for item in value:
            prompts.extend(_prompts_from_json(item))
        return prompts
    if isinstance(value, dict):
        for key in ("prompt", "prompt_text", "source_prompt_text"):
            prompt = value.get(key)
            if isinstance(prompt, str):
                return [prompt]
        prompts = []
        for key in ("prompts", "rows", "entries", "cases"):
            if key in value:
                prompts.extend(_prompts_from_json(value[key]))
        return prompts
    return []


def load_protected_prompts(paths: Iterable[Path]) -> list[str]:
    """Load holdout prompts from JSON, JSONL, or one-prompt-per-line text files."""
    protected: list[str] = []
    for path in paths:
        if path.suffix.casefold() == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    protected.extend(_prompts_from_json(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        elif path.suffix.casefold() == ".json":
            try:
                protected.extend(_prompts_from_json(json.loads(path.read_text(encoding="utf-8"))))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        else:
            protected.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return protected


def build_catalog_file(
    output: Path,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
    protected_prompt_paths: Iterable[Path] = (),
    hard_similarity_threshold: float = DEFAULT_HARD_SIMILARITY_THRESHOLD,
    review_similarity_threshold: float = DEFAULT_REVIEW_SIMILARITY_THRESHOLD,
    check_only: bool = False,
    report_output: Path | None = None,
) -> CatalogValidationReport:
    """Build, validate, and optionally write the catalog JSONL."""
    entries = assign_family_splits(ORIGINAL_PROMPT_FAMILIES, seed=seed)
    protected_prompts = load_protected_prompts(protected_prompt_paths)
    report = validate_prompt_catalog(
        entries,
        protected_prompts=protected_prompts,
        hard_similarity_threshold=hard_similarity_threshold,
        review_similarity_threshold=review_similarity_threshold,
    )
    report.raise_for_issues()
    if not check_only:
        write_jsonl(output, entries)
        report_path = report_output or output.with_name(f"{output.stem}_audit.json")
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/prompt_catalog_v1.jsonl"),
        help="Destination JSONL. Parent directories are created automatically.",
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--protected-prompts",
        type=Path,
        action="append",
        default=[],
        help="JSON, JSONL, or text file of holdout prompts; may be repeated.",
    )
    parser.add_argument(
        "--hard-similarity-threshold",
        type=float,
        default=DEFAULT_HARD_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--review-similarity-threshold",
        type=float,
        default=DEFAULT_REVIEW_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without writing the output file.",
    )
    args = parser.parse_args()

    report = build_catalog_file(
        args.output,
        seed=args.seed,
        protected_prompt_paths=args.protected_prompts,
        hard_similarity_threshold=args.hard_similarity_threshold,
        review_similarity_threshold=args.review_similarity_threshold,
        check_only=args.check_only,
        report_output=args.report_output,
    )
    counts = ", ".join(
        f"{split.value}={report.split_counts[split]}" for split in PromptSplit
    )
    action = "Validated" if args.check_only else f"Wrote {args.output} with"
    print(
        f"{action} {report.family_count} original prompt families "
        f"({counts}); max cross-split token similarity="
        f"{report.max_cross_split_similarity:.3f}"
    )


if __name__ == "__main__":
    main()
