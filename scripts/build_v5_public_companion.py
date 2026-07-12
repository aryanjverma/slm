"""Build the redistribution-safe public companion dataset (never the private v5 corpus)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from apush_frq_grader_slm.dataset_v4 import OFFICIAL_SOURCE_TAGS
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


NOTICE = (
    "This is a redistribution-safe companion built from project-authored synthetic data. "
    "It is not the private corpus used for final v5 training."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/data/train_cases.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/public/apush-leq-grader-public"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    if "v5/private" in source.as_posix().lower():
        raise PermissionError("private v5 artifacts cannot be used for the public companion")
    cases = [FRQCase.model_validate(row) for row in read_jsonl(source)]
    if not cases:
        raise ValueError("public companion source is empty")
    for case in cases:
        forbidden = set(case.tags) & OFFICIAL_SOURCE_TAGS
        provenance_type = case.provenance.source_type if case.provenance else ""
        if forbidden or provenance_type in OFFICIAL_SOURCE_TAGS:
            raise PermissionError(f"non-redistributable source marker on {case.id}: {forbidden}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "public_cases.jsonl"
    schema_path = output / "schema.json"
    write_jsonl(data_path, cases)
    schema_path.write_text(
        json.dumps(
            {
                "notice": NOTICE,
                "case_schema": FRQCase.model_json_schema(),
                "runtime_output_contract": {
                    "scores": ["thesis", "contextualization", "evidence", "analysis_reasoning"],
                    "total": "deterministic sum of the four scores",
                    "feedback": ["thesis", "contextualization", "evidence", "analysis_reasoning"],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "apush-leq-grader-public-companion-v1",
        "notice": NOTICE,
        "intended_hub_repo": "aryanjverma/apush-leq-grader-public",
        "private_v5_training_corpus": False,
        "source": "project-authored deterministic synthetic generator output",
        "row_count": len(cases),
        "failure_type_distribution": dict(sorted(Counter(case.failure_type.value for case in cases).items())),
        "artifacts": {
            data_path.name: sha256(data_path),
            schema_path.name: sha256(schema_path),
        },
        "contains": {
            "college_board_essays": False,
            "style_references": False,
            "private_v5_labels": False,
            "manual_review_records": False,
            "per_case_v5_predictions": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# APUSH LEQ Grader Public Companion\n\n"
        f"> {NOTICE}\n\n"
        "This repository demonstrates the FRQCase schema, structured grading target, and "
        "public synthetic baseline data. It contains no private v5 essays, labels, style "
        "references, review packets, or predictions. See `manifest.json` for hashes and counts.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
