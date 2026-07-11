"""Select, review-gate, and finalize the private v5 data corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.dataset_v5 import (
    PRIVATE_USE_NOTICE, annotate_distribution_match, assemble_v5_selection,
    assert_manual_approval, candidate_to_case, file_sha256, manual_review_packet,
    select_v4_replay, style_distribution_audit,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.training_v5 import export_v5_chat_rows


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare-review", "finalize"))
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Private cloud-reviewed candidates (reviews must be real, not inferred).")
    parser.add_argument("--v4-cases", type=Path,
                        default=Path("artifacts/data/v4/judging/train_cases_v4_judged_reviewed.jsonl"))
    parser.add_argument("--golden-cases", type=Path,
                        default=Path("artifacts/data/eval_cb_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data/v5/private"))
    parser.add_argument("--approval", type=Path,
                        default=Path("artifacts/data/v5/private/manual_review_approval_v5.json"))
    parser.add_argument("--overlap-corpus", type=Path, action="append", default=[],
                        help="Repeat for v4 and golden/private case JSONL files used only for dedup.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provisional = args.output_dir / "selected_cases_v5_provisional.jsonl"
    packet_path = args.output_dir / "manual_review_packet_v5.jsonl"

    if args.stage == "prepare-review":
        golden_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.golden_cases)]
        overlap_texts = [case.student_response for case in golden_cases]
        for path in args.overlap_corpus:
            overlap_texts.extend(str(r.get("student_response") or r.get("essay") or "") for r in read_jsonl(path))
        candidates = annotate_distribution_match(read_jsonl(args.candidates), golden_cases)
        train, dev = assemble_v5_selection(
            candidates, source_texts=overlap_texts, golden_cases=golden_cases
        )
        selected = train + dev
        style_audit = style_distribution_audit(
            [row for row in selected if row["selection_class"] == "golden_matched"],
            golden_cases,
        )
        write_jsonl(provisional, selected)
        write_jsonl(packet_path, manual_review_packet(selected))
        _json(args.output_dir / "private_use_manifest_v5.json", {
            "private_use_notice": PRIVATE_USE_NOTICE, "redistribute_rows": False,
            "selected": 600, "train": 540, "dev": 60, "manual_review": 60,
            "approval_required_before_finalize": True,
            "aggregate_style_audit": style_audit,
        })
        print(f"Prepared {packet_path}; review all rows and create the approval file before finalize.")
        return

    assert_manual_approval(packet_path, args.approval)
    if not provisional.exists():
        raise FileNotFoundError("run prepare-review before finalize")
    selected = read_jsonl(provisional)
    packet = {row["task_id"]: row for row in read_jsonl(packet_path)}
    # Apply human-corrected reviewed records to the provisional corpus.
    selected = [packet.get(row["task_id"], row) for row in selected]
    golden_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.golden_cases)]
    selected = annotate_distribution_match(selected, golden_cases)
    overlap_texts = [case.student_response for case in golden_cases]
    for path in args.overlap_corpus:
        overlap_texts.extend(
            str(row.get("student_response") or row.get("essay") or "")
            for row in read_jsonl(path)
        )
    final_train, final_dev = assemble_v5_selection(
        selected, source_texts=overlap_texts, golden_cases=golden_cases
    )
    train = [candidate_to_case(row, split="train") for row in final_train]
    dev = [candidate_to_case(row, split="dev") for row in final_dev]
    replay_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.v4_cases)]
    replay = select_v4_replay(replay_cases)
    train_path = args.output_dir / "train_cases_v5.jsonl"
    dev_path = args.output_dir / "dev_cases_v5.jsonl"
    replay_path = args.output_dir / "replay_cases_v4_for_v5.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    write_jsonl(replay_path, replay)
    train_and_replay = list(train) + list(replay)
    chat_paths = {
        "train_chat_v5_scorer.jsonl": export_v5_chat_rows(train_and_replay, "scorer"),
        "train_chat_v5_feedback.jsonl": export_v5_chat_rows(train_and_replay, "feedback"),
        "dev_chat_v5_scorer.jsonl": export_v5_chat_rows(dev, "scorer"),
        "dev_chat_v5_feedback.jsonl": export_v5_chat_rows(dev, "feedback"),
    }
    artifacts = {
        "train_cases_v5.jsonl": file_sha256(train_path),
        "dev_cases_v5.jsonl": file_sha256(dev_path),
        "replay_cases_v4_for_v5.jsonl": file_sha256(replay_path),
    }
    for name, rows in chat_paths.items():
        path = args.output_dir / name
        write_jsonl(path, rows)
        artifacts[name] = file_sha256(path)
    _json(args.output_dir / "assembly_audit_v5.json", {
        "approved": True, "new_train": len(train), "new_dev": len(dev),
        "v4_replay": len(replay), "training_rows_total": len(train) + len(replay),
        "golden_eval_rows_in_training": 0,
        "artifacts": artifacts,
    })
    print("Finalized 540 v5 train, 60 v5 dev, and 75 v4 replay rows.")


if __name__ == "__main__":
    main()
