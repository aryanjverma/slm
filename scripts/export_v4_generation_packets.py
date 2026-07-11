"""Export per-batch generation packets with AMSCO memory for parallel writers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.knowledge.amsco import facts_for_prompt, load_kb
from apush_frq_grader_slm.prompts_v4 import (
    LEQ_RUBRIC_TEXT,
    STUDENT_SYSTEM_PROMPT_V4,
    format_student_user_message,
)


def _memory_block(bundle: dict, persona: dict) -> str:
    knowledge = str(persona.get("historical_knowledge", "competent"))
    misconception = str(persona.get("misconception", "none"))
    facts = list(bundle.get("key_facts") or [])
    evidence = list(bundle.get("evidence_bank") or [])
    context = list(bundle.get("context_hooks") or [])
    wrong = list(bundle.get("misconceptions") or [])

    keep = {"weak": 4, "uneven": 8, "competent": 14, "strong": 22}.get(knowledge, 12)
    lines: list[str] = []
    if context:
        lines.append("Broader context you vaguely remember:")
        lines.extend(f"- {item}" for item in context[:4])
    if facts:
        lines.append("Facts you remember (may be incomplete):")
        lines.extend(f"- {item}" for item in facts[:keep])
    if evidence and knowledge in {"competent", "strong"}:
        lines.append("Specific evidence names you might use:")
        lines.extend(f"- {item}" for item in evidence[:8])
    if misconception != "none" and wrong:
        lines.append(
            "You may confuse or misstate some of the following (do not label them as wrong):"
        )
        lines.extend(f"- {item}" for item in wrong[:3])
    return "\n".join(lines)


def _target_guidance(scores: dict, total: int) -> str:
    return (
        f"Hidden calibration target {total}/6 with scores "
        f"thesis={scores['thesis']}, contextualization={scores['contextualization']}, "
        f"evidence={scores['evidence']}, analysis_reasoning={scores['analysis_reasoning']}. "
        "Earn exactly these points: if a criterion is 0, that element must truly be absent "
        "(no thesis / no broader context / fewer than two specific examples / no reasoning "
        "structure). If evidence is 1, name-drop at least two facts without arguing them; "
        "if 2, use them to prove the thesis. If analysis_reasoning is 1, structure with "
        "causation/comparison/CCOT but skip complexity; if 2, add nuance or a counterargument."
    )


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks)
    kb = load_kb(args.kb)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    packets: list[dict] = []
    for task in tasks:
        persona = task.get("persona") or {}
        bundle = facts_for_prompt(kb, task["prompt"], max_facts=30)
        memory = _memory_block(bundle, persona)
        system = STUDENT_SYSTEM_PROMPT_V4.format(
            time_budget_minutes=persona.get("time_budget_minutes", 30)
        )
        user = format_student_user_message(
            prompt=task["prompt"],
            persona_dict=persona,
            amsco_memory_block=memory,
            style_reference=str(task.get("style_excerpt") or ""),
            target_guidance=_target_guidance(task["target_scores"], int(task["target_total"])),
        )
        packets.append(
            {
                "task_id": task["task_id"],
                "seed_id": task["seed_id"],
                "prompt": task["prompt"],
                "target_scores": task["target_scores"],
                "target_total": task["target_total"],
                "failure_type": task["failure_type"],
                "length_band": task["length_band"],
                "persona": persona,
                "period": task.get("period"),
                "amsco_chapter_ids": bundle.get("chapter_ids") or task.get("amsco_chapter_ids"),
                "system_prompt": system,
                "user_message": user,
                "rubric_text": LEQ_RUBRIC_TEXT,
            }
        )

    n = len(packets)
    batch_size = max(1, (n + args.batches - 1) // args.batches)
    for batch_index in range(args.batches):
        start = batch_index * batch_size
        chunk = packets[start : start + batch_size]
        if not chunk:
            break
        path = args.output_dir / f"gen_packet_batch_{batch_index:02d}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in chunk),
            encoding="utf-8",
        )
        print(f"Wrote {len(chunk)} packets -> {path}")
    manifest = {
        "total_packets": n,
        "batches": args.batches,
        "batch_size": batch_size,
        "rubric_embedded": True,
    }
    (args.output_dir / "packets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=Path("artifacts/data/v4/synth_tasks_v4.jsonl")
    )
    parser.add_argument(
        "--kb", type=Path, default=Path("artifacts/knowledge/amsco_2016_kb.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/data/v4/packets")
    )
    parser.add_argument("--batches", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
