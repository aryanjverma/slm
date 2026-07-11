"""Evaluate one v3 candidate on set1, or on the explicitly locked set2 exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.dataset_v3 import format_v3_user_message
from apush_frq_grader_slm.eval_v3 import (
    V3EvalRecord,
    assert_final_lock,
    build_run_identity,
    load_resumable_records,
    make_record,
    model_fingerprint,
    select_official_split,
    sha256_path,
    summarize_v3,
)
from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.structured_output_v3 import (
    V3_SYSTEM_PROMPT,
    build_balanced_json_stopping_criteria,
    build_score_enum_logits_processor,
    first_complete_json_end,
    has_repetition,
    trim_after_first_json,
)


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model_name_or_path"], dtype=dtype, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = PeftModel.from_pretrained(base, path).merge_and_unload()
        model = model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
    if device == "cpu" and not adapter_config.exists():
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def main() -> None:
    import torch
    from transformers import LogitsProcessorList, StoppingCriteriaList

    args = parse_args()
    all_cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]
    cases = select_official_split(all_cases, final_evaluation=args.final_evaluation)
    split = "set2" if args.final_evaluation else "set1"
    settings = {
        "max_new_tokens": 320,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
        "balanced_json_stopping": True,
        "score_enum_constraint": "scores_only_v2",
    }
    identity = build_run_identity(
        model_name=args.model_name,
        model_hash=model_fingerprint(args.model),
        data_hash=sha256_path(args.eval_path),
        split=split,
        decoding_settings=settings,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{identity.run_id}_{split}_results.jsonl"
    receipt_path = args.output_dir / "set2_final_receipt.json"
    if args.final_evaluation:
        if args.lock_manifest is None:
            raise SystemExit("--final-evaluation requires --lock-manifest")
        assert_final_lock(args.lock_manifest, identity, receipt_path)
    existing = load_resumable_records(results_path, identity=identity, cases=cases)
    completed = {record.case_id for record in existing}
    pending = [case for case in cases if case.id not in completed]
    model, tokenizer, device = (
        load_model(args.model) if pending else (None, None, "not_loaded")
    )
    print(
        f"v3 {split}: total={len(cases)} complete={len(existing)} pending={len(pending)} "
        f"device={device} run_id={identity.run_id}"
    )
    records: dict[str, V3EvalRecord] = {record.case_id: record for record in existing}
    with results_path.open("a", encoding="utf-8", newline="\n") as output:
        for index, case in enumerate(pending, start=1):
            messages = [
                {"role": "system", "content": V3_SYSTEM_PROMPT},
                {"role": "user", "content": format_v3_user_message(case)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = int(inputs["input_ids"].shape[-1])
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList(
                        [build_score_enum_logits_processor(tokenizer, prompt_length)]
                    ),
                    stopping_criteria=StoppingCriteriaList(
                        [build_balanced_json_stopping_criteria(tokenizer, prompt_length)]
                    ),
                )
            completion_ids = generated[0][prompt_length:]
            raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            _, has_trailing_text = trim_after_first_json(raw)
            if first_complete_json_end(raw) is not None:
                finish = "balanced_json"
            elif len(completion_ids) >= 320:
                finish = "length"
            else:
                finish = "eos"
            record = make_record(
                case_id=case.id,
                identity=identity,
                raw_response=raw,
                prompt_tokens=prompt_length,
                completion_tokens=len(completion_ids),
                finish_reason=finish,
                repetition_detected=has_repetition(raw),
            )
            if has_trailing_text:
                record.normalization_actions.insert(0, "ignored_trailing_after_balanced_object")
            records[case.id] = record
            output.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=True) + "\n")
            output.flush()
            print(f"{index}/{len(pending)} {case.id}: {finish}", flush=True)

    ordered = [records[case.id] for case in cases]
    summary = summarize_v3(ordered, cases, identity)
    summary_payload = summary.model_dump(mode="json")
    summary_payload["identity"] = identity.model_dump(mode="json")
    summary_path = args.output_dir / f"{identity.run_id}_{split}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.final_evaluation:
        receipt_path.write_text(
            json.dumps(
                {"run_id": identity.run_id, "results": results_path.as_posix()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--eval-path", type=Path, default=Path("artifacts/data/eval_cb_cases.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/v3"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument("--lock-manifest", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
