"""Evaluate a Hugging Face or local fine-tuned model against held-out LEQ cases."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from pydantic import BaseModel

from apush_frq_grader_slm.behavior import SYSTEM_PROMPT
from apush_frq_grader_slm.data import format_user_message
from apush_frq_grader_slm.eval import (
    score_response,
    summarize,
    summarize_by_dimensions,
    summarize_real_eval,
    summarize_real_eval_by_rubric,
)
from apush_frq_grader_slm.eval_diagnostics import diagnose_rows
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.rubric import rubric_version_for_year
from apush_frq_grader_slm.schemas import EvalResult, FRQCase


CB_CASE_ID = re.compile(
    r"^ap_central_(?P<year>\d{4})_leq(?P<leq>\d+)_set(?P<set>\d+)_(?P<sample>\d+[A-C])$"
)


def load_model_and_tokenizer(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_model_id = json.loads(adapter_config.read_text())["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, path)
        model = model.merge_and_unload()  # fold LoRA into base -> faster inference
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                dtype=dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )

    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    if args.real_eval is None:
        args.real_eval = args.eval_path.name == "eval_cb_cases.jsonl"
    cases = [
        hydrate_cb_provenance(FRQCase.model_validate(row))
        for row in read_jsonl(args.eval_path)
    ]
    if len({case.id for case in cases}) != len(cases):
        raise RuntimeError(f"Evaluation file contains duplicate case IDs: {args.eval_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_real_results" if args.real_eval else "_results"
    results_path = args.output_dir / f"{args.model_name}{suffix}.jsonl"
    existing = load_existing_results(results_path, cases, args.model_name) if args.resume else []
    if not args.resume and results_path.exists():
        results_path.unlink()
    pending_cases = select_pending_cases(cases, existing)
    model = tokenizer = device = None
    if pending_cases:
        model, tokenizer, device = load_model_and_tokenizer(args.model)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        device = "not_loaded"
    started_at = time.monotonic()
    print(
        f"Starting {args.model_name}: total={len(cases)}, completed={len(existing)}, "
        f"pending={len(pending_cases)}, device={device}, "
        f"max_new_tokens={args.max_new_tokens}, output={results_path}",
        flush=True,
    )

    results_by_id = {result.case_id: result for result in existing}
    with results_path.open("a", encoding="utf-8", newline="\n") as results_file:
        progress = tqdm(pending_cases, desc=args.model_name, unit="case", dynamic_ncols=True)
        for index, case in enumerate(progress, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": format_user_message(case.prompt, case.student_response),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            result = score_response(case, response.strip(), args.model_name)
            results_by_id[case.id] = result
            payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
            results_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            results_file.flush()
            elapsed = time.monotonic() - started_at
            seconds_per_case = elapsed / index
            eta_seconds = seconds_per_case * (len(pending_cases) - index)
            progress.set_postfix(
                elapsed=f"{elapsed / 60:.1f}m",
                eta=f"{eta_seconds / 60:.1f}m",
            )
            if index % args.log_every == 0 or index == len(pending_cases):
                print(
                    f"Progress {args.model_name}: {len(existing) + index}/{len(cases)} cases, "
                    f"elapsed={elapsed / 60:.1f}m, eta={eta_seconds / 60:.1f}m",
                    flush=True,
                )

    results = [results_by_id[case.id] for case in cases if case.id in results_by_id]
    diagnostic_path = args.output_dir / f"{args.model_name}{suffix}_diagnostics.json"
    diagnostics = diagnose_rows(
        [result.model_dump(mode="json") for result in results],
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        token_margin=2,
    )
    diagnostic_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Diagnostics: {diagnostic_path}", flush=True)

    # Generation is the expensive part and is already saved to results_path. If
    # summarizing fails, do not lose the run — report how to recompute the
    # summary from the saved per-case results without re-generating.
    try:
        dimensions = summarize_by_dimensions(results, cases)
        (args.output_dir / f"{args.model_name}{suffix}_dimensions.json").write_text(
            json.dumps(dimensions, indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.real_eval:
            summary = summarize_real_eval(results, cases)
            write_jsonl(args.output_dir / f"{args.model_name}_real_summary.jsonl", [summary])
            by_rubric = summarize_real_eval_by_rubric(results, cases)
            write_jsonl(
                args.output_dir / f"{args.model_name}_real_by_rubric.jsonl",
                [
                    {"rubric_version": version, **item.model_dump(mode="json")}
                    for version, item in by_rubric.items()
                ],
            )
        else:
            write_jsonl(
                args.output_dir / f"{args.model_name}_summary.jsonl",
                [summarize(results, args.model_name)],
            )
    except Exception:
        real_flag = " --real-eval" if args.real_eval else ""
        print(
            f"\nGeneration finished and results are saved to {results_path}, but "
            f"summarizing failed (traceback below). Recompute the summary without "
            f"re-generating:\n"
            f"  python scripts/summarize_from_results.py "
            f"--results {results_path} --eval-path {args.eval_path} "
            f"--model-name {args.model_name} --output-dir {args.output_dir}{real_flag}\n"
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--model-name", default="hf_model")
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("artifacts/data/eval_cb_cases.jsonl"),
    )
    parser.add_argument(
        "--real-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use real CB eval metrics (row agreement, MAE, QWK, and rubric slices).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_existing_results(
    results_path: Path,
    cases: list[FRQCase],
    model_name: str,
) -> list[EvalResult]:
    """Validate and order saved results so resume never duplicates or mixes runs."""
    if not results_path.exists():
        return []
    results = [EvalResult.model_validate(row) for row in read_jsonl(results_path)]
    allowed_ids = {case.id for case in cases}
    seen: set[str] = set()
    by_id: dict[str, EvalResult] = {}
    for result in results:
        if result.case_id not in allowed_ids:
            raise RuntimeError(f"Saved result has unknown case ID: {result.case_id}")
        if result.case_id in seen:
            raise RuntimeError(f"Saved results contain duplicate case ID: {result.case_id}")
        if result.model_name != model_name:
            raise RuntimeError(
                f"Saved result model mismatch for {result.case_id}: "
                f"{result.model_name!r} != {model_name!r}"
            )
        seen.add(result.case_id)
        by_id[result.case_id] = result
    return [by_id[case.id] for case in cases if case.id in by_id]


def select_pending_cases(cases: list[FRQCase], results: list[EvalResult]) -> list[FRQCase]:
    completed_ids = {result.case_id for result in results}
    return [case for case in cases if case.id not in completed_ids]


def hydrate_cb_provenance(case: FRQCase) -> FRQCase:
    """Recover evaluation-only year/set/rubric metadata encoded in legacy CB case IDs."""
    match = CB_CASE_ID.fullmatch(case.id)
    if match is None:
        return case
    hydrated = case.model_copy(deep=True)
    year = int(match.group("year"))
    leq_number = int(match.group("leq"))
    set_number = int(match.group("set"))
    filename = f"ap{str(year)[-2:]}-apc-us-history-leq{leq_number}-set-{set_number}.pdf"
    hydrated.provenance.source_type = "college_board"
    hydrated.provenance.source_id = case.id
    hydrated.provenance.source_url = f"https://apcentral.collegeboard.org/media/pdf/{filename}"
    hydrated.provenance.year = year
    hydrated.provenance.leq_number = leq_number
    hydrated.provenance.set_number = set_number
    hydrated.provenance.sample_id = match.group("sample")
    hydrated.provenance.rubric_version = rubric_version_for_year(year)
    hydrated.provenance.prompt_family_id = (
        f"ap_central_{year}_leq{leq_number}_set{set_number}"
    )
    return hydrated


if __name__ == "__main__":
    main()
