"""Evaluate a Hugging Face or local fine-tuned model against held-out LEQ cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from apush_frq_grader_slm.behavior import SYSTEM_PROMPT
from apush_frq_grader_slm.data import format_user_message
from apush_frq_grader_slm.eval import (
    score_response,
    summarize,
    summarize_by_dimensions,
    summarize_real_eval,
    summarize_real_eval_by_rubric,
)
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def load_model_and_tokenizer(model_id: str):
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
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep litmus and real per-case results in separate files so a real run does
    # not overwrite the litmus results (both tracks share --model-name).
    suffix = "_real_results" if args.real_eval else "_results"
    results_path = args.output_dir / f"{args.model_name}{suffix}.jsonl"

    results = []
    # Write each result as it is scored (and flush) so an interrupt or runtime
    # disconnect leaves a partial results file instead of losing the whole run.
    with results_path.open("w", encoding="utf-8") as results_file:
        for case in tqdm(cases, desc=args.model_name, unit="case"):
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
            results.append(result)
            payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
            results_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
            results_file.flush()

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
    parser.add_argument("--eval-path", type=Path, default=Path("artifacts/data/eval_cases.jsonl"))
    parser.add_argument(
        "--real-eval",
        action="store_true",
        help="Use real CB eval metrics (row agreement, QWK) on eval_real_cases.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
