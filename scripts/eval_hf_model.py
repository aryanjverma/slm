"""Evaluate a Hugging Face or local fine-tuned model against held-out LEQ cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from apush_frq_grader_slm.behavior import SYSTEM_PROMPT
from apush_frq_grader_slm.data import format_user_message
from apush_frq_grader_slm.eval import score_response, summarize
from apush_frq_grader_slm.io import read_jsonl, write_jsonl
from apush_frq_grader_slm.schemas import FRQCase


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    cases = [FRQCase.model_validate(row) for row in read_jsonl(args.eval_path)]
    results = []
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": format_user_message(case.prompt, case.student_response),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        results.append(score_response(case, response.strip(), args.model_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.model_name}_results.jsonl", results)
    write_jsonl(args.output_dir / f"{args.model_name}_summary.jsonl", [summarize(results, args.model_name)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--model-name", default="hf_model")
    parser.add_argument("--eval-path", type=Path, default=Path("artifacts/data/eval_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
