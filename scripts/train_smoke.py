"""CPU-friendly LoRA smoke training for the Day 2 end-to-end loop.

Uses transformers + PEFT (no Unsloth / 4-bit) so the smoke path runs on machines
without a GPU. For production training, use scripts/train_qlora.py instead.

Example:
    python scripts/train_smoke.py \\
        --data artifacts/smoke/train_chat.jsonl \\
        --output artifacts/models/apush-frq-grader-v1-smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apush_frq_grader_slm.io import read_jsonl


def main() -> None:
    args = parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Training dependencies are missing. Run: pip install -e ".[train]"'
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    texts = [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for row in read_jsonl(args.data)
    ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    step = 0
    while step < args.max_steps:
        for text in texts:
            if step >= args.max_steps:
                break
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length,
            ).to(device)
            labels = batch["input_ids"].clone()
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 5 == 0:
                print(f"step {step}/{args.max_steps} loss={outputs.loss.item():.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved smoke adapter to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("artifacts/smoke/train_chat.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/apush-frq-grader-v1-smoke"),
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main()
