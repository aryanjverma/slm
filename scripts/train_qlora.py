"""QLoRA fine-tuning script for the arithmetic tutor SLM.

Install training extras first:
    pip install -e ".[train]"

Example:
    python scripts/train_qlora.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --data artifacts/data/train_chat.jsonl \
        --output artifacts/models/arithmetic-tutor-v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    args = parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Run: pip install -e \".[train]\""
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def formatting_prompts_func(batch):
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=20,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=max(50, args.max_steps // 2),
            optim="adamw_8bit",
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/train_chat.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/arithmetic-tutor-v1"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    main()
