"""Train the reviewed v4 dataset with assistant-only QLoRA loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.training_v3 import (
    AssistantOnlyDataCollator,
    assert_assistant_only_example,
    tokenize_assistant_only,
)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Training dependencies missing; run pip install -e ".[train]"') from exc

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No training rows found in {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_rows = []
    supervised_tokens = 0
    for index, row in enumerate(rows):
        example = tokenize_assistant_only(
            row["messages"], tokenizer, max_length=args.max_seq_length
        )
        assert_assistant_only_example(example)
        supervised_tokens += sum(label != -100 for label in example["labels"])
        tokenized_rows.append(example)
    print(
        f"assistant-only preflight: rows={len(rows)} "
        f"supervised_tokens={supervised_tokens} max_seq_length={args.max_seq_length}",
        flush=True,
    )

    class TokenizedDataset:
        def __len__(self) -> int:
            return len(tokenized_rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return tokenized_rows[index]

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
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
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    final_dir = args.output / "final"
    if final_dir.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Final adapter already exists: {final_dir}; choose a fresh --output or "
            "pass --overwrite-output intentionally"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=TokenizedDataset(),
        data_collator=AssistantOnlyDataCollator(tokenizer),
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="epoch",
            save_total_limit=3,
            bf16=args.bf16,
            fp16=not args.bf16,
            optim="paged_adamw_8bit",
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "v4_training_metadata.json").write_text(
        json.dumps(
            {
                "assistant_only_loss": True,
                "rows": len(rows),
                "supervised_tokens": supervised_tokens,
                "epochs": args.epochs,
                "warmup_ratio": args.warmup_ratio,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved fresh v4 adapter to {final_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
