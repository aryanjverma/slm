"""Train a v3 Qwen candidate with identical, assistant-only QLoRA settings."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from datasets import load_dataset

from apush_frq_grader_slm.training_v3 import (
    AssistantOnlyDataCollator,
    assert_assistant_only_example,
    tokenize_assistant_only,
)

ALLOWED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
}


def main() -> None:
    args = parse_args()
    if args.model not in ALLOWED_MODELS and not args.allow_custom_model:
        raise SystemExit(f"v3 comparison model must be one of {sorted(ALLOWED_MODELS)}")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Training dependencies missing; run pip install -e ".[train]"') from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def tokenize(row):
        example = tokenize_assistant_only(row["messages"], tokenizer, max_length=3072)
        assert_assistant_only_example(example)
        return example

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    for index in range(min(10, len(tokenized))):
        assert_assistant_only_example(tokenized[index])

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization, device_map="auto"
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

    callbacks = []
    if args.dev_eval_command:
        command_template = args.dev_eval_command

        class DevGenerationCallback(TrainerCallback):
            def on_save(self, callback_args, state, control, **kwargs):
                checkpoint = Path(callback_args.output_dir) / f"checkpoint-{state.global_step}"
                command = command_template.format(checkpoint=str(checkpoint))
                subprocess.run(shlex.split(command), check=True)
                return control

        callbacks.append(DevGenerationCallback())

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        data_collator=AssistantOnlyDataCollator(tokenizer),
        callbacks=callbacks,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=3,
            warmup_ratio=0.05,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            logging_steps=10,
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
    trainer.train()
    trainer.save_model(str(args.output / "final"))
    tokenizer.save_pretrained(str(args.output / "final"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-custom-model", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/v3/train_chat_v3.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dev-eval-command",
        required=True,
        help="Command run after each checkpoint; use {checkpoint} as the saved path placeholder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
