"""Train a fresh v5 scorer or score-conditioned feedback QLoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apush_frq_grader_slm.io import read_jsonl
from apush_frq_grader_slm.schemas import FRQCase
from apush_frq_grader_slm.training_v5 import (
    V5_MAX_SEQ_LENGTH,
    V5_SCORE_TOKEN_WEIGHT,
    WeightedAssistantCollator,
    build_v5_chat_row,
    sha256_tree,
    tokenize_v5_row,
    weighted_causal_lm_loss,
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
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Training dependencies missing; run pip install -e ".[train]"') from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_rows = prepare_rows(args.data, tokenizer, args.task, args)
    eval_rows = (
        prepare_rows(args.eval_data, tokenizer, args.task, args) if args.eval_data else None
    )

    class Rows:
        def __init__(self, values: list[dict[str, Any]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.values[index]

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    class WeightedTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs=False, **kwargs):
            weights = inputs.pop("loss_weights")
            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = weighted_causal_lm_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    final_dir = args.output / "final"
    if final_dir.exists() and not args.overwrite_output:
        raise FileExistsError(f"Final adapter already exists: {final_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    has_eval = eval_rows is not None
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps" if has_eval else "epoch",
        save_steps=args.eval_steps,
        save_total_limit=3,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    callbacks = [EarlyStoppingCallback(args.early_stopping_patience)] if has_eval else []
    trainer = WeightedTrainer(
        model=model,
        train_dataset=Rows(train_rows),
        eval_dataset=Rows(eval_rows) if eval_rows is not None else None,
        data_collator=WeightedAssistantCollator(tokenizer),
        args=training_args,
        callbacks=callbacks,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metadata = {
        "format": "apush-frq-grader-v5-adapter",
        "task": args.task,
        "inherited_base": str(Path(args.model).resolve()),
        "inherited_base_sha256": sha256_tree(Path(args.model)),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_tree(args.data),
        "eval_data": str(args.eval_data.resolve()) if args.eval_data else None,
        "eval_data_sha256": sha256_tree(args.eval_data) if args.eval_data else None,
        "rows": len(train_rows),
        "max_seq_length": args.max_seq_length,
        "score_token_weight": args.score_token_weight if args.task == "scorer" else 1.0,
        "lora_rank": args.lora_rank,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.grad_accum,
        "seed": args.seed,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
    }
    (final_dir / "v5_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved v5 {args.task} adapter to {final_dir}", flush=True)


def prepare_rows(path: Path, tokenizer: Any, task: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = [FRQCase.model_validate(row) for row in read_jsonl(path)]
    if not cases:
        raise ValueError(f"No cases found in {path}")
    values = []
    for case in cases:
        row = build_v5_chat_row(case, task)  # type: ignore[arg-type]
        values.append(
            tokenize_v5_row(
                row["messages"], tokenizer, task=task, max_length=args.max_seq_length,
                score_token_weight=args.score_token_weight,
            )
        )
    supervised = sum(sum(label != -100 for label in row["labels"]) for row in values)
    print(f"{task} preflight: rows={len(values)} supervised_tokens={supervised}", flush=True)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("scorer", "feedback"), required=True)
    parser.add_argument("--model", required=True, help="Merged v5 inherited base path")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=V5_MAX_SEQ_LENGTH)
    parser.add_argument("--score-token-weight", type=float, default=V5_SCORE_TOKEN_WEIGHT)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 4.0 if args.task == "scorer" else 2.0
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.task == "scorer" else 5e-5
    return args


if __name__ == "__main__":
    main()
