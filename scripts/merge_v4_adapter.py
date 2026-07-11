"""Merge the final v4 PEFT adapter into its Qwen base for v5 inheritance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apush_frq_grader_slm.training_v5 import sha256_tree


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Merge dependencies missing; run pip install -e ".[train]"') from exc

    adapter_config = json.loads((args.v4_adapter / "adapter_config.json").read_text())
    configured_base = adapter_config.get("base_model_name_or_path")
    base_model = args.base_model or configured_base
    if not base_model:
        raise ValueError("No base model supplied and adapter_config.json has no base model")
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(base, args.v4_adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.v4_adapter)
    tokenizer.save_pretrained(args.output)
    inherited_hash = sha256_tree(args.output, exclude_names={"v5_inherited_base.json"})
    metadata = {
        "format": "apush-frq-grader-v5-inherited-base",
        "base_model": str(base_model),
        "v4_adapter": str(args.v4_adapter.resolve()),
        "v4_adapter_sha256": sha256_tree(args.v4_adapter),
        "inherited_base_sha256": inherited_hash,
        "dtype": "bfloat16" if args.bf16 else "float16",
    }
    (args.output / "v5_inherited_base.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved inherited v5 base to {args.output} ({inherited_hash})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-adapter", type=Path, required=True)
    parser.add_argument("--base-model", help="Defaults to adapter_config.json base_model_name_or_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
