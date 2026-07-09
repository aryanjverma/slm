# v1 QLoRA Run Report — Colab (T4), 2026-07-09

## TL;DR

The run **did not produce a gradeable model.** Two independent infrastructure bugs —
not model quality — blocked the Day-3 gate:

1. **Training crashed at step 150/300** (the first checkpoint save) with a `PicklingError`
   on TRL's `SFTConfig`. The loss curve up to that point was excellent, but the final
   adapter was **never saved**.
2. **Both eval cells then failed** because there was no complete adapter to load and the
   eval environment was missing the Qwen tokenizer's `sentencepiece`/`tiktoken` dependency.

Net: the only numbers on the board are the two deterministic baselines. There is **zero
tuned-model performance to report** — the gate is blocked, not lost.

## Cell-by-cell outcome

| Cell | Step | Result |
|------|------|--------|
| 1 | GPU check | OK — Tesla T4, CUDA 13.0 |
| 2 | Clone | OK — data present (1000 train / 198 litmus / 72 real) |
| 3 | Install | OK |
| 4 | HF token | WARN — none; anonymous downloads (non-fatal) |
| 5 | **Train** | **FAIL — crashed at 150/300** |
| 6 | Baselines | OK |
| 7 | **Litmus eval (tuned)** | FAIL — tokenizer error |
| 8 | **Real eval (tuned)** | FAIL — tokenizer error |
| 9 | Summary table | WARN — only baselines printed (tuned row missing) |
| 10 | Save to Drive | FAIL — mount failed (irrelevant; nothing to save) |

## Bug 1 — training died at the checkpoint save

Training itself was healthy. Loss converged cleanly:

```
step  20  loss 2.339
step  40  loss 1.566
step  60  loss 0.607
step  80  loss 0.287
step 100  loss 0.136
...
step 150  loss ~0.012   <-- crash here
```

Then, exactly at step 150:

```
_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>:
it's not the same object as trl.trainer.sft_config.SFTConfig
```

**Root cause.** `train_qlora.py` set `save_steps = max(50, max_steps // 2) = 150`, so the
first checkpoint save fired at step 150. Unsloth recompiles the trainer into
`unsloth_compiled_cache/UnslothSFTTrainer.py`, so the `SFTConfig` instance in
`trainer.args` is an instance of Unsloth's *recompiled* class, whose identity differs from
the importable `trl.trainer.sft_config.SFTConfig`. When `Trainer._save_checkpoint` calls
`torch.save(self.args, "training_args.bin")`, pickle's class-identity check fails. This is a
TRL <-> Unsloth version-skew bug; `pyproject.toml` pinned neither (`trl>=0.9.0`, `unsloth`
from bleeding-edge git).

**Consequence.** The crash is inside `_save_checkpoint`, so `trainer.save_model()` and
`tokenizer.save_pretrained()` never executed. The target dir
`artifacts/models/apush-frq-grader-v1/` had no final adapter. A partial `checkpoint-150/`
was written before the `training_args.bin` step that crashed, but no complete final artifact.

## Bug 2 — eval couldn't load a tokenizer

```
ValueError: Couldn't instantiate the backend tokenizer ...
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
```

Two things compounded:

- **No adapter to point at.** `eval_hf_model.py` checks for `adapter_config.json` in the
  target dir. Because Bug 1 skipped the final save, that file was absent, so the loader fell
  into the plain-HF-model branch.
- **Missing tokenizer deps.** The Qwen2.5 tokenizer needs `sentencepiece` or `tiktoken` to
  build the fast tokenizer when only slow files are present. Neither was declared in the
  `[train]` extra, so the fresh `eval_hf_model.py` subprocess couldn't convert it. Training
  worked only because Unsloth had loaded the fast tokenizer directly from the HF download.

## Baseline numbers (the only ones that ran)

| model | n | json | rubric | ground | robust | total |
|-------|---|------|--------|--------|--------|-------|
| inflated_prompted_base | 198 | 1.00 | 0.82 | 0.17 | 0.93 | 0.69 |
| apush_grader_reference | 198 | 1.00 | 1.00 | 1.00 | 2.00 | 1.00 |

The tuned model must move grounding 0.17 -> ~1.0 and robustness upward. Its row does not
exist yet.

## Fixes applied

1. **Stop the save crash** — `scripts/train_qlora.py`. The real culprit is
   `torch.save(self.args, "training_args.bin")` inside `Trainer._save`, which pickles the
   `SFTConfig` and hits the class-identity mismatch. `_save` runs on *every* save, so it
   fires at both the mid-run checkpoint (step 150) and the final `trainer.save_model()`.
   Two changes: (a) `save_strategy="no"` to drop mid-run checkpoints, and (b) replace
   `trainer.save_model(...)` with `model.save_pretrained(...)` so the final save writes the
   PEFT adapter directly and never pickles `self.args`. This also yields a complete
   `adapter_config.json`, resolving the missing-config half of Bug 2. A more durable
   follow-up is to pin compatible `trl`/`unsloth` versions.

   > Note: the first patch used only `save_strategy="no"`, which merely relocated the crash
   > from step 150 to the final save. The `model.save_pretrained(...)` change is the fix.
2. **Eval tokenizer deps** — added `sentencepiece>=0.2.0` and `tiktoken>=0.7.0` to the
   `[train]` extra in `pyproject.toml`, and added a `!pip install -q sentencepiece tiktoken`
   cell (section 6b) to `notebooks/colab_train_eval_v1.ipynb` before the eval cells.

## Watch on the re-run

Loss plateaued at ~0.012 by epoch 2 across 5 epochs on 1000 synthetic rows — a strong
**overfitting** signal. Once the run completes, watch the litmus adversarial slices
(`grade_inflation_request`, `prompt_injection`). If robustness is weak, fix it in the data
(more/harder adversarial examples), not the hyperparameters — per the project thesis.
