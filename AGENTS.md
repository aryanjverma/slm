# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python project (`requires-python >=3.10`; the VM has Python 3.12). The core
data/eval/demo loop is CPU-only and needs no GPU, no network, and no API keys. Standard commands
are documented in `README.md` and `CLAUDE.md` — reference those first. Notes below are only the
non-obvious caveats for this environment.

- **Use `python3`, not `python`.** Only `python3`/`pip3` are on PATH. The `README.md`/`CLAUDE.md`
  examples use PowerShell syntax and bare `python`; translate them to `python3` on this Linux VM.
- **Invoke via `python3 -m ...`.** The console scripts (`apush-grader-generate`, `apush-grader-eval`,
  `apush-grader-demo`) install to `~/.local/bin`, which is not on PATH. Use the module form instead,
  e.g. `python3 -m apush_frq_grader_slm.cli.demo`.
- **`ruff` and `pytest` are not declared dependencies.** The update script installs them so lint/test
  work. Lint: `python3 -m ruff check .` — note it currently reports ~66 pre-existing errors, almost
  all `E70x`/`E402` in notebook-embedded scripts under `scripts/` and `ingest/`; these predate env
  setup and are not caused by dependency installation.
- **`demo` is an interactive REPL.** It reads a prompt then an essay from stdin. To run
  non-interactively, pipe input, e.g. `printf 'PROMPT\nESSAY\nquit\n' | python3 -m apush_frq_grader_slm.cli.demo`.
- **Tests:** `python3 -m pytest` (~205 pass, 1 skipped). `tests/test_v5_model_pipeline.py` self-skips
  without the `[train]` extra (torch); ingest tests self-skip when `tests/fixtures/` files are absent.
- **Real training/ingestion is out of scope for CPU-only setup.** The `[train]` extra pulls Unsloth
  from a git URL and needs a CUDA GPU (none here). CPU fallbacks (`scripts/train_smoke.py`,
  `scripts/eval_hf_model.py`) still download `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face (network).
  The `[judge]`/`[ingest]` scripts require `OPENAI_API_KEY`.
