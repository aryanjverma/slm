# V5 Final Runbook

Status on 2026-07-12: **private dataset finalized; GPU training pending**. The 60-row review,
540/60/75 assembly, strict hash preflight, zero-leakage check, and CPU smoke pass.

## 1. Complete and bind the review

```powershell
python scripts/review_v5_manual_packet.py --reviewer YOUR_NAME
python scripts/assemble_v5_dataset.py finalize `
  --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
python scripts/smoke_v5_pipeline.py
python -m pytest
```

Finalization must produce 540 new train cases, 60 development cases, 75 v4 replay cases, and the
audited 615-row `train_cases_v5_with_replay.jsonl`. Both `train_v5.py` and the Colab notebook fail
closed unless approval, packet hash, artifact hashes, counts, and zero golden leakage all pass.

## 2. Colab / Drive run

Open `notebooks/colab_train_v5.ipynb`. The fixed Drive root is
`/content/drive/MyDrive/apush-frq-grader-v5`; place the completed v4 adapter and finalized private
directory at the paths configured in cell 1. The frozen settings are scorer 4 epochs at 1e-4,
feedback 2 epochs at 5e-5, LoRA rank 16, batch 1, accumulation 4, warmup 0.03, seed 13, 4096 tokens,
and scorer score-token weight 4.0.

Run the ten-case GPU smoke, then the full 60-case base/v4/v5 development comparison. Freeze the
selected configuration before the one permitted 53-case development-informed golden evaluation.
Never tune against golden answers.

## 3. Release decision

All gates must pass: QWK ≥ 0.40, MAE ≤ 1.50, within-one ≥ 60%, every criterion exact rate above
v4, mean-total drift ≤ 0.50, structured validity ≥ 98%, and grounding ≥ 85%. A failure is published
as non-production-ready, without retuning.

Only aggregate reports may leave private storage. Never upload essays, style references, labels,
review packets, or per-case predictions.

## 4. Publish after a passing run

```powershell
huggingface-cli upload aryanjverma/apush-frq-grader-v5 PATH_TO_BUNDLE . --repo-type model
python scripts/build_v5_public_companion.py
huggingface-cli upload aryanjverma/apush-leq-grader-public artifacts/public/apush-leq-grader-public . --repo-type dataset
huggingface-cli upload aryanjverma/apush-frq-grader-v5-demo PATH_TO_SPACE . --repo-type space
```

Required public targets are the model `aryanjverma/apush-frq-grader-v5`, companion dataset
`aryanjverma/apush-leq-grader-public`, and Space `aryanjverma/apush-frq-grader-v5-demo`.
