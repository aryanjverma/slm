# V5 Manual Review — READY FOR YOU

Pilot approval is in place, the remaining 1,470 essays were generated and judged, and
`assemble_v5_dataset.py prepare-review` has produced a fresh 60-row packet from the
authentic corpus (r1 contaminated packet voided).

## Review now

```powershell
python scripts/review_v5_manual_packet.py --reviewer YOUR_NAME
```

Packet: `artifacts/data/v5/private/manual_review_packet_v5.jsonl`  
Provisional corpus: `artifacts/data/v5/private/selected_cases_v5_provisional.jsonl`  
Manifest: `artifacts/data/v5/private/private_use_manifest_v5.json`

Accept, correct, or reject each of the 60 rows. Training finalize stays blocked until
`manual_review_approval_v5.json` is hash-bound with all 60 accepted or corrected.

After approval:

```powershell
python scripts/assemble_v5_dataset.py finalize --candidates artifacts/data/v5/private/validated_candidates_r2.jsonl
```

## Assembly notes

- Validated pool: 1,308 accepted (916 golden_matched, 392 boundary).
- Selected: 420 golden-matched + 180 boundary → 540 train / 60 dev.
- Aggregate style audit passes (mean within 10% of golden; median/quartiles within the
  assembly band). CB goldens lack paragraph breaks, so paragraph_count is skipped in
  the aggregate style gate.
- Short style-reference length bands were widened for future regenerations so extreme
  short CB stubs do not force sub-100-word essays when rebuilding.
