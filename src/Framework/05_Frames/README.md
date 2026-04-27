# 05_Frames

`05_Frames` is Phase 5 of the framework. It builds adapted function-frame resources from audited substitutions and maps them back to per-evaluation records.

## What It Does

1. `build_dedup_pairs.py`:
- Collects eligible CSI substitutions from audit results.
- Clusters near-duplicate adapted spans.
- Produces deduplicated source/adapted pairs and dedup-to-eval mapping.

2. `run_adapted_frames.py`:
- Generates adapted-side function frames for each dedup pair using model calls.
- Writes checkpointed and final frame outputs.

3. `build_paired_frames.py`:
- Maps deduplicated adapted frames back to each original evaluation item.
- Produces paired source/adapted function-frame records for downstream analysis.

## Inputs and Outputs

- Inputs include `01_CSI_Extraction/source_csi_final_google.jsonl`, `02_Hygiene/adaptations_with_hygiene.jsonl`, and `03_Audit/audit_results.jsonl`
- Main outputs include `05_Frames/dedup_pairs.jsonl`, `05_Frames/05_frames.jsonl`, and `05_Frames/paired_frames.jsonl`

## Example Usage

```bash
python src/Framework/05_Frames/build_dedup_pairs.py
python src/Framework/05_Frames/run_adapted_frames.py
python src/Framework/05_Frames/build_paired_frames.py
```
