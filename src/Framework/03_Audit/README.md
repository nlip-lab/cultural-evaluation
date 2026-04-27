# 03_Audit

`03_Audit` is Phase 3 of the framework. It audits how each adaptation handles source CSIs and then computes record-level transfer metrics.

## What It Does

1. `run_audit.py`:
- Audits each source CSI alignment in an adaptation.
- Labels strategy usage such as `retained`, `localized`, `globalized`, `transformed`, `omitted`, and `distorted`.
- Produces per-record audit outputs with counts and unsupported additions.

2. `compute_csi_metrics.py`:
- Computes RF, TCT, and SAT metrics from audit outputs.
- Writes CSI metric records and aggregated summaries by model, strategy, and domain.

## Inputs and Outputs

- Inputs: `01_CSI_Extraction/source_csi_final_google.jsonl` and `02_Hygiene/adaptations_with_hygiene.jsonl`
- Main outputs: `03_Audit/audit_results.jsonl`, `03_Audit/csi_metrics.jsonl`, and summary JSON files in the same directory

## Example Usage

```bash
python src/Framework/03_Audit/run_audit.py --pilot 500
python src/Framework/03_Audit/compute_csi_metrics.py
```
