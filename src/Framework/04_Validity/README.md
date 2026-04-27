# 04_Validity

`04_Validity` is Phase 4 of the framework. It evaluates substitution quality for adapted CSI spans and assigns C2 validity scores.

## What It Does

1. `run_c2_judge.py`:
- Loads eligible audited substitutions.
- Scores each substitution on affordance, scale transfer, dependency propagation, identity fit, and figurative equivalence.
- Produces per-record C2 outputs and summary statistics.

2. `scale_checker.py`:
- Runs deterministic numeric and unit checks.
- Detects provable scale and currency conversion issues used to override invalid model scores.

## Inputs and Outputs

- Inputs: `03_Audit/audit_results.jsonl`, `01_CSI_Extraction/source_csi_final_google.jsonl`, and `02_Hygiene/adaptations_with_hygiene.jsonl`
- Main outputs: `04_Validity/c2_scores.jsonl` and `04_Validity/c2_stats.json`

## Example Usage

```bash
python src/Framework/04_Validity/run_c2_judge.py --pilot 1000
```
