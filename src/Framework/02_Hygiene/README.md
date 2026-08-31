# 02_Hygiene

`02_Hygiene` is Phase 2 of the framework. It evaluates each adapted output for language quality and basic generation safety before downstream audit and scoring.

## What It Does

1. `hygiene_checker.py`:
- Runs deterministic checks for language validity, refusals, degeneration, meta-commentary, and length ratio.
- Produces an overall label: `pass`, `fail_soft`, or `fail_hard`, with optional CRAFT label.

2. `run_hygiene.py`:
- Applies hygiene checks to indexed adaptations.
- Writes per-record hygiene results and aggregate distributions.

3. `run_phase2.py`:
- Runs full phase processing with optional resume/checkpoint behavior.
- Produces merged adaptation records with Phase 2 hygiene annotations and summary stats.

## Inputs and Outputs

- Input index: `src/Framework/all_adaptations_indexed.jsonl`
- Main outputs: `src/Framework/02_Hygiene/adaptations_with_hygiene.jsonl` and `src/Framework/02_Hygiene/hygiene_stats.json`

## Example Usage

```bash
python src/Framework/02_Hygiene/run_hygiene.py
python src/Framework/02_Hygiene/run_phase2.py --resume
```
