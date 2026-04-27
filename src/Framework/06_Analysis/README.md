# 06_Analysis

`06_Analysis` is Phase 6 of the framework. It computes function-level similarity and preservation metrics for adapted outputs.

## What It Does

1. `compute_fps.py`:
- Uses judge-based scoring for function preservation on sampled adaptation pairs.
- Produces per-pair and per-adaptation FPS outputs.

2. `compute_fes.py`:
- Computes embedding-based function similarity for contextual and holistic function text.
- Produces per-pair and per-adaptation FES outputs.

3. `compute_fbs.py`:
- Computes BERTScore-based function similarity for contextual and holistic function text.
- Produces per-pair and per-adaptation FBS outputs.

4. `compute_bleu_rouge.py`:
- Computes sentence-level BLEU and ROUGE-L over function descriptions.
- Produces per-pair and per-adaptation BLEU/ROUGE outputs.

## Inputs and Outputs

- Input: `05_Frames/paired_frames.jsonl`
- Main outputs are written in `06_Analysis/` as `*_scores.jsonl` and `*_per_adaptation.jsonl` artifacts

## Example Usage

```bash
python src/Framework/06_Analysis/compute_fps.py
python src/Framework/06_Analysis/compute_fes.py
python src/Framework/06_Analysis/compute_fbs.py
python src/Framework/06_Analysis/compute_bleu_rouge.py
```
