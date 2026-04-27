# 07_Reverse

`07_Reverse` is Phase 7 of the framework. It evaluates round-trip consistency by reverse-adapting each passing output back to the source culture and comparing the result against the original at three levels: surface overlap, CSI alignment, and semantic fidelity.

## What It Does

1. `run_csi_extraction.py`:
   - Extracts CSI spans from reverse-adapted outputs using the same extraction pipeline as Phase 1.
   - Produces a CSI inventory for the reverse-direction texts.

2. `run_9a_surface_metrics.py`:
   - Computes string-level surface metrics (BLEU, ROUGE-L, BERTScore) between reverse-adapted text and the original American source.

3. `run_9c_csi_alignment.py`:
   - Aligns CSIs extracted from reverse-adapted outputs against the original source CSI inventory.
   - Labels each item as recovered, substituted, leaked, or missing.
   - Computes Rec, Sub, Lk, and Miss rates.

4. `run_9d_llm_judge.py`:
   - Scores meaning, structural, detail, and overall equivalence between the reverse-adapted output and original source on a 0–10 scale using an LLM judge.

## Inputs and Outputs

- Inputs:
  - `03_data/processed/reverse_adapted_data/runs/{strategy}/{model}_{domain}.jsonl`
  - `01_CSI_Extraction/source_csi_final_google.jsonl`
  - `02_Hygiene/adaptations_with_hygiene.jsonl`
- Main outputs: `07_Reverse/reverse_csi.jsonl`, `07_Reverse/surface_metrics.jsonl`, `07_Reverse/csi_alignment.jsonl`, `07_Reverse/llm_judge_scores.jsonl`

## Interpretation

The reverse direction (Indian→American) is structurally easier given models' English-centric priors. Degraded round-trip fidelity therefore implicates the forward adaptation pass rather than the reverse. Near-zero leakage with a high missing rate places the information bottleneck in the forward pass.

## Example Usage

```bash
python src/Framework/07_Reverse/run_csi_extraction.py
python src/Framework/07_Reverse/run_9a_surface_metrics.py
python src/Framework/07_Reverse/run_9c_csi_alignment.py
python src/Framework/07_Reverse/run_9d_llm_judge.py
```
