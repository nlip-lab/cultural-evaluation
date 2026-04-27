# 01_CSI_Extraction

`01_CSI_Extraction` is Phase 1 of the evaluation framework. It builds structured CSI annotations from the source benchmark text and prepares those annotations for downstream scoring.

## What It Does

1. `run_extraction.py`:
- Reads `03_data/processed/benchmark.jsonl`.
- Extracts CSI spans and metadata using either Google or OpenAI backend.
- Writes backend-specific inventories and extraction statistics.

2. `run_function_frames.py`:
- Enriches extracted CSI spans with denotation and functional-role fields.
- Produces frame-enriched CSI inventory outputs.

3. `run_wikidata_grounding.py`:
- Grounds selected CSI categories to Wikidata entities.
- Adds `wikidata_qid`, description, and type metadata to CSI entries.

4. `prompt_utils.py`:
- Loads and formats the source CSI extraction prompt template.

## Inputs and Outputs

- Input benchmark: `03_data/processed/benchmark.jsonl`
- Prompt templates: `src/Framework/prompts/`
- Main outputs: phase-local JSONL and JSON stats files inside `src/Framework/01_CSI_Extraction/`

## Example Usage

```bash
python src/Framework/01_CSI_Extraction/run_extraction.py --backend google
python src/Framework/01_CSI_Extraction/run_function_frames.py --backend google
python src/Framework/01_CSI_Extraction/run_wikidata_grounding.py
```
