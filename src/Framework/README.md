# Evaluation Framework

Seven-phase pipeline for evaluating cultural adaptation quality across the
forward adaptation, reverse adaptation, and scoring stages.

## Pipeline Overview

```text
build_index.py              Join adaptations to benchmark source records
build_sample_ids_16k.py     Sample 16K item IDs for evaluation subset
01_CSI_Extraction/          Extract CSIs from source texts
02_Hygiene/                 Hygiene checks on adapted texts
03_Audit/                   CSI audit: coverage, accuracy, hallucination
04_Validity/                Substitution validity
05_Frames/                  Build adapted functional frames
06_Analysis/                Function analysis metrics
07_Reverse/                 Reverse adaptation evaluation
build_unified_scores.py     Aggregate phase outputs into final scores
```

## Directory Structure

```text
src/Framework/
├── build_index.py
├── build_sample_ids_16k.py
├── build_unified_scores.py
├── prompts/
├── 01_CSI_Extraction/
├── 02_Hygiene/
├── 03_Audit/
├── 04_Validity/
├── 05_Frames/
├── 06_Analysis/
└── 07_Reverse/
```

## Quick Start

```bash
python build_index.py
python build_sample_ids_16k.py
python 01_CSI_Extraction/run_extraction.py --backend google
python 01_CSI_Extraction/run_function_frames.py
python 02_Hygiene/run_hygiene.py
python 03_Audit/run_audit.py
python 04_Validity/run_c2_judge.py
python 05_Frames/run_adapted_frames.py
python 06_Analysis/compute_fps.py
python 06_Analysis/compute_fbs.py
python 06_Analysis/compute_fes.py
python 06_Analysis/compute_bleu_rouge.py
python 07_Reverse/run_csi_extraction.py
python 07_Reverse/run_9a_surface_metrics.py
python 07_Reverse/run_9c_csi_alignment.py
python 07_Reverse/run_9d_llm_judge.py
python build_unified_scores.py
```

## Data Paths

| Description | Path |
|---|---|
| Benchmark | `03_data/processed/benchmark.jsonl` |
| Forward adaptations | `03_data/processed/adapted_data/runs/{strategy}/{model}_{domain}.jsonl` |
| Reverse adaptations | `03_data/processed/reverse_adapted_data/runs/{strategy}/{model}_{domain}.jsonl` |
| Phase outputs | `03_data/results/` |

## LLM APIs Required

- Phase 1 extraction: Gemini 2.5 Flash Lite or GPT-4o
- Phase 3 audit: GPT-4o
- Phase 4 validity: GPT-4o
- Phase 7 judge: GPT-4o
