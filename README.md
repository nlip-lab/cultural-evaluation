# Lost in Localization: Benchmarking Intralingual Cultural Adaptation in Large Language Models

This repository contains the code and infrastructure for **CulturalAdapt-6K** and the end-to-end evaluation pipeline described in our paper. The framework evaluates *function-preserving cultural transfer* — whether a model can adapt the contextual roles of culture-specific items, not merely their surface forms.

---

## Overview

Cultural adaptation requires more than lexical substitution: a model must decide *what* to change, *whether* each replacement is locally valid, and *whether* it preserves the original's functional role. We formalize this as **function-preserving cultural transfer** and build the infrastructure to evaluate it:

- **CulturalAdapt-6K** — 6,000 English texts across four discourse domains (educational math, narrative stories, conversational dialogue, procedural recipes), curated from American-source material for American→Indian intralingual adaptation.
- **IAET** (Intercultural Adaptation Error Taxonomy) — five stage-aligned error categories (Scope, Substitution Validity, Function Preservation, Target-Culture Plausibility, Output Integrity) with severity grades.
- **CSI-level evaluation pipeline** — seven sequential phases that operationalize each IAET stage through function frames, which capture *what job* a cultural item does in context.

We evaluate 191,968 adaptations from eight instruction-tuned models across four prompting conditions.

---

## Repository Structure

```
.
├── README.md                    ← this file
├── src/
│   ├── Dataset/                 ← benchmark construction (Steps 1–5)
│   ├── Adaptations/             ← adaptation generation (forward & reverse)
│   └── Framework/               ← seven-phase evaluation pipeline
│       ├── build_index.py
│       ├── build_sample_ids_16k.py
│       ├── build_unified_scores.py
│       ├── prompts/             ← prompt templates (.txt)
│       ├── 01_CSI_Extraction/
│       ├── 02_Hygiene/
│       ├── 03_Audit/
│       ├── 04_Validity/
│       ├── 05_Frames/
│       ├── 06_Analysis/
│       └── 07_Reverse/
```

Each subdirectory under `src/` contains its own `README.md` with full usage instructions.

---

## Full Pipeline

### Stage 0 — Benchmark Construction (`src/Dataset/`)

Constructs `benchmark.jsonl` (6,000 items, 1,500 per domain) from four publicly available datasets.

```bash
python src/Dataset/01_download.py
python src/Dataset/02_sieve_recipes.py
python src/Dataset/02_sieve_stories.py
bash   src/Dataset/run_pipeline.sh          # GPU required for CSI detection
python src/Dataset/04_diversity_sample.py --datasets gsm8k rocstories dailydialog recipenlg
python src/Dataset/05_verify.py
```

Output: `03_data/processed/benchmark.jsonl`

### Stage 1 — Adaptation Generation (`src/Adaptations/`)

Generates forward (American→Indian) and reverse (Indian→American) adaptations across eight models and four prompt strategies.

```bash
bash src/Adaptations/run_adapt.sh      # forward; requires GPU
bash src/Adaptations/run_reverse.sh    # reverse; requires GPU
```

Outputs: `03_data/processed/adapted_data/runs/{strategy}/{model}_{domain}.jsonl`

### Stage 2 — Evaluation Framework (`src/Framework/`)

Seven sequential phases; run from the repository root.

```bash
# Join adaptations to benchmark records
python src/Framework/build_index.py
python src/Framework/build_sample_ids_16k.py

# Phase 1: CSI extraction and function-frame construction
python src/Framework/01_CSI_Extraction/run_extraction.py --backend google
python src/Framework/01_CSI_Extraction/run_function_frames.py --backend google
python src/Framework/01_CSI_Extraction/run_wikidata_grounding.py

# Phase 2: Output integrity filtering
python src/Framework/02_Hygiene/run_hygiene.py
python src/Framework/02_Hygiene/run_phase2.py --resume

# Phase 3: Adaptation audit (RF, TCT, SAT metrics)
python src/Framework/03_Audit/run_audit.py
python src/Framework/03_Audit/compute_csi_metrics.py

# Phase 4: Substitution validity (C2 scoring)
python src/Framework/04_Validity/run_c2_judge.py

# Phase 5: Adapted function frames
python src/Framework/05_Frames/build_dedup_pairs.py
python src/Framework/05_Frames/run_adapted_frames.py
python src/Framework/05_Frames/build_paired_frames.py

# Phase 6: Function similarity metrics (FES, FBS, FPS, BLEU/ROUGE)
python src/Framework/06_Analysis/compute_fps.py
python src/Framework/06_Analysis/compute_fes.py
python src/Framework/06_Analysis/compute_fbs.py
python src/Framework/06_Analysis/compute_bleu_rouge.py

# Phase 7: Round-trip consistency evaluation
python src/Framework/07_Reverse/run_csi_extraction.py
python src/Framework/07_Reverse/run_9a_surface_metrics.py
python src/Framework/07_Reverse/run_9c_csi_alignment.py
python src/Framework/07_Reverse/run_9d_llm_judge.py

# Aggregate all phase outputs
python src/Framework/build_unified_scores.py
```

---

## Data Layout Convention

The pipeline expects the following directory layout at the repository root (not committed; populated by running the pipeline):

```
03_data/
├── raw/                         ← downloaded source datasets
└── processed/
    ├── benchmark.jsonl          ← 6,000-item benchmark
    ├── adapted_data/
    │   └── runs/{strategy}/{model}_{domain}.jsonl
    └── reverse_adapted_data/
        └── runs/{strategy}/{model}_{domain}.jsonl

04_experiments/
└── logs/                        ← run logs
```

---

## Models Evaluated

| Model | Parameters | Origin |
|---|---|---|
| Llama-3.3-70B-Instruct | 70B AWQ | Meta |
| Qwen2.5-72B-Instruct | 72B AWQ | Alibaba |
| Aya-Expanse-32B | 32B | Cohere |
| Gemma-3-27B-it | 27B | Google DeepMind |
| Sarvam-30B | 30B MoE | Sarvam AI |
| Llama-3.1-8B-Instruct | 8B | Meta |
| Gemma-2-9B-it | 9B | Google |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral AI |

---

## Prompt Strategies

| Strategy | Description |
|---|---|
| `simple` | Minimal one-line instruction |
| `instructed` | Detailed constraint-aware system prompt |
| `oneshot` | System prompt with one hand-crafted example per domain |
| `cot` | Chain-of-thought (IDENTIFY → MAP → VERIFY) |

---

## APIs Required

| Phase | API |
|---|---|
| Phase 1 (CSI extraction, frames) | Google Gemini 2.5 Flash Lite or GPT-4o |
| Phase 3 (audit) | GPT-4o |
| Phase 4 (validity) | GPT-4o |
| Phase 6 (FPS) | GPT-4o |
| Phase 7 (LLM judge) | GPT-4o |
| Evaluation judge (round-trip) | Gemini 2.5 Pro |

Set `OPENAI_API_KEY` and `GOOGLE_API_KEY` as environment variables before running.

---

## Hardware

- **Dataset construction** (Step 3): CUDA GPU with ≥16 GB VRAM
- **Adaptation generation**: 2× GPU recommended for 70B/72B models (~40 GB VRAM each)
- **Framework phases 1–7**: CPU-runnable except where noted; API calls are async-batched

---

## Error Taxonomy (IAET)

| Category | Subtypes | Severity |
|---|---|---|
| **C1 Scope** | Residual retention, saturation, over-globalization, collapse | ★★–★★★ |
| **C2 Substitution Validity** | Affordance, scale transfer, dependency, identity, figurative | ★–★★★ |
| **C3 Function Preservation** | Intent shift, temporal disruption, scope distortion, pragmatic drift | ★★–★★★ |
| **C4 Target-Culture Plausibility** | Incoherence, norm violation, stereotyping, implausibility, inauthenticity | ★–★★ |
| **C5 Output Integrity** | Degenerate output, hallucinated insertion, foreignized diction | ★–★★★ |

---

## Citation

```bibtex
@inproceedings{anonymous2025lostinlocalization,
  title     = {Lost in Localization: Benchmarking Intralingual Cultural Adaptation in Large Language Models},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of ACL 2025},
  year      = {2025}
}
```
