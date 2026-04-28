# Lost in Localization: Benchmarking Intralingual Cultural Adaptation in Large Language Models

> *Anonymous submission — EMNLP 2026*

---

## Overview

This repository contains the full pipeline for **CulturalAdapt-6K**, a benchmark for evaluating *function-preserving intralingual cultural adaptation* (American English → Indian English) in large language models.

The project has three components:

| Component | Description |
|---|---|
| **CulturalAdapt-6K** | 6,000 American-source texts across four discourse domains with CSI-level annotations |
| **IAET** | Intercultural Adaptation Error Taxonomy — 21 subtypes across 5 stage-aligned error categories |
| **Evaluation pipeline** | Seven-phase, CSI-level evaluation framework operationalizing each IAET category |

---

## Repository Structure

```
.
├── README.md
├── REPRODUCIBILITY.md           ← step-by-step environment and run guide
├── requirements.txt             ← pip dependencies
├── environment.yml              ← conda environment (Python 3.10, CUDA 11.8)
├── .env.example                 ← API key template
└── src/
    ├── Dataset/                 ← benchmark construction (5 stages)
    │   ├── 01_download.py       ← download raw source corpora
    │   ├── 02_sieve_recipes.py  ← filter RecipeNLG (2.2M → 50K)
    │   ├── 02_sieve_stories.py  ← filter ROCStories (98K → 8,960)
    │   ├── 03_process.py        ← 4-tier CSI detection (GPU required)
    │   ├── 04_diversity_sample.py ← KMeans diversity sampling → 6,000 items
    │   ├── 05_verify.py         ← benchmark integrity report
    │   └── run_pipeline.sh      ← shell wrapper for step 3
    ├── Adaptations/             ← forward and reverse generation
    │   ├── adapt.py             ← main adaptation driver (8 models × 4 strategies)
    │   ├── reverse_adapt.py     ← Indian → American reverse pass
    │   ├── model_*.py           ← one adapter per model (vLLM / OpenAI)
    │   ├── run_adapt.sh         ← forward adaptation shell wrapper
    │   └── run_reverse.sh       ← reverse adaptation shell wrapper
    └── Framework/               ← seven-phase evaluation pipeline
        ├── build_index.py
        ├── build_sample_ids_16k.py
        ├── build_unified_scores.py
        ├── prompts/             ← prompt templates (.txt)
        ├── 01_CSI_Extraction/   ← extract and enrich CSI spans
        ├── 02_Hygiene/          ← output integrity checks
        ├── 03_Audit/            ← CSI strategy audit + RF/TCT/SAT metrics
        ├── 04_Validity/         ← substitution validity (C2 score)
        ├── 05_Frames/           ← adapted function-frame construction
        ├── 06_Analysis/         ← FPS, FES, FBS, BLEU/ROUGE
        └── 07_Reverse/          ← round-trip consistency analysis
```

Each `src/` subdirectory has its own `README.md` with inputs, outputs, and usage.

---

## Benchmark: CulturalAdapt-6K

Four discourse domains, 1,500 items each:

| Domain | Source dataset | Raw size |
|---|---|---|
| Educational / math | GSM8K | 8,792 |
| Narrative stories | ROCStories | 98,161 |
| Conversational dialogue | DailyDialog | 13,118 |
| Procedural recipes | RecipeNLG | 2,231,142 |

Items are annotated with **Culturally Sensitive Items (CSIs)** across 8 categories: `food_drink`, `place_institution_brand`, `unit_currency_scale`, `holiday_event_ritual`, `personal_name_identity`, `social_norm_address`, `pop_culture_sport_media`, `idiom_figurative`.

Each CSI carries: category, adaptation necessity (`must_adapt` / `may_adapt` / `should_retain`), salience (0–1), and functional-role fields.

---

## Error Taxonomy (IAET)

| Category | Subtypes | Severity |
|---|---|:---:|
| **C1 Scope** | C1.1 Residual Source Retention · C1.2 Cultural Saturation · C1.3 Over-Globalization · C1.4 Adaptation Collapse | ●● – ●●● |
| **C2 Substitution Validity** | C2.1 Affordance Violation · C2.2 Scale Transfer Error · C2.3 Dependency Propagation Failure · C2.4 Identity-Marker Mismatch · C2.5 Figurative Equivalence Failure | ●● – ●●● |
| **C3 Function Preservation** | C3.1 Core Intent Shift · C3.2 Temporal Logic Disruption · C3.3 Scope Distortion · C3.4 Pragmatic Force Drift | ●● – ●●● |
| **C4 Target-Culture Plausibility** | C4.1 Internal Cultural Incoherence · C4.2 Norm Violation · C4.3 Stereotypical Portrayal · C4.4 Contextual Implausibility · C4.5 Perspective Inauthenticity | ● – ●● |
| **C5 Output Integrity** | C5.1 Degenerate Output · C5.2 Hallucinated Insertion · C5.3 Foreignized Diction | ● – ●●● |

*Severity: ●●● Critical · ●● Substantive · ● Surface*

---

## Models and Prompt Strategies

### Models (8 total)

| Tier | Model | Params | Backend |
|---|---|---|---|
| High | Llama-3.3-70B-Instruct (AWQ) | 70B | vLLM, TP=2 |
| High | Qwen2.5-72B-Instruct (AWQ) | 72B | vLLM, TP=2 |
| Mid | Aya-Expanse-32B | 32B | vLLM, TP=2 |
| Mid | Gemma-3-27B-it | 27B | vLLM, TP=2 |
| Mid | Sarvam-30B | 30B MoE | vLLM, TP=2 |
| Low | Llama-3.1-8B-Instruct | 8B | vLLM, TP=1 |
| Low | Gemma-2-9B-it | 9B | vLLM, TP=1 |
| Low | Mistral-7B-Instruct-v0.3 | 7B | vLLM, TP=1 |

GPT-4o is also supported via the OpenAI API (`model_gpt4o.py`).

### Prompt Strategies (4 total)

| Strategy | Description |
|---|---|
| `simple` | Minimal one-line instruction |
| `instructed` | Detailed constraint-aware rewriting prompt |
| `oneshot` | Instructed + one hand-crafted example per domain |
| `cot` | Chain-of-thought: IDENTIFY → MAP → VERIFY |

---

## Evaluation Pipeline

Seven sequential phases, each operationalizing one or more IAET categories:

| Phase | Scripts | IAET | Key outputs |
|---|---|---|---|
| **1 · CSI Extraction** | `run_extraction.py`, `run_function_frames.py`, `run_wikidata_grounding.py` | C1–C5 setup | CSI spans + functional frames |
| **2 · Output Integrity** | `run_hygiene.py`, `run_phase2.py` | C5.1 | Pass / fail labels per adaptation |
| **3 · Adaptation Audit** | `run_audit.py`, `compute_csi_metrics.py` | C1 | RF, TCT, SAT metrics |
| **4 · Substitution Validity** | `run_c2_judge.py`, `scale_checker.py` | C2 | C2 composite score |
| **5 · Adapted Frames** | `build_dedup_pairs.py`, `run_adapted_frames.py`, `build_paired_frames.py` | C3 setup | Source↔adapted frame pairs |
| **6 · Function Preservation** | `compute_fps.py`, `compute_fes.py`, `compute_fbs.py`, `compute_bleu_rouge.py` | C3 | FPS, FES, FBS, BLEU, ROUGE |
| **7 · Round-Trip** | `run_csi_extraction.py`, `run_9a_surface_metrics.py`, `run_9c_csi_alignment.py`, `run_9d_llm_judge.py` | C4, C5.2 | CSI recovery, round-trip scores |

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd <repo>

# Option A — conda (recommended)
conda env create -f environment.yml
conda activate cultural-eval

# Option B — pip
python -m pip install -r requirements.txt
```

### 2. Download spaCy NER model

```bash
python -m spacy download en_core_web_trf
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env: fill in OPENAI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS
```

The adaptation scripts load `.env` from the repository root automatically. For framework scripts, export the keys before running:

```bash
export $(grep -v '^#' .env | xargs)
```

### 4. Create data directories

```bash
mkdir -p 03_data/raw 03_data/processed/adapted_data 03_data/processed/reverse_adapted_data
mkdir -p 04_experiments/logs
```

### 5. (Optional) Download NLTK data

```python
import nltk
nltk.download('punkt_tab')
```

---

## Running the Full Pipeline

### Stage 0 — Build the benchmark

```bash
python src/Dataset/01_download.py
python src/Dataset/02_sieve_recipes.py
python src/Dataset/02_sieve_stories.py
bash   src/Dataset/run_pipeline.sh          # GPU required; ~6–8 h on A100
python src/Dataset/04_diversity_sample.py --datasets gsm8k rocstories dailydialog recipenlg
python src/Dataset/05_verify.py
```

Output: `03_data/processed/benchmark.jsonl`

### Stage 1 — Generate adaptations

```bash
bash src/Adaptations/run_adapt.sh      # forward (American → Indian); GPU
bash src/Adaptations/run_reverse.sh    # reverse (Indian → American); GPU
```

Smoke test (2 items/domain, no files written):

```bash
CUDA_VISIBLE_DEVICES=0 python src/Adaptations/adapt.py \
    --model mistral --prompt_level simple --smoke_test
```

### Stage 2 — Evaluate

```bash
python src/Framework/build_index.py
python src/Framework/build_sample_ids_16k.py

python src/Framework/01_CSI_Extraction/run_extraction.py --backend google
python src/Framework/01_CSI_Extraction/run_function_frames.py --backend google
python src/Framework/01_CSI_Extraction/run_wikidata_grounding.py

python src/Framework/02_Hygiene/run_hygiene.py
python src/Framework/02_Hygiene/run_phase2.py --resume

python src/Framework/03_Audit/run_audit.py
python src/Framework/03_Audit/compute_csi_metrics.py

python src/Framework/04_Validity/run_c2_judge.py

python src/Framework/05_Frames/build_dedup_pairs.py
python src/Framework/05_Frames/run_adapted_frames.py
python src/Framework/05_Frames/build_paired_frames.py

python src/Framework/06_Analysis/compute_fps.py
python src/Framework/06_Analysis/compute_fes.py
python src/Framework/06_Analysis/compute_fbs.py
python src/Framework/06_Analysis/compute_bleu_rouge.py

python src/Framework/07_Reverse/run_csi_extraction.py
python src/Framework/07_Reverse/run_9a_surface_metrics.py
python src/Framework/07_Reverse/run_9c_csi_alignment.py
python src/Framework/07_Reverse/run_9d_llm_judge.py

python src/Framework/build_unified_scores.py
```

All framework scripts must be run **from the repository root** (`python src/Framework/...`), not from within their subdirectory.

---

## Data Layout

Populated by running the pipeline; not committed to the repository.

```
03_data/
├── raw/                                  ← source corpora from download step
└── processed/
    ├── benchmark.jsonl                   ← 6,000-item annotated benchmark
    ├── adapted_data/
    │   └── runs/{strategy}/{model}_{domain}.jsonl
    └── reverse_adapted_data/
        └── runs/{strategy}/{model}_{domain}.jsonl

04_experiments/
└── logs/                                 ← run logs and timing
```

---

## APIs and Hardware

### APIs Required

| Used in | API | Purpose |
|---|---|---|
| Phase 1 | Google Gemini 2.5 Flash Lite | CSI extraction, function frames |
| Phases 3, 4, 6 | GPT-4o | Audit, validity, FPS judge |
| Phase 7 | Gemini 2.5 Pro | Round-trip LLM judge |

Both APIs are called asynchronously with retry logic and configurable concurrency limits. All phases support `--pilot N` to run on the first N items for cost estimation.

### Hardware Requirements

| Stage | Minimum | Notes |
|---|---|---|
| Dataset construction (step 3) | 1× GPU, ≥16 GB VRAM | Qwen2.5-7B fp16 + spaCy transformer |
| Adaptation — 7B/8B/9B models | 1× GPU, ≥16 GB VRAM | tensor parallel = 1 |
| Adaptation — 27B/32B models | 2× GPU, ≥32 GB VRAM each | tensor parallel = 2 |
| Adaptation — 70B/72B (AWQ) | 2× GPU, ≥40 GB VRAM each | tensor parallel = 2 |
| Framework phases 1–7 | CPU only | except FBS (GPU speeds BERTScore) |

---

## Reproducibility

- All pipeline scripts are **checkpoint-resumable**: restart with `--resume` to skip completed items.
- Randomness is seeded (`PYTHONHASHSEED=42`, `random.seed(42)`, `numpy.random.seed(42)`).
- CUDA determinism is enforced via `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- vLLM runs at `temperature=0.0` (greedy decoding).
- See `REPRODUCIBILITY.md` for step-by-step environment setup and troubleshooting.

---

## Citation

```bibtex
@inproceedings{anonymous2025lostinlocalization,
  title     = {Lost in Localization: Benchmarking Intralingual Cultural Adaptation
               in Large Language Models},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP 2026)},
  year      = {2026},
  note      = {Anonymous submission}
}
```
