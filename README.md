# Lost in Localization: Benchmarking Intralingual Cultural Adaptation in Large Language Models

> *Anonymous submission — ACL 2025*
> Code: [https://anonymous.4open.science/r/cultural-evaluation-framework/](https://anonymous.4open.science/r/cultural-evaluation-framework/)

---

## What This Paper Is About

Cultural adaptation requires more than lexical substitution. A model must decide *what* to change, *whether* each replacement is locally valid, and *whether* it preserves the original's **functional role** — what job the cultural item is doing in context, not merely what it denotes. We call this **function-preserving cultural transfer** and argue it is the right unit of evaluation.

Existing benchmarks conflate these three decisions into a single quality judgment. We disentangle them with three contributions:

| Contribution | Description |
|---|---|
| **CulturalAdapt-6K** | 6,000 American-source texts across four discourse domains, with CSI-level annotations |
| **IAET** | Intercultural Adaptation Error Taxonomy — 21 subtypes across 5 stage-aligned categories |
| **Evaluation pipeline** | Seven-phase, CSI-level pipeline grounded in function frames and operationalizing each IAET stage |

We evaluate **191,968 adaptations** from **eight instruction-tuned models** under **four prompting conditions** on the American→Indian axis.

**Central finding:** models that substitute the most are not the ones that substitute best. High localization inflates failure exposure; conservative retention inflates validity scores; yet both poles converge to the same functional quality. Current LLMs can dress text in a target culture's vocabulary but cannot reliably preserve the contextual roles that vocabulary is meant to serve.

---

## The Intercultural Adaptation Error Taxonomy (IAET)

Each error category maps to one pipeline stage and follows a **root-cause** principle: when a failure cascades across stages, it is assigned the earliest applicable label.

| Category | Subtype | Severity |
|---|---|:---:|
| **C1 Scope** | C1.1 Residual Source Retention | ●● |
| | C1.2 Cultural Saturation | ●● |
| | C1.3 Over-Globalization | ● |
| | C1.4 Adaptation Collapse | ●●● |
| **C2 Substitution Validity** | C2.1 Affordance Violation | ●●● |
| | C2.2 Scale Transfer Error | ●●● |
| | C2.3 Dependency Propagation Failure | ●● |
| | C2.4 Identity-Marker Mismatch | ●● |
| | C2.5 Figurative Equivalence Failure | ● |
| **C3 Function Preservation** | C3.1 Core Intent Shift | ●●● |
| | C3.2 Temporal Logic Disruption | ●●● |
| | C3.3 Scope Distortion | ●● |
| | C3.4 Pragmatic Force Drift | ●● |
| **C4 Target-Culture Plausibility** | C4.1 Internal Cultural Incoherence | ●● |
| | C4.2 Norm Violation | ●● |
| | C4.3 Stereotypical Portrayal | ●● |
| | C4.4 Contextual Implausibility | ● |
| | C4.5 Perspective Inauthenticity | ● |
| **C5 Output Integrity** | C5.1 Degenerate Output | ●●● |
| | C5.2 Hallucinated Insertion | ●● |
| | C5.3 Foreignized Diction | ● |

*Severity: ●●● Critical · ●● Substantive · ● Surface*

**Example.** Source: *"Sarah bought 3 pounds of ground beef at Safeway for $4.99/lb. She used half to make cheeseburgers for the Fourth of July barbecue."*

| Code | Adapted excerpt | Diagnosis |
|---|---|---|
| **C1.1** | *"Priya bought 3 **pounds** at **Safeway** for **$4.99** … **Fourth of July**."* | Name swapped; unit, store, currency, and holiday remain American. |
| **C2.2** | *"… mutton at **Rs. 4.99/kg** for **Independence Day**."* | Labels localized, values unchanged: Rs. 4.99/kg is implausible; arithmetic breaks. |
| **C5.3** | *"… **spiced meat patties** for the **Republic Day barbecue**."* | Correct referents, unnatural register: no Indian speaker says "Republic Day barbecue." |

---

## CulturalAdapt-6K

A benchmark of **6,000 English texts** drawn from four American-grounded datasets, curated for intralingual (English→English) American→Indian adaptation.

### Source Datasets

| Domain | Dataset | Raw size | Candidates | Final |
|---|---|---|---|---|
| Educational / math | GSM8K | 8,792 | 8,792 | 1,500 |
| Narrative stories | ROCStories | 98,161 | 8,960 | 1,500 |
| Conversational dialogue | DailyDialog | 13,118 | 13,118 | 1,500 |
| Procedural recipes | RecipeNLG | 2,231,142 | 50,000 | 1,500 |

### Construction Pipeline

```
01_download.py       ─── download raw corpora
02_sieve_*.py        ─── structural filtering (RecipeNLG, ROCStories only)
03_process.py        ─── 4-tier CSI detection  ← GPU required
                          Tier 0: compiled regex (instant YES)
                          Tier 1: Qwen2.5-7B + spaCy NER
                          Tier 2: conflict / confidence gate
                          Tier 3: LLM reconsideration with NER evidence
04_diversity_sample.py── KMeans (K=1500/domain) + weighted centroid scoring
05_verify.py         ─── benchmark integrity report
```

Manual validation on 500 items (two annotators): **κ = 0.86**, 98.4% agreement with pipeline.

### CSI Distribution (33,549 items across 6,000 texts)

| Category | Count | Must% | May% | Retain% | Avg. Salience |
|---|---:|---:|---:|---:|---:|
| Unit / currency | 10,174 | **69.7** | 6.9 | 23.4 | 0.557 |
| Food & drink | 9,788 | 9.1 | 56.4 | 34.5 | 0.534 |
| Place / brand | 6,647 | 11.7 | 47.6 | 40.6 | 0.475 |
| Personal name | 2,433 | 4.7 | 8.5 | **86.8** | 0.290 |
| Pop culture | 1,544 | 26.9 | 29.0 | 44.1 | 0.540 |
| Holiday / ritual | 1,204 | 47.1 | 19.6 | 33.3 | **0.636** |
| Social norm | 1,125 | 13.2 | 59.2 | 27.6 | 0.448 |
| Idiom / figurative | **634** | 14.0 | **70.0** | 15.9 | 0.532 |
| **Total** | **33,549** | 30.1 | 33.9 | 36.0 | 0.513 |

Only 30.1% of CSIs are mandatory adaptation targets. Uniform conversion of all American references would trigger saturation (C1.2/SAT) penalties on the majority of edits.

---

## Models and Prompt Strategies

### Models (8 total)

| Tier | Model | Params | Origin |
|---|---|---|---|
| High | Llama-3.3-70B-Instruct (AWQ) | 70B | Meta |
| High | Qwen2.5-72B-Instruct (AWQ) | 72B | Alibaba |
| Mid | Aya-Expanse-32B | 32B | Cohere |
| Mid | Gemma-3-27B-it | 27B | Google DeepMind |
| Mid | Sarvam-30B | 30B MoE | Sarvam AI (India) |
| Low | Llama-3.1-8B-Instruct | 8B | Meta |
| Low | Gemma-2-9B-it | 9B | Google |
| Low | Mistral-7B-Instruct-v0.3 | 7B | Mistral AI |

Sarvam-30B is an India-origin model, providing a culturally proximate training prior for the target context.

### Prompt Strategies (4 total)

| Strategy | Description |
|---|---|
| `simple` | Minimal one-line instruction |
| `instructed` | Detailed constraint-aware rewriting prompt |
| `oneshot` | Instructed + one hand-crafted example per domain |
| `cot` | Chain-of-thought: IDENTIFY → MAP → VERIFY |

Total: 8 models × 4 strategies × 1,500 items × 4 domains = **192,000** (191,968 after exemplar leakage control in `oneshot`).

---

## Evaluation Pipeline

Seven sequential phases, each operationalizing one or more IAET categories.

```
build_index.py              ── join adaptations to benchmark source records
build_sample_ids_16k.py     ── stratified 16K sample for judge-intensive phases
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 1 · CSI Extraction                              (C1–C5 setup)   │
     │  run_extraction.py   → CSI spans + salience + necessity               │
     │  run_function_frames.py → function frames (denotation + role)         │
     │  run_wikidata_grounding.py → Wikidata QID injection for 28% of CSIs   │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 2 · Output Integrity Filtering                        (C5.1)    │
     │  run_hygiene.py / run_phase2.py                                        │
     │  Labels: pass · fail_soft · fail_hard                                  │
     │  Hard failures receive C5 label and exit the pipeline                  │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 3 · Adaptation Audit                                  (C1)      │
     │  run_audit.py   → per-CSI labels: retained / localized / globalized /  │
     │                    transformed / omitted / distorted                   │
     │  compute_csi_metrics.py → RF, TCT, SAT                                │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 4 · Substitution Validity                             (C2)      │
     │  run_c2_judge.py → affordance · scale · dependency · identity ·        │
     │                    figurative; scale_checker overrides arithmetic       │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 5 · Adapted Function Frames                         (C3 setup)  │
     │  build_dedup_pairs.py → cluster near-duplicate adapted spans           │
     │  run_adapted_frames.py → generate adapted-side function frames         │
     │  build_paired_frames.py → project back to evaluation items             │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 6 · Semantic Function Preservation                    (C3)      │
     │  compute_fps.py → FPS (LLM judge, stratified 16K)                     │
     │  compute_fes.py → FES (embedding cosine, contextual + holistic)       │
     │  compute_fbs.py → FBS (BERTScore-F1)                                  │
     │  compute_bleu_rouge.py → BLEU / ROUGE-L (lexical baselines)           │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
     ┌──────────────────────────────────▼──────────────────────────────────────┐
     │  Phase 7 · Round-Trip Consistency                          (C4, C5.2)  │
     │  run_csi_extraction.py → CSI extraction on reverse outputs            │
     │  run_9a_surface_metrics.py → BLEU / ROUGE-L / BERTScore              │
     │  run_9c_csi_alignment.py → Rec / Sub / Leak / Miss rates             │
     │  run_9d_llm_judge.py → meaning · structure · detail · overall (0–10)  │
     └──────────────────────────────────┬──────────────────────────────────────┘
                                        │
                             build_unified_scores.py
```

### Metrics at a Glance

| Phase | Metric | What it measures |
|---|---|---|
| 3 | **RF** (↓) | Salience-weighted residual source retention |
| 3 | **TCT** (↑) | Salience-weighted target-culture transfer credit |
| 3 | **SAT** (↓) | Unsupported cultural insertions relative to source salience |
| 4 | **C2** (↑) | Weighted composite of five substitution quality dimensions |
| 4 | **SW-C2** | Salience-weighted C2 |
| 6 | **FES-C / FES-H** (↑) | Frame embedding cosine similarity (contextual / holistic) |
| 6 | **FBS-C / FBS-H** (↑) | BERTScore-F1 over function frame text |
| 6 | **FPS** (↑) | Structured LLM judgment of functional equivalence (stratified 16K) |
| 7 | **Rec / Lk / Ms** | CSI recovered / leaked / missing in round-trip |
| 7 | **Mn / St / Dt / Ov** | LLM judge: meaning, structure, detail, overall (0–10) |

---

## Key Results

### Output Integrity (Phase 2)

Pass rates fracture along model origin and prompt structure. The simple prompt induces widespread meta-commentary failures in top models and collapses Sarvam further.

| Model | CoT | 1-shot | Inst. | Simple | **All** |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-72B | **96.3** | 95.9 | 95.8 | 14.4 | 75.6 |
| Gemma-3-27B | 94.9 | **96.1** | **95.9** | 2.5 | 72.4 |
| Llama-3.3-70B | 95.5 | 95.7 | 95.6 | 9.9 | 74.2 |
| Mistral-7B | 91.7 | 94.7 | 85.5 | **49.5** | **80.3** |
| Gemma-2-9B | 92.2 | 94.9 | 94.7 | 29.6 | 77.9 |
| Llama-3.1-8B | 91.6 | 94.2 | 79.5 | 12.8 | 69.5 |
| Aya-Expanse-32B | 73.2 | 21.2 | 3.3 | 27.6 | 31.3 |
| Sarvam-30B | 12.8 | 9.6 | 4.5 | 14.7 | 10.4 |
| *Overall* | 81.0 | 75.3 | 69.4 | 20.1 | 61.5 |

### Adaptation Audit (Phase 3)

RF, TCT, and SAT form a three-way trade-off rather than a single quality axis. Qwen's low RF traces to its 39.2% retention rate — model-intrinsic conservatism, not prompt sensitivity. Aya inserts nearly one unjustified Indian item per source CSI (SAT ≈ 1.0).

| Model | RF ↓ | TCT ↑ | SAT ↓ | Loc% ↑ | Ret% ↓ |
|---|---:|---:|---:|---:|---:|
| Llama-3.3-70B | 0.055 | **0.903** | 0.660 | **77.0** | **19.4** |
| Gemma-3-27B | 0.055 | 0.876 | 0.637 | 71.7 | 21.4 |
| Sarvam-30B | **0.053** | 0.872 | **0.289** | 63.8 | 28.7 |
| Llama-3.1-8B | 0.077 | 0.870 | 0.700 | 71.9 | 24.1 |
| Gemma-2-9B | 0.073 | 0.840 | 0.515 | 65.3 | 27.1 |
| Mistral-7B | 0.081 | 0.859 | 0.857 | 69.4 | 24.9 |
| Aya-Expanse-32B | 0.074 | 0.852 | 0.997 | 67.2 | 25.4 |
| Qwen2.5-72B | 0.122 | 0.799 | 0.313 | 57.5 | **39.2** |

### Substitution Validity (Phase 4)

Mean C2 = **0.6104** across 155,555 passing adaptations. All non-trivial failures reduce to two subtypes: C2.1 affordance violations (11.8%) and C2.2 scale transfer errors (21.8%). Qwen leads C2 (0.653) because retention bias suppresses substitution volume — its high score is an artifact of doing less. The salience-weighted degradation (Δ = −0.024) is uniform across models, confirming failures concentrate on high-salience CSIs.

Unit/currency is the only genuinely hard category: mean C2 = 0.570 with a bimodal distribution reflecting all-or-nothing arithmetic outcomes.

### Semantic Function Preservation (Phase 6)

FES (0.54–0.56) and FBS (0.60) are stable across all slices. BLEU (0.04) and ROUGE-L (0.23) confirm function preservation cannot be reduced to surface overlap.

| Model | FES-C ↑ | FES-H ↑ | FBS-C ↑ | FBS-H ↑ | FPS ↑ |
|---|---:|---:|---:|---:|---:|
| Gemma-3-27B | 0.544 | **0.566** | **0.603** | **0.605** | **0.660** |
| Aya-Expanse-32B | **0.545** | 0.563 | **0.603** | 0.605 | 0.655 |
| Sarvam-30B | 0.533 | **0.568** | 0.598 | 0.602 | 0.657 |
| Mistral-7B | 0.544 | 0.561 | 0.604 | 0.604 | 0.654 |
| Llama-3.3-70B | 0.536 | 0.556 | 0.599 | 0.601 | 0.648 |
| Qwen2.5-72B | 0.533 | 0.559 | 0.599 | 0.602 | 0.645 |
| *Global* | 0.540 | 0.560 | 0.601 | 0.603 | 0.650 |

**Key finding:** audit poles Qwen (conservative) and Llama-3.3-70B (aggressive) converge in function space (FPS 0.645 vs. 0.648), confirming substitution quantity does not predict functional quality.

### Round-Trip Consistency (Phase 7)

Near-zero leakage (1.3%) shows the reverse pass reliably strips Indian CSIs. The high missing rate (30.7%) places the information bottleneck in the **forward** pass. Detail preservation collapses to 4.80/10 across every model and domain — the primary failure mode.

| Slice | BLEU ↑ | ROUGE-L ↑ | BS-F1 ↑ | Rec ↑ | Lk ↓ | Ms ↓ | Detail ↑ | Overall ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.3-70B | 0.618 | **0.750** | **0.868** | **0.726** | **0.008** | **0.290** | **5.26** | **7.82** |
| Gemma-2-9B | **0.620** | 0.734 | 0.852 | 0.689 | 0.010 | 0.298 | 4.92 | 7.50 |
| Mistral-7B | 0.495 | 0.676 | 0.835 | 0.714 | 0.020 | 0.319 | 4.44 | 7.17 |
| One-shot | **0.739** | **0.859** | **0.928** | 0.634 | **0.003** | 0.295 | **5.34** | **7.93** |
| CoT | 0.706 | 0.836 | 0.910 | 0.652 | 0.008 | **0.292** | 5.15 | 7.64 |
| Simple | 0.163 | 0.351 | 0.651 | **0.910** | 0.036 | 0.355 | 3.57 | 6.56 |
| *Global avg.* | 0.574 | 0.717 | 0.849 | 0.706 | 0.013 | 0.307 | 4.80 | 7.46 |

---

## Repository Structure

```
.
├── README.md                        ← this file
├── requirements.txt                 ← pip dependencies
├── environment.yml                  ← conda environment
├── .env.example                     ← API key template
├── .gitattributes
├── .gitignore
└── src/
    ├── Dataset/                     ← benchmark construction
    │   ├── 01_download.py
    │   ├── 02_sieve_recipes.py
    │   ├── 02_sieve_stories.py
    │   ├── 03_process.py            ← 4-tier CSI detection (GPU)
    │   ├── 04_diversity_sample.py
    │   ├── 05_verify.py
    │   ├── run_pipeline.sh
    │   └── README.md
    ├── Adaptations/                 ← forward & reverse generation
    │   ├── adapt.py
    │   ├── reverse_adapt.py
    │   ├── model_{aya,gemma,gemma2,gpt4o,llama,llama8b,mistral,qwen,sarvam}.py
    │   ├── run_adapt.sh
    │   ├── run_reverse.sh
    │   └── README.md
    └── Framework/                   ← seven-phase evaluation
        ├── build_index.py
        ├── build_sample_ids_16k.py
        ├── build_unified_scores.py
        ├── prompts/                 ← prompt templates (.txt)
        ├── 01_CSI_Extraction/
        ├── 02_Hygiene/
        ├── 03_Audit/
        ├── 04_Validity/
        ├── 05_Frames/
        ├── 06_Analysis/
        ├── 07_Reverse/
        └── README.md
```

Each `src/` subdirectory has its own `README.md` with inputs, outputs, and usage.

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
# Edit .env and fill in OPENAI_API_KEY and GOOGLE_APPLICATION_CREDENTIALS
```

The adaptation scripts load `.env` from the repository root automatically. For framework phases, export the keys in your shell or activate the `.env` before running.

### 4. (Optional) Download NLTK data

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
bash   src/Dataset/run_pipeline.sh          # GPU; ~6–8 h on A100
python src/Dataset/04_diversity_sample.py --datasets gsm8k rocstories dailydialog recipenlg
python src/Dataset/05_verify.py
```

Output: `03_data/processed/benchmark.jsonl`

### Stage 1 — Generate adaptations

```bash
bash src/Adaptations/run_adapt.sh      # forward (American→Indian); GPU
bash src/Adaptations/run_reverse.sh    # reverse (Indian→American); GPU
```

For a smoke test (2 items/domain, no files written):

```bash
CUDA_VISIBLE_DEVICES=0 python src/Adaptations/adapt.py \
    --model mistral --prompt_level simple --smoke_test
```

### Stage 2 — Evaluate

```bash
# Index
python src/Framework/build_index.py
python src/Framework/build_sample_ids_16k.py

# Phase 1
python src/Framework/01_CSI_Extraction/run_extraction.py --backend google
python src/Framework/01_CSI_Extraction/run_function_frames.py --backend google
python src/Framework/01_CSI_Extraction/run_wikidata_grounding.py

# Phase 2
python src/Framework/02_Hygiene/run_hygiene.py
python src/Framework/02_Hygiene/run_phase2.py --resume

# Phase 3
python src/Framework/03_Audit/run_audit.py
python src/Framework/03_Audit/compute_csi_metrics.py

# Phase 4
python src/Framework/04_Validity/run_c2_judge.py

# Phase 5
python src/Framework/05_Frames/build_dedup_pairs.py
python src/Framework/05_Frames/run_adapted_frames.py
python src/Framework/05_Frames/build_paired_frames.py

# Phase 6
python src/Framework/06_Analysis/compute_fps.py
python src/Framework/06_Analysis/compute_fes.py
python src/Framework/06_Analysis/compute_fbs.py
python src/Framework/06_Analysis/compute_bleu_rouge.py

# Phase 7
python src/Framework/07_Reverse/run_csi_extraction.py
python src/Framework/07_Reverse/run_9a_surface_metrics.py
python src/Framework/07_Reverse/run_9c_csi_alignment.py
python src/Framework/07_Reverse/run_9d_llm_judge.py

# Aggregate
python src/Framework/build_unified_scores.py
```

---

## Data Layout

Populated by running the pipeline; not committed to the repository.

```
03_data/
├── raw/                                  ← source corpora from download step
└── processed/
    ├── benchmark.jsonl                   ← 6,000-item benchmark
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

Both APIs are called asynchronously with retry logic and configurable concurrency limits.

### Hardware Requirements

| Stage | Minimum | Notes |
|---|---|---|
| Dataset construction (Step 3) | 1× GPU, ≥16 GB VRAM | Qwen2.5-7B fp16 + spaCy transformer |
| Adaptation — 7B/8B/9B models | 1× GPU, ≥16 GB VRAM | tensor parallel = 1 |
| Adaptation — 27B/32B models | 2× GPU, ≥32 GB VRAM each | tensor parallel = 2 |
| Adaptation — 70B/72B (AWQ) | 2× GPU, ≥40 GB VRAM each | tensor parallel = 2 |
| Framework phases 1–7 | CPU only | except FBS (GPU speeds BERTScore) |

---

## Reproducibility Notes

- All pipeline scripts are **checkpoint-resumable**: interrupted runs can be restarted with `--resume` and will skip already-completed items.
- Randomness is seeded (`PYTHONHASHSEED=42`, `random.seed(42)`) throughout.
- CUDA determinism is enforced via `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- The one-shot exemplars are hand-crafted (one per domain) and included in `src/Adaptations/adapt.py`; no external exemplar dataset is required.
- Wikidata grounding uses the public SPARQL endpoint (`query.wikidata.org`) with no authentication.
- See `REPRODUCIBILITY.md` for step-by-step environment setup and common troubleshooting.

---

## Citation

```bibtex
@inproceedings{anonymous2025lostinlocalization,
  title     = {Lost in Localization: Benchmarking Intralingual Cultural Adaptation
               in Large Language Models},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for
               Computational Linguistics (ACL 2025)},
  year      = {2025},
  note      = {Anonymous submission}
}
```
