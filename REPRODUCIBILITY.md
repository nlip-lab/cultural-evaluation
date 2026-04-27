# Reproducibility Guide

Step-by-step instructions for setting up and running the full pipeline from scratch.

---

## 1. Environment

**Python 3.10** is recommended. The code is tested on Linux with CUDA 11.8/12.1.

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate cultural-eval

# pip only
python -m pip install -r requirements.txt
```

After installation, download the spaCy NER model used in dataset construction:

```bash
python -m spacy download en_core_web_trf
```

And NLTK punkt data used by the surface-metrics phase:

```python
import nltk
nltk.download('punkt_tab')
```

---

## 2. API Credentials

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API keys |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP console → IAM → Service Accounts → JSON key with `generativelanguage.googleapis.com` enabled |
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read-only) — only needed for `gemma-2-9b-it` and `llama-3.1-8b` |

The adaptation scripts load `.env` automatically. For framework scripts, export before running:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## 3. Data Directory

Create the expected directory skeleton before running any pipeline step:

```bash
mkdir -p 03_data/raw 03_data/processed/adapted_data 03_data/processed/reverse_adapted_data
mkdir -p 04_experiments/logs
```

---

## 4. Dataset Construction

Estimated wall-clock time on a single A100-80GB:

| Step | Time |
|---|---|
| `01_download.py` | ~10 min (network-bound) |
| `02_sieve_recipes.py` | ~8 min |
| `02_sieve_stories.py` | ~2 min |
| `03_process.py` (all datasets) | ~6–8 h |
| `04_diversity_sample.py` | ~20 min |
| `05_verify.py` | <1 min |

Step 3 checkpoints every 100 items and is safely resumable after interruption.

---

## 5. Adaptation Generation

Each model is run sequentially within `adapt.py`. Estimated time per model on 2× A100:

| Model tier | Time for all 4 domains × 1 strategy |
|---|---|
| 70B/72B (AWQ, TP=2) | ~4–5 h |
| 27B–32B (TP=2) | ~3–4 h |
| 7B–9B (TP=1) | ~1–2 h |

For a quick sanity check (no files written):

```bash
CUDA_VISIBLE_DEVICES=0 python src/Adaptations/adapt.py \
    --model mistral --prompt_level cot --smoke_test
```

---

## 6. Framework Phases

Phases 1, 3, 4, 5, 6 (FPS), and 7 make LLM API calls. Expected API usage at full scale:

| Phase | Calls | Approx. cost |
|---|---|---|
| 1 — extraction (Gemini Flash Lite) | ~6,000 | low |
| 1 — frames (Gemini Flash Lite) | ~33,000 | low |
| 3 — audit (GPT-4o) | ~155,000 | moderate |
| 4 — validity (GPT-4o) | ~95,000 | moderate |
| 6 — FPS (GPT-4o, 16K subset) | ~16,000 | low |
| 7 — judge (Gemini Pro) | ~100,000 | moderate |

All phases support `--pilot N` to run on the first N items for cost estimation before committing to the full run. All phases checkpoint incrementally and support `--resume`.

---

## 7. Common Issues

**`CUDA out of memory` during adaptation**
- Reduce `gpu_memory_utilization` in the relevant `model_*.py` (default 0.90).
- For 70B/72B models ensure both GPUs have ≥40 GB free before launching.

**spaCy `en_core_web_trf` not found**
```bash
python -m spacy download en_core_web_trf
```

**Google API `403 PERMISSION_DENIED`**
- Ensure the service account has `Generative Language API` enabled in GCP and the JSON path in `GOOGLE_APPLICATION_CREDENTIALS` is correct.

**`openai.AuthenticationError`**
- Check `OPENAI_API_KEY` is exported or present in `.env` at the repo root.

**Framework scripts fail to find input files**
- All framework scripts are designed to run from the **repository root**, not from within their subdirectory. Use `python src/Framework/...` from the root.

---

## 8. Seeding and Determinism

| Mechanism | Value |
|---|---|
| `PYTHONHASHSEED` | 42 |
| `random.seed` | 42 |
| `numpy.random.seed` | 42 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| vLLM temperature | 0.0 (greedy decoding) |

These are set automatically inside the relevant scripts. The only non-determinism source is LLM API responses, which are not seeded.
