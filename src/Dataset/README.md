# Dataset Pipeline

Constructs the 6,000-item benchmark from four American-grounded source datasets, annotating each item with Culturally Sensitive Items (CSIs) and curating for diversity.

## Pipeline Steps

```
01_download.py          Download raw data (GSM8K, ROCStories, DailyDialog, RecipeNLG)
        |
02_sieve_recipes.py     8-stage sieve: RecipeNLG 2.2M -> 50K candidates
02_sieve_stories.py     5-stage sieve: ROCStories 98K -> 8,960 candidates
        |
03_process.py           Tier 0-3 CSI detection on all 4 datasets (GPU required)
run_pipeline.sh         Shell launcher for 03_process.py
        |
04_diversity_sample.py  KMeans diversity sampling -> 1,500 items per dataset
        |
05_verify.py            Verification report on final benchmark.jsonl
```

Sieving is only applied to RecipeNLG and ROCStories because their raw sizes (2.2M and 98K) would make GPU-based detection prohibitively slow without pre-filtering.

## Quick Start

```bash
# Step 1: Download
python 01_download.py

# Step 2: Sieve large datasets
python 02_sieve_recipes.py
python 02_sieve_stories.py

# Step 3: CSI detection (GPU)
bash run_pipeline.sh

# Step 4: Diversity sampling
python 04_diversity_sample.py --datasets gsm8k rocstories dailydialog recipenlg

# Step 5: Verify
python 05_verify.py
```

## Source Datasets

| Domain | Dataset | Raw Size |
|---|---|---|
| educational | GSM8K | 8,792 |
| stories | ROCStories | 98,161 |
| dialogue | DailyDialog | 13,118 conversations |
| procedural | RecipeNLG | 2,231,142 recipes |

## CSI Detection Architecture (03_process.py)

- **Tier 0**: Fast compiled regex matching for high-precision cultural markers (instant YES, no GPU)
- **Tier 1**: GPU LLM classification (Qwen2.5-7B-Instruct, fp16) + spaCy NER (en_core_web_trf)
- **Tier 2**: Conflict/confidence gate - triggers Tier 3 on LLM/NER disagreement or low confidence
- **Tier 3**: LLM reconsideration with NER evidence highlighted in prompt

Checkpointing is built in: the pipeline saves progress every 100 items and resumes safely after interruption.

## Output

`03_data/processed/benchmark.jsonl` - 6,000 items (1,500 per domain), each containing:
- `culturally_sensitive`: boolean
- `markers_found`: list of detected cultural marker strings
- `resolution_tier`: 0 (Tier 0 match) / 1 (Tier 1 high confidence) / 3 (Tier 3 refined)
- `confidence`: HIGH / MEDIUM / LOW

## Hardware

Step 3 requires a CUDA GPU with >=16 GB VRAM (Qwen2.5-7B fp16 + spaCy transformer on the same device). Steps 1, 2, 4, and 5 run on CPU.
