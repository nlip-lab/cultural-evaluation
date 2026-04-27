# Adaptations Pipeline

Generates and evaluates cultural adaptations of benchmark items using nine LLMs across four prompt strategies.

## Direction

- **Forward**: American -> Indian (`adapt.py` / `run_adapt.sh`)
- **Reverse**: Indian -> American (`reverse_adapt.py` / `run_reverse.sh`)

## Files

```
adapt.py            Forward adaptation controller (orchestrates all models)
reverse_adapt.py    Reverse adaptation controller (Indian -> American)
model_aya.py        Aya-Expanse-32B (CohereLabs) - multilingual
model_gemma.py      Gemma-3-27B-it (Google DeepMind)
model_gemma2.py     Gemma-2-9B-it (Google) - scale ablation
model_gpt4o.py      GPT-4o (OpenAI) - API-based ceiling model
model_llama.py      Llama-3.3-70B-Instruct-AWQ (Meta) - large Western baseline
model_llama8b.py    Llama-3.1-8B-Instruct (Meta) - scale ablation
model_mistral.py    Mistral-7B-Instruct-v0.3 - floor model
model_qwen.py       Qwen2.5-72B-Instruct-AWQ (Alibaba) - Eastern-origin
model_sarvam.py     Sarvam-30B (Sarvam AI) - Indian-origin MoE
run_adapt.sh        Launcher for adapt.py (cot mode, GPUs 1+2)
run_reverse.sh      Launcher for reverse_adapt.py with GPU auto-detection
```

## Quick Start

```bash
# Forward adaptation (all models, cot prompt)
bash run_adapt.sh

# Or directly:
python adapt.py --input ../../../03_data/processed/benchmark.jsonl \
    --model llama --prompt_level instructed --temperature 0.0 --gpus "1,2"

# Smoke test (2 items/domain, no files written):
CUDA_VISIBLE_DEVICES=0 python adapt.py --model mistral \
    --prompt_level simple --smoke_test

# Reverse adaptation:
bash run_reverse.sh
```

## Models

| Name | Model ID | Params | VRAM | TP |
|---|---|---|---|---|
| gpt4o | gpt-4o | - | API | - |
| llama | Llama-3.3-70B-Instruct-AWQ | 70B AWQ | ~40 GB | 2 |
| qwen | Qwen2.5-72B-Instruct-AWQ | 72B AWQ | ~40 GB | 2 |
| aya | aya-expanse-32b | 32B | ~64 GB | 2 |
| sarvam | sarvam-30b | 32B MoE | ~64 GB | 2 |
| gemma | gemma-3-27b-it | 27B | ~54 GB | 2 |
| llama8b | Llama-3.1-8B-Instruct | 8B | ~16 GB | 1 |
| gemma2 | gemma-2-9b-it | 9B | ~18 GB | 1 |
| mistral | Mistral-7B-Instruct-v0.3 | 7B | ~14 GB | 1 |

## Prompt Strategies

| Level | Description |
|---|---|
| `simple` | Minimal one-line instruction |
| `instructed` | Detailed 3-rule system prompt |
| `oneshot` | System prompt + one hand-crafted example per domain |
| `cot` | Chain-of-thought with IDENTIFY -> MAP -> VERIFY steps |

## Output

```
03_data/processed/adapted_data/runs/{prompt_level}/{model}_{domain}.jsonl
03_data/processed/reverse_adapted_data/runs/{prompt_level}/{model}_{domain}.jsonl
```

Each output file begins with a config header line (`is_config_header: true`) followed by one record per adapted item, containing `original_text`, `adapted_text`, `cot_reasoning` (CoT only), and metadata.

Checkpointing is built in: files are written incrementally and can be safely resumed after interruption.
