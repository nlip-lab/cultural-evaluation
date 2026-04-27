"""
Culturally Sensitive Item (CSI) detection pipeline.

Architecture: Tier 0-3 cascade with checkpointing.

  Tier 0 (keyword matching):
    Fast regex patterns for obvious cultural markers -> instant YES

  Tier 1 (sequential inference on same GPU):
    LLM classification + NER entity extraction

  Tier 2 (robust decision gate):
    Conflict detection (LLM=NO and NER=YES), confidence assessment,
    and verification triggers

  Tier 3 (conditional refinement):
    LLM reconsideration with NER entities highlighted -> FINAL VERDICT

Checkpointing saves progress every N items; auto-resumes on restart.
All four source datasets are processed: gsm8k, dailydialog, rocstories, recipenlg.
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import psutil
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "03_data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

DATASET_FILES = {
    "gsm8k": RAW_DIR / "gsm8k" / "gsm8k.jsonl",
    "rocstories": PROC_DIR / "rocstories" / "rocstories_cultural_adaptation_8960_v3.jsonl",
    "dailydialog": RAW_DIR / "dailydialog" / "dailydialog.jsonl",
    "recipenlg": PROC_DIR / "recipenlg" / "recipenlg_cultural_adaptation_50k_v3.jsonl",
}

DATASET_TO_DOMAIN = {
    "gsm8k": "educational",
    "rocstories": "stories",
    "dailydialog": "dialogue",
    "recipenlg": "procedural",
}

TEXT_LABELS = {
    "educational": "Question",
    "stories": "Story",
    "dialogue": "Dialogue",
    "procedural": "Recipe",
}

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 240
DEFAULT_BATCH_SIZE = 8
MIN_FREE_GB_FOR_MODEL = 12.0

MAX_PROMPT_LENGTH = 32768
TRUNCATION_WARNING_THRESHOLD = 0.95
SAFE_BATCH_TOKEN_THRESHOLD = 8192

ACTIVE_MODEL_NAME = MODEL_NAME
ACTIVE_QUANTIZATION = "none"

CSI_NER_TYPES = {"PERSON", "GPE", "LOC", "NORP", "FAC"}

TIER0_PATTERNS = {
    "educational": [
        r"\b(?:u\.?s\.?\s*dollars?|usd)\s*\d+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:usd|us\s*dollars?)\b",
        r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:pounds?|lbs?|ounces?|oz)\b",
        r"\b\d+(?:\.\d+)?\s*(?:fahrenheit|°\s*f)\b",
        r"\b(?:thanksgiving|black\s+friday|fourth\s+of\s+july|independence\s+day|super\s+bowl|memorial\s+day|labor\s+day)\b",
        r"\b(?:irs|dmv|usps|medicare|medicaid|social\s+security)\b",
        r"\b(?:walmart|best\s*buy)\b",
        r"\b(?:united\s+states|u\.s\.|u\.s\.a\.|usa|american)\b",
    ],
    "stories": [
        r"\b(?:u\.?s\.?\s*dollars?|usd)\s*\d+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:usd|us\s*dollars?)\b",
        r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
        r"\b(?:thanksgiving|black\s+friday|fourth\s+of\s+july|independence\s+day|super\s+bowl|memorial\s+day|labor\s+day)\b",
        r"\b(?:walmart|best\s*buy)\b",
        r"\b(?:nfl|nba|mlb|ncaa)\b",
        r"\b(?:united\s+states|u\.s\.|u\.s\.a\.|usa|american)\b",
    ],
    "dialogue": [
        r"\b(?:u\.?s\.?\s*dollars?|usd)\s*\d+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:usd|us\s*dollars?)\b",
        r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",
        r"\b(?:thanksgiving|black\s+friday|fourth\s+of\s+july|independence\s+day|super\s+bowl|memorial\s+day|labor\s+day)\b",
        r"\b(?:walmart|best\s*buy)\b",
        r"\b(?:nfl|nba|mlb|ncaa)\b",
        r"\b(?:united\s+states|u\.s\.|u\.s\.a\.|usa|american)\b",
    ],
    "procedural": [
        r"\b(?:thanksgiving|fourth\s+of\s+july|super\s+bowl|memorial\s+day|labor\s+day)\b",
        r"\b(?:bisquick|jell-o|cool\s*whip|crisco|velveeta|kraft\s+mac\s*(?:and|&)\s+cheese|campbell\'?s|hellmann\'?s|pillsbury|betty\s*crocker|libby\'?s\s+pumpkin)\b",
        r"\b(?:cornbread|mac\s*(?:and|&)\s*cheese|s\'?mores|buffalo\s+wings|pecan\s+pie|pumpkin\s+pie|meatloaf)\b",
        r"\b(?:united\s+states|u\.s\.|u\.s\.a\.|usa|american)\b",
    ],
}

TIER0_COMPILED = {
    domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for domain, patterns in TIER0_PATTERNS.items()
}

_SPEAKER_LABEL_RE = re.compile(r"^\s*Speaker\s+[A-Z]\s*:\s*", re.MULTILINE | re.IGNORECASE)


def strip_speaker_labels(text: str) -> str:
    """Remove speaker prefixes used in dialogue formatting."""
    return _SPEAKER_LABEL_RE.sub("", text)


def _header(title: str, char: str = "#"):
    """Print a full-width section header."""
    width = 80
    print(f"\n{char * width}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{char * width}", flush=True)


def _section(title: str):
    """Print a subsection separator line."""
    print(f"\n  -- {title} {'-' * max(1, 60 - len(title))}", flush=True)


def _kv(key: str, value, indent: int = 4):
    """Print one aligned key-value pair."""
    print(f"{' ' * indent}{key:<28s}: {value}", flush=True)


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Render progress as ASCII bar and percentage."""
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "." * (width - filled)
    pct = current / max(total, 1) * 100
    return f"[{bar}] {pct:5.1f}%"


def _one_line(text: str, max_len: int = 220) -> str:
    """Collapse whitespace and clip long strings for terminal output."""
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _print_smoke_summary(record: dict):
    """Print concise sample output for smoke test mode."""
    print("\n  [SMOKE SAMPLE]", flush=True)
    _kv("Text", _one_line(record.get("original_text", ""), 220), indent=6)
    _kv(
        "Judgement",
        f"{record.get('culturally_sensitive', 'NO')} "
        f"(tier={record.get('resolution_tier', '?')}, conf={record.get('confidence', 'LOW')})",
        indent=6,
    )
    _kv("Reasoning", _one_line(record.get("reasoning", ""), 320), indent=6)


SYSTEM_PROMPT = """You are a Culturally Sensitive Item (CSI) detector for cross-cultural NLP research.

Your task: detect ONLY items that are UNDENIABLY, HEAVILY grounded in specific culture.
A text is culturally sensitive ONLY if it contains markers that would REQUIRE active adaptation when transplanting to a different culture.

STRICT INCLUSION CRITERIA:
An item is YES only if it contains ANY of:
1. UNAMBIGUOUS PERSON NAMES — Proper nouns can contribute signal, but a name alone is NOT sufficient for YES.
2. SPECIFIC PLACE NAMES — Real geographies: countries, states, cities, landmarks (e.g., Texas, London, California). NOT generic locations like "town", "city", "school".
3. CURRENCY AMOUNTS — Real currency with amounts ($50, €20, ¥100). A currency symbol or word alone is NOT enough.
4. IMPERIAL/US MEASUREMENTS — US customary units ONLY: ounces, pounds, Fahrenheit, miles. NOT metric (kg, km, Celsius).
5. US-SPECIFIC HOLIDAYS — Holidays unique/distinct to American culture: Thanksgiving, Halloween, Fourth of July, Super Bowl, Black Friday. NOT global holidays like Christmas, Easter.
6. AMERICAN CULTURAL PRODUCTS — Brands/foods iconic to US culture (Jell-O, Bisquick, cornbread, s'mores, barbecue). NOT generic words.
7. AMERICAN SPORTS — Distinctly American sports (American football, baseball, NFL). NOT universal sports like soccer.

DECISION RULE FOR BENCHMARK QUALITY:
- A single weak marker is not enough.
- YES requires clear, non-trivial cultural grounding that would force meaningful American -> Indian adaptation.
- Example hard negative: "John went to school." should be NO.

STRICT EXCLUSION CRITERIA:
Reject (NO) if:
- Markers are TRIVIAL or INCIDENTAL (a name appears once in a massive text that's otherwise generic)
- Markers are GENERIC/UNIVERSAL (generic roles, common nouns, metric units, global food)
- The cultural marker does NOT actively force adaptation (e.g., "John ate rice" — generic, minimal US signal)
- Pronouns (he, she, they, I, you), dialogue labels (Speaker A), or formatting are mistaken for content
- The text is otherwise culturally neutral (adapting to another culture would require almost no changes)

You MUST respond in EXACTLY this format:
MARKERS_FOUND: [comma-separated list of specific markers, or "none"]
REASONING: [Explain why these markers require adaptation to an Indian context, or why they are too generic]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
VERDICT: [YES or NO]"""


DOMAIN_FEW_SHOT_EXAMPLES = {
    "educational": """EXAMPLES (Educational Questions):

Question: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
MARKERS_FOUND: Weng (person name), $12 (currency amount)
REASONING: The explicit dollar amount is culturally anchored and requires adaptation to local currency conventions.
CONFIDENCE: HIGH
VERDICT: YES

Question: "A library received 50 visitors on Monday. On Tuesday it received twice as many. How many total visitors over both days?"
MARKERS_FOUND: none
REASONING: No culturally anchored entities, institutions, currency, or units are present.
CONFIDENCE: HIGH
VERDICT: NO
""",
    "stories": """EXAMPLES (Narrative Stories):

Story: "John walked home and ate dinner."
MARKERS_FOUND: John (person name)
REASONING: A person name alone is a weak marker and does not force meaningful cultural adaptation.
CONFIDENCE: HIGH
VERDICT: NO

Story: "They bought a turkey at Walmart for Thanksgiving dinner and paid $45."
MARKERS_FOUND: Walmart (brand), Thanksgiving (holiday), $45 (currency amount)
REASONING: Multiple strong US anchors jointly require non-trivial adaptation of holiday, retail, and currency references.
CONFIDENCE: HIGH
VERDICT: YES
""",
    "dialogue": """EXAMPLES (Dialogue):

Dialogue: "Speaker A: Hey John, are you coming to Thanksgiving dinner? Speaker B: Yes, I'll bring pumpkin pie."
MARKERS_FOUND: John (person name), Thanksgiving (holiday)
REASONING: Thanksgiving is a strong US cultural anchor that requires adaptation beyond simple name substitution.
CONFIDENCE: HIGH
VERDICT: YES

Dialogue: "Speaker A: Are you free tomorrow? Speaker B: Yes, let's meet after class."
MARKERS_FOUND: none
REASONING: The dialogue is generic and culturally neutral, with no strong US-specific anchor requiring adaptation.
CONFIDENCE: HIGH
VERDICT: NO
""",
    "procedural": """EXAMPLES (Recipes/Procedures):

Recipe: "Grandma Smith's Southern Cornbread with Jell-O topping. Bake at 350°F."
MARKERS_FOUND: Smith (person name), cornbread (US cultural food), Jell-O (US cultural product), 350°F (US temperature unit)
REASONING: Multiple strong US-specific food and product anchors require concrete cultural adaptation.
CONFIDENCE: HIGH
VERDICT: YES

Recipe: "Vegetable Soup. Ingredients: 1 liter water, 500 g carrots, 300 g potatoes."
MARKERS_FOUND: none
REASONING: Metric units and generic ingredients are globally transferable with minimal adaptation.
CONFIDENCE: HIGH
VERDICT: NO
""",
}


RECONSIDER_SYSTEM_PROMPT = """You are a Culturally Sensitive Item (CSI) detector performing a rigorous SECOND-PASS review.

An initial LLM pass produced a classification. An independent NER system detected named entities.

Your task: provide a FINAL classification with maximum rigor. ONLY accept items that are DEFINITIVELY culturally grounded.

STRICT RULES FOR REFINEMENT:
- Distinguish genuine cultural markers from false alarms and generic false positives.
- Names: only count if they are unambiguous proper nouns (not common nouns, not dialogue labels).
- Places: only count if they are real geographic locations.
- Measurements: only count if they are imperial/US-specific units with numeric context.
- Holidays: only count if distinctly American (Thanksgiving, Halloween, Fourth of July). NOT global.
- Foods/Brands: only count if iconic to US culture.
- If ANY doubt -> REJECT (NO). We want highest precision.

CRITICAL: Even if items passed Tier 1, this is your chance to CORRECT false acceptances.
Reject any items where cultural grounding is weak, trivial, or incidental.

You MUST respond in EXACTLY this format:
MARKERS_FOUND: [comma-separated list of genuine markers, or "none"]
REASONING: [Explain why these markers require adaptation to an Indian context, or why they are too generic]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
VERDICT: [YES or NO]"""


def detect_gpus():
    """Log all visible GPUs and their free/total memory."""
    if not torch.cuda.is_available():
        print("  No GPUs available.", flush=True)
        return
    visible_count = torch.cuda.device_count()
    print(f"  Detected {visible_count} visible GPU(s):", flush=True)
    for idx in range(visible_count):
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        print(f"    GPU {idx}: {props.name}  "
              f"{free_bytes / 1024**3:.1f}GB free / {total_bytes / 1024**3:.1f}GB total", flush=True)


def ensure_device_ready(device: str, min_free_gb: float = MIN_FREE_GB_FOR_MODEL):
    """Validate CUDA device id and minimum free VRAM."""
    if device == "auto" or not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    try:
        index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Invalid device: {device}") from e
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gb = free_bytes / 1024**3
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient VRAM on {device}: {free_gb:.1f}GB free, need {min_free_gb:.1f}GB")


def load_items(dataset_name: str, max_items: int = 0) -> list[dict]:
    """Load input JSONL items for a dataset with optional truncation."""
    path = DATASET_FILES[dataset_name]
    if not path.exists():
        print(f"  ERROR: {path} not found. Run 01_download.py first.", flush=True)
        sys.exit(1)
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
            if max_items and len(items) >= max_items:
                break
    return items


def load_model(device: str = "cuda:0"):
    """Load the LLM and tokenizer on the selected device."""
    global ACTIVE_MODEL_NAME, ACTIVE_QUANTIZATION
    _section("Loading LLM")
    _kv("Model", MODEL_NAME)
    _kv("Target device", device)
    ensure_device_ready(device)

    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    ACTIVE_MODEL_NAME = MODEL_NAME
    ACTIVE_QUANTIZATION = "none"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.num_beams = 1

    _kv("Model loaded on", str(model.device))
    _kv("LLM load time", f"{time.time() - load_start:.1f}s")
    if torch.cuda.is_available():
        idx = int(device.split(":")[1]) if ":" in device else 0
        alloc_gb = torch.cuda.memory_allocated(idx) / 1024**3
        _kv("VRAM allocated", f"{alloc_gb:.1f} GB")
    return model, tokenizer


def load_ner(gpu_id: int = 0):
    """Load spaCy transformer NER model on GPU or CPU."""
    _section("Loading spaCy NER (en_core_web_trf)")
    load_start = time.time()

    if gpu_id >= 0 and torch.cuda.is_available():
        preferred = spacy.prefer_gpu(gpu_id)
        if not preferred:
            raise RuntimeError(f"spaCy could not bind to GPU {gpu_id}")
        spacy.require_gpu(gpu_id)
        _kv("NER device", f"cuda:{gpu_id} (enforced)")
    else:
        _kv("NER device", "cpu")

    nlp = spacy.load("en_core_web_trf")
    doc = nlp("John went to Texas.")
    ents = [(e.text, e.label_) for e in doc.ents]
    _kv("NER model", "en_core_web_trf")
    _kv("Test entities", str(ents))
    _kv("NER load time", f"{time.time() - load_start:.1f}s")
    return nlp


def parse_llm_response(raw: str) -> dict:
    """Parse LLM output robustly, preferring the last structured block if echoes exist."""
    result = {
        "markers_found": [],
        "confidence": "LOW",
        "verdict": "NO",
        "reasoning": "",
        "raw_output": raw,
    }

    marker_hits = list(re.finditer(r"MARKERS_FOUND\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE))
    if marker_hits:
        markers_str = marker_hits[-1].group(1).strip()
        if markers_str.lower() != "none":
            result["markers_found"] = [s.strip() for s in markers_str.split(",") if s.strip()]

    reasoning_hits = list(
        re.finditer(
            r"REASONING\s*:\s*(.+?)(?=\n\s*(?:CONFIDENCE|VERDICT|MARKERS_FOUND)\s*:|$)",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
    )
    if reasoning_hits:
        result["reasoning"] = reasoning_hits[-1].group(1).strip()[:700]
    else:
        result["reasoning"] = raw.strip()[:700]

    conf_hits = list(re.finditer(r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", raw, re.IGNORECASE))
    if conf_hits:
        result["confidence"] = conf_hits[-1].group(1).upper()

    verdict_hits = list(re.finditer(r"VERDICT\s*:\s*(YES|NO)", raw, re.IGNORECASE))
    if verdict_hits:
        result["verdict"] = verdict_hits[-1].group(1).upper()
    else:
        if re.search(r"\bYES\b", raw, re.IGNORECASE) and not re.search(r"\bNO\b", raw, re.IGNORECASE):
            result["verdict"] = "YES"

    return result


def _text_chunks(text: str, chunk_chars: int = 10000, overlap_chars: int = 800) -> list[str]:
    """Split long text into overlapping chunks for inference safety."""
    text = text or ""
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _aggregate_chunk_results(results: list[dict]) -> dict:
    """Aggregate chunk-level results into one conservative verdict."""
    if not results:
        return {
            "markers_found": [],
            "confidence": "LOW",
            "verdict": "NO",
            "reasoning": "No chunk results produced.",
            "raw_output": "",
        }

    yes_rows = [r for r in results if r.get("verdict") == "YES"]
    if yes_rows:
        merged_markers = []
        seen = set()
        for r in yes_rows:
            for m in r.get("markers_found", []):
                ml = m.lower()
                if ml not in seen:
                    seen.add(ml)
                    merged_markers.append(m)
        return {
            "markers_found": merged_markers,
            "confidence": "HIGH" if any(r.get("confidence") == "HIGH" for r in yes_rows) else "MEDIUM",
            "verdict": "YES",
            "reasoning": "At least one chunk contained strong cultural anchors requiring adaptation.",
            "raw_output": "\n---\n".join(r.get("raw_output", "") for r in yes_rows[:3]),
        }

    return {
        "markers_found": [],
        "confidence": "HIGH" if all(r.get("confidence") == "HIGH" for r in results) else "MEDIUM",
        "verdict": "NO",
        "reasoning": "All text chunks were culturally neutral or insufficiently grounded.",
        "raw_output": "\n---\n".join(r.get("raw_output", "") for r in results[:3]),
    }


def check_tier0_match(text: str, domain: str) -> dict:
    """Tier 0: Fast keyword matching for obvious cultural markers."""
    if domain not in TIER0_COMPILED:
        return {"matched": False, "matched_patterns": [], "matched_keywords": []}

    text_for_match = strip_speaker_labels(text) if domain == "dialogue" else text

    matched_keywords = []
    for pattern in TIER0_COMPILED[domain]:
        matches = pattern.findall(text_for_match)
        matched_keywords.extend(matches)

    seen = set()
    unique_keywords = []
    for kw in matched_keywords:
        kw_lower = kw.lower()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return {
        "matched": len(unique_keywords) > 0,
        "matched_patterns": [f"Tier0_{domain}"],
        "matched_keywords": unique_keywords,
    }


def load_checkpoint(dataset_out_dir: Path) -> dict:
    """Load checkpoint from disk, checking both temporary and final output files."""
    checkpoint_path = dataset_out_dir / "_checkpoint.jsonl"
    all_path = dataset_out_dir / "all_classified.jsonl"

    target_path = checkpoint_path if checkpoint_path.exists() else (all_path if all_path.exists() else None)

    if not target_path:
        return {}

    try:
        done = {}
        with open(target_path, "r") as f:
            for line in f:
                r = json.loads(line)
                iid = r.get("id")
                if iid is None:
                    continue
                done[iid] = {
                    "culturally_sensitive": r.get("culturally_sensitive", "NO"),
                    "resolution_tier": r.get("resolution_tier", 1),
                    "tier3_triggered": r.get("tier3_triggered", False),
                    "tier1_verdict": r.get("tier1_verdict", ""),
                    "confidence": r.get("confidence", "UNKNOWN"),
                }
        return done
    except Exception as e:
        print(f"  Warning: Failed to load checkpoint from {target_path.name}: {e}", flush=True)
        return {}


def save_checkpoint_append(new_results: list[dict], checkpoint_path: Path):
    """Append new results to checkpoint file."""
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "a") as f:
            for r in new_results:
                rec = dict(r)
                if "ner_entities" in rec:
                    rec["ner_entities"] = [list(e) for e in rec["ner_entities"]]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}", flush=True)


def get_memory_stats(device: str = "cuda:0") -> dict:
    """Get current memory usage statistics."""
    stats = {}

    ram = psutil.virtual_memory()
    stats["ram_used_gb"] = ram.used / (1024**3)
    stats["ram_total_gb"] = ram.total / (1024**3)
    stats["ram_percent"] = ram.percent

    if torch.cuda.is_available() and device.startswith("cuda"):
        try:
            gpu_idx = int(device.split(":")[1]) if ":" in device else 0
            stats["vram_allocated_gb"] = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            stats["vram_reserved_gb"] = torch.cuda.memory_reserved(gpu_idx) / (1024**3)
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
            stats["vram_free_gb"] = free_bytes / (1024**3)
            stats["vram_total_gb"] = total_bytes / (1024**3)
            stats["vram_percent"] = (1 - free_bytes / total_bytes) * 100
        except Exception as e:
            stats["vram_error"] = str(e)

    return stats


def run_pass1_llm_batch(texts: list[str], domain: str, model, tokenizer) -> list[dict]:
    """Batch-mode Tier 1 LLM inference with OOM fallback to single-item processing."""
    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_stage1_prompt(text, domain)},
        ]
        for text in texts
    ]

    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    prompt_meta = tokenizer(prompts, add_special_tokens=True, truncation=False)
    if any(len(ids) > MAX_PROMPT_LENGTH for ids in prompt_meta["input_ids"]):
        print("  Long prompt in batch detected; falling back to single-item full-text handling.", flush=True)
        return [run_pass1_llm_single_item(text, domain, model, tokenizer) for text in texts]

    if any(len(ids) > SAFE_BATCH_TOKEN_THRESHOLD for ids in prompt_meta["input_ids"]):
        print(
            f"  Batch contains very long prompt (>{SAFE_BATCH_TOKEN_THRESHOLD} tokens); "
            "falling back to 1-by-1 processing for safety.",
            flush=True,
        )
        return [run_pass1_llm_single_item(text, domain, model, tokenizer) for text in texts]

    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_PROMPT_LENGTH,
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        batch_results = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            parsed = parse_llm_response(raw)
            batch_results.append(parsed)

        return batch_results

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during batched LLM inference for batch of {len(texts)}, falling back to 1-by-1", flush=True)
        torch.cuda.empty_cache()
        return [run_pass1_llm_single_item(text, domain, model, tokenizer) for text in texts]


def run_pass1_llm_single_item(text: str, domain: str, model, tokenizer) -> dict:
    """Single-item Tier 1 LLM inference (fallback for OOM or oversized prompts)."""
    user_msg = _build_stage1_prompt(text, domain)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    prompt_meta = tokenizer([prompt], add_special_tokens=True, truncation=False)
    if len(prompt_meta["input_ids"][0]) > MAX_PROMPT_LENGTH:
        print("  Long text detected; using chunked full-text evaluation.", flush=True)
        chunks = _text_chunks(text)
        if len(chunks) <= 1:
            return {
                "markers_found": [],
                "confidence": "LOW",
                "verdict": "NO",
                "reasoning": "Input exceeded safe context window and could not be chunked reliably.",
                "raw_output": "",
            }
        return _aggregate_chunk_results([run_pass1_llm_single_item(ch, domain, model, tokenizer) for ch in chunks])

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [prompt], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_PROMPT_LENGTH,
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "markers_found": [],
            "confidence": "LOW",
            "verdict": "NO",
            "reasoning": "OOM during inference",
            "raw_output": "MARKERS_FOUND: none\nCONFIDENCE: LOW\nVERDICT: NO\nREASONING: Out of memory",
        }

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = parse_llm_response(raw)
    return parsed


def run_pass1_ner_batch(texts: list[str], domain: str, nlp, ner_batch_size: int) -> list[dict]:
    """Batch-mode Tier 1 NER extraction via spaCy."""
    texts_clean = [strip_speaker_labels(t) if domain == "dialogue" else t for t in texts]
    spacy_batch_size = max(1, min(len(texts_clean), max(1, ner_batch_size)))
    batch_results = []
    for doc in nlp.pipe(texts_clean, batch_size=spacy_batch_size):
        ents = [(e.text, e.label_) for e in doc.ents if e.label_ in CSI_NER_TYPES]
        batch_results.append({
            "entities": ents,
            "has_csi_entities": len(ents) > 0,
            "entity_types": list({e[1] for e in ents}),
        })
    return batch_results


def run_pass3_llm_batch(
    texts: list[str],
    domain: str,
    llm_results: list[dict],
    ner_results: list[dict],
    trigger_reasons_batch: list[list[str]],
    model,
    tokenizer,
) -> list[dict]:
    """Batch-mode Tier 3 refinement with OOM fallback to single-item processing."""
    label = TEXT_LABELS.get(domain, "Text")
    prompts = []
    for text, llm_result, ner_result, trigger_reasons in zip(
        texts, llm_results, ner_results, trigger_reasons_batch
    ):
        text_clean = strip_speaker_labels(text) if domain == "dialogue" else text
        ner_entities_str = ", ".join(
            [f"{e[0]} ({e[1]})" for e in ner_result["entities"]]
        ) if ner_result["entities"] else "none"
        user_msg = f"""Re-evaluate this {label} with explicit attention to detected entities.

{label}: \"{text_clean}\"

DETECTED NAMED ENTITIES:
{ner_entities_str}

Previous LLM verdict: {llm_result['verdict']} (confidence: {llm_result['confidence']})
Trigger reasons: {'; '.join(trigger_reasons)}

Apply strict precision criteria. Reason first, then confidence, then verdict."""
        messages = [
            {"role": "system", "content": RECONSIDER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    prompt_meta = tokenizer(prompts, add_special_tokens=True, truncation=False)
    if any(len(ids) > MAX_PROMPT_LENGTH for ids in prompt_meta["input_ids"]):
        print("  Long Tier 3 prompt in batch; falling back to single-item full-text handling.", flush=True)
        return [
            run_pass3_llm(text, domain, llm_result, ner_result, model, tokenizer, trigger_reasons)
            for text, llm_result, ner_result, trigger_reasons in zip(
                texts, llm_results, ner_results, trigger_reasons_batch
            )
        ]

    if any(len(ids) > SAFE_BATCH_TOKEN_THRESHOLD for ids in prompt_meta["input_ids"]):
        print(
            f"  Tier 3 batch contains very long prompt (>{SAFE_BATCH_TOKEN_THRESHOLD} tokens); "
            "falling back to 1-by-1 processing for safety.",
            flush=True,
        )
        return [
            run_pass3_llm(text, domain, llm_result, ner_result, model, tokenizer, trigger_reasons)
            for text, llm_result, ner_result, trigger_reasons in zip(
                texts, llm_results, ner_results, trigger_reasons_batch
            )
        ]

    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        parsed = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            parsed_item = parse_llm_response(raw)
            parsed.append(parsed_item)
        return parsed
    except torch.cuda.OutOfMemoryError:
        print(
            f"  OOM during Tier 3 batched inference for batch of {len(texts)}; falling back to 1-by-1",
            flush=True,
        )
        torch.cuda.empty_cache()
        fallback = []
        for text, llm_result, ner_result, trigger_reasons in zip(
            texts, llm_results, ner_results, trigger_reasons_batch
        ):
            fallback.append(
                run_pass3_llm(text, domain, llm_result, ner_result, model, tokenizer, trigger_reasons)
            )
        return fallback


def _build_stage1_prompt(text: str, domain: str) -> str:
    """Construct pass-1 user prompt with domain-specific examples."""
    label = TEXT_LABELS.get(domain, "Text")
    few_shot = DOMAIN_FEW_SHOT_EXAMPLES.get(domain, DOMAIN_FEW_SHOT_EXAMPLES["educational"])
    text_clean = strip_speaker_labels(text) if domain == "dialogue" else text
    return (
        f"{few_shot}\n\n"
        f"Now classify this {label}.\n"
        f"Domain: {domain}\n\n"
        f"{label}: \"{text_clean}\""
    )


def run_pass1_ner(text: str, domain: str, nlp) -> dict:
    """Single-item Tier 1 NER extraction."""
    text_clean = strip_speaker_labels(text) if domain == "dialogue" else text
    doc = nlp(text_clean)
    ents = [(e.text, e.label_) for e in doc.ents if e.label_ in CSI_NER_TYPES]
    return {
        "entities": ents,
        "has_csi_entities": len(ents) > 0,
        "entity_types": list({e[1] for e in ents}),
    }


def should_trigger_pass3(llm_result: dict, ner_result: dict) -> tuple[bool, list[str]]:
    """Determine whether Tier 3 refinement should run for this item.

    Triggers when:
      - LLM=NO but NER found non-person CSI entities (conflict)
      - LLM confidence is MEDIUM or LOW
      - YES verdict with no markers or only person-entity NER support
    """
    reasons = []
    entity_types = set(ner_result.get("entity_types", []))
    non_person_entities = [e for e in ner_result.get("entities", []) if len(e) > 1 and e[1] != "PERSON"]

    if llm_result["verdict"] == "NO" and non_person_entities:
        non_person_types = ", ".join(sorted(entity_types - {"PERSON"}))
        reasons.append(
            "CONFLICT: LLM=NO but NER found "
            f"{len(non_person_entities)} non-person entities"
            + (f" ({non_person_types})" if non_person_types else "")
        )

    if (
        llm_result["verdict"] == "YES"
        and llm_result["confidence"] == "HIGH"
        and not ner_result["has_csi_entities"]
    ):
        reasons.append("VERIFY: confident YES but NER found zero entities")

    if llm_result["confidence"] in ("MEDIUM", "LOW"):
        reasons.append(f"LOW_CONF: LLM confidence={llm_result['confidence']}")

    markers = llm_result.get("markers_found") or []
    if llm_result["verdict"] == "YES":
        if not markers:
            reasons.append("VERIFY: YES verdict with empty marker set")

        only_person_entities = ner_result["has_csi_entities"] and entity_types.issubset({"PERSON"})
        if only_person_entities:
            reasons.append("VERIFY: YES verdict with person-only NER; no geography or cultural affiliation found")

    return len(reasons) > 0, reasons


def run_pass3_llm(text: str, domain: str, llm_result: dict, ner_result: dict,
                  model, tokenizer, trigger_reasons: list[str]) -> dict:
    """Single-item Tier 3 LLM reconsideration with NER entities highlighted."""
    text_clean = strip_speaker_labels(text) if domain == "dialogue" else text
    label = TEXT_LABELS.get(domain, "Text")

    ner_entities_str = ", ".join([f"{e[0]} ({e[1]})" for e in ner_result["entities"]]) if ner_result["entities"] else "none"

    pass3_msg = f"""Re-evaluate this {label} with explicit attention to the detected named entities:

{label}: "{text_clean}"

DETECTED NAMED ENTITIES (from independent NER system):
{ner_entities_str}

Previous LLM verdict: {llm_result['verdict']} (confidence: {llm_result['confidence']})
Trigger for re-evaluation: {'; '.join(trigger_reasons)}

Provide your FINAL classification. Be rigorous in distinguishing genuine cultural markers from false alarms."""

    messages = [
        {"role": "system", "content": RECONSIDER_SYSTEM_PROMPT},
        {"role": "user", "content": pass3_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    prompt_meta = tokenizer([prompt], add_special_tokens=True, truncation=False)
    if len(prompt_meta["input_ids"][0]) > MAX_PROMPT_LENGTH:
        print("  Long Tier 3 input detected; using chunked full-text refinement.", flush=True)
        chunks = _text_chunks(text)
        if len(chunks) <= 1:
            return {
                "markers_found": llm_result.get("markers_found", []),
                "confidence": llm_result.get("confidence", "LOW"),
                "verdict": llm_result.get("verdict", "NO"),
                "reasoning": "Tier 3 input exceeded safe context window; fell back to Tier 1 decision.",
                "raw_output": "",
            }
        return _aggregate_chunk_results([
            run_pass3_llm(ch, domain, llm_result, ner_result, model, tokenizer, trigger_reasons)
            for ch in chunks
        ])

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [prompt], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_PROMPT_LENGTH,
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length >= MAX_PROMPT_LENGTH * TRUNCATION_WARNING_THRESHOLD:
        print(f"  WARNING: Tier 3 prompt length {prompt_length}/{MAX_PROMPT_LENGTH} tokens - text may be truncated!", flush=True)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "markers_found": llm_result["markers_found"],
            "confidence": llm_result["confidence"],
            "verdict": llm_result["verdict"],
            "reasoning": "Pass 3 OOM; fell back to Pass 1",
            "raw_output": "",
        }

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = parse_llm_response(raw)
    return parsed


def process_dataset(dataset_name: str, model, tokenizer, nlp,
                    test_size: int, batch_size: int, checkpoint_freq: int,
                    device: str, smoke_test: bool = False) -> list[dict]:
    """Process one dataset through the Tier 0-3 CSI detection cascade.

    Tier 0: Keyword matching (instant YES)
    Tier 1: LLM + NER sequential on the same GPU
    Tier 2: Robust decision gate
    Tier 3: LLM with NER evidence (conditional)
    """
    domain = DATASET_TO_DOMAIN[dataset_name]
    items = load_items(dataset_name, max_items=test_size if test_size else 0)

    _header(f"DATASET: {dataset_name}  (domain={domain}, items={len(items):,})", "=")

    _section("Random Samples")
    for it in random.sample(items, min(3, len(items))):
        print(f"    [{it['id']}] {it['text'][:180]}")
        print()

    _section("Pipeline Configuration")
    _kv("Execution mode", "Tier 0-3 (keywords -> LLM/NER -> gate -> refinement)")
    _kv("GPU allocation", device)
    _kv("Checkpoint frequency", f"Every {checkpoint_freq} items")
    _kv("Tier 0", f"{len(TIER0_PATTERNS.get(domain, []))} keyword patterns")
    _kv("Smoke test", smoke_test)

    dataset_out_dir = (PROC_DIR / dataset_name / "smoke_test") if smoke_test else (PROC_DIR / dataset_name)
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dataset_out_dir / "_checkpoint.jsonl"

    if smoke_test:
        for p in [
            dataset_out_dir / "_checkpoint.jsonl",
            dataset_out_dir / "all_classified.jsonl",
            dataset_out_dir / "culturally_sensitive.jsonl",
            dataset_out_dir / "not_sensitive.jsonl",
            dataset_out_dir / "summary.json",
        ]:
            if p.exists():
                p.unlink()
        done = {}
    else:
        done = load_checkpoint(dataset_out_dir)

    if done:
        print(f"  Checkpoint loaded: {len(done):,} items already processed", flush=True)

    items_remaining = [item for item in items if item["id"] not in done]
    total = len(items)
    already_done = len(done)

    _kv("Total items", f"{total:,}")
    _kv("Already completed", f"{already_done:,}")
    _kv("Remaining", f"{len(items_remaining):,}")

    tier0_yes_count = 0
    llm_path_yes_count = 0
    tier1_no_count = 0
    tier3_triggered_count = 0
    tier3_upgraded_count = 0

    for r in done.values():
        if r["culturally_sensitive"] == "YES":
            if r.get("resolution_tier") == 0:
                tier0_yes_count += 1
            elif r.get("resolution_tier") in (1, 3):
                llm_path_yes_count += 1
        else:
            tier1_no_count += 1
        if r.get("tier3_triggered"):
            tier3_triggered_count += 1
        if r.get("tier1_verdict") == "NO" and r.get("culturally_sensitive") == "YES":
            tier3_upgraded_count += 1

    final_yes_count = tier0_yes_count + llm_path_yes_count
    final_no_count = tier1_no_count

    confidence_dist = defaultdict(int)
    for r in done.values():
        confidence_dist[r.get("confidence", "UNKNOWN")] += 1

    start_time = time.time()

    if len(items_remaining) > 0:
        _header(f"PROCESSING ({len(items_remaining):,} remaining items)", "#")
        print()

        checkpoint_buffer = []
        batch_idx = 0
        for batch_start in range(0, len(items_remaining), batch_size):
            batch_end = min(batch_start + batch_size, len(items_remaining))
            batch_items = items_remaining[batch_start:batch_end]
            batch_idx += 1

            batch_ids = [item.get("id", str(item.get("index", idx))) for idx, item in enumerate(batch_items)]
            batch_texts = [item.get("text", "") for item in batch_items]

            if not any(text.strip() for text in batch_texts):
                continue

            batch_tier0_results = [check_tier0_match(text, domain) for text in batch_texts]

            tier1_llm_results: list[Optional[dict]] = [None] * len(batch_texts)
            tier1_ner_results: list[Optional[dict]] = [None] * len(batch_texts)

            tier1_indices = [i for i, t0 in enumerate(batch_tier0_results) if not t0["matched"]]
            if tier1_indices:
                tier1_texts = [batch_texts[i] for i in tier1_indices]
                tier1_llm_subset = run_pass1_llm_batch(tier1_texts, domain, model, tokenizer)
                tier1_ner_subset = run_pass1_ner_batch(tier1_texts, domain, nlp, batch_size)

                for subset_idx, batch_idx_in_full in enumerate(tier1_indices):
                    tier1_llm_results[batch_idx_in_full] = tier1_llm_subset[subset_idx]
                    tier1_ner_results[batch_idx_in_full] = tier1_ner_subset[subset_idx]

            tier3_needed_batch = []
            tier3_indices_global = []

            for tier1_idx, (llm_result, ner_result) in enumerate(zip(tier1_llm_results, tier1_ner_results)):
                if llm_result is None or ner_result is None:
                    continue
                should_run_tier3, trigger_reasons = should_trigger_pass3(llm_result, ner_result)
                if should_run_tier3:
                    tier3_needed_batch.append((llm_result, ner_result, trigger_reasons))
                    tier3_indices_global.append(tier1_idx)

            tier3_results = {}
            if tier3_needed_batch:
                tier3_texts = [batch_texts[i] for i in tier3_indices_global]
                tier3_llm_results = [x[0] for x in tier3_needed_batch]
                tier3_ner_results = [x[1] for x in tier3_needed_batch]
                tier3_trigger_reasons = [x[2] for x in tier3_needed_batch]

                tier3_final_results = run_pass3_llm_batch(
                    tier3_texts,
                    domain,
                    tier3_llm_results,
                    tier3_ner_results,
                    tier3_trigger_reasons,
                    model,
                    tokenizer,
                )

                for global_idx, final_result, trigger_reasons in zip(
                    tier3_indices_global, tier3_final_results, tier3_trigger_reasons
                ):
                    final_result["_trigger_reasons"] = trigger_reasons
                    tier3_results[global_idx] = final_result

            for item_idx_in_batch, item in enumerate(batch_items):
                iid = batch_ids[item_idx_in_batch]
                text = batch_texts[item_idx_in_batch]
                tier0_result = batch_tier0_results[item_idx_in_batch]

                if not text.strip():
                    continue

                if tier0_result["matched"]:
                    record = {
                        "id": iid,
                        "split": item.get("split", ""),
                        "index": item.get("index", -1),
                        "source": item.get("source", ""),
                        "domain": domain,
                        "original_text": text,
                        "culturally_sensitive": "YES",
                        "resolution_tier": 0,
                        "confidence": "HIGH",
                        "markers_found": tier0_result["matched_keywords"],
                        "reasoning": f"Tier 0 keyword match: {', '.join(tier0_result['matched_keywords'][:5])}",
                        "ner_entities": [],
                        "tier0_matched": True,
                        "tier0_keywords": tier0_result["matched_keywords"],
                        "tier1_verdict": "",
                        "tier1_confidence": "",
                        "tier1_markers_found": [],
                        "tier1_reasoning": "",
                        "tier1_raw_output": "",
                        "tier3_triggered": False,
                        "tier3_trigger_reasons": [],
                        "tier3_verdict": "",
                        "tier3_confidence": "",
                        "tier3_markers_found": [],
                        "tier3_reasoning": "",
                        "tier3_raw_output": "",
                        "tier1_tier3_agreement": True,
                    }

                    tier0_yes_count += 1
                    final_yes_count += 1
                    confidence_dist["HIGH"] += 1

                else:
                    llm_result = tier1_llm_results[item_idx_in_batch]
                    ner_result = tier1_ner_results[item_idx_in_batch]
                    if llm_result is None or ner_result is None:
                        raise RuntimeError(
                            f"Internal error: missing Tier 1 results for non-Tier-0 item {iid}"
                        )

                    if item_idx_in_batch in tier3_results:
                        final_result = tier3_results[item_idx_in_batch]
                        final_tier = 3
                        tier3_triggered_count += 1
                        if llm_result["verdict"] == "NO" and final_result["verdict"] == "YES":
                            tier3_upgraded_count += 1
                    else:
                        final_result = llm_result
                        final_tier = 1

                    if final_result["verdict"] == "YES":
                        llm_path_yes_count += 1
                        final_yes_count += 1
                    else:
                        tier1_no_count += 1
                        final_no_count += 1

                    confidence_dist[final_result["confidence"]] += 1

                    record = {
                        "id": iid,
                        "split": item.get("split", ""),
                        "index": item.get("index", -1),
                        "source": item.get("source", ""),
                        "domain": domain,
                        "original_text": text,
                        "culturally_sensitive": final_result["verdict"],
                        "resolution_tier": final_tier,
                        "confidence": final_result["confidence"],
                        "markers_found": final_result["markers_found"],
                        "reasoning": final_result["reasoning"],
                        "ner_entities": ner_result["entities"],
                        "tier0_matched": False,
                        "tier0_keywords": [],
                        "tier3_triggered": (final_tier == 3),
                        "tier3_trigger_reasons": tier3_results[item_idx_in_batch].get("_trigger_reasons", []) if item_idx_in_batch in tier3_results else [],
                        "tier1_verdict": llm_result["verdict"],
                        "tier1_confidence": llm_result["confidence"],
                        "tier1_markers_found": llm_result["markers_found"],
                        "tier1_reasoning": llm_result["reasoning"],
                        "tier1_raw_output": llm_result.get("raw_output", ""),
                    }

                    if final_tier == 3:
                        record["tier3_verdict"] = final_result["verdict"]
                        record["tier3_confidence"] = final_result["confidence"]
                        record["tier3_markers_found"] = final_result["markers_found"]
                        record["tier3_reasoning"] = final_result["reasoning"]
                        record["tier3_raw_output"] = final_result.get("raw_output", "")
                        record["tier1_tier3_agreement"] = (llm_result["verdict"] == final_result["verdict"])
                    else:
                        record["tier3_verdict"] = ""
                        record["tier3_confidence"] = ""
                        record["tier3_markers_found"] = []
                        record["tier3_reasoning"] = ""
                        record["tier3_raw_output"] = ""
                        record["tier1_tier3_agreement"] = True

                if domain == "procedural":
                    record["title"] = item.get("title", "")
                    record["ingredients"] = item.get("ingredients", "")
                    record["directions"] = item.get("directions", "")

                checkpoint_buffer.append(record)

                if smoke_test:
                    _print_smoke_summary(record)

            if (batch_idx % max(1, checkpoint_freq // batch_size)) == 0 or batch_end >= len(items_remaining):
                if checkpoint_buffer:
                    save_checkpoint_append(checkpoint_buffer, checkpoint_path)
                    checkpoint_buffer = []

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_processed = already_done + batch_end
            elapsed = time.time() - start_time
            rate = batch_end / max(elapsed, 0.1)
            eta_seconds = (len(items_remaining) - batch_end) / max(rate, 0.01)
            eta_hours = eta_seconds / 3600

            mem = get_memory_stats(device)

            bar = _progress_bar(total_processed, total, width=30)

            if batch_idx % 10 == 0 or batch_end >= len(items_remaining):
                print(f"  {bar}  {total_processed:>7,}/{total:,}", flush=True)
                print(f"    Batch {batch_idx} | Rate: {rate:.2f} items/s  |  ETA: {eta_hours:.1f}h", flush=True)
                print(f"    Tier0={tier0_yes_count:,} Tier1+={llm_path_yes_count:,} Tier1-={tier1_no_count:,} Tier3={tier3_triggered_count:,}", flush=True)
                print(f"    Confidence: HIGH={confidence_dist.get('HIGH', 0):,} "
                      f"MED={confidence_dist.get('MEDIUM', 0):,} LOW={confidence_dist.get('LOW', 0):,}", flush=True)

                if "vram_allocated_gb" in mem:
                    print(f"    Memory: VRAM {mem['vram_allocated_gb']:.1f}GB/{mem['vram_total_gb']:.1f}GB "
                          f"({mem.get('vram_percent', 0):.0f}%)  |  RAM {mem['ram_used_gb']:.1f}GB "
                          f"({mem['ram_percent']:.0f}%)", flush=True)
                print(flush=True)

    elapsed = time.time() - start_time
    _section("Processing Summary")
    _kv("Items processed", f"{total:,}")
    _kv("Tier 0 YES (keywords)", f"{tier0_yes_count:,} ({tier0_yes_count/max(total,1)*100:.1f}%)")
    _kv("Tier 1+ YES (LLM path)", f"{llm_path_yes_count:,} ({llm_path_yes_count/max(total,1)*100:.1f}%)")
    _kv("Final YES (sensitive)", f"{final_yes_count:,} ({final_yes_count/max(total,1)*100:.1f}%)")
    _kv("Final NO (neutral)", f"{final_no_count:,} ({final_no_count/max(total,1)*100:.1f}%)")
    _kv("Tier 3 triggered", f"{tier3_triggered_count:,} ({tier3_triggered_count/max(total,1)*100:.1f}%)")
    _kv("  -> Upgraded NO->YES", f"{tier3_upgraded_count:,}")
    _kv("Time elapsed", f"{elapsed:.1f}s ({total/max(elapsed,0.1):.2f} items/s)")

    _header(f"{dataset_name} FINAL RESULTS", "=")
    _kv("Total items", f"{total:,}")
    _kv("SENSITIVE (YES)", f"{final_yes_count:,} ({final_yes_count/max(total,1)*100:.1f}%)")
    _kv("NEUTRAL (NO)", f"{final_no_count:,} ({final_no_count/max(total,1)*100:.1f}%)")
    _kv("Tier 0 (keywords)", f"{tier0_yes_count:,}")
    _kv("Tier 1 final", f"{total - tier0_yes_count - tier3_triggered_count:,}")
    _kv("Tier 3 refined", f"{tier3_triggered_count:,}")
    _kv("Upgraded by Tier 3", f"{tier3_upgraded_count:,}")

    print(f"  Generating final output files from checkpoint...", flush=True)

    all_path = dataset_out_dir / "all_classified.jsonl"
    yes_path = dataset_out_dir / "culturally_sensitive.jsonl"
    no_path = dataset_out_dir / "not_sensitive.jsonl"

    if checkpoint_path.exists():
        import shutil
        shutil.copy2(checkpoint_path, all_path)

    if all_path.exists():
        with open(all_path, "r") as f_in, \
             open(yes_path, "w") as f_yes, \
             open(no_path, "w") as f_no:
            for line in f_in:
                record = json.loads(line)
                if record["culturally_sensitive"] == "YES":
                    f_yes.write(line)
                else:
                    f_no.write(line)

    print(f"  Saved: {all_path}  ({total:,} items)", flush=True)
    print(f"  Saved: {yes_path}  ({final_yes_count:,} items)", flush=True)
    print(f"  Saved: {no_path}  ({final_no_count:,} items)", flush=True)

    summary = {
        "dataset": dataset_name,
        "domain": domain,
        "total_items": total,
        "total_sensitive": final_yes_count,
        "total_neutral": final_no_count,
        "pct_sensitive": round(final_yes_count / max(total, 1) * 100, 2),
        "pipeline_mode": "tier0-keywords_tier1-llm-ner_tier2-gate_tier3-refinement",
        "tier0_yes": tier0_yes_count,
        "tier1_yes": llm_path_yes_count,
        "tier1_no": tier1_no_count,
        "tier3_triggered": tier3_triggered_count,
        "tier3_upgraded": tier3_upgraded_count,
        "confidence_distribution": dict(confidence_dist),
        "model": ACTIVE_MODEL_NAME,
        "quantization": ACTIVE_QUANTIZATION,
        "ner_model": "en_core_web_trf",
        "checkpoint_frequency": checkpoint_freq,
    }
    summary_path = dataset_out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}", flush=True)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Cleaned up checkpoint file", flush=True)

    return []


def main():
    """CLI entry point for running tiered CSI detection."""
    parser = argparse.ArgumentParser(description="Tier 0-3 CSI Detection Pipeline")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "dailydialog", "rocstories", "recipenlg"],
        choices=["gsm8k", "rocstories", "dailydialog", "recipenlg"],
    )
    parser.add_argument("--test_size", type=int, default=0,
                        help="Process only N items per dataset (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ner_gpu", type=int, default=0)
    parser.add_argument("--checkpoint_freq", type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only first 8 items from each dataset")
    args = parser.parse_args()

    ner_gpu_explicit = "--ner_gpu" in sys.argv
    if (not ner_gpu_explicit) and args.device.startswith("cuda:"):
        try:
            args.ner_gpu = int(args.device.split(":", 1)[1])
        except (IndexError, ValueError):
            pass

    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    _header("TIER 0-3 CSI DETECTION PIPELINE", "=")
    detect_gpus()
    print()
    _kv("Pipeline Mode", "Tier 0 (keywords) -> Tier 1 (LLM/NER) -> Tier 2 (gate) -> Tier 3 (refinement)")
    _kv("LLM Model", MODEL_NAME)
    _kv("NER Model", "en_core_web_trf")
    _kv("Batch Size", args.batch_size)
    _kv("Device", args.device)
    _kv("NER GPU", args.ner_gpu)
    _kv("Checkpoint frequency", f"Every {args.checkpoint_freq} items")
    _kv("Smoke test", args.smoke_test)
    _kv("Datasets", str(args.datasets))
    _kv("Test size", str(args.test_size or "FULL"))
    _kv("Output dir", str(PROC_DIR))

    if args.smoke_test:
        args.datasets = ["gsm8k", "dailydialog", "rocstories", "recipenlg"]
        args.test_size = 8
        _section("Smoke Test Overrides")
        _kv("Datasets", str(args.datasets))
        _kv("Items per dataset", 8)
        _kv("Total target items", 32)

    model, tokenizer = load_model(args.device)
    nlp = load_ner(gpu_id=args.ner_gpu)

    grand_summary = {}

    for dataset_name in args.datasets:
        ds_start = time.time()
        _ = process_dataset(
            dataset_name=dataset_name,
            model=model,
            tokenizer=tokenizer,
            nlp=nlp,
            test_size=args.test_size,
            batch_size=args.batch_size,
            checkpoint_freq=args.checkpoint_freq,
            device=args.device,
            smoke_test=args.smoke_test,
        )
        elapsed = round(time.time() - ds_start, 1)

        summary_path = (PROC_DIR / dataset_name / "smoke_test" / "summary.json") if args.smoke_test else (PROC_DIR / dataset_name / "summary.json")
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
                grand_summary[dataset_name] = {
                    "total": summary["total_items"],
                    "sensitive": summary["total_sensitive"],
                    "time_s": elapsed,
                }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _header("GRAND SUMMARY", "=")
    print(f"  {'Dataset':<15s}  {'Total':>9s}  {'Sensitive':>10s}  {'%':>6s}  {'Time':>8s}")
    print(f"  {'-'*15}  {'-'*9}  {'-'*10}  {'-'*6}  {'-'*8}")
    for ds, info in grand_summary.items():
        pct = info["sensitive"] / max(info["total"], 1) * 100
        print(f"  {ds:<15s}  {info['total']:>9,}  {info['sensitive']:>10,}  "
              f"{pct:>5.1f}%  {info['time_s']:>7.1f}s")

    print(f"\n  Pipeline complete. Output: {PROC_DIR}", flush=True)


if __name__ == "__main__":
    main()



