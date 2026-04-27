"""
Reverse adaptation controller: reads Indian-adapted texts produced by adapt.py
and asks each LLM to reverse-adapt them back to American cultural context
(Indian → American). All four prompt strategies are mirrored; one-shot examples
swap original/adapted roles to show the Indian→American direction.

Input:  03_data/processed/adapted_data/runs/{prompt_level}/{model}_{domain}.jsonl
Output: 03_data/processed/reverse_adapted_data/runs/{prompt_level}/{model}_{domain}.jsonl
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
PROC_DIR     = PROJECT_ROOT / "03_data" / "processed"
ADAPTED_RUNS_DIR = PROC_DIR / "adapted_data" / "runs"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "03_data" / "processed" / "reverse_adapted_data"

PROMPT_VERSION = "CAP-v3.0-REV"

SIMPLE_SYSTEM_TEMPLATE = """\
You are a cultural adaptation specialist.\
"""

SIMPLE_USER_TEMPLATE = """\
{domain_label}: {text}

Adapt this from {source_culture} to {target_culture} context.\
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are a cultural adaptation specialist.

Replace every culture-specific item (CSI) in the text with a natural {target_culture} equivalent. The source text is grounded in {source_culture} cultural context. CSIs include: ecology, material culture (food, clothing, objects), social practices, institutions, currency, measurements, and names.

Rules:
1. Preserve all numeric values, logical dependencies, and causal structure exactly.
2. Output must be entirely in the same language as the input.
3. Do not solve, answer, or explain. Output the adapted text only.\
"""

COT_USER_TEMPLATE = """\
{domain_label}: {text}

Adapt this text step by step.

STEP 1 — IDENTIFY:
List each culture-specific item (CSI) and its type (ecology / material culture / social practice / institution / currency / measurement / name / other).

STEP 2 — MAP:
For each CSI, state the {target_culture} equivalent replacing the {source_culture} original.

STEP 3 — VERIFY:
Confirm all numeric values and logical dependencies remain valid after substitution.

Finally, write ONLY the final adapted text after the marker below. No explanations, no notes.

ADAPTED TEXT:\
"""

DIRECT_USER_TEMPLATE = """\
{domain_label}: {text}

Write the culturally adapted version. Output the adapted text only.\
"""

ONESHOT_USER_TEMPLATE = """\
Here is an example of adapting from {source_culture} to {target_culture} context:

{domain_label}: {example_original}

Adapted {domain_label}: {example_adapted}

Now adapt the following text in the same way. Output the adapted text only with no explanations or notes.

{domain_label}: {text}\
"""


ONESHOT_EXAMPLES = {
    "educational": {
        "id": "gsm8k_train_6713",
        "original": (
            "Ananya is making a pie. She can't decide if she will make "
            "guava or mango. So she decides to make whichever pie is "
            "cheapest. Both pies will need ₹160 for flour, ₹80 for sugar and "
            "₹120 for eggs and butter to make the crust. The guava pie "
            "needs 1.5 kg of guavas. The guavas come in 250 g packs for "
            "₹180. For the mango pie, she needs 2 kg and they come in a "
            "two-kilogram box for ₹1,120. What is the total price to make "
            "the cheapest pie?"
        ),
        "adapted": (
            "Charlotte is making a pie. She can't decide if she will make "
            "cherry or blueberry. So she decides to make whichever pie is "
            "cheapest. Both pies will need $2 for flour, $1 for sugar and "
            "$1.5 for eggs and butter to make the crust. The blueberry pie "
            "needs 3 pounds of blueberries. The blueberries come in 8 oz "
            "containers for $2.25. For the cherry pie, she needs 4 pounds "
            "and they come in a four-pound bag for $14. What is the total "
            "price to make the cheapest pie?"
        ),
    },
    "stories": {
        "id": "rocstories_train_46523",
        "original": (
            "Jaipur is Priya's favorite place to visit. She was raised on "
            "Rajasthani folk music and loved it still. When there she "
            "always tried to go to Hawa Mahal and Johari Bazaar. One time "
            "she actually got up and sing at one of the chai stalls! She "
            "eventually decided to relocate to Jaipur."
        ),
        "adapted": (
            "Nashville is Robyn's favorite place to visit. She was raised "
            "on country music and loved it still. When there she always "
            "tried to go to the Ryman and 2nd Ave. One time she actually "
            "got up and sing at one of the watering holes! She eventually "
            "decided to relocate to Nashville."
        ),
    },
    "dialogue": {
        "id": "dailydialog_train_8500",
        "original": (
            "Speaker A: We need to figure out how much money we've spent "
            "and what we've spent it on .\n"
            "Speaker B: Why ?\n"
            "Speaker A: Well , I think we should save more . If we save "
            "more money we can retire earlier and enjoy our life better .\n"
            "Speaker B: Really , well , OK . Get the receipts out .\n"
            "Speaker A: Let's see . We spent ₹25,000 for our home loan "
            "EMI , ₹12,000 on groceries , ₹3,000 on utilities , ₹8,000 on "
            "petrol , ₹4,000 on books , and ₹15,000 on entertainment last "
            "month .\n"
            "Speaker B: That's ₹67,000 .\n"
            "Speaker A: We put ₹20,000 into the savings account , and "
            "₹30,000 into our mutual fund SIP .\n"
            "Speaker B: ₹1,17,000 . Our salaries are ₹1,25,000 , combined . "
            "That means we don't have any receipts for ₹8,000 .\n"
            "Speaker A: Right . Stop thinking about it . Where do you "
            "think we can save more money ?\n"
            "Speaker B: Well , we could eat in more often . That ₹15,000 "
            "for entertainment is mostly going to restaurants .\n"
            "Speaker A: Yeah , that's a good idea , and you like to cook .\n"
            "Speaker B: All right ."
        ),
        "adapted": (
            "Speaker A: We need to figure out how much money we've spent "
            "and what we've spent it on .\n"
            "Speaker B: Why ?\n"
            "Speaker A: Well , I think we should save more . If we save "
            "more money we can retire earlier and enjoy our life better .\n"
            "Speaker B: Really , well , OK . Get the receipts out .\n"
            "Speaker A: Let's see . We spent $ 700 for our home loan "
            "payment , $ 300 0n groceries , $ 75 0n utilities , $ 250 n "
            "gasoline , $ 100 on books , and $ 400 0n entertainment last "
            "month .\n"
            "Speaker B: That's $ 1 825 .\n"
            "Speaker A: We put $ 500 into the savings account , and $ 750 "
            "into our stock account .\n"
            "Speaker B: $ 3 075 . Our paychecks is $ 3 300 , combined . "
            "That means we don't have any receipts for $ 225 .\n"
            "Speaker A: Right . Stop thinking about it . Where do you "
            "think we can save more money ?\n"
            "Speaker B: Well , we could eat in more often . That $ 400 "
            "for entertainment is mostly going to restaurants .\n"
            "Speaker A: Yeah , that's a good idea , and you like to cook .\n"
            "Speaker B: All right ."
        ),
    },
    "procedural": {
        "id": "recipenlg_train_2012030",
        "original": (
            "Title: Festive Cheese Tree\n"
            "Ingredients: 2 pkg. (225 g each) Amul Cream Cheese, softened, "
            "1 pkg. (225 g) Amul Shredded Cheddar Cheese, 1 Tbsp. chopped "
            "red bell pepper D-Mart 1 pkg For ₹199 thru 02/06, 1 Tbsp. "
            "finely chopped onion Reliance Fresh 500 g For ₹49 thru 02/09, "
            "2 tsp. Weikfield Worcestershire Sauce, 1 tsp. lemon juice "
            "DMart 4 ct For ₹299 thru 02/09, Dash ground red pepper, "
            "Toasted bread or mathri\n"
            "Directions: 1. Mix cream cheese and cheddar cheese with "
            "electric mixer on medium speed until well blended. 2. Blend "
            "in red bell pepper, onion, Worcestershire sauce, lemon juice "
            "and ground red pepper. 3. Refrigerate several hours or "
            "overnight. 4. Drop 6 (80 ml) measures of mixture into "
            "triangle shape on serving platter. 5. Drop remaining mixture "
            "at base of triangle; smooth to form tree. 6. Garnish as "
            "desired. 7. Serve with toasted breads or mathri."
        ),
        "adapted": (
            "Title: Holiday Cheese Tree\n"
            "Ingredients: 2 pkg. (8 oz. each) PHILADELPHIA Cream Cheese, "
            "softened, 1 pkg. (8 oz.) KRAFT Shredded Sharp Cheddar Cheese, "
            "1 Tbsp. chopped red bell pepper Target 1 pkg For $2.99 thru "
            "02/06, 1 Tbsp. finely chopped onion King Sooper's 1 lb For "
            "$0.99 thru 02/09, 2 tsp. LEA & PERRINS Worcestershire Sauce, "
            "1 tsp. lemon juice Safeway 4 ct For $5.00 thru 02/09, Dash "
            "ground red pepper, Toasted bread or crackers\n"
            "Directions: 1. Mix cream cheese and cheddar cheese with "
            "electric mixer on medium speed until well blended. 2. Blend "
            "in red bell pepper, onion, Worcestershire sauce, lemon juice "
            "and ground red pepper. 3. Refrigerate several hours or "
            "overnight. 4. Drop 6 (1/3-cup) measures of mixture into "
            "triangle shape on serving platter. 5. Drop remaining mixture "
            "at base of triangle; smooth to form tree. 6. Garnish as "
            "desired. 7. Serve with toasted breads or crackers."
        ),
    },
}

ONESHOT_EXAMPLE_IDS = {ex["id"] for ex in ONESHOT_EXAMPLES.values()}

DOMAIN_LABELS = {
    "educational": "Question",
    "stories":     "Story",
    "dialogue":    "Dialogue",
    "procedural":  "Recipe",
}

DOMAIN_MAX_TOKENS = {
    "educational": 256,
    "stories":     256,
    "dialogue":    512,
    "procedural":  768,
}

DATASET_TO_DOMAIN = {
    "gsm8k":       "educational",
    "rocstories":  "stories",
    "dailydialog": "dialogue",
    "recipenlg":   "procedural",
}
DOMAIN_TO_DATASET = {v: k for k, v in DATASET_TO_DOMAIN.items()}

COT_EXTRA_TOKENS = 400

TP2_MODELS = {"llama", "qwen", "aya", "sarvam", "gemma"}
TP1_MODELS = {"llama8b", "gemma2", "mistral"}
ALL_MODELS = ["llama", "qwen", "aya", "sarvam", "gemma", "llama8b", "gemma2", "mistral"]

REVERSE_BATCH_SIZES: dict[str, int] = {
    "llama":   8,
    "qwen":    8,
    "aya":     16,
    "sarvam":  16,
    "gemma":   16,
    "llama8b": 32,
    "gemma2":  32,
    "mistral": 32,
}

def load_env_file(path: Path) -> None:
    """Load KEY=VALUE entries from a .env file into process environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)

def _nvidia_smi_free_mib(gpu_indices: list[int]) -> dict[int, float]:
    """Query free VRAM via nvidia-smi subprocess — avoids initializing CUDA in parent."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        result = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                idx, free_mib, total_mib = int(parts[0]), float(parts[1]), float(parts[2])
                if idx in gpu_indices:
                    result[idx] = (free_mib / 1024, total_mib / 1024)
        return result
    except Exception as e:
        print(f"  ⚠ nvidia-smi query failed: {e}", flush=True)
        return {}

def check_gpu_memory_or_abort(min_free_gb: float) -> None:
    """Abort when visible GPUs do not satisfy minimum free VRAM."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpu_indices = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    else:
        import subprocess
        try:
            n_str = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True
            )
            gpu_indices = [int(l.strip()) for l in n_str.strip().splitlines()]
        except Exception:
            gpu_indices = []
    if not gpu_indices:
        print("  ⚠ No visible GPUs to check.", flush=True)
        return
    print(f"  GPU free-memory check (threshold: {min_free_gb:.1f} GB):", flush=True)
    info = _nvidia_smi_free_mib(gpu_indices)
    low = []
    for idx in gpu_indices:
        if idx in info:
            free_gb, total_gb = info[idx]
            print(f"    GPU {idx}  free={free_gb:.1f} GB / total={total_gb:.1f} GB", flush=True)
            if free_gb < min_free_gb:
                low.append((idx, free_gb))
        else:
            print(f"    GPU {idx}  (no data from nvidia-smi)", flush=True)
    if low:
        details = ", ".join([f"GPU {i}={gb:.1f}GB" for i, gb in low])
        print(
            f"\n  ERROR: Insufficient free VRAM ({details}). "
            f"Need at least {min_free_gb:.1f} GB free per visible GPU. Aborting.",
            flush=True,
        )
        sys.exit(1)

def log_gpu_free_memory(prefix: str) -> None:
    """Print free VRAM for visible GPUs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        gpu_indices = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    else:
        gpu_indices = list(range(3))
    info = _nvidia_smi_free_mib(gpu_indices)
    for idx, (free_gb, _) in sorted(info.items()):
        print(f"  [{prefix}] GPU {idx} free: {free_gb:.1f} GB", flush=True)

def build_messages(
    text: str,
    domain: str,
    source_culture: str,
    target_culture: str,
    prompt_level: str,
) -> list[dict]:
    """Build chat messages for reverse adaptation."""
    label = DOMAIN_LABELS.get(domain, "Text")

    if prompt_level == "simple":
        system = SIMPLE_SYSTEM_TEMPLATE
        user   = SIMPLE_USER_TEMPLATE.format(
            domain_label=label,
            text=text,
            source_culture=source_culture,
            target_culture=target_culture,
        )
    elif prompt_level == "oneshot":
        system  = SYSTEM_PROMPT_TEMPLATE.format(
            source_culture=source_culture,
            target_culture=target_culture,
        )
        example = ONESHOT_EXAMPLES[domain]
        user    = ONESHOT_USER_TEMPLATE.format(
            domain_label=label,
            text=text,
            source_culture=source_culture,
            target_culture=target_culture,
            example_original=example["original"],
            example_adapted=example["adapted"],
        )
    else:
        system   = SYSTEM_PROMPT_TEMPLATE.format(
            source_culture=source_culture,
            target_culture=target_culture,
        )
        user_tmpl = COT_USER_TEMPLATE if prompt_level == "cot" else DIRECT_USER_TEMPLATE
        user = user_tmpl.format(
            domain_label=label,
            text=text,
            source_culture=source_culture,
            target_culture=target_culture,
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def strip_adapted_prefix(text: str, domain_label: str) -> str:
    """Remove copied one-shot prefix from generated text."""
    pattern = rf"^Adapted\s+{re.escape(domain_label)}\s*:\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()

def parse_cot_output(raw: str) -> tuple[str, str]:
    """Extract CoT reasoning and final adapted text from model output."""
    matches = list(re.finditer(
        r"(?:\*{1,3}|#{1,4}\s*)?ADAPTED\s+TEXT(?:\*{1,3})?\s*:?\s*\n?",
        raw,
        re.IGNORECASE,
    ))
    m = matches[-1] if matches else None
    if m:
        reasoning = raw[: m.start()].strip()
        adapted   = raw[m.end() :].strip()
        adapted   = re.sub(r"^\s*(?:\*{1,3}|#{1,4})\s*", "", adapted)
        adapted   = re.sub(r"\s*\*{1,3}\s*$", "", adapted)
        adapted   = adapted.strip()
        if adapted:
            return reasoning, adapted
    paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 20]
    if paragraphs:
        return raw.strip(), paragraphs[-1]
    return raw.strip(), raw.strip()

def get_max_tokens(domain: str, prompt_level: str) -> int:
    """Return max generation tokens for a domain and prompt level."""
    base = DOMAIN_MAX_TOKENS.get(domain, 512)
    return base + COT_EXTRA_TOKENS if prompt_level == "cot" else base

def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSONL rows from path."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_checkpoint_ids(path: Path) -> set:
    """Load processed reverse IDs from checkpoint/final output."""
    if not path.exists():
        return set()
    ids = set()
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("is_config_header"):
                    continue
                if "rev_id" in data:
                    ids.add(data["rev_id"])
                elif "id" in data:
                    ids.add(data["id"])
            except Exception:
                skipped += 1
    if skipped:
        print(f"  ⚠ {path.name}: skipped {skipped} malformed lines", flush=True)
    return ids

def _make_config_header(model_name: str, model_id: str, args, prompt_level: str) -> dict:
    """Build configuration header record written to output JSONL."""
    return {
        "is_config_header": True,
        "model": model_name,
        "model_id": model_id,
        "prompt_level": prompt_level,
        "prompt_version": PROMPT_VERSION,
        "temperature": args.temperature,
        "source_culture": "Indian",
        "target_culture": "American",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def _write_records(
    path: Path,
    records: list[dict],
    config_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Append or overwrite records in JSONL output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    write_header = config_header is not None and (overwrite or not path.exists())
    with path.open(mode, encoding="utf-8") as f:
        if write_header:
            f.write(json.dumps(config_header, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_adapter(model_name: str):
    """Instantiate the adapter class for a model name."""
    if model_name == "llama":
        from model_llama import LlamaAdapter
        return LlamaAdapter()
    elif model_name == "llama8b":
        from model_llama8b import Llama8BAdapter
        return Llama8BAdapter()
    elif model_name == "qwen":
        from model_qwen import QwenAdapter
        return QwenAdapter()
    elif model_name == "aya":
        from model_aya import AyaAdapter
        return AyaAdapter()
    elif model_name == "gemma":
        from model_gemma import GemmaAdapter
        return GemmaAdapter()
    elif model_name == "sarvam":
        from model_sarvam import SarvamAdapter
        return SarvamAdapter()
    elif model_name == "gemma2":
        from model_gemma2 import Gemma2Adapter
        return Gemma2Adapter()
    elif model_name == "mistral":
        from model_mistral import MistralAdapter
        return MistralAdapter()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_input_rows_for_model(
    model_name: str,
    prompt_level: str,
    domain_filter: str,
    adapted_runs_dir: Path,
) -> list[dict]:
    """Load adapted texts for one model across all domains and a given prompt level.

    Each record gets a synthetic rev_id = "{id}__rev__{model}__{prompt_level}__to_american"
    so checkpoints are unique even across re-runs.
    The `adapted_text` field becomes the source text for the reverse adaptation.
    """
    rows = []
    domains = (
        [domain_filter]
        if domain_filter != "all"
        else list(DATASET_TO_DOMAIN.values())
    )

    for domain in domains:
        fpath = adapted_runs_dir / prompt_level / f"{model_name}_{domain}.jsonl"
        if not fpath.exists():
            print(f"  ⚠ Input file not found (skipping): {fpath}", flush=True)
            continue

        records = load_jsonl(fpath)
        loaded = 0
        for r in records:
            if r.get("is_config_header"):
                continue
            adapted_text = r.get("adapted_text", "").strip()
            if not adapted_text:
                continue
            rows.append({
                "id":           r["id"],
                "rev_id":       f"{r['id']}__rev__{model_name}__{prompt_level}__to_american",
                "domain":       domain,
                "original_adapted_text": adapted_text,
                "original_text": r.get("original_text", ""),
                "model":        model_name,
                "fwd_prompt_level": prompt_level,
            })
            loaded += 1
        print(f"  [{model_name}][{domain}][{prompt_level}] loaded {loaded:,} rows", flush=True)

    return rows

def _get_domain(row: dict) -> str:
    """Return normalized domain label from a row."""
    return row.get("domain", "educational")

def process_model(
    model_name: str,
    all_prompt_rows: dict[str, list[dict]],
    args,
    output_dir: Path,
    checkpoint_freq: int = 50,
) -> None:
    """Process all prompt levels for one model sequentially (model loaded once)."""

    print(f"\n{'═' * 70}", flush=True)
    print(f"  MODEL: {model_name.upper()}", flush=True)
    print(f"{'═' * 70}", flush=True)

    adapter = load_adapter(model_name)
    adapter.load()
    model_id = adapter.model_id
    batch_size = REVERSE_BATCH_SIZES.get(model_name, getattr(adapter, "BATCH_SIZE", 1))
    has_batch  = hasattr(adapter, "generate_batch") and batch_size > 1
    print(
        f"  Adapter loaded: {model_id}  "
        f"batch={'enabled (' + str(batch_size) + ')' if has_batch else 'disabled'}",
        flush=True,
    )

    try:
        for prompt_level, rows in all_prompt_rows.items():
            if not rows:
                print(f"  [{model_name}][{prompt_level}] No rows — skipping.", flush=True)
                continue

            print(f"\n  ── Prompt level: {prompt_level} ({len(rows):,} items) ──", flush=True)

            active_domains = sorted(set(_get_domain(r) for r in rows))

            base_dir = output_dir / "runs" / prompt_level
            final_paths = {
                d: base_dir / f"{model_name}_{d}.jsonl" for d in active_domains
            }
            checkpoint_paths = {
                d: base_dir / f"checkpoint_{model_name}_{d}.jsonl" for d in active_domains
            }
            base_dir.mkdir(parents=True, exist_ok=True)

            done_ids: set = set()
            skip_domains: set = set()
            for d in active_domains:
                if final_paths[d].exists():
                    skip_domains.add(d)
                    print(f"  [{model_name}][{prompt_level}] {d}: final file exists — skipping.", flush=True)
                elif checkpoint_paths[d].exists():
                    ids = load_checkpoint_ids(checkpoint_paths[d])
                    done_ids.update(ids)
                    print(f"  [{model_name}][{prompt_level}] {d}: resuming ({len(ids):,} done).", flush=True)

            remaining = [
                r for r in rows
                if r["rev_id"] not in done_ids and _get_domain(r) not in skip_domains
            ]
            print(
                f"  Done: {len(done_ids):,}  |  Skip domains: {skip_domains or 'none'}  |  "
                f"Remaining: {len(remaining):,}",
                flush=True,
            )
            if not remaining:
                print("  Nothing to do.", flush=True)
                continue

            domain_buffers: dict[str, list[dict]] = {
                d: [] for d in active_domains if d not in skip_domains
            }
            domain_groups: dict[str, list[dict]] = {}
            for r in remaining:
                d = _get_domain(r)
                domain_groups.setdefault(d, []).append(r)

            effective_batch = batch_size if has_batch else 1
            t_start = time.time()
            processed_count = 0
            total_remaining = len(remaining)

            for domain in sorted(domain_groups.keys()):
                domain_rows = domain_groups[domain]
                max_tokens  = get_max_tokens(domain, prompt_level)
                domain_label = DOMAIN_LABELS.get(domain, "Text")

                print(
                    f"\n  ── Domain: {domain} ({len(domain_rows)} items, "
                    f"max_tokens={max_tokens}) ──",
                    flush=True,
                )

                for batch_start in range(0, len(domain_rows), effective_batch):
                    batch_rows = domain_rows[batch_start : batch_start + effective_batch]
                    batch_start_time = time.time()

                    batch_messages = []
                    batch_texts    = []
                    for row in batch_rows:
                        text = row["original_adapted_text"]
                        batch_texts.append(text)
                        batch_messages.append(
                            build_messages(
                                text=text,
                                domain=domain,
                                source_culture="Indian",
                                target_culture="American",
                                prompt_level=prompt_level,
                            )
                        )

                    if has_batch and len(batch_messages) > 1:
                        try:
                            raw_outputs = adapter.generate_batch(
                                batch_messages=batch_messages,
                                temperature=args.temperature,
                                max_new_tokens=max_tokens,
                            )
                            if len(raw_outputs) != len(batch_messages):
                                raise RuntimeError(
                                    f"Batch output count mismatch: {len(raw_outputs)} vs {len(batch_messages)}"
                                )
                        except Exception as e:
                            print(f"  ⚠ Batch failed ({e}), falling back to sequential...", flush=True)
                            raw_outputs = []
                            for msgs in batch_messages:
                                try:
                                    raw_outputs.append(
                                        adapter.generate(
                                            messages=msgs,
                                            temperature=args.temperature,
                                            max_new_tokens=max_tokens,
                                        )
                                    )
                                except Exception as e2:
                                    print(f"  ⚠ Sequential fallback error: {e2}", flush=True)
                                    raw_outputs.append("")
                    else:
                        raw_outputs = []
                        for msgs in batch_messages:
                            try:
                                raw_outputs.append(
                                    adapter.generate(
                                        messages=msgs,
                                        temperature=args.temperature,
                                        max_new_tokens=max_tokens,
                                    )
                                )
                            except Exception as e:
                                print(f"  ⚠ generation error: {e}", flush=True)
                                raw_outputs.append("")

                    batch_gen_time = round(time.time() - batch_start_time, 3)
                    per_item_time  = round(batch_gen_time / max(len(batch_rows), 1), 3)

                    for i, (row, raw) in enumerate(zip(batch_rows, raw_outputs)):
                        processed_count += 1

                        if prompt_level == "cot":
                            cot_reasoning, adapted_text = parse_cot_output(raw)
                        else:
                            cot_reasoning = None
                            adapted_text  = raw.strip()
                            if prompt_level == "oneshot":
                                adapted_text = strip_adapted_prefix(adapted_text, domain_label)

                        out_record = {
                            "rev_id":           row["rev_id"],
                            "id":               row["id"],
                            "domain":           domain,
                            "source_culture":   "Indian",
                            "target_culture":   "American",
                            "model":            model_name,
                            "model_id":         model_id,
                            "prompt_version":   PROMPT_VERSION,
                            "prompt_level":     prompt_level,
                            "fwd_prompt_level": row["fwd_prompt_level"],
                            "temperature":      args.temperature,
                            "max_new_tokens":   max_tokens,
                            "timestamp":        datetime.now(timezone.utc).isoformat(),
                            "generation_time_s": per_item_time,
                            "indian_text":      batch_texts[i],
                            "original_american_text": row.get("original_text", ""),
                            "adapted_text":     adapted_text,
                            "cot_reasoning":    cot_reasoning,
                            "output_words":     len(raw.split()),
                            "raw_output":       raw,
                        }
                        domain_buffers[domain].append(out_record)

                    elapsed = time.time() - t_start
                    rate    = processed_count / max(elapsed, 0.1)
                    eta     = (total_remaining - processed_count) / max(rate, 0.01)
                    print(
                        f"  [{processed_count:>5}/{total_remaining}]  "
                        f"{rate:.2f} items/s  ETA {eta / 60:.1f}min  "
                        f"batch={len(batch_rows)} in {batch_gen_time:.1f}s "
                        f"({per_item_time:.2f}s/item)",
                        flush=True,
                    )

                    header = _make_config_header(model_name, model_id, args, prompt_level)
                    for d, buf in domain_buffers.items():
                        if buf:
                            _write_records(checkpoint_paths[d], buf, config_header=header)
                            buf.clear()

            for d in active_domains:
                if d in skip_domains:
                    continue
                ckpt  = checkpoint_paths[d]
                final = final_paths[d]
                if ckpt.exists():
                    ckpt.rename(final)
                    print(f"  [{model_name}][{prompt_level}] {d}: checkpoint → {final.name}", flush=True)

    finally:
        try:
            adapter.unload()
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    print(f"\n  [{model_name}] Done.", flush=True)

def parse_temperature(val: str) -> float:
    """Parse and validate temperature in [0.0, 1.0]."""
    t = float(val)
    if not (0.0 <= t <= 1.0):
        raise argparse.ArgumentTypeError(f"Temperature must be in [0, 1], got {t}")
    return t

def parse_args():
    """Parse CLI arguments for reverse adaptation runs."""
    p = argparse.ArgumentParser(
        description="Reverse Cultural Adaptation Controller (Indian → American)"
    )
    p.add_argument(
        "--adapted_runs_dir",
        type=Path,
        default=ADAPTED_RUNS_DIR,
        help="Root of forward-adapted runs: runs/{prompt_level}/{model}_{domain}.jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for reverse adaptations",
    )
    p.add_argument(
        "--model",
        type=str,
        default="all",
        choices=ALL_MODELS + ["all"],
        help="Model to run (or 'all' for all 8 models)",
    )
    p.add_argument(
        "--prompt_level",
        type=str,
        default="all",
        choices=["all", "simple", "instructed", "oneshot", "cot"],
        help="Prompt strategy (or 'all' for all 4 strategies)",
    )
    p.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "educational", "stories", "dialogue", "procedural"],
        help="Filter to a single domain",
    )
    p.add_argument(
        "--temperature",
        type=parse_temperature,
        default=0.0,
        help="Generation temperature (default: 0 = deterministic)",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="2",
        help="Comma-separated visible GPU indices (e.g. '2' or '0,2')",
    )
    p.add_argument(
        "--max_tp",
        type=int,
        default=None,
        help=(
            "Override tensor_parallel_size for all models. "
            "Defaults to min(num_visible_gpus, model_native_tp). "
            "Use --max_tp 1 to force single-GPU for all models."
        ),
    )
    p.add_argument(
        "--skip_tp2",
        action="store_true",
        help=(
            "Skip models that require TP=2 (aya, sarvam, gemma, llama, qwen) "
            "when only 1 GPU is available."
        ),
    )
    return p.parse_args()

def main():
    """Entry point for reverse adaptation execution."""
    load_env_file(SCRIPT_DIR / ".env")
    load_env_file(SCRIPT_DIR.parent / ".env")

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

    if args.max_tp is not None:
        os.environ["REVERSE_MAX_TP"] = str(args.max_tp)

    num_visible_gpus = len([g.strip() for g in args.gpus.split(",") if g.strip()])
    effective_max_tp = args.max_tp if args.max_tp is not None else num_visible_gpus

    check_gpu_memory_or_abort(min_free_gb=20.0)

    print(f"\n{'#' * 70}", flush=True)
    print("  Reverse Cultural Adaptation (Indian → American)", flush=True)
    print(f"{'#' * 70}", flush=True)
    print(f"  Adapted runs dir:  {args.adapted_runs_dir}", flush=True)
    print(f"  Output dir:        {args.output_dir}", flush=True)
    print(f"  GPUs visible:      {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}", flush=True)
    print(f"  Effective max TP:  {effective_max_tp}", flush=True)
    print(f"  Model(s):          {args.model}", flush=True)
    print(f"  Prompt level(s):   {args.prompt_level}", flush=True)
    print(f"  Domain filter:     {args.domain}", flush=True)
    print(f"  Temperature:       {args.temperature}", flush=True)
    print(f"  Skip TP2 models:   {args.skip_tp2}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_levels = (
        ["simple", "instructed", "oneshot", "cot"]
        if args.prompt_level == "all"
        else [args.prompt_level]
    )
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    if effective_max_tp < 2 and args.skip_tp2:
        skipped = [m for m in models_to_run if m in TP2_MODELS]
        models_to_run = [m for m in models_to_run if m not in TP2_MODELS]
        if skipped:
            print(f"\n  ⚠ Skipping TP=2 models (--skip_tp2): {skipped}", flush=True)

    failed_models = []

    for model_name in models_to_run:
        all_prompt_rows: dict[str, list[dict]] = {}
        for pl in prompt_levels:
            rows = load_input_rows_for_model(
                model_name=model_name,
                prompt_level=pl,
                domain_filter=args.domain,
                adapted_runs_dir=args.adapted_runs_dir,
            )
            all_prompt_rows[pl] = rows

        total_rows = sum(len(v) for v in all_prompt_rows.values())
        if total_rows == 0:
            print(f"\n  [{model_name}] No input rows found — skipping.", flush=True)
            continue

        print(f"\n  [{model_name}] Total rows across all prompt levels: {total_rows:,}", flush=True)

        try:
            process_model(
                model_name=model_name,
                all_prompt_rows=all_prompt_rows,
                args=args,
                output_dir=args.output_dir,
            )
        except (Exception, SystemExit) as e:
            failed_models.append((model_name, str(e)))
            print(f"\n  ⚠ Model {model_name} failed: {e}", flush=True)
            print("  Continuing with remaining models...", flush=True)
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log_gpu_free_memory("cleanup")

    print(f"\n{'#' * 70}", flush=True)
    print("  Reverse adaptation complete.", flush=True)
    if failed_models:
        print("  Failed models:", flush=True)
        for mn, err in failed_models:
            print(f"    {mn}: {err}", flush=True)

    run_dir = args.output_dir / "runs"
    if run_dir.exists():
        for pl in prompt_levels:
            for mn in models_to_run:
                for domain in DATASET_TO_DOMAIN.values():
                    fpath = run_dir / pl / f"{mn}_{domain}.jsonl"
                    if fpath.exists():
                        try:
                            n = sum(
                                1 for line in fpath.open()
                                if line.strip() and not json.loads(line).get("is_config_header")
                            )
                            print(f"  {pl}/{mn}/{domain}: {n:,} items", flush=True)
                        except Exception:
                            pass

if __name__ == "__main__":
    main()
