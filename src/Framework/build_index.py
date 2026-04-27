"""
Build all_adaptations_indexed.jsonl by joining each adaptation back to its
benchmark source record.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROC_DIR / "adapted_data" / "runs"
BENCHMARK = PROC_DIR / "benchmark.jsonl"
OUT_PATH = EVAL_DIR / "all_adaptations_indexed.jsonl"

DOMAIN_SLUG = {
    "dialogue": "dailydialog",
    "educational": "gsm8k",
    "procedural": "recipenlg",
    "stories": "rocstories",
}
STRATEGIES = {"simple", "instructed", "oneshot", "cot"}

print("Loading benchmark ... ", end="", flush=True)
benchmark: dict[str, dict] = {}
with open(BENCHMARK, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            item = json.loads(line)
            benchmark[item["id"]] = item
print(f"{len(benchmark):,} items loaded")

print("\nIndexing adapted files ...")

written = 0
skipped_no_join = 0
skipped_no_text = 0
domain_counts: dict[str, int] = defaultdict(int)
join_failures: list[str] = []

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for strategy_dir in sorted(RUNS_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        if strategy not in STRATEGIES:
            print(f"  SKIP unexpected dir: {strategy_dir.name}")
            continue

        for jsonl_file in sorted(strategy_dir.glob("*.jsonl")):
            stem = jsonl_file.stem
            parts = stem.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in DOMAIN_SLUG:
                print(f"  SKIP unrecognised filename: {jsonl_file.name}")
                continue
            model_name = parts[0]
            domain = DOMAIN_SLUG[parts[1]]

            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    adapted = json.loads(line)
                    if adapted.get("is_config_header"):
                        continue

                    source_id = adapted.get("id")
                    src = benchmark.get(source_id)
                    if src is None:
                        msg = f"No benchmark match: {source_id} ({jsonl_file.name})"
                        join_failures.append(msg)
                        skipped_no_join += 1
                        continue

                    source_text = src.get("original_text", "").strip()
                    if not source_text:
                        msg = f"Empty source_text: {source_id}"
                        join_failures.append(msg)
                        skipped_no_text += 1
                        continue

                    eval_id = f"{model_name}__{strategy}__{domain}__{source_id}"
                    record = {
                        "eval_id": eval_id,
                        "source_id": source_id,
                        "domain": domain,
                        "model": model_name,
                        "strategy": strategy,
                        "source_text": source_text,
                        "adapted_text": adapted.get("adapted_text", ""),
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    domain_counts[domain] += 1

print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")
print(f"  Records written      : {written:,}")
print(f"  Join failures        : {skipped_no_join:,}")
print(f"  Empty source_text    : {skipped_no_text:,}")
total_skipped = skipped_no_join + skipped_no_text
print(f"  Total skipped        : {total_skipped:,}")

print("\n  Records per domain:")
for d in ["gsm8k", "rocstories", "dailydialog", "recipenlg"]:
    print(f"    {d:<20} {domain_counts.get(d, 0):>8,}")

if join_failures:
    print("\n  Join failure details (first 20):")
    for msg in join_failures[:20]:
        print(f"    - {msg}")
    if len(join_failures) > 20:
        print(f"    ... and {len(join_failures) - 20} more")

print("\n  Spot-check - first record:")
with open(OUT_PATH, encoding="utf-8") as f:
    sample = json.loads(f.readline())
for k, v in sample.items():
    val = str(v)[:80] + "..." if v and len(str(v)) > 80 else v
    print(f"    {k:<20} {val}")

print("\n  Verifying source_text populated for all records ... ", end="", flush=True)
empty_src = 0
with open(OUT_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("source_text", "").strip():
            empty_src += 1
if empty_src:
    print(f"FAIL - {empty_src} records have empty source_text!")
    sys.exit(1)
print("OK")

print(f"\n  Output: {OUT_PATH}")
