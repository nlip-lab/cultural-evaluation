"""
Phase 2 — Run Hygiene Checks on All Adaptations

Input:
  src/Framework/all_adaptations_indexed.jsonl (from Task 1)

Output:
  03_data/results/hygiene_scores.jsonl
  (one record per adaptation with hygiene results)

Usage:
  python run_hygiene.py [--test] [--sample N]

  --test     Run on first 3 records only (quick validation)
  --sample N Run on random sample of N records
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from hygiene_checker import HygieneChecker

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "03_data" / "results"

INDEX_PATH = EVAL_DIR / "all_adaptations_indexed.jsonl"
OUT_PATH = RESULTS_DIR / "hygiene_scores.jsonl"
STATS_PATH = RESULTS_DIR / "hygiene_stats.json"

def parse_args():
    """Parse CLI arguments for hygiene execution mode."""
    p = argparse.ArgumentParser(description="Phase 2 — Hygiene checker")
    p.add_argument("--test", action="store_true", help="Run on first 3 records only")
    p.add_argument("--sample", type=int, help="Run on random sample of N records")
    return p.parse_args()

def main():
    """Run hygiene checks and write outputs and summary statistics."""
    args = parse_args()

    print("=" * 70)
    print("Phase 2 — Output Hygiene Checks")
    print("=" * 70)

    print(f"\n[0] Loading {INDEX_PATH.name} …")
    records = []
    with open(INDEX_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"    {len(records):,} records loaded")

    if args.test:
        records = records[:3]
        print(f"    TEST MODE: running on {len(records)} records")
    elif args.sample:
        records = random.sample(records, min(args.sample, len(records)))
        print(f"    SAMPLE MODE: running on {len(records):,} records")

    print(f"\n[1] Initializing HygieneChecker …")
    checker = HygieneChecker()
    print(f"    Ready")

    print(f"\n[2] Running hygiene checks ({len(records):,} records) …")

    written = 0
    errors = 0
    t0 = time.time()

    overall_dist = defaultdict(int)
    lang_dist = defaultdict(int)
    degen_dist = defaultdict(int)
    meta_dist = defaultdict(int)
    refusal_dist = defaultdict(int)
    length_dist = defaultdict(int)
    label_dist = defaultdict(int)
    domain_dist = defaultdict(int)

    with open(OUT_PATH, "w") as out_f:
        for i, record in enumerate(records, 1):
            try:
                eval_id = record.get("eval_id", "")
                source_text = record.get("source_text", "")
                adapted_text = record.get("adapted_text", "")
                domain = record.get("domain", "unknown")

                hygiene = checker.check(source_text, adapted_text, domain)

                output = {
                    "eval_id": eval_id,
                    "02_hygiene": hygiene,
                }

                out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
                written += 1

                overall_dist[hygiene["overall"]] += 1
                lang_dist[hygiene["lang_check"]] += 1
                degen_dist[hygiene["degeneration"]] += 1
                meta_dist[hygiene["meta_commentary"]] += 1
                refusal_dist[hygiene["refusal"]] += 1
                length_dist[hygiene["length_check"]] += 1
                domain_dist[domain] += 1
                if hygiene["iaet_label"]:
                    label_dist[hygiene["iaet_label"]] += 1

            except Exception as e:
                errors += 1
                print(f"\n  ERROR at record {i}: {e}")

            if i % 1000 == 0 or i == len(records):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(records) - i) / rate if rate > 0 else 0
                print(f"    {i:>6}/{len(records)} | {rate:.0f} rec/s | ETA {eta/60:.1f}m")

    print(f"\n[3] Computing summary stats …")

    stats = {
        "total_records": len(records),
        "records_written": written,
        "errors": errors,
        "overall_distribution": dict(overall_dist),
        "lang_check_distribution": dict(lang_dist),
        "degeneration_distribution": dict(degen_dist),
        "meta_commentary_distribution": dict(meta_dist),
        "refusal_distribution": dict(refusal_dist),
        "length_check_distribution": dict(length_dist),
        "iaet_label_distribution": dict(label_dist),
        "domain_distribution": dict(domain_dist),
    }

    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Records written       : {written:,}")
    print(f"  Errors                : {errors:,}")

    print(f"\n  Overall classification:")
    for status in ["pass", "fail_soft", "fail_hard"]:
        count = overall_dist.get(status, 0)
        pct = 100 * count / written if written else 0
        print(f"    {status:<20} {count:>6,}  ({pct:>5.1f}%)")

    print(f"\n  Language checks:")
    for status in sorted(lang_dist.keys()):
        count = lang_dist.get(status, 0)
        pct = 100 * count / written if written else 0
        print(f"    {status:<20} {count:>6,}  ({pct:>5.1f}%)")

    print(f"\n  Degeneration types:")
    for status in sorted(degen_dist.keys()):
        count = degen_dist.get(status, 0)
        pct = 100 * count / written if written else 0
        print(f"    {status:<20} {count:>6,}  ({pct:>5.1f}%)")

    print(f"\n  IAET failure labels (hard failures only):")
    for label in sorted(label_dist.keys()):
        count = label_dist[label]
        print(f"    {label:<20} {count:>6,}")

    print(f"\n  Output : {OUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")

if __name__ == "__main__":
    main()
