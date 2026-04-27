"""
Phase 2 runner — apply HygieneChecker to every adaptation.

Input  : src/Framework/all_adaptations_indexed.jsonl
Output : src/Framework/hygiene/adaptations_with_hygiene.jsonl
         src/Framework/hygiene/hygiene_stats.json

Usage:
  python run_phase2.py [--resume] [--test]
  --resume   skip already-written eval_ids (uses a .checkpoint set)
  --test     run on first 200 records and print sample output
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR    = EVAL_DIR / "02_Hygiene"

INDEX_PATH   = EVAL_DIR / "all_adaptations_indexed.jsonl"
OUT_PATH     = PHASE_DIR / "adaptations_with_hygiene.jsonl"
STATS_PATH   = PHASE_DIR / "hygiene_stats.json"
CKPT_PATH    = PHASE_DIR / ".hygiene_checkpoint.json"

def parse_args():
    """Parse CLI arguments for full or resumed hygiene evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="Continue from checkpoint (skip already-written records)")
    p.add_argument("--test", action="store_true",
                   help="Run on first 200 records only")
    return p.parse_args()

def load_checkpoint() -> set[str]:
    """Load the checkpoint file as a set of completed eval IDs."""
    if CKPT_PATH.exists():
        data = json.loads(CKPT_PATH.read_text())
        return set(data.get("done_ids", []))
    return set()

def save_checkpoint(done_ids: set[str]):
    """Persist completed eval IDs to checkpoint file."""
    CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))

class StatsAccumulator:
    """Thread-safe (single-threaded) stats collector."""

    def __init__(self):
        """Initialize counters for overall and stratified hygiene statistics."""
        self.total          = 0
        self.by_overall     = defaultdict(int)
        self.by_model       = defaultdict(lambda: defaultdict(int))
        self.by_strategy    = defaultdict(lambda: defaultdict(int))
        self.by_domain      = defaultdict(lambda: defaultdict(int))
        self.by_model_x_strategy: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.fail_type      = defaultdict(int)

    def add(self, record: dict, h: dict):
        """Accumulate one evaluated record into all reporting buckets."""
        model    = record.get("model", "unknown")
        strategy = record.get("strategy", "unknown")
        domain   = record.get("domain", "unknown")
        overall  = h["overall"]

        self.total += 1
        self.by_overall[overall]                         += 1
        self.by_model[model][overall]                    += 1
        self.by_strategy[strategy][overall]              += 1
        self.by_domain[domain][overall]                  += 1
        self.by_model_x_strategy[model][strategy][overall] += 1

        if overall == "fail_hard":
            if h["lang_check"] == "fail_non_english":
                self.fail_type["C5.1_non_english"] += 1
            elif h["degeneration"] in ("empty", "truncated", "repetitive"):
                self.fail_type["C5.1_degenerate"] += 1
            elif h["refusal"] == "flagged":
                self.fail_type["C5.1_refusal"] += 1
            elif h.get("iaet_label") == "C1.4":
                self.fail_type["C1.4_meta_only"] += 1
            else:
                self.fail_type["C5.1_other"] += 1
        elif overall == "fail_soft":
            if h["meta_commentary"] == "flagged":
                self.fail_type["soft_meta"] += 1
            elif h["length_check"] != "pass":
                self.fail_type["soft_length"] += 1
            elif h["lang_check"] == "fail_mixed":
                self.fail_type["soft_lang_mixed"] += 1
            else:
                self.fail_type["soft_other"] += 1

    def to_dict(self) -> dict:
        """Convert nested counters to JSON-serializable dictionaries."""
        mx = {
            m: {s: dict(ov) for s, ov in sv.items()}
            for m, sv in self.by_model_x_strategy.items()
        }
        return {
            "total":      self.total,
            "pass":       self.by_overall.get("pass", 0),
            "fail_hard":  self.by_overall.get("fail_hard", 0),
            "fail_soft":  self.by_overall.get("fail_soft", 0),
            "by_model": {
                m: dict(v) for m, v in sorted(self.by_model.items())
            },
            "by_strategy": {
                s: dict(v) for s, v in sorted(self.by_strategy.items())
            },
            "by_domain": {
                d: dict(v) for d, v in sorted(self.by_domain.items())
            },
            "by_model_x_strategy": mx,
            "by_failure_type": dict(self.fail_type),
        }

def print_summary(stats: dict):
    """Print a compact terminal report for hygiene outcomes."""
    total = stats["total"]
    if total == 0:
        print("  No records processed.")
        return

    pct = lambda n: f"{100 * n / total:5.1f}%"

    print(f"\n{'='*78}")
    print("HYGIENE RESULTS — PHASE 2")
    print(f"{'='*78}")
    print(f"  Total records : {total:>10,}")
    print(f"  pass          : {stats['pass']:>10,}  ({pct(stats['pass'])})")
    print(f"  fail_soft     : {stats['fail_soft']:>10,}  ({pct(stats['fail_soft'])})")
    print(f"  fail_hard     : {stats['fail_hard']:>10,}  ({pct(stats['fail_hard'])})")

    print(f"\n  Failure type breakdown:")
    for ft, cnt in sorted(stats["by_failure_type"].items(), key=lambda x: -x[1]):
        print(f"    {ft:<30} {cnt:>7,}  ({pct(cnt)})")

    models    = sorted(stats["by_model"].keys())
    strategies= sorted(stats["by_strategy"].keys())

    print(f"\n  Fail rate by model (% fail_hard | % fail_soft):")
    print(f"  {'Model':<14}", end="")
    for s in strategies:
        print(f"  {s:>18}", end="")
    print()
    print(f"  {'-'*14}", end="")
    for _ in strategies:
        print(f"  {'------------------':>18}", end="")
    print()

    cross = stats.get("by_model_x_strategy", {})
    for m in models:
        row = f"  {m:<14}"
        for s in strategies:
            sv = cross.get(m, {}).get(s, {})
            tot = sum(sv.values())
            if tot:
                fh = sv.get("fail_hard", 0)
                fs = sv.get("fail_soft", 0)
                row += f"  {100*fh/tot:5.1f}h {100*fs/tot:5.1f}s"
            else:
                row += f"  {'N/A':>18}"
        print(row)

    print(f"\n  Fail rate by strategy:")
    print(f"  {'Strategy':<14}  {'pass':>8}  {'fail_soft':>10}  {'fail_hard':>10}  {'hard%':>7}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}")
    for s in strategies:
        sv  = stats["by_strategy"].get(s, {})
        tot = sum(sv.values())
        fh  = sv.get("fail_hard", 0)
        fs  = sv.get("fail_soft", 0)
        pa  = sv.get("pass", 0)
        hard_pct = f"{100*fh/tot:.1f}%" if tot else "N/A"
        print(f"  {s:<14}  {pa:>8,}  {fs:>10,}  {fh:>10,}  {hard_pct:>7}")

    print(f"\n  Fail rate by domain:")
    print(f"  {'Domain':<20}  {'pass':>8}  {'fail_soft':>10}  {'fail_hard':>10}  {'hard%':>7}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}")
    for d in sorted(stats["by_domain"].keys()):
        dv  = stats["by_domain"][d]
        tot = sum(dv.values())
        fh  = dv.get("fail_hard", 0)
        fs  = dv.get("fail_soft", 0)
        pa  = dv.get("pass", 0)
        hard_pct = f"{100*fh/tot:.1f}%" if tot else "N/A"
        print(f"  {d:<20}  {pa:>8,}  {fs:>10,}  {fh:>10,}  {hard_pct:>7}")

    print(f"\n  Output : {OUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")
    print(f"{'='*78}")

def main():
    """Run phase 2 hygiene checks over indexed adaptations."""
    args = parse_args()

    print("=" * 70)
    print("Phase 2 — Output Hygiene Check (full run)")
    print("=" * 70)

    sys.path.insert(0, str(PHASE_DIR))
    from hygiene_checker import HygieneChecker

    done_ids: set[str] = set()
    if args.resume and not args.test:
        done_ids = load_checkpoint()
        print(f"\n[resume] {len(done_ids):,} records already done — will skip them")

    print("\n[0] Initialising HygieneChecker (loading fastText model) …")
    checker = HygieneChecker()
    _ = checker.check("warmup text", "warmup text", "rocstories")
    print("    Ready")

    print(f"\n[1] Streaming {INDEX_PATH.name} …")
    all_records: list[dict] = []
    with open(INDEX_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    print(f"    {len(all_records):,} records loaded")

    if args.test:
        all_records = all_records[:200]
        print(f"    TEST MODE: running on {len(all_records)} records")

    if done_ids:
        all_records = [r for r in all_records if r["eval_id"] not in done_ids]
        print(f"    After checkpoint filter: {len(all_records):,} remaining")

    print(f"\n[2] Running hygiene checks …")

    stats   = StatsAccumulator()
    errors  = 0
    written = 0
    t0      = time.time()

    CHECKPOINT_EVERY = 10_000
    buffer: list[str] = []

    write_mode = "a" if (args.resume and not args.test) else "w"

    if args.resume and not args.test and OUT_PATH.exists():
        print(f"    [resume] Rebuilding stats from existing output file …")
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stats.add(rec, rec["02_hygiene"])
                written += 1
        print(f"    {written:,} records already counted in stats")

    with open(OUT_PATH, write_mode) as out_f:
        for i, record in enumerate(all_records, 1):
            try:
                h = checker.check(
                    record["source_text"],
                    record.get("adapted_text", ""),
                    record.get("domain", "unknown"),
                )

                out_record = {**record, "02_hygiene": h}
                buffer.append(json.dumps(out_record, ensure_ascii=False))

                stats.add(record, h)
                done_ids.add(record["eval_id"])
                written += 1

            except Exception as e:
                errors += 1
                print(f"\n  ERROR at record {i}: {e}")
                continue

            if i % 5_000 == 0 or i == len(all_records):
                elapsed = time.time() - t0
                rate    = i / elapsed
                eta     = (len(all_records) - i) / rate if rate > 0 else 0
                total_done = written
                hd = stats.by_overall.get("fail_hard", 0)
                hs = stats.by_overall.get("fail_soft", 0)
                pa = stats.by_overall.get("pass", 0)
                print(
                    f"    {i:>7}/{len(all_records)} | "
                    f"{rate:>6.0f} rec/s | ETA {eta/60:>4.1f}m | "
                    f"pass={pa:,}  soft={hs:,}  hard={hd:,}"
                )

            if i % CHECKPOINT_EVERY == 0:
                out_f.write("\n".join(buffer) + "\n")
                buffer.clear()
                if not args.test:
                    save_checkpoint(done_ids)

        if buffer:
            out_f.write("\n".join(buffer) + "\n")
        if not args.test:
            save_checkpoint(done_ids)

    print(f"\n[3] Saving stats …")
    stats_dict = stats.to_dict()
    STATS_PATH.write_text(json.dumps(stats_dict, indent=2))

    if errors:
        print(f"  Errors (skipped records): {errors:,}")

    print_summary(stats_dict)

    if args.test:
        print("\n  Sample records (first 3):")
        with open(OUT_PATH) as f:
            for j, line in enumerate(f):
                if j >= 3:
                    break
                rec = json.loads(line.strip())
                h   = rec["02_hygiene"]
                print(f"\n  [{j+1}] {rec['eval_id']}")
                print(f"       overall={h['overall']}  lang={h['lang_check']}"
                      f"  degen={h['degeneration']}  meta={h['meta_commentary']}")
                print(f"       ratio={h['length_ratio']:.2f}  "
                      f"iaet={h['iaet_label']}  reason={h['failure_reason']}")

if __name__ == "__main__":
    main()
