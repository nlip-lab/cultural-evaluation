"""
Phase 5 — Map deduplicated adapted function frames back to every adaptation.

Reads:
  - 05_Frames/05_frames.jsonl
  - 05_Frames/dedup_to_eval_map.json
  - 01_CSI_Extraction/source_csi_final_google.jsonl
  - 03_Audit/audit_results.jsonl

Writes:
  - 05_Frames/paired_frames.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
PHASE1_DIR = EVAL_DIR / "01_CSI_Extraction"
PHASE3_DIR = EVAL_DIR / "03_Audit"

ADAPTED_FRAMES_PATH = PHASE5_DIR / "05_frames.jsonl"
DEDUP_MAP_PATH = PHASE5_DIR / "dedup_to_eval_map.json"
SOURCE_CSI_PATH = PHASE1_DIR / "source_csi_final_google.jsonl"
AUDIT_PATH = PHASE3_DIR / "audit_results.jsonl"
OUT_PATH = PHASE5_DIR / "paired_frames.jsonl"

TARGET_STRATEGIES = {"localized", "globalized", "transformed"}

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building paired frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted-frames-path", type=Path, default=ADAPTED_FRAMES_PATH)
    parser.add_argument("--dedup-map-path", type=Path, default=DEDUP_MAP_PATH)
    parser.add_argument("--source-csi-path", type=Path, default=SOURCE_CSI_PATH)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    return parser.parse_args()

def load_phase5_frames(path: Path) -> dict[str, dict]:
    """Load adapted frame rows indexed by dedup_id."""
    idx: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx[row["dedup_id"]] = row
    return idx

def load_dedup_map(path: Path) -> dict[str, list[dict]]:
    """Load dedup_id to eval/csi references map."""
    with path.open() as handle:
        return json.load(handle)

def load_source_index(path: Path) -> dict[str, dict]:
    """Load source CSI records indexed by source_id."""
    idx: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row.get("source_id") or row.get("id")
            if source_id:
                idx[source_id] = row
    return idx

def load_audit_index(path: Path) -> dict[str, dict]:
    """Load audit records indexed by eval_id."""
    idx: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            eval_id = row.get("eval_id")
            if eval_id:
                idx[eval_id] = row
    return idx

def build_source_function(csi: dict) -> dict:
    """Build normalized source-side function fields."""
    return {
        "denotation": csi.get("denotation", ""),
        "cultural_role": csi.get("cultural_role", ""),
        "contextual_function": csi.get("contextual_function", ""),
        "affordances": list(csi.get("required_affordances") or []),
    }

def build_adapted_function(frame: dict) -> dict:
    """Build normalized adapted-side function fields."""
    return {
        "denotation": frame.get("adapted_denotation", ""),
        "cultural_role": frame.get("adapted_cultural_role", ""),
        "contextual_function": frame.get("adapted_contextual_function", ""),
        "affordances": list(frame.get("adapted_affordances") or []),
    }

def build_paired_rows(
    phase5_frames: dict[str, dict],
    dedup_map: dict[str, list[dict]],
    source_index: dict[str, dict],
    audit_index: dict[str, dict],
) -> tuple[list[dict], dict[str, Counter]]:
    """Build paired source/adapted function rows and summary counters."""
    rows: list[dict] = []
    strategy_counts = Counter()
    domain_counts = Counter()
    model_counts = Counter()

    for dedup_id, eval_refs in dedup_map.items():
        adapted_frame = phase5_frames.get(dedup_id)
        if adapted_frame is None:
            continue

        for ref in eval_refs:
            eval_id = ref["eval_id"]
            csi_index = ref["csi_index"]
            audit = audit_index.get(eval_id)
            if audit is None:
                continue

            source_id = audit.get("source_id")
            source_record = source_index.get(source_id)
            if source_record is None:
                continue

            csis = source_record.get("csis") or []
            alignments = audit.get("csi_alignments") or []
            if csi_index >= len(csis) or csi_index >= len(alignments):
                continue

            csi = csis[csi_index]
            alignment = alignments[csi_index]
            strategy = alignment.get("strategy")
            if strategy not in TARGET_STRATEGIES:
                continue

            row = {
                "eval_id": eval_id,
                "source_id": source_id,
                "domain": audit.get("domain", source_record.get("domain", "")),
                "model": audit.get("model", ""),
                "prompt_strategy": audit.get("strategy", ""),
                "csi_index": csi_index,
                "source_span": csi.get("span", alignment.get("source_span", "")),
                "adapted_span": adapted_frame.get("adapted_span", alignment.get("adapted_span", "")),
                "source_category": csi.get("category", ""),
                "strategy": strategy,
                "source_function": build_source_function(csi),
                "adapted_function": build_adapted_function(adapted_frame),
            }
            rows.append(row)
            strategy_counts[strategy] += 1
            domain_counts[row["domain"]] += 1
            model_counts[row["model"]] += 1

    rows.sort(key=lambda r: (r["eval_id"], r["csi_index"]))
    return rows, {
        "strategy": strategy_counts,
        "domain": domain_counts,
        "model": model_counts,
    }

def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_counter(title: str, counter: Counter) -> None:
    """Print one counter distribution with sorted keys."""
    print(title)
    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")

def main() -> None:
    """Run phase-5 pairing pipeline."""
    args = parse_args()
    phase5_frames = load_phase5_frames(args.adapted_frames_path)
    dedup_map = load_dedup_map(args.dedup_map_path)
    source_index = load_source_index(args.source_csi_path)
    audit_index = load_audit_index(args.audit_path)

    rows, counters = build_paired_rows(phase5_frames, dedup_map, source_index, audit_index)
    write_jsonl(args.output_path, rows)

    print(f"Total paired records: {len(rows)}")
    print_counter("Records per strategy:", counters["strategy"])
    print_counter("Records per domain:", counters["domain"])
    print_counter("Records per model:", counters["model"])

if __name__ == "__main__":
    main()
