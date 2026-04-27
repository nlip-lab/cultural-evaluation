#!/usr/bin/env python3
"""
Build the stratified ~16K eval_id sample used for downstream LLM judge phases.

Sampling per (model, prompt_strategy, domain) cell:
  - all items if cell has fewer than 125 success rows
  - otherwise sample 125 unique eval_ids with this priority:
      32 from top RF quartile
      32 from bottom RF quartile
      31 from middle RF band
      30 from SAT > median SAT

Any shortfall from a bucket is backfilled from remaining unsampled items
within the same cell.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PHASE3_DIR = EVAL_DIR / "03_Audit"

AUDIT_PATH = PHASE3_DIR / "audit_results.jsonl"
METRICS_PATH = PHASE3_DIR / "csi_metrics.jsonl"
OUT_PATH = EVAL_DIR / "sample_ids_16k.json"

TARGET_PER_CELL = 125
TARGET_TOTAL = 16000
DEFAULT_SEED = 42

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()

def load_success_eval_ids(path: Path) -> set[str]:
    success_ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audit_status") == "success":
                success_ids.add(row["eval_id"])
    return success_ids

def load_metric_rows(path: Path, success_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["eval_id"] in success_ids:
                rows.append(row)
    return rows

def group_by_cell(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cell = f"{row['model']}__{row['prompt_strategy']}__{row['domain']}"
        grouped[cell].append(row)
    return grouped

def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2

def choose_random(
    candidates: list[dict],
    k: int,
    rng: random.Random,
    selected_ids: set[str],
) -> list[dict]:
    pool = [row for row in candidates if row["eval_id"] not in selected_ids]
    if not pool or k <= 0:
        return []
    if len(pool) <= k:
        chosen = pool
    else:
        chosen = rng.sample(pool, k)
    selected_ids.update(row["eval_id"] for row in chosen)
    return chosen

def choose_head(
    candidates: list[dict],
    k: int,
    selected_ids: set[str],
) -> list[dict]:
    chosen: list[dict] = []
    for row in candidates:
        if row["eval_id"] in selected_ids:
            continue
        chosen.append(row)
        selected_ids.add(row["eval_id"])
        if len(chosen) >= k:
            break
    return chosen

def sample_cell(rows: list[dict], rng: random.Random) -> list[str]:
    if len(rows) <= TARGET_PER_CELL:
        return sorted((row["eval_id"] for row in rows))

    rows_by_rf_desc = sorted(rows, key=lambda r: (-r["rf"], r["eval_id"]))
    n = len(rows_by_rf_desc)
    quartile = max(n // 4, 1)
    top_quartile = rows_by_rf_desc[:quartile]
    bottom_quartile = sorted(rows_by_rf_desc[-quartile:], key=lambda r: (r["rf"], r["eval_id"]))
    middle_band = rows_by_rf_desc[quartile : n - quartile]

    sat_med = median([row["sat"] for row in rows])
    sat_high = [row for row in rows if row["sat"] > sat_med]

    selected_ids: set[str] = set()
    sampled: list[dict] = []
    sampled.extend(choose_head(top_quartile, 32, selected_ids))
    sampled.extend(choose_head(bottom_quartile, 32, selected_ids))
    sampled.extend(choose_random(middle_band, 31, rng, selected_ids))

    sat_high_sorted = sorted(sat_high, key=lambda r: (-r["sat"], r["eval_id"]))
    sampled.extend(choose_random(sat_high_sorted, 30, rng, selected_ids))

    if len(sampled) < TARGET_PER_CELL:
        remaining = sorted(rows, key=lambda r: (-r["sat"], -r["rf"], r["eval_id"]))
        sampled.extend(choose_random(remaining, TARGET_PER_CELL - len(sampled), rng, selected_ids))

    return sorted(row["eval_id"] for row in sampled[:TARGET_PER_CELL])

def fill_to_target_total(
    grouped: dict[str, list[dict]],
    by_cell: dict[str, list[str]],
    rng: random.Random,
) -> None:
    current_total = sum(len(ids) for ids in by_cell.values())
    if current_total >= TARGET_TOTAL:
        return

    deficit = TARGET_TOTAL - current_total
    expandable = []
    for cell, rows in grouped.items():
        chosen = set(by_cell[cell])
        extras = [row["eval_id"] for row in rows if row["eval_id"] not in chosen]
        if extras:
            expandable.append((cell, extras))

    cell_order = [cell for cell, _ in sorted(expandable, key=lambda item: item[0])]
    extras_by_cell = {cell: extras for cell, extras in expandable}

    while deficit > 0 and cell_order:
        progressed = False
        for cell in cell_order:
            extras = extras_by_cell[cell]
            if not extras:
                continue
            pick_idx = rng.randrange(len(extras))
            picked = extras.pop(pick_idx)
            by_cell[cell].append(picked)
            deficit -= 1
            progressed = True
            if deficit == 0:
                break
        if not progressed:
            break

    for cell in by_cell:
        by_cell[cell] = sorted(by_cell[cell])

def build_sample(rows: list[dict], seed: int) -> tuple[dict[str, list[str]], list[str], list[str]]:
    rng = random.Random(seed)
    grouped = group_by_cell(rows)
    by_cell: dict[str, list[str]] = {}
    short_cells: list[str] = []

    for cell in sorted(grouped):
        sampled_ids = sample_cell(grouped[cell], rng)
        by_cell[cell] = sampled_ids
        if len(grouped[cell]) < TARGET_PER_CELL:
            short_cells.append(f"{cell}: {len(grouped[cell])}")

    fill_to_target_total(grouped, by_cell, rng)
    all_ids = sorted({eval_id for ids in by_cell.values() for eval_id in ids})
    return by_cell, all_ids, short_cells

def write_output(path: Path, by_cell: dict[str, list[str]], all_ids: list[str]) -> None:
    payload = {
        "total": len(all_ids),
        "by_cell": by_cell,
        "all_ids": all_ids,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

def main() -> None:
    args = parse_args()
    success_ids = load_success_eval_ids(args.audit_path)
    metric_rows = load_metric_rows(args.metrics_path, success_ids)
    by_cell, all_ids, short_cells = build_sample(metric_rows, args.seed)

    write_output(args.output_path, by_cell, all_ids)

    print(f"Total sampled: {len(all_ids)}")
    print("Per-cell counts:")
    for cell in sorted(by_cell):
        print(f"  {cell}: {len(by_cell[cell])}")

    if short_cells:
        print("Cells with < 125 items:")
        for cell in short_cells:
            print(f"  {cell}")
    else:
        print("Cells with < 125 items: none")

if __name__ == "__main__":
    main()
