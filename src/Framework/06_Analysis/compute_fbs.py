"""
Phase 6 — BERTScore on paired function-frame strings.

Computes BERTScore on:
  - contextual_function only
  - holistic concatenation: denotation + cultural_role + contextual_function

Writes:
  - 06_Analysis/fbs_scores.jsonl
  - 06_Analysis/fbs_per_adaptation.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from bert_score import score as bertscore_score

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
PHASE6_DIR = EVAL_DIR / "06_Analysis"

PAIRED_FRAMES_PATH = PHASE5_DIR / "paired_frames.jsonl"
FBS_SCORES_PATH = PHASE6_DIR / "fbs_scores.jsonl"
FBS_ADAPT_PATH = PHASE6_DIR / "fbs_per_adaptation.jsonl"
MODEL_TYPE = "microsoft/deberta-xlarge-mnli"

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BERTScore evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-frames-path", type=Path, default=PAIRED_FRAMES_PATH)
    parser.add_argument("--scores-out", type=Path, default=FBS_SCORES_PATH)
    parser.add_argument("--adaptation-out", type=Path, default=FBS_ADAPT_PATH)
    parser.add_argument("--model-type", default=MODEL_TYPE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()

def load_rows(path: Path) -> list[dict]:
    """Load paired frame rows from JSONL."""
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_holistic_text(frame: dict) -> str:
    """Build holistic function text from denotation, role, and contextual function."""
    parts = [
        str(frame.get("denotation", "")).strip(),
        str(frame.get("cultural_role", "")).strip(),
        str(frame.get("contextual_function", "")).strip(),
    ]
    return ". ".join(part for part in parts if part)

def run_bertscore(
    refs: list[str],
    cands: list[str],
    *,
    model_type: str,
    batch_size: int,
    device: str,
    num_layers: int | None,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BERTScore and return precision, recall, and F1 arrays."""
    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    kwargs = {
        "cands": cands,
        "refs": refs,
        "lang": "en",
        "model_type": model_type,
        "batch_size": batch_size,
        "device": device,
        "idf": False,
        "verbose": True,
        "rescale_with_baseline": False,
    }
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    p, r, f1 = bertscore_score(**kwargs)
    return (
        p.detach().cpu().numpy(),
        r.detach().cpu().numpy(),
        f1.detach().cpu().numpy(),
    )

def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def mean(values: list[float]) -> float:
    """Return mean for a list of float values."""
    return float(np.mean(values)) if values else 0.0

def score_unique_pairs(
    refs: list[str],
    cands: list[str],
    *,
    model_type: str,
    batch_size: int,
    device: str,
    num_layers: int | None,
    local_files_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score deduplicated text pairs and map scores back to full row order."""
    pair_to_idx: dict[tuple[str, str], int] = {}
    unique_refs: list[str] = []
    unique_cands: list[str] = []
    row_lookup: list[int] = []

    for ref, cand in zip(refs, cands, strict=True):
        key = (ref, cand)
        idx = pair_to_idx.get(key)
        if idx is None:
            idx = len(unique_refs)
            pair_to_idx[key] = idx
            unique_refs.append(ref)
            unique_cands.append(cand)
        row_lookup.append(idx)

    print(f"  unique pairs: {len(unique_refs)}")
    p_unique, r_unique, f1_unique = run_bertscore(
        refs=unique_refs,
        cands=unique_cands,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        num_layers=num_layers,
        local_files_only=local_files_only,
    )
    row_indices = np.array(row_lookup)
    return p_unique[row_indices], r_unique[row_indices], f1_unique[row_indices]

def main() -> None:
    """Run FBS scoring pipeline for paired function frames."""
    args = parse_args()
    rows = load_rows(args.paired_frames_path)

    source_contextual = [row["source_function"].get("contextual_function", "") for row in rows]
    adapted_contextual = [row["adapted_function"].get("contextual_function", "") for row in rows]
    source_holistic = [build_holistic_text(row["source_function"]) for row in rows]
    adapted_holistic = [build_holistic_text(row["adapted_function"]) for row in rows]

    print(f"Loaded {len(rows)} paired rows")
    print(f"Running contextual BERTScore with {args.model_type} on {args.device}")
    p_ctx, r_ctx, f1_ctx = score_unique_pairs(
        refs=source_contextual,
        cands=adapted_contextual,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
        num_layers=args.num_layers,
        local_files_only=args.local_files_only,
    )

    torch.cuda.empty_cache()

    print("Running holistic BERTScore")
    _, _, f1_hol = score_unique_pairs(
        refs=source_holistic,
        cands=adapted_holistic,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device,
        num_layers=args.num_layers,
        local_files_only=args.local_files_only,
    )

    score_rows: list[dict] = []
    by_eval: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)

    for row, p_val, r_val, f1_val, f1_h_val in zip(
        rows, p_ctx.tolist(), r_ctx.tolist(), f1_ctx.tolist(), f1_hol.tolist(), strict=True
    ):
        score_row = {
            "eval_id": row["eval_id"],
            "csi_index": row["csi_index"],
            "source_span": row["source_span"],
            "adapted_span": row["adapted_span"],
            "fbs_precision": round(float(p_val), 6),
            "fbs_recall": round(float(r_val), 6),
            "fbs_f1": round(float(f1_val), 6),
            "fbs_holistic_f1": round(float(f1_h_val), 6),
        }
        score_rows.append(score_row)
        by_eval[row["eval_id"]].append(
            {
                **score_row,
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "domain": row["domain"],
                "strategy": row["strategy"],
                "source_category": row["source_category"],
            }
        )
        by_domain[row["domain"]].append(float(f1_val))
        by_strategy[row["strategy"]].append(float(f1_val))
        by_category[row["source_category"]].append(float(f1_val))

    adaptation_rows: list[dict] = []
    for eval_id, items in sorted(by_eval.items()):
        adaptation_rows.append(
            {
                "eval_id": eval_id,
                "model": items[0]["model"],
                "prompt_strategy": items[0]["prompt_strategy"],
                "domain": items[0]["domain"],
                "mean_fbs_f1": round(mean([item["fbs_f1"] for item in items]), 6),
                "mean_fbs_holistic_f1": round(mean([item["fbs_holistic_f1"] for item in items]), 6),
                "min_fbs_f1": round(min(item["fbs_f1"] for item in items), 6),
                "num_pairs": len(items),
            }
        )

    write_jsonl(args.scores_out, score_rows)
    write_jsonl(args.adaptation_out, adaptation_rows)

    print(f"Overall mean FBS F1: {mean([row['fbs_f1'] for row in score_rows]):.6f}")
    print("FBS by domain:")
    for key in sorted(by_domain):
        print(f"  {key}: {mean(by_domain[key]):.6f}")
    print("FBS by strategy type:")
    for key in sorted(by_strategy):
        print(f"  {key}: {mean(by_strategy[key]):.6f}")
    print("FBS by CSI category:")
    for key in sorted(by_category):
        print(f"  {key}: {mean(by_category[key]):.6f}")

if __name__ == "__main__":
    main()
