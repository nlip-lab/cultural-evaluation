"""
Phase 6 — Sentence-level BLEU and ROUGE-L on paired function descriptions.

Computes metrics on:
  - contextual_function only
  - holistic concatenation: denotation + cultural_role + contextual_function

Writes:
  - 06_Analysis/bleu_rouge_scores.jsonl
  - 06_Analysis/bleu_rouge_per_adaptation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
PHASE6_DIR = EVAL_DIR / "06_Analysis"

PAIRED_FRAMES_PATH = PHASE5_DIR / "paired_frames.jsonl"
SCORES_OUT_PATH = PHASE6_DIR / "bleu_rouge_scores.jsonl"
ADAPT_OUT_PATH = PHASE6_DIR / "bleu_rouge_per_adaptation.jsonl"

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BLEU/ROUGE scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-frames-path", type=Path, default=PAIRED_FRAMES_PATH)
    parser.add_argument("--scores-out", type=Path, default=SCORES_OUT_PATH)
    parser.add_argument("--adaptation-out", type=Path, default=ADAPT_OUT_PATH)
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

def tokenize(text: str) -> list[str]:
    """Tokenize text for sentence-level BLEU computation."""
    return text.lower().split()

def compute_bleu(reference: str, candidate: str, smoother) -> float:
    """Compute sentence BLEU between reference and candidate strings."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=smoother.method1,
        )
    )

def compute_rouge_l(reference: str, candidate: str, scorer: rouge_scorer.RougeScorer) -> float:
    """Compute ROUGE-L F1 between reference and candidate strings."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    result = scorer.score(reference, candidate)
    return float(result["rougeL"].fmeasure)

def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def mean(values: list[float]) -> float:
    """Return mean for a list of float values."""
    return float(np.mean(values)) if values else 0.0

def main() -> None:
    """Run BLEU/ROUGE scoring pipeline for paired function frames."""
    args = parse_args()
    rows = load_rows(args.paired_frames_path)

    smoother = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    score_rows: list[dict] = []
    by_eval: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        source_contextual = row["source_function"].get("contextual_function", "")
        adapted_contextual = row["adapted_function"].get("contextual_function", "")
        source_holistic = build_holistic_text(row["source_function"])
        adapted_holistic = build_holistic_text(row["adapted_function"])

        bleu_contextual = compute_bleu(source_contextual, adapted_contextual, smoother)
        rouge_contextual = compute_rouge_l(source_contextual, adapted_contextual, scorer)
        bleu_holistic = compute_bleu(source_holistic, adapted_holistic, smoother)
        rouge_holistic = compute_rouge_l(source_holistic, adapted_holistic, scorer)

        score_row = {
            "eval_id": row["eval_id"],
            "csi_index": row["csi_index"],
            "source_span": row["source_span"],
            "adapted_span": row["adapted_span"],
            "bleu_contextual": round(bleu_contextual, 6),
            "rouge_l_contextual": round(rouge_contextual, 6),
            "bleu_holistic": round(bleu_holistic, 6),
            "rouge_l_holistic": round(rouge_holistic, 6),
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
        by_domain[row["domain"]].append(bleu_contextual)
        by_strategy[row["strategy"]].append(bleu_contextual)
        by_category[row["source_category"]].append(bleu_contextual)

    adaptation_rows: list[dict] = []
    for eval_id, items in sorted(by_eval.items()):
        adaptation_rows.append(
            {
                "eval_id": eval_id,
                "model": items[0]["model"],
                "prompt_strategy": items[0]["prompt_strategy"],
                "domain": items[0]["domain"],
                "mean_bleu_contextual": round(mean([item["bleu_contextual"] for item in items]), 6),
                "mean_rouge_l_contextual": round(mean([item["rouge_l_contextual"] for item in items]), 6),
                "mean_bleu_holistic": round(mean([item["bleu_holistic"] for item in items]), 6),
                "mean_rouge_l_holistic": round(mean([item["rouge_l_holistic"] for item in items]), 6),
                "num_pairs": len(items),
            }
        )

    write_jsonl(args.scores_out, score_rows)
    write_jsonl(args.adaptation_out, adaptation_rows)

    print(f"Overall mean BLEU contextual: {mean([row['bleu_contextual'] for row in score_rows]):.6f}")
    print(f"Overall mean ROUGE-L contextual: {mean([row['rouge_l_contextual'] for row in score_rows]):.6f}")
    print(f"Overall mean BLEU holistic: {mean([row['bleu_holistic'] for row in score_rows]):.6f}")
    print(f"Overall mean ROUGE-L holistic: {mean([row['rouge_l_holistic'] for row in score_rows]):.6f}")

    print("BLEU by domain:")
    for key in sorted(by_domain):
        print(f"  {key}: {mean(by_domain[key]):.6f}")

    print("BLEU by strategy type:")
    for key in sorted(by_strategy):
        print(f"  {key}: {mean(by_strategy[key]):.6f}")

    print("BLEU by CSI category:")
    for key in sorted(by_category):
        print(f"  {key}: {mean(by_category[key]):.6f}")

if __name__ == "__main__":
    main()
