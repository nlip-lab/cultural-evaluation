"""
Phase 6 — Function Embedding Similarity (FES).

Computes:
  - contextual similarity on source/adapted contextual_function
  - holistic similarity on concatenated denotation + cultural_role + contextual_function

Writes:
  - 06_Analysis/fes_scores.jsonl
  - 06_Analysis/fes_per_adaptation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
PHASE6_DIR = EVAL_DIR / "06_Analysis"

PAIRED_FRAMES_PATH = PHASE5_DIR / "paired_frames.jsonl"
FES_SCORES_PATH = PHASE6_DIR / "fes_scores.jsonl"
FES_ADAPT_PATH = PHASE6_DIR / "fes_per_adaptation.jsonl"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for embedding-similarity evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-frames-path", type=Path, default=PAIRED_FRAMES_PATH)
    parser.add_argument("--scores-out", type=Path, default=FES_SCORES_PATH)
    parser.add_argument("--adaptation-out", type=Path, default=FES_ADAPT_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
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

def cosine_rowwise(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity for normalized embeddings."""
    return np.sum(left * right, axis=1)

def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode a list of texts into normalized embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

def encode_unique_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Encode unique texts and return lookup mapping to embeddings."""
    unique_texts = sorted(set(texts))
    print(f"  unique strings: {len(unique_texts)}")
    embeddings = encode_texts(model, unique_texts, batch_size)
    return {text: emb for text, emb in zip(unique_texts, embeddings, strict=True)}

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
    """Run FES scoring pipeline for paired function frames."""
    args = parse_args()
    rows = load_rows(args.paired_frames_path)

    source_contextual = [
        row["source_function"].get("contextual_function", "")
        for row in rows
    ]
    adapted_contextual = [
        row["adapted_function"].get("contextual_function", "")
        for row in rows
    ]
    source_holistic = [build_holistic_text(row["source_function"]) for row in rows]
    adapted_holistic = [build_holistic_text(row["adapted_function"]) for row in rows]

    model = SentenceTransformer(args.model_name, local_files_only=args.local_files_only)

    print(f"Loaded {len(rows)} paired rows")
    print(f"Encoding contextual functions with {args.model_name}")
    print("Source contextual:")
    source_contextual_lookup = encode_unique_texts(model, source_contextual, args.batch_size)
    print("Adapted contextual:")
    adapted_contextual_lookup = encode_unique_texts(model, adapted_contextual, args.batch_size)

    print("Encoding holistic function strings")
    print("Source holistic:")
    source_holistic_lookup = encode_unique_texts(model, source_holistic, args.batch_size)
    print("Adapted holistic:")
    adapted_holistic_lookup = encode_unique_texts(model, adapted_holistic, args.batch_size)

    source_contextual_emb = np.vstack([source_contextual_lookup[text] for text in source_contextual])
    adapted_contextual_emb = np.vstack([adapted_contextual_lookup[text] for text in adapted_contextual])
    source_holistic_emb = np.vstack([source_holistic_lookup[text] for text in source_holistic])
    adapted_holistic_emb = np.vstack([adapted_holistic_lookup[text] for text in adapted_holistic])

    contextual_scores = cosine_rowwise(source_contextual_emb, adapted_contextual_emb)
    holistic_scores = cosine_rowwise(source_holistic_emb, adapted_holistic_emb)

    score_rows: list[dict] = []
    by_eval: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)

    for row, contextual, holistic in zip(rows, contextual_scores.tolist(), holistic_scores.tolist(), strict=True):
        score_row = {
            "eval_id": row["eval_id"],
            "csi_index": row["csi_index"],
            "source_span": row["source_span"],
            "adapted_span": row["adapted_span"],
            "strategy": row["strategy"],
            "source_category": row["source_category"],
            "fes_contextual": round(float(contextual), 6),
            "fes_holistic": round(float(holistic), 6),
        }
        score_rows.append(score_row)
        by_eval[row["eval_id"]].append({**score_row, "model": row["model"], "prompt_strategy": row["prompt_strategy"], "domain": row["domain"]})
        by_domain[row["domain"]].append(float(contextual))
        by_strategy[row["strategy"]].append(float(contextual))
        by_category[row["source_category"]].append(float(contextual))

    adaptation_rows: list[dict] = []
    for eval_id, items in sorted(by_eval.items()):
        adaptation_rows.append(
            {
                "eval_id": eval_id,
                "model": items[0]["model"],
                "prompt_strategy": items[0]["prompt_strategy"],
                "domain": items[0]["domain"],
                "mean_fes_contextual": round(mean([item["fes_contextual"] for item in items]), 6),
                "mean_fes_holistic": round(mean([item["fes_holistic"] for item in items]), 6),
                "min_fes_contextual": round(min(item["fes_contextual"] for item in items), 6),
                "num_pairs": len(items),
            }
        )

    write_jsonl(args.scores_out, score_rows)
    write_jsonl(args.adaptation_out, adaptation_rows)

    print(f"Overall mean FES contextual: {mean([row['fes_contextual'] for row in score_rows]):.6f}")
    print(f"Overall mean FES holistic: {mean([row['fes_holistic'] for row in score_rows]):.6f}")

    print("FES by domain:")
    for key in sorted(by_domain):
        print(f"  {key}: {mean(by_domain[key]):.6f}")

    print("FES by strategy type:")
    for key in sorted(by_strategy):
        print(f"  {key}: {mean(by_strategy[key]):.6f}")

    print("FES by CSI category:")
    for key in sorted(by_category):
        print(f"  {key}: {mean(by_category[key]):.6f}")

if __name__ == "__main__":
    main()
