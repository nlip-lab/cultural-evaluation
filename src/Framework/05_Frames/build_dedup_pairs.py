"""
Phase 5 — Deduplicate adapted CSI pairs before adapted frame generation.

Builds a deduplicated inventory of source/adapted CSI pairs from:
  - 01_CSI_Extraction/source_csi_final_google.jsonl
  - 03_Audit/audit_results.jsonl

Outputs:
  - 05_Frames/dedup_pairs.jsonl
  - 05_Frames/dedup_to_eval_map.json

Clustering rule:
  - Keep only localized/globalized/transformed alignments
  - Group first by (source_span, strategy)
  - Within each group, cluster unique adapted spans using cosine similarity
    over `all-mpnet-base-v2` embeddings with threshold > 0.92
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE1_PATH = EVAL_DIR / "01_CSI_Extraction" / "source_csi_final_google.jsonl"
PHASE3_PATH = EVAL_DIR / "03_Audit" / "audit_results.jsonl"
PHASE2_PATH = EVAL_DIR / "02_Hygiene" / "adaptations_with_hygiene.jsonl"
PHASE5_DIR = EVAL_DIR / "05_Frames"
OUT_PAIRS_PATH = PHASE5_DIR / "dedup_pairs.jsonl"
OUT_MAP_PATH = PHASE5_DIR / "dedup_to_eval_map.json"

TARGET_STRATEGIES = {"localized", "globalized", "transformed"}
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.92

@dataclass(frozen=True)
class Occurrence:
    """Container for one eligible aligned CSI occurrence."""
    eval_id: str
    csi_index: int
    source_id: str
    domain: str
    source_span: str
    adapted_span: str
    strategy: str
    source_category: str
    source_contextual_function: str
    source_required_affordances: tuple[str, ...]
    source_text: str
    adapted_text: str

class UnionFind:
    """Disjoint-set structure used for span similarity clustering."""
    def __init__(self, size: int) -> None:
        """Initialize parent/rank arrays for given set size."""
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, idx: int) -> int:
        """Find canonical root for an index with path compression."""
        while self.parent[idx] != idx:
            self.parent[idx] = self.parent[self.parent[idx]]
            idx = self.parent[idx]
        return idx

    def union(self, left: int, right: int) -> None:
        """Union two indices by rank."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dedup pair construction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-path",
        type=Path,
        default=PHASE1_PATH,
        help="Path to source CSI inventory JSONL.",
    )
    parser.add_argument(
        "--phase3-path",
        type=Path,
        default=PHASE3_PATH,
        help="Path to audit results JSONL.",
    )
    parser.add_argument(
        "--phase2-path",
        type=Path,
        default=PHASE2_PATH,
        help="Path to hygiene output JSONL for adapted_text lookup.",
    )
    parser.add_argument(
        "--pairs-out",
        type=Path,
        default=OUT_PAIRS_PATH,
        help="Output JSONL for deduplicated source/adapted pairs.",
    )
    parser.add_argument(
        "--map-out",
        type=Path,
        default=OUT_MAP_PATH,
        help="Output JSON for dedup_id -> eval_id/csi_index map.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for clustering within a source-span group.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--model-name",
        default=EMBED_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load embedding model from local cache only.",
    )
    parser.add_argument(
        "--pilot-groups",
        type=int,
        default=None,
        help="Optional limit on source-span groups for a smoke run.",
    )
    return parser.parse_args()

def load_source_index(path: Path) -> dict[str, dict]:
    """Load source-side CSI inventory indexed by source_id."""
    index: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = record.get("source_id") or record.get("id")
            if not source_id:
                continue
            index[source_id] = record
    return index

def load_hygiene_index(path: Path) -> dict[str, dict]:
    """Load hygiene records indexed by eval_id."""
    index: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            eval_id = record.get("eval_id")
            if eval_id:
                index[eval_id] = record
    return index

def load_occurrences(
    phase3_path: Path,
    source_index: dict[str, dict],
    hygiene_index: dict[str, dict],
) -> list[Occurrence]:
    """Load eligible CSI occurrences from audit alignments."""
    occurrences: list[Occurrence] = []

    with phase3_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue

            audit = json.loads(line)
            if audit.get("audit_status") != "success":
                continue

            source_id = audit.get("source_id")
            source_record = source_index.get(source_id)
            if not source_record:
                continue
            hygiene_record = hygiene_index.get(audit.get("eval_id"), {})

            source_text = source_record.get("source_text", "")
            source_csis = source_record.get("csis") or []
            alignments = audit.get("csi_alignments") or []
            adapted_text = hygiene_record.get("adapted_text", "")

            for csi_index, source_csi in enumerate(source_csis):
                if csi_index >= len(alignments):
                    break

                alignment = alignments[csi_index]
                strategy = (alignment.get("strategy") or "").strip()
                adapted_span = (alignment.get("adapted_span") or "").strip()
                source_span = (source_csi.get("span") or alignment.get("source_span") or "").strip()

                if strategy not in TARGET_STRATEGIES:
                    continue
                if not source_span or not adapted_span:
                    continue

                occurrences.append(
                    Occurrence(
                        eval_id=audit["eval_id"],
                        csi_index=csi_index,
                        source_id=source_id,
                        domain=audit.get("domain") or source_record.get("domain", ""),
                        source_span=source_span,
                        adapted_span=adapted_span,
                        strategy=strategy,
                        source_category=source_csi.get("category", ""),
                        source_contextual_function=source_csi.get("contextual_function", ""),
                        source_required_affordances=tuple(source_csi.get("required_affordances") or []),
                        source_text=source_text,
                        adapted_text=adapted_text,
                    )
                )

    return occurrences

def encode_spans(
    unique_spans: list[str],
    *,
    model_name: str,
    batch_size: int,
    local_files_only: bool,
) -> dict[str, np.ndarray]:
    """Encode unique adapted spans into normalized embedding vectors."""
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    embeddings = model.encode(
        unique_spans,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {span: vector for span, vector in zip(unique_spans, embeddings, strict=True)}

def cluster_variants(
    spans: list[str],
    embedding_lookup: dict[str, np.ndarray],
    threshold: float,
) -> list[list[str]]:
    """Cluster variant spans by cosine similarity threshold."""
    if len(spans) == 1:
        return [spans]

    matrix = np.vstack([embedding_lookup[span] for span in spans])
    similarities = matrix @ matrix.T
    union_find = UnionFind(len(spans))

    upper_rows, upper_cols = np.where(np.triu(similarities > threshold, k=1))
    for row, col in zip(upper_rows.tolist(), upper_cols.tolist(), strict=True):
        union_find.union(row, col)

    grouped: dict[int, list[str]] = defaultdict(list)
    for idx, span in enumerate(spans):
        grouped[union_find.find(idx)].append(span)
    return list(grouped.values())

def choose_representative_span(
    variants: list[str],
    span_counts: Counter[str],
) -> str:
    """Select representative span from variants by frequency and tie-breakers."""
    return max(
        variants,
        key=lambda span: (span_counts[span], -len(span), span),
    )

def choose_representative_occurrence(
    members: list[Occurrence],
    representative_span: str,
) -> Occurrence:
    """Select representative occurrence matching representative span when possible."""
    rep_candidates = [member for member in members if member.adapted_span == representative_span]
    if rep_candidates:
        members = rep_candidates
    return members[0]

def select_occurrences_for_run(
    occurrences: list[Occurrence],
    pilot_groups: int | None,
) -> list[Occurrence]:
    """Select occurrence subset for full or pilot execution."""
    grouped_occurrences: dict[tuple[str, str], list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped_occurrences[(occurrence.source_span, occurrence.strategy)].append(occurrence)

    sorted_group_items = sorted(
        grouped_occurrences.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )
    if pilot_groups is not None:
        sorted_group_items = sorted_group_items[:pilot_groups]

    selected: list[Occurrence] = []
    for _, group_occurrences in sorted_group_items:
        selected.extend(group_occurrences)
    return selected

def build_outputs(
    occurrences: list[Occurrence],
    embedding_lookup: dict[str, np.ndarray],
    threshold: float,
) -> tuple[list[dict], dict[str, list[dict]], dict[str, float]]:
    """Build deduplicated rows, back-mapping, and summary stats."""
    grouped_occurrences: dict[tuple[str, str], list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped_occurrences[(occurrence.source_span, occurrence.strategy)].append(occurrence)

    sorted_group_items = sorted(
        grouped_occurrences.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )

    dedup_rows: list[dict] = []
    dedup_map: dict[str, list[dict]] = {}
    dedup_counter = 1

    unique_pairs_before = len(
        {
            (occ.source_span, occ.adapted_span, occ.strategy)
            for _, group_occurrences in sorted_group_items
            for occ in group_occurrences
        }
    )

    for (_, _), group_members in sorted_group_items:
        span_counts = Counter(member.adapted_span for member in group_members)
        unique_spans = sorted(span_counts)
        clusters = cluster_variants(unique_spans, embedding_lookup, threshold)

        for variants in sorted(clusters, key=lambda cluster: (-sum(span_counts[v] for v in cluster), cluster)):
            representative_span = choose_representative_span(variants, span_counts)
            cluster_members = [member for member in group_members if member.adapted_span in set(variants)]
            representative = choose_representative_occurrence(cluster_members, representative_span)
            dedup_id = f"dedup_{dedup_counter:04d}"
            dedup_counter += 1

            dedup_rows.append(
                {
                    "dedup_id": dedup_id,
                    "source_span": representative.source_span,
                    "source_category": representative.source_category,
                    "source_contextual_function": representative.source_contextual_function,
                    "source_required_affordances": list(representative.source_required_affordances),
                    "adapted_span_representative": representative_span,
                    "adapted_span_variants": sorted(variants),
                    "strategy": representative.strategy,
                    "frequency": len(cluster_members),
                    "representative_source_text": representative.source_text,
                    "representative_adapted_text": representative.adapted_text,
                    "representative_domain": representative.domain,
                }
            )
            dedup_map[dedup_id] = [
                {"eval_id": member.eval_id, "csi_index": member.csi_index}
                for member in cluster_members
            ]

    total_dedup_groups = len(dedup_rows)
    reduction_ratio = (
        1.0 - (total_dedup_groups / unique_pairs_before)
        if unique_pairs_before
        else 0.0
    )
    stats = {
        "unique_pairs_before_dedup": unique_pairs_before,
        "dedup_groups_after_clustering": total_dedup_groups,
        "reduction_ratio": reduction_ratio,
    }
    dedup_rows.sort(key=lambda row: (-row["frequency"], row["source_span"], row["adapted_span_representative"]))
    return dedup_rows, dedup_map, stats

def write_outputs(
    dedup_rows: list[dict],
    dedup_map: dict[str, list[dict]],
    *,
    pairs_out: Path,
    map_out: Path,
) -> None:
    """Write dedup JSONL rows and dedup map JSON file."""
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with pairs_out.open("w") as handle:
        for row in dedup_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with map_out.open("w") as handle:
        json.dump(dedup_map, handle, ensure_ascii=False, indent=2)

def print_summary(dedup_rows: list[dict], stats: dict[str, float]) -> None:
    """Print high-level deduplication outcome summary."""
    print(f"Total unique (source_span, adapted_span, strategy) pairs before dedup: {stats['unique_pairs_before_dedup']}")
    print(f"Total dedup groups after clustering: {stats['dedup_groups_after_clustering']}")
    print(f"Reduction ratio: {stats['reduction_ratio']:.4f}")
    print("Top 20 most frequent dedup groups:")
    for row in dedup_rows[:20]:
        print(
            f"  {row['dedup_id']}: "
            f"{row['source_span']} -> {row['adapted_span_representative']} "
            f"[{row['strategy']}] freq={row['frequency']} variants={len(row['adapted_span_variants'])}"
        )

def main() -> None:
    """Run phase-5 dedup pair generation workflow."""
    args = parse_args()

    source_index = load_source_index(args.phase1_path)
    hygiene_index = load_hygiene_index(args.phase2_path)
    occurrences = load_occurrences(args.phase3_path, source_index, hygiene_index)
    selected_occurrences = select_occurrences_for_run(occurrences, args.pilot_groups)
    unique_adapted_spans = sorted({occ.adapted_span for occ in selected_occurrences})

    print(f"Loaded {len(source_index):,} source records")
    print(f"Loaded {len(hygiene_index):,} hygiene records")
    print(f"Collected {len(occurrences):,} eligible CSI occurrences")
    if args.pilot_groups is not None:
        print(
            f"Pilot mode selected {len(selected_occurrences):,} occurrences "
            f"across {args.pilot_groups:,} source-span groups"
        )
    print(f"Encoding {len(unique_adapted_spans):,} unique adapted spans with {args.model_name}")

    embedding_lookup = encode_spans(
        unique_adapted_spans,
        model_name=args.model_name,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    dedup_rows, dedup_map, stats = build_outputs(
        selected_occurrences,
        embedding_lookup,
        threshold=args.threshold,
    )
    write_outputs(dedup_rows, dedup_map, pairs_out=args.pairs_out, map_out=args.map_out)
    print_summary(dedup_rows, stats)

if __name__ == "__main__":
    main()
