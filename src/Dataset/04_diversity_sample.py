"""
Diversity sampling for final benchmark curation.

Selects up to 1,500 culturally sensitive items per dataset with:
  1) Semantic diversity (embedding-space coverage via KMeans)
  2) Cultural richness (signal quality from the CSI detection pipeline)

Input:  03_data/processed/<dataset>/culturally_sensitive.jsonl
Output: 03_data/processed/<dataset>/final_3k_benchmark.jsonl
        03_data/processed/benchmark.jsonl  (combined across all datasets)

Selection algorithm:
  1. If N <= target_k, keep all items without clustering.
  2. Embed each text with sentence-transformers; L2-normalize.
  3. Run KMeans(K=target_k, random_state=fixed).
  4. Per cluster, score each item:
       P_i  = normalized cosine similarity to centroid (proximity)
       R_i  = mean of percentile ranks for:
                marker count, confidence ordinal, tier ordinal, unique NER entity count
       S_i  = lambda * P_i + (1 - lambda) * R_i
  5. Pick argmax(S_i) per cluster as representative.
  6. Deterministic maximin+quality backfill if empty clusters reduce total picks.

Fixed random_state guarantees identical output across runs.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "03_data"
PROC_DIR = DATA_DIR / "processed"

DEFAULT_DATASETS = ["gsm8k", "rocstories", "dailydialog", "recipenlg"]


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSONL rows from path."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to UTF-8 JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def confidence_to_score(conf: str) -> float:
    """Map confidence label to ordinal score for richness ranking."""
    conf = (conf or "").upper()
    if conf == "HIGH":
        return 3.0
    if conf == "MEDIUM":
        return 2.0
    return 1.0


def tier_to_score(tier_val) -> float:
    """Map resolution tier to ordinal score for richness ranking."""
    try:
        tier = int(tier_val)
    except Exception:
        tier = 1
    if tier == 0:
        return 3.0
    if tier == 1:
        return 2.0
    return 1.0


def count_markers(row: dict) -> int:
    """Return number of marker strings attached to a row."""
    return len(row.get("markers_found") or [])


def unique_ner_count(row: dict) -> int:
    """Count unique normalized NER entity surface forms."""
    ents = row.get("ner_entities") or []
    uniq = set()
    for e in ents:
        if isinstance(e, (list, tuple)) and len(e) >= 1:
            uniq.add(str(e[0]).strip().lower())
        elif isinstance(e, str):
            uniq.add(e.strip().lower())
    return len(uniq)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Percentile ranks in [0,1] with average ranks for tied values."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([1.0], dtype=np.float32)

    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return (ranks / (n - 1)).astype(np.float32)


def build_richness_scores(rows: list[dict]) -> np.ndarray:
    """Compute [0,1] richness scores from marker/confidence/tier/entity signals."""
    marker_counts = np.array([count_markers(r) for r in rows], dtype=np.float32)
    conf_scores = np.array([confidence_to_score(r.get("confidence", "")) for r in rows], dtype=np.float32)
    tier_scores = np.array([tier_to_score(r.get("resolution_tier", 1)) for r in rows], dtype=np.float32)
    ent_counts = np.array([unique_ner_count(r) for r in rows], dtype=np.float32)
    m_pr = percentile_ranks(marker_counts)
    c_pr = percentile_ranks(conf_scores)
    t_pr = percentile_ranks(tier_scores)
    e_pr = percentile_ranks(ent_counts)

    richness = (m_pr + c_pr + t_pr + e_pr) / 4.0
    return np.clip(richness, 0.0, 1.0).astype(np.float32)


def build_all_kept_rows(rows: list[dict], dataset: str) -> list[dict]:
    """Build deterministic metadata when candidate pool is already below target."""
    n = len(rows)
    richness_all = build_richness_scores(rows)
    out = []
    for idx, r in enumerate(rows):
        rc = dict(r)
        rc["domain"] = dataset
        rc["step3_selection_mode"] = "all_kept_below_target"
        rc["step3_cluster_id"] = -1
        rc["step3_cluster_size"] = n
        rc["step3_centroid_similarity"] = None
        rc["step3_proximity_score"] = None
        rc["step3_richness_score"] = float(richness_all[idx])
        rc["step3_selection_score"] = float(richness_all[idx])
        rc["step3_selected_by"] = "all_kept"
        out.append(rc)
    return out


def print_selected_distribution(selected_rows: list[dict]) -> None:
    """Print compact diagnostics for selected subset composition."""
    if not selected_rows:
        print("  Selected distribution: no rows", flush=True)
        return

    conf_counter = Counter((r.get("confidence") or "UNKNOWN").upper() for r in selected_rows)
    tier_counter = Counter(int(r.get("resolution_tier", -1)) for r in selected_rows)
    marker_counts = np.array([len(r.get("markers_found") or []) for r in selected_rows], dtype=np.float32)
    score_vals = np.array([float(r.get("step3_selection_score", 0.0)) for r in selected_rows], dtype=np.float32)

    print("  Selected distribution:", flush=True)
    print(
        f"    confidence  HIGH={conf_counter.get('HIGH', 0):,} "
        f"MEDIUM={conf_counter.get('MEDIUM', 0):,} "
        f"LOW={conf_counter.get('LOW', 0):,} "
        f"UNKNOWN={conf_counter.get('UNKNOWN', 0):,}",
        flush=True,
    )
    print(
        f"    resolution_tier  T0={tier_counter.get(0, 0):,} "
        f"T1={tier_counter.get(1, 0):,} T3={tier_counter.get(3, 0):,}",
        flush=True,
    )
    print(
        f"    marker_count  mean={marker_counts.mean():.2f} "
        f"p50={np.percentile(marker_counts, 50):.1f} "
        f"p90={np.percentile(marker_counts, 90):.1f}",
        flush=True,
    )
    print(
        f"    selection_score  mean={score_vals.mean():.4f} "
        f"min={score_vals.min():.4f} max={score_vals.max():.4f}",
        flush=True,
    )


def cluster_and_select(
    rows: list[dict],
    embeddings: np.ndarray,
    target_k: int,
    random_state: int,
    lambda_proximity: float,
    dataset: str,
) -> list[dict]:
    """Select cluster representatives using proximity-richness scoring."""
    n = len(rows)
    if n <= target_k:
        return build_all_kept_rows(rows, dataset)

    print(f"  Running KMeans with K={target_k:,} on N={n:,} items ...", flush=True)
    t0 = time.time()
    kmeans = KMeans(
        n_clusters=target_k,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    centers = normalize(centers)
    print(f"  KMeans completed in {time.time() - t0:.1f}s", flush=True)

    richness = build_richness_scores(rows)

    by_cluster = defaultdict(list)
    for idx, c in enumerate(labels):
        by_cluster[int(c)].append(idx)

    selected_idx = []
    selected_set = set()

    for c in range(target_k):
        members = by_cluster.get(c, [])
        if not members:
            continue

        center = centers[c]
        sims = embeddings[members] @ center
        smin = float(np.min(sims))
        smax = float(np.max(sims))
        denom = max(smax - smin, 1e-8)
        prox = (sims - smin) / denom

        member_rich = richness[members]
        score = lambda_proximity * prox + (1.0 - lambda_proximity) * member_rich

        local_best = int(np.argmax(score))
        best_idx = members[local_best]

        selected_idx.append(best_idx)
        selected_set.add(best_idx)

    if len(selected_idx) < target_k:
        need = target_k - len(selected_idx)
        print(f"  Backfill needed: {need} items (rare empty-cluster case)", flush=True)
        remaining = [i for i in range(n) if i not in selected_set]

        if selected_idx:
            selected_emb = embeddings[selected_idx]
            max_sim = np.max(embeddings[remaining] @ selected_emb.T, axis=1)
        else:
            max_sim = np.zeros(len(remaining), dtype=np.float32)

        novelty = 1.0 - max_sim
        backfill_score = 0.70 * novelty + 0.30 * richness[remaining]
        order = np.argsort(-backfill_score)
        for j in order[:need]:
            chosen = remaining[int(j)]
            selected_idx.append(chosen)
            selected_set.add(chosen)

    output = []
    selected_idx_sorted = sorted(selected_idx, key=lambda i: (int(labels[i]), i))
    cluster_sizes = {c: len(v) for c, v in by_cluster.items()}

    for i in selected_idx_sorted:
        c = int(labels[i])
        center = centers[c]
        sim = float(np.dot(embeddings[i], center))

        members = by_cluster[c]
        sims = embeddings[members] @ center
        smin = float(np.min(sims))
        smax = float(np.max(sims))
        prox = float((sim - smin) / max(smax - smin, 1e-8))

        rr = float(richness[i])
        ss = float(lambda_proximity * prox + (1.0 - lambda_proximity) * rr)

        row = dict(rows[i])
        row["domain"] = dataset
        row["step3_selection_mode"] = "kmeans_diversity_sampling"
        row["step3_cluster_id"] = c
        row["step3_cluster_size"] = int(cluster_sizes.get(c, 1))
        row["step3_centroid_similarity"] = round(sim, 6)
        row["step3_proximity_score"] = round(prox, 6)
        row["step3_richness_score"] = round(rr, 6)
        row["step3_selection_score"] = round(ss, 6)
        row["step3_selected_by"] = "cluster_representative"
        output.append(row)

    return output


def run_for_dataset(
    dataset: str,
    target_k: int,
    random_state: int,
    lambda_proximity: float,
    model_name: str,
    batch_size: int,
) -> Path:
    """Run full diversity sampling workflow for one dataset."""
    in_path = PROC_DIR / dataset / "culturally_sensitive.jsonl"
    out_path = PROC_DIR / dataset / "final_3k_benchmark.jsonl"

    print("\n" + "=" * 90, flush=True)
    print(f"DIVERSITY SAMPLING :: {dataset}", flush=True)
    print("=" * 90, flush=True)
    print(f"  Input:  {in_path}", flush=True)
    print(f"  Output: {out_path}", flush=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    t0 = time.time()
    rows = load_jsonl(in_path)
    n = len(rows)
    print(f"  Loaded {n:,} candidates", flush=True)
    print(
        f"  Selection objective: S_i = {lambda_proximity:.2f}*P_i + {1.0 - lambda_proximity:.2f}*R_i",
        flush=True,
    )

    if n == 0:
        print("  No candidates available; writing empty output.", flush=True)
        save_jsonl(out_path, [])
        return out_path

    if n <= target_k:
        print(
            f"  Candidate pool <= target ({n:,} <= {target_k:,}); keeping 100% without clustering.",
            flush=True,
        )
        selected = build_all_kept_rows(rows, dataset)
        save_jsonl(out_path, selected)
        print_selected_distribution(selected)
        print(f"  Saved {len(selected):,} items in {time.time() - t0:.1f}s", flush=True)
        return out_path

    texts = [str(r.get("original_text") or "") for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading embedding model: {model_name}", flush=True)
    print(f"  Embedding device: {device}", flush=True)
    model = SentenceTransformer(model_name, device=device)

    print("  Encoding texts ...", flush=True)
    t_embed = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = normalize(np.asarray(emb, dtype=np.float32))
    print(f"  Embedding completed in {time.time() - t_embed:.1f}s", flush=True)

    selected = cluster_and_select(
        rows=rows,
        embeddings=emb,
        target_k=target_k,
        random_state=random_state,
        lambda_proximity=lambda_proximity,
        dataset=dataset,
    )

    save_jsonl(out_path, selected)
    print(f"  Final selected: {len(selected):,}", flush=True)
    print_selected_distribution(selected)
    print(f"  Total time: {time.time() - t0:.1f}s", flush=True)
    return out_path


def main():
    """CLI entry point for diversity sampling across requested datasets."""
    parser = argparse.ArgumentParser(
        description="Diversity Sampling: KMeans + richness-aware representative selection"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k"],
        choices=DEFAULT_DATASETS,
        help="Datasets to process",
    )
    parser.add_argument("--target_k", type=int, default=1500, help="Target benchmark size per dataset")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--lambda_proximity",
        type=float,
        default=0.65,
        help="Weight for centroid proximity in final score S_i",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--embed_batch_size", type=int, default=128, help="Embedding batch size")

    args = parser.parse_args()

    print("\n" + "#" * 90, flush=True)
    print("DIVERSITY SAMPLING", flush=True)
    print("#" * 90, flush=True)
    print(f"  Datasets: {args.datasets}", flush=True)
    print(f"  target_k: {args.target_k}", flush=True)
    print(f"  random_state: {args.random_state}", flush=True)
    print(f"  lambda_proximity: {args.lambda_proximity}", flush=True)
    print(f"  embedding_model: {args.embedding_model}", flush=True)

    outputs = []
    for ds in args.datasets:
        outputs.append(
            run_for_dataset(
                dataset=ds,
                target_k=args.target_k,
                random_state=args.random_state,
                lambda_proximity=args.lambda_proximity,
                model_name=args.embedding_model,
                batch_size=args.embed_batch_size,
            )
        )

    print("\nCompleted diversity sampling for all requested datasets.", flush=True)
    all_rows = []
    for p in outputs:
        print(f"  - {p}", flush=True)
        if p.exists():
            all_rows.extend(load_jsonl(p))

    combined_path = PROC_DIR / "benchmark.jsonl"
    print(f"\nWriting combined benchmark to: {combined_path}", flush=True)
    save_jsonl(combined_path, all_rows)
    print(f"Total benchmark size: {len(all_rows):,} items", flush=True)


if __name__ == "__main__":
    main()
