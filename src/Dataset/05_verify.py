"""
Benchmark verification: validates domain counts, samples items by tier,
inspects marker distributions, and runs targeted spot-checks on the
final benchmark.jsonl produced by 04_diversity_sample.py.
"""

import json
import random
from collections import Counter
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[3] / "03_data" / "processed" / "benchmark.jsonl"
random.seed(42)


def load(path):
    """Load non-empty JSONL rows from disk."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def section(title):
    """Print a top-level report section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def subsection(title):
    """Print a subsection header."""
    print(f"\n  -- {title} {'-'*(60 - len(title))}")


def main():
    """Run verification checks and print a human-readable audit report."""
    rows = load(BENCHMARK)

    domains = sorted(set(r.get("domain", "?") for r in rows))
    print(f"Actual domain values in file: {domains}")
    print(f"Total: {len(rows):,}\n")

    section("A. DOMAIN COUNTS")
    for d in domains:
        count = sum(1 for r in rows if r.get("domain") == d)
        print(f"    {d:<20s}: {count:,}")

    section("B. RANDOM SAMPLES (3 per domain x per tier)")
    for domain in domains:
        subset = [r for r in rows if r.get("domain") == domain]
        for tier in [0, 1, 3]:
            tier_subset = [r for r in subset if r.get("resolution_tier") == tier]
            if not tier_subset:
                continue
            samples = random.sample(tier_subset, min(3, len(tier_subset)))
            subsection(f"{domain} / Tier {tier}  (pool={len(tier_subset):,})")
            for s in samples:
                text = (s.get("original_text") or "")
                text_preview = " ".join(text.split())[:300]
                markers = s.get("markers_found") or []
                markers_str = ", ".join(str(m) for m in markers[:10])
                if len(markers) > 10:
                    markers_str += f" ... (+{len(markers)-10} more)"
                print(f"      ID: {s.get('id','?')}")
                print(f"      Text: {text_preview}")
                print(f"      Markers: [{markers_str}]")
                print(f"      Confidence: {s.get('confidence','?')}  |  "
                      f"Score: {s.get('step3_selection_score','?')}")
                if tier == 3:
                    print(f"      Tier1 verdict: {s.get('tier1_verdict','?')} -> "
                          f"Final: {s.get('culturally_sensitive','?')}")
                    print(f"      Trigger: {s.get('tier3_trigger_reasons','?')}")
                print()

    section("C. TOP 3 ITEMS BY MARKER COUNT (per domain)")
    for domain in domains:
        subset = [r for r in rows if r.get("domain") == domain]
        subset.sort(key=lambda r: len(r.get("markers_found") or []), reverse=True)
        subsection(f"{domain} -- top 3 by marker count")
        for s in subset[:3]:
            text = " ".join((s.get("original_text") or "").split())[:250]
            markers = s.get("markers_found") or []
            print(f"      [{len(markers)} markers] {text}")
            print(f"      Markers: {markers[:12]}")
            print()

    section("D. EMPTY-MARKER ITEMS")
    empty_marker_items = [r for r in rows if not (r.get("markers_found") or [])]
    print(f"  Total items with empty markers_found: {len(empty_marker_items)}")
    for r in empty_marker_items:
        text = " ".join((r.get("original_text") or "").split())[:200]
        print(f"\n    ID: {r.get('id','?')}  domain={r.get('domain','?')}  "
              f"tier={r.get('resolution_tier','?')}  conf={r.get('confidence','?')}")
        print(f"    Text: {text}")
        print(f"    Reasoning: {(r.get('reasoning') or '')[:200]}")

    section("E. CROSS-DOMAIN MARKER OVERLAP")
    domain_markers = {}
    for domain in domains:
        subset = [r for r in rows if r.get("domain") == domain]
        markers = set()
        for r in subset:
            for m in (r.get("markers_found") or []):
                markers.add(str(m).lower().strip())
        domain_markers[domain] = markers
        print(f"    {domain:<20s}: {len(markers):,} unique marker strings")

    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            d1, d2 = domains[i], domains[j]
            overlap = domain_markers[d1] & domain_markers[d2]
            print(f"    {d1} vs {d2}: {len(overlap):,} shared")
            if overlap:
                print(f"      examples: {list(overlap)[:10]}")

    section("F. SELECTION SCORE INVESTIGATION")
    for domain in domains:
        subset = [r for r in rows if r.get("domain") == domain]
        subset.sort(key=lambda r: r.get("step3_selection_score", 0))
        worst = subset[0]
        best = subset[-1]

        subsection(f"{domain}")
        print(f"    LOWEST score: {worst.get('step3_selection_score')}")
        print(f"      richness={worst.get('step3_richness_score')}  "
              f"proximity={worst.get('step3_proximity_score')}  "
              f"cluster_size={worst.get('step3_cluster_size')}")
        text = " ".join((worst.get("original_text") or "").split())[:200]
        print(f"      Text: {text}")

        print(f"    HIGHEST score: {best.get('step3_selection_score')}")
        print(f"      richness={best.get('step3_richness_score')}  "
              f"proximity={best.get('step3_proximity_score')}  "
              f"cluster_size={best.get('step3_cluster_size')}")
        text = " ".join((best.get("original_text") or "").split())[:200]
        print(f"      Text: {text}")

    print("\n" + "=" * 80)
    print("  END OF VERIFICATION REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
