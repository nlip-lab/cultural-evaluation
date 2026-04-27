"""
ROCStories sieve: reduces 98K stories to 8,960 high-quality candidates
optimized for cultural adaptation benchmarking.

Pipeline (5 stages):
  F1A: Structural — exactly 5 sentences
  F1B: Structural — 32-58 words
  F2:  Marker extraction — strong + contextual US cultural markers
  F3:  Adaptability tiering + mandatory cultural gate (tier >= 1)
  F4:  Deduplication — SHA1 fingerprint-based exact dedup
  F5:  Deterministic ranking — score-sorted top-N with seeded tie-breaking

Input:  03_data/raw/rocstories/rocstories.jsonl
Output: 03_data/processed/rocstories/rocstories_cultural_adaptation_8960_v3.jsonl
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b\w+(?:[-']\w+)?\b")


def sentence_count(text):
    """Count sentence-like segments using punctuation boundary splitting."""
    text = (text or "").strip()
    if not text:
        return 0
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return len(parts) if parts else 1


def word_count(text):
    """Count tokens using a conservative word regex."""
    return len(WORD_RE.findall(text or ""))


def seeded_tie_break(identifier, seed):
    """Return deterministic integer tie-break key from seed and identifier."""
    token = f"{seed}:{identifier}".encode("utf-8")
    return int(hashlib.md5(token).hexdigest()[:8], 16)


def histogram(counter, low, high):
    """Build an inclusive histogram slice as a serializable dict."""
    return {str(i): counter.get(i, 0) for i in range(low, high + 1)}


def build_patterns():
    """
    Build high-precision American cultural marker lexicon.

    Strong markers are distinctly US anchors unlikely to appear in non-US English
    stories. Contextual markers carry American social/institutional signal but may
    appear globally.

    Returns:
        (compiled_strong, compiled_contextual, strong_category_map,
         contextual_category_map, name_re)
    """
    strong_patterns = {
        "thanksgiving": r"\bthanksgiving\b",
        "halloween": r"\bhalloween\b",
        "fourth_of_july": r"\bfourth\s+of\s+july\b|\bindependence\s+day\b",
        "black_friday": r"\bblack\s+friday\b",
        "trick_or_treat": r"\btrick\s*(?:-| )?or\s*(?:-| )?treat\b",
        "super_bowl": r"\bsuper\s*bowl\b",
        "prom": r"\bprom\b",
        "homecoming": r"\bhomecoming\b",
        "walmart": r"\bwalmart\b",
        "target_store": r"\btarget\s+(?:store|superstore|supermarket)\b",
        "mcdonalds": r"\bmcdonalds?\b|\bmc\s*d(?:onald)?s?\b",
        "starbucks": r"\bstarbucks\b",
        "kfc": r"\bkfc\b",
        "costco": r"\bcostco\b",
        "best_buy": r"\bbest\s+buy\b",
        "hot_dog": r"\bhot\s*dogs?\b",
        "bbq": r"\bbbq\b|\bbarbecue\b",
        "peanut_butter": r"\bpeanut\s+butter\b",
        "apple_pie": r"\bapple\s+pie\b",
        "pecan_pie": r"\bpecan\s+pie\b",
        "baked_beans": r"\bbaked\s+beans\b",
        "potato_salad": r"\bpotato\s+salad\b",
        "coleslaw": r"\bcoleslaw\b",
        "deviled_eggs": r"\bdeviled\s+eggs\b",
        "corn_bread": r"\bcornbread\b",
        "smores": r"\bs['']?mores\b",
        "american_football": r"\bamerican\s+football\b",
        "quarterback": r"\bquarterback\b",
        "cheerleader": r"\bcheerleaders?\b",
        "baseball": r"\bbaseball\b",
        "nfl": r"\bnfl\b",
        "nba": r"\bnba\b",
        "nhl": r"\bnhl\b",
        "pickup_truck": r"\bpickup\s+truck\b",
        "lawnmower": r"\blawn\s*mowers?\b|\blawnmower\b",
        "suburb": r"\bsuburb(?:an|s)?\b",
        "dollar_word": r"\bdollars?\b|\bbucks\b",
        "dollar_symbol": r"\$",
    }

    contextual_patterns = {
        "christmas": r"\bchristmas\b",
        "easter": r"\beaster\b",
        "valentines_day": r"\bvalentine(?:'s)?\s+day\b",
        "mothers_day": r"\bmother(?:'s)?\s+day\b",
        "fathers_day": r"\bfather(?:'s)?\s+day\b",
        "labor_day": r"\blabor\s+day\b",
        "memorial_day": r"\bmemorial\s+day\b",
        "high_school": r"\bhigh\s+school\b",
        "middle_school": r"\bmiddle\s+school\b",
        "college": r"\bcollege\b",
        "sat_exam": r"\bSAT(?:\s+(?:exam|test|score))\b",
        "scholarship": r"\bscholarship\b",
        "dmv": r"\bdmv\b",
        "diner": r"\bdiner\b",
        "dollar_store": r"\bdollar\s+store\b",
        "campus": r"\bcampus\b",
        "dorm": r"\bdorm\b",
        "principal": r"\bprincipal\b",
        "coach": r"\bcoach\b",
        "burger": r"\bburgers?\b",
        "french_fries": r"\bfrench\s+fries\b|\bfries\b",
        "cubicle": r"\bcubicles?\b",
        "minimum_wage": r"\bminimum\s+wage\b",
        "paycheck": r"\bpaycheck\b",
        "tailgate": r"\btailgat(?:e|ing)\b",
        "report_card": r"\breport\s+card\b",
        "gpa": r"\bgpa\b",
        "freeway": r"\bfreeways?\b",
        "road_trip": r"\broad\s+trip\b",
        "country_music": r"\bcountry\s+(?:music|band|song)?\b",
        "hip_hop": r"\bhip\s+hops?\b",
        "tip_money": r"\bleft\s+(?:a\s+)?tip\b|\btip(?:ped|s|ping)?\s+(?:the\s+)?(?:waiter|waitress|server|bartender)\b",
        "jock": r"\bjocks?\b",
        "graduation": r"\bgraduation\b",
    }

    compiled_strong = {}
    for key, pattern in strong_patterns.items():
        if isinstance(pattern, tuple):
            expr, flags = pattern
            compiled_strong[key] = re.compile(expr, flags)
        else:
            compiled_strong[key] = re.compile(pattern, re.IGNORECASE)

    compiled_contextual = {}
    for key, pattern in contextual_patterns.items():
        if isinstance(pattern, tuple):
            expr, flags = pattern
            compiled_contextual[key] = re.compile(expr, flags)
        else:
            compiled_contextual[key] = re.compile(pattern, re.IGNORECASE)

    strong_category_map = {
        "holidays_traditions": {"thanksgiving", "halloween", "fourth_of_july", "black_friday",
                                "trick_or_treat", "super_bowl"},
        "education": {"prom", "homecoming"},
        "brands_retail": {"walmart", "target_store", "mcdonalds", "starbucks", "kfc", "costco", "best_buy"},
        "distinctive_food": {"hot_dog", "bbq", "peanut_butter", "apple_pie"},
        "sports_recreation": {"baseball", "american_football", "quarterback", "cheerleader", "nfl", "nba", "nhl"},
        "suburban_finance": {"pickup_truck", "lawnmower", "suburb", "dollar_word", "dollar_symbol"},
    }

    contextual_category_map = {
        "holidays_secondary": {"christmas", "easter", "valentines_day", "mothers_day", "fathers_day",
                               "labor_day", "memorial_day"},
        "education_institutions": {"high_school", "middle_school", "college", "sat_exam", "scholarship"},
        "institutions_places": {"dmv", "diner", "dollar_store", "campus", "dorm", "principal", "coach"},
        "education_artifacts": {"report_card", "gpa"},
        "work_economy": {"cubicle", "minimum_wage", "paycheck"},
        "cultural_food": {"burger", "french_fries"},
        "social_events": {"tailgate", "graduation"},
        "travel_entertainment": {"freeway", "road_trip", "country_music", "hip_hop"},
        "customs": {"tip_money", "jock"},
    }

    name_re = re.compile(
        r"\b(?:john|mary|james|robert|michael|william|david|richard|thomas|charles|"
        r"joseph|daniel|matthew|andrew|joshua|emily|emma|olivia|ava|sophia|isabella|"
        r"mia|abigail|madison|chris|mike|sarah|anna|kate|kevin|brian|lisa|jessica|"
        r"jennifer|ashley|kimberly|mark|donald|steven|paul|george|edward|ronald|"
        r"anthony|frank|ryan|gary|nicholas|eric|jonathan|stephen|larry|carlos|"
        r"michelle|carol|amanda|melissa|deborah|stephanie|rebecca|nancy)\b",
        re.IGNORECASE,
    )

    return compiled_strong, compiled_contextual, strong_category_map, contextual_category_map, name_re


def collect_distinct_hits(text, compiled_patterns):
    """Return pattern keys that appear at least once in text."""
    hits = set()
    for key, cre in compiled_patterns.items():
        if cre.search(text):
            hits.add(key)
    return hits


def count_occurrences(text, compiled_patterns):
    """Count all regex matches across compiled pattern values."""
    total = 0
    for cre in compiled_patterns.values():
        total += len(cre.findall(text))
    return total


def adaptability_tier(strong_count, contextual_count, total_occ, category_count):
    """
    Assign adaptation difficulty tier.

    Returns:
        3: high — strong US anchor + moderate density + category diversity
        2: medium — at least one marker with plausible adaptation load
        1: low — culturally marked but weakly grounded or overloaded
        0: neutral — no cultural anchors
    """
    if strong_count == 0 and contextual_count == 0:
        return 0
    if strong_count >= 1 and 1 <= total_occ <= 7 and 1 <= category_count <= 5:
        return 3
    if (strong_count + contextual_count) >= 1 and 1 <= total_occ <= 9 and category_count >= 1:
        return 2
    return 1


def analyze(input_path, output_json, export_jsonl, sample_size, random_seed, log_every):
    """
    Main sieve pipeline: load -> filter -> tier -> dedup -> rank -> select -> export.
    """
    compiled_strong, compiled_contextual, strong_category_map, contextual_category_map, name_re = build_patterns()

    print(f"[setup] input={input_path}")
    print(f"[setup] output_json={output_json}")
    print(f"[setup] export_jsonl={export_jsonl}")
    print(f"[setup] sample_size={sample_size}")
    print(f"[setup] random_seed={random_seed}")
    print("[setup] counting rows for progress logging...")
    with input_path.open("r", encoding="utf-8") as f_count:
        total_expected = sum(1 for _ in f_count)
    print(f"[setup] total_rows={total_expected:,}")

    total_rows = 0
    json_errors = 0

    split_counts = Counter()
    sentence_hist = Counter()
    word_hist = Counter()

    after_f1a_exact_sentences = 0
    after_f1b_word_window = 0
    rows_with_any_marker = 0
    rows_with_strong_marker = 0
    rows_with_contextual_only = 0
    rows_name_only_no_markers = 0

    tier_hist = Counter()
    survivors = []
    seen_fingerprints = set()
    dropped_duplicates = 0
    after_f3_min_tier_gate = 0
    after_f4_deduplication = 0

    strong_key_freq = Counter()
    contextual_key_freq = Counter()
    category_freq = Counter()

    start_time = time.time()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            total_rows += 1
            if total_rows % log_every == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * total_rows / total_expected if total_expected else 0.0
                print(f"[progress] processed={total_rows:,} ({pct:5.2f}%) rate={rate:,.0f} rows/s elapsed={elapsed:.1f}s")

            try:
                obj = json.loads(line)
            except Exception:
                json_errors += 1
                continue

            text = (obj.get("text") or "").strip()
            sc = sentence_count(text)
            wc = word_count(text)

            split_counts[obj.get("split") or "unknown"] += 1
            sentence_hist[sc] += 1
            word_hist[wc] += 1

            if sc != 5:
                continue
            after_f1a_exact_sentences += 1

            if not (32 <= wc <= 58):
                continue
            after_f1b_word_window += 1

            strong_hits = collect_distinct_hits(text, compiled_strong)
            contextual_hits = collect_distinct_hits(text, compiled_contextual)
            any_hits = strong_hits | contextual_hits

            if any_hits:
                rows_with_any_marker += 1
            if strong_hits:
                rows_with_strong_marker += 1
            if contextual_hits and not strong_hits:
                rows_with_contextual_only += 1

            has_name = bool(name_re.search(text))
            if has_name and not any_hits:
                rows_name_only_no_markers += 1

            distinct_categories = set()
            for cat, keys in strong_category_map.items():
                if strong_hits & keys:
                    distinct_categories.add(cat)
                    category_freq[cat] += 1
            for cat, keys in contextual_category_map.items():
                if contextual_hits & keys:
                    distinct_categories.add(cat)
                    category_freq[cat] += 1

            for m in strong_hits:
                strong_key_freq[m] += 1
            for m in contextual_hits:
                contextual_key_freq[m] += 1

            strong_occ = count_occurrences(text, {k: compiled_strong[k] for k in strong_hits})
            contextual_occ = count_occurrences(text, {k: compiled_contextual[k] for k in contextual_hits})
            total_occ = strong_occ + contextual_occ

            tier = adaptability_tier(
                strong_count=len(strong_hits),
                contextual_count=len(contextual_hits),
                total_occ=total_occ,
                category_count=len(distinct_categories),
            )
            tier_hist[tier] += 1

            if tier < 1:
                continue
            after_f3_min_tier_gate += 1

            fp = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if fp in seen_fingerprints:
                dropped_duplicates += 1
                continue
            seen_fingerprints.add(fp)
            after_f4_deduplication += 1

            score = (
                250 * tier
                + 200 * len(strong_hits)
                + 80 * len(contextual_hits)
                + 30 * len(distinct_categories)
                + min(strong_occ, 6)
                + min(contextual_occ, 4)
            )

            survivors.append({
                "line_idx": line_idx,
                "id": obj.get("id"),
                "tie_key": seeded_tie_break(obj.get("id") or line_idx, random_seed),
                "word_count": wc,
                "strong_marker_count": len(strong_hits),
                "contextual_marker_count": len(contextual_hits),
                "distinct_marker_count": len(any_hits),
                "distinct_category_count": len(distinct_categories),
                "strong_marker_occurrences": strong_occ,
                "contextual_marker_occurrences": contextual_occ,
                "marker_occurrences": total_occ,
                "adaptability_tier": tier,
                "has_common_name": has_name,
                "score": score,
                "obj": obj,
            })

    elapsed = time.time() - start_time

    print(f"[diagnostic] tier3={tier_hist.get(3, 0):,}  tier2={tier_hist.get(2, 0):,}  tier1={tier_hist.get(1, 0):,}  tier0={tier_hist.get(0, 0):,}")
    print(f"[diagnostic] tier2_plus={tier_hist.get(3, 0) + tier_hist.get(2, 0):,}")

    ranked = sorted(
        survivors,
        key=lambda r: (
            -r["score"],
            abs(r["word_count"] - 43),
            r["tie_key"],
            str(r["id"]),
        ),
    )

    selected = ranked[:sample_size]
    selected_count = len(selected)

    selected_tier_hist = Counter(r["adaptability_tier"] for r in selected)
    selected_with_any_marker = sum(1 for r in selected if r["distinct_marker_count"] > 0)
    selected_with_strong_marker = sum(1 for r in selected if r["strong_marker_count"] > 0)

    summary = {
        "input_file": str(input_path),
        "run_config": {
            "sample_size": sample_size,
            "random_seed": random_seed,
            "log_every": log_every,
            "word_window": [32, 58],
            "strong_marker_keys": len(compiled_strong),
            "contextual_marker_keys": len(compiled_contextual),
        },
        "counts": {
            "start": total_rows,
            "after_f1a_exact_sentences": after_f1a_exact_sentences,
            "after_f1b_word_window": after_f1b_word_window,
            "after_f2_with_any_marker": rows_with_any_marker,
            "after_f2_with_strong_marker": rows_with_strong_marker,
            "after_f2_contextual_only": rows_with_contextual_only,
            "names_only_no_markers": rows_name_only_no_markers,
            "tier_3_high": tier_hist.get(3, 0),
            "tier_2_medium": tier_hist.get(2, 0),
            "tier_1_low": tier_hist.get(1, 0),
            "tier_0_neutral": tier_hist.get(0, 0),
            "after_f3_min_tier_gate": after_f3_min_tier_gate,
            "after_f4_deduplication": after_f4_deduplication,
            "after_f5_selection": selected_count,
        },
        "quality_checks": {
            "json_errors": json_errors,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_rows_per_sec": round(total_rows / elapsed, 2) if elapsed > 0 else None,
            "selected_exact_target": selected_count == sample_size,
            "dropped_duplicates": dropped_duplicates,
        },
        "selected_quality": {
            "selected_with_any_marker": selected_with_any_marker,
            "selected_with_strong_marker": selected_with_strong_marker,
            "avg_strong_markers": round(sum(r["strong_marker_count"] for r in selected) / selected_count, 2) if selected_count else 0.0,
            "avg_contextual_markers": round(sum(r["contextual_marker_count"] for r in selected) / selected_count, 2) if selected_count else 0.0,
        },
        "marker_coverage": {
            "distinct_strong_marker_keys": len(strong_key_freq),
            "distinct_contextual_marker_keys": len(contextual_key_freq),
            "top_20_strong_marker_keys": strong_key_freq.most_common(20),
            "top_20_contextual_marker_keys": contextual_key_freq.most_common(20),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[export] wrote summary: {output_json}")

    export_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with export_jsonl.open("w", encoding="utf-8") as fout:
        for r in selected:
            fout.write(json.dumps(r["obj"]) + "\n")
    print(f"[export] wrote selected JSONL: {export_jsonl}")

    print("\n" + "=" * 80)
    print("ROCStories Sieve - COMPLETE")
    print("=" * 80)
    print(f"[done] start={total_rows:,}")
    print(f"[done] after_f1a_exact_sentences={after_f1a_exact_sentences:,} ({100.0*after_f1a_exact_sentences/total_rows:.2f}%)")
    print(f"[done] after_f1b_word_window_32_58={after_f1b_word_window:,} ({100.0*after_f1b_word_window/total_rows:.2f}%)")
    print(f"[done] after_f3_min_tier_gate={after_f3_min_tier_gate:,} ({100.0*after_f3_min_tier_gate/total_rows:.2f}%)")
    print(f"[done] after_f4_deduplication={after_f4_deduplication:,} ({100.0*after_f4_deduplication/total_rows:.2f}%)")
    print(f"[done] selected_exact={selected_count:,}")
    print(f"[done] elapsed={elapsed:.2f}s throughput={total_rows/elapsed:.0f} rows/s")
    print("=" * 80)


def parse_args():
    """Parse CLI arguments for ROCStories sieve execution."""
    parser = argparse.ArgumentParser(description="ROCStories cultural adaptation sieve")
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "03_data" / "raw" / "rocstories" / "rocstories.jsonl",
        help="Path to ROCStories JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "04_experiments" / "analysis" / "rocstories_sieve_summary.json",
        help="Path to write summary JSON.",
    )
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        default=_ROOT / "03_data" / "processed" / "rocstories" / "rocstories_cultural_adaptation_8960_v3.jsonl",
        help="Path to write selected stories as JSONL.",
    )
    parser.add_argument("--sample-size", type=int, default=8960)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.output, args.export_jsonl, args.sample_size, args.random_seed, args.log_every)
