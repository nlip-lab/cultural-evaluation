"""
RecipeNLG sieve: reduces 2.2M recipes to 50,000 high-quality candidates
optimized for cultural adaptation benchmarking.

Pipeline (8 stages):
  F1A: Structural — exactly 5-8 directions
  F1B: Structural — exactly 5-10 ingredients
  F1C: Linguistic — max 40 words per direction step
  F2:  Physics mandatory — at least 2 distinct US/Imperial unit markers
  F3:  Entity mandatory — at least 1 distinct American cultural entity marker
  F4:  Density control — at most 50 total marker occurrences
  F5:  Deduplication — SHA1 fingerprint-based exact dedup
  F6:  Deterministic VIP ranking — score-sorted selection with seeded tie-breaking

Input:  03_data/raw/recipenlg/recipenlg.jsonl
Output: 03_data/processed/recipenlg/recipenlg_cultural_adaptation_50k_v3.jsonl
"""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def build_patterns():
    """
    Build Western/American culinary markers organized by adaptation challenge type.

    Physics patterns capture US/Imperial measurement and temperature signals that
    force quantitative adaptation (cups→ml, pounds→grams, Fahrenheit→Celsius).
    Entity patterns capture distinctive US cultural/brand/holiday anchors.

    Returns:
        (compiled_physics, compiled_entities, physics_category_map, entity_category_map)
    """
    physics_patterns = {
        "cup": r"\bcups?\b",
        "fluid_ounce": r"\bfl\.?\s*oz\.?\b",
        "quart": r"\bquarts?\b|\bqt\.?\b",
        "gallon": r"\bgallons?\b|\bgal\.?\b",
        "pint": r"\bpints?\b|\bpt\.?\b",
        "ounce": r"\b(?:oz\.?|ounces?)\b",
        "pound": r"\b(?:lb\.?|lbs\.?|pounds?)\b",
        "fahrenheit": r"\b(?:fahrenheit|°F)\b",
        "degree_symbol": r"(?<!\w)°(?!C)",
    }

    entity_patterns = {
        "thanksgiving": r"\bthanksgiving\b",
        "halloween": r"\bhalloween\b",
        "fourth_of_july": r"\bfourth\s+of\s+july\b|\bindependence\s+day\b",
        "black_friday": r"\bblack\s+friday\b",
        "super_bowl": r"\bsuper\s*bowl\b",
        "mac_and_cheese": r"\bmac(?:\s+and)?\s+cheese\b",
        "casserole": r"\bcasseroles?\b",
        "meatloaf": r"\bmeatloaf\b",
        "hamburger": r"\bhamburgers?\b",
        "hot_dog": r"\bhot\s*dogs?\b",
        "buffalo_wings": r"\bbuffalo\s+wings?\b",
        "sloppy_joe": r"\bsloppy\s+joes?\b",
        "bbq": r"\bbbq\b|\bbarbecue\b",
        "fried_chicken": r"\bfried\s+chicken\b",
        "brownies": r"\bbrownies?\b",
        "peach_cobbler": r"\bpeach\s+cobbler\b",
        "apple_pie": r"\bapple\s+pie\b",
        "pumpkin_pie": r"\bpumpkin\s+pie\b",
        "pecan_pie": r"\bpecan\s+pie\b",
        "baked_beans": r"\bbaked\s+beans\b",
        "potato_salad": r"\bpotato\s+salad\b",
        "coleslaw": r"\bcoleslaw\b",
        "deviled_eggs": r"\bdeviled\s+eggs\b",
        "corn_bread": r"\bcornbread\b",
        "smores": r"\bs['']?mores\b",
        "walmart": r"\bwalmart\b",
        "target": r"\btarget\s+(?:store|superstore|supermarket)\b",
        "mcdonalds": r"\bmcdonalds?\b",
        "starbucks": r"\bstarbucks\b",
        "kfc": r"\bkfc\b",
        "bisquick": r"\bbisquick\b",
        "jell_o": r"\bjell[\s-]?o\b",
        "cool_whip": r"\bcool\s+whip\b",
        "campbells": r"\bcampbell(?:'s)?\b",
        "kraft": r"\bkraft\b",
        "kraft_mac_cheese": r"\bkraft\s+mac\s*(?:and|&)\s+cheese\b",
        "pillsbury": r"\bpillsbury\b",
        "betty_crocker": r"\bbetty\s+crocker\b",
        "velveeta": r"\bvelveeta\b",
        "crisco": r"\bcrisco\b",
        "hellmanns": r"\bhellmann(?:'s)?\b",
        "libbys_pumpkin": r"\blibby(?:'s)?\s+pumpkin\b",
        "cream_cheese": r"\bcream\s+cheese\b",
        "sour_cream": r"\bsour\s+cream\b",
        "peanut_butter": r"\bpeanut\s+butter\b",
        "marshmallow": r"\bmarshmallows?\b",
        "graham_cracker": r"\bgraham\s+(?:cracker|crumb)s?\b",
        "chocolate_chip": r"\bchocolate\s+chips?\b",
        "maple_syrup": r"\bmaple\s+syrup\b",
        "pecan": r"\bpecans?\b|\bpecan\b",
        "american_football": r"\bamerican\s+football\b|\bnfl\b|\bquarterback\b|\btouchdown\b",
        "baseball": r"\bbaseball\b|\bmlb\b",
        "tailgate": r"\btailgat(?:e|ing)\b",
        "pickup_truck": r"\bpickup\s+truck\b",
        "high_school": r"\bhigh\s+school\b",
        "sat_exam": r"\bSAT(?:\s+(?:exam|test|score))\b",
        "home_ec": r"\bhome\s+ec\b",
        "usd_amount": r"\busd\s*\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*usd\b|\$\s*\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*dollars?\b",
        "diner": r"\bdiner\b",
        "dollar_store": r"\bdollar\s+store\b",
        "us_reference": r"\b(?:united\s+states|u\.s\.?a?\.?|usa|american)\b",
        "grits": r"\bgrits\b",
        "collard_greens": r"\bcollard\s+greens\b",
        "biscuits": r"\bbuttermilk\s+biscuits?\b",
        "pulled_pork": r"\bpulled\s+pork\b",
        "brisket": r"\bbrisket\b",
        "catfish": r"\bcatfish\b",
        "crawfish": r"\bcrawfish\b",
        "shrimp_boil": r"\bshrimp\s+boil\b",
        "fried_okra": r"\bfried\s+okra\b",
        "black_eyed_peas": r"\bblack[\s-]?eyed\s+peas\b",
        "soul_food": r"\bsoul\s+food\b",
        "cajun": r"\bcajun\b",
        "creole": r"\bcreole\b",
        "eggnog": r"\beggnog\b",
        "sweet_tea": r"\bsweet\s+tea\b",
        "fried_green_tomatoes": r"\bfried\s+green\s+tomatoes\b",
        "texas_sheet_cake": r"\btexas\s+sheet\s+cake\b",
        "red_velvet": r"\bred\s+velvet\b",
        "cornbread_stuffing": r"\bcornbread\s+stuffing\b",
        "green_bean_casserole": r"\bgreen\s+bean\s+casserole\b",
        "candied_yams": r"\bcandied\s+yams\b",
        "sweet_potato_pie": r"\bsweet\s+potato\s+pie\b",
        "nilla_wafers": r"\bnilla\s+wafers\b",
    }

    compiled_physics = {
        key: re.compile(pattern, re.IGNORECASE) for key, pattern in physics_patterns.items()
    }

    compiled_entities = {
        key: re.compile(pattern, re.IGNORECASE) for key, pattern in entity_patterns.items()
    }

    physics_category_map = {
        "volume_measurements": {"cup", "fluid_ounce", "quart", "gallon", "pint"},
        "weight_measurements": {"ounce", "pound"},
        "temperature": {"fahrenheit", "degree_symbol"},
    }

    entity_category_map = {
        "us_holidays_events": {"thanksgiving", "halloween", "fourth_of_july", "black_friday", "super_bowl"},
        "us_dishes": {"mac_and_cheese", "casserole", "meatloaf", "hamburger", "hot_dog", "buffalo_wings",
                      "sloppy_joe", "bbq", "fried_chicken", "apple_pie", "pumpkin_pie", "pecan_pie",
                      "smores", "peach_cobbler", "deviled_eggs"},
        "us_sides_salads": {"potato_salad", "coleslaw", "baked_beans", "corn_bread"},
        "us_desserts": {"brownies", "texas_sheet_cake", "red_velvet"},
        "us_brands_products": {"walmart", "target", "mcdonalds", "starbucks", "kfc", "bisquick", "jell_o",
                               "cool_whip", "campbells", "kraft", "kraft_mac_cheese", "pillsbury",
                               "betty_crocker", "velveeta", "crisco", "hellmanns", "libbys_pumpkin", "nilla_wafers"},
        "us_ingredient_conventions": {"cream_cheese", "sour_cream", "peanut_butter", "marshmallow",
                                      "graham_cracker", "chocolate_chip", "maple_syrup", "pecan", "eggnog"},
        "us_sports_suburban_cues": {"american_football", "baseball", "tailgate", "pickup_truck"},
        "us_institutions_currency": {"diner", "dollar_store", "usd_amount", "us_reference", "high_school",
                                     "sat_exam", "home_ec"},
        "us_regional_cuisine": {"grits", "collard_greens", "biscuits", "pulled_pork", "brisket", "catfish",
                                "crawfish", "shrimp_boil", "fried_okra", "black_eyed_peas", "soul_food",
                                "cajun", "creole", "sweet_tea", "fried_green_tomatoes", "green_bean_casserole",
                                "candied_yams", "sweet_potato_pie", "cornbread_stuffing"},
    }

    return compiled_physics, compiled_entities, physics_category_map, entity_category_map


def collect_distinct_hits(text, compiled_patterns):
    """Extract distinct marker keys found in text."""
    hits = set()
    for key, cre in compiled_patterns.items():
        if cre.search(text):
            hits.add(key)
    return hits


def count_occurrences(text, compiled_patterns):
    """Count total marker occurrences in text."""
    total = 0
    for cre in compiled_patterns.values():
        total += len(cre.findall(text))
    return total


def recipe_fingerprint(title, ingredients, directions):
    """Generate deterministic SHA1 fingerprint for deduplication."""
    normalized = lambda s: re.sub(r"\s+", " ", (s or "").lower().strip())
    key = "||".join([
        normalized(title),
        "|".join(normalized(x) for x in (ingredients or [])),
        "|".join(normalized(x) for x in (directions or [])),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def seeded_tie_break(identifier, seed):
    """Generate seeded MD5-based tie-break value for deterministic ranking."""
    token = f"{seed}:{identifier}".encode("utf-8")
    return int(hashlib.md5(token).hexdigest()[:8], 16)


def analyze(input_path, output_json, export_jsonl, sample_size, random_seed, log_every):
    """
    Main sieve pipeline: filter -> rank -> select -> export.

    Applies 8 sequential stages (F1A through F6) to produce a deterministic
    sample of recipes with measurable US cultural adaptation challenges.
    """
    compiled_physics, compiled_entities, physics_category_map, entity_category_map = build_patterns()

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
    after_f1a = 0
    after_f1b = 0
    after_f1c = 0
    after_f2_physics = 0
    after_f3_entity = 0
    after_f4_density = 0
    after_f5_dedup = 0

    survivors = []
    seen_fingerprints = set()
    dropped_duplicates = 0

    physics_freq = Counter()
    entity_freq = Counter()
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

            directions = [str(x).strip() for x in (obj.get("directions") or []) if x]
            ingredients = [str(x).strip() for x in (obj.get("ingredients") or []) if x]

            if not (5 <= len(directions) <= 8):
                continue
            after_f1a += 1

            if not (5 <= len(ingredients) <= 10):
                continue
            after_f1b += 1

            max_step_words = max(
                (len(re.findall(r"\S+", step)) for step in directions),
                default=0
            )
            if max_step_words > 40:
                continue
            after_f1c += 1

            search_text = "\n".join(
                [obj.get("title") or "", " ".join(ingredients), " ".join(directions)]
            )

            physics_hits = collect_distinct_hits(search_text, compiled_physics)
            if len(physics_hits) < 2:
                continue
            after_f2_physics += 1

            entity_hits = collect_distinct_hits(search_text, compiled_entities)
            if len(entity_hits) < 1:
                continue
            after_f3_entity += 1

            physics_occ = count_occurrences(search_text, {k: compiled_physics[k] for k in physics_hits})
            entity_occ = count_occurrences(search_text, {k: compiled_entities[k] for k in entity_hits})
            total_occ = physics_occ + entity_occ

            if total_occ > 50:
                continue
            after_f4_density += 1

            fp = recipe_fingerprint(obj.get("title"), ingredients, directions)
            if fp in seen_fingerprints:
                dropped_duplicates += 1
                continue
            seen_fingerprints.add(fp)
            after_f5_dedup += 1

            for m in physics_hits:
                physics_freq[m] += 1
            for m in entity_hits:
                entity_freq[m] += 1

            distinct_categories = set()
            for cat, keys in physics_category_map.items():
                if physics_hits & keys:
                    category_freq[cat] += 1
                    distinct_categories.add(cat)
            for cat, keys in entity_category_map.items():
                if entity_hits & keys:
                    category_freq[cat] += 1
                    distinct_categories.add(cat)

            score = (
                200 * len(entity_hits)
                + 140 * len(physics_hits)
                + 50 * len(distinct_categories)
                + entity_occ
                + physics_occ
            )

            survivors.append({
                "line_idx": line_idx,
                "id": obj.get("id"),
                "tie_key": seeded_tie_break(obj.get("id") or line_idx, random_seed),
                "physics_types": len(physics_hits),
                "entity_types": len(entity_hits),
                "physics_occ": physics_occ,
                "entity_occ": entity_occ,
                "total_occ": total_occ,
                "categories": len(distinct_categories),
                "directions_count": len(directions),
                "ingredients_count": len(ingredients),
                "score": score,
                "obj": obj,
            })

    elapsed = time.time() - start_time

    ranked = sorted(
        survivors,
        key=lambda r: (
            -r["score"],
            abs(r["directions_count"] - 6),
            abs(r["ingredients_count"] - 7),
            r["tie_key"],
            str(r["id"]),
        ),
    )

    if len(ranked) < sample_size:
        raise ValueError(
            f"Only {len(ranked):,} survivors after filtering; cannot meet target sample_size={sample_size:,}."
        )

    selected = ranked[:sample_size]
    selected_count = len(selected)

    selected_physics_stats = {
        "any": sum(1 for r in selected if r["physics_types"] > 0),
        "avg_types": round(sum(r["physics_types"] for r in selected) / selected_count, 2) if selected_count else 0,
        "avg_occ": round(sum(r["physics_occ"] for r in selected) / selected_count, 2) if selected_count else 0,
    }

    selected_entity_stats = {
        "any": sum(1 for r in selected if r["entity_types"] > 0),
        "avg_types": round(sum(r["entity_types"] for r in selected) / selected_count, 2) if selected_count else 0,
        "avg_occ": round(sum(r["entity_occ"] for r in selected) / selected_count, 2) if selected_count else 0,
    }

    monotonic_ok = (
        total_rows >= after_f1a >= after_f1b >= after_f1c >= after_f2_physics
        >= after_f3_entity >= after_f4_density >= after_f5_dedup >= selected_count
    )

    summary = {
        "configuration": {
            "input": str(input_path),
            "sample_size": sample_size,
            "random_seed": random_seed,
            "output_jsonl": str(export_jsonl),
            "output_json": str(output_json),
        },
        "filter_logic": {
            "F1A": "5-8 directions",
            "F1B": "5-10 ingredients",
            "F1C": "max 40 words/step",
            "F2": ">=2 distinct physics markers",
            "F3": ">=1 distinct entity marker",
            "F4": "<=50 total marker occurrences",
            "F5": "SHA1 deduplication",
            "F6": "VIP ranking: score>balance>seed>id",
        },
        "stage_counts": {
            "start": total_rows,
            "after_f1a_directions_5_to_8": after_f1a,
            "after_f1b_ingredients_5_to_10": after_f1b,
            "after_f1c_max_40_words_per_step": after_f1c,
            "after_f2_physics_2_plus": after_f2_physics,
            "after_f3_entity_mandatory_1_plus": after_f3_entity,
            "after_f4_density_up_to_50": after_f4_density,
            "after_f5_deduplication": after_f5_dedup,
            "selected_f6_top_n": selected_count,
        },
        "quality_checks": {
            "json_errors": json_errors,
            "dropped_duplicates": dropped_duplicates,
            "monotonic_counts_ok": monotonic_ok,
            "elapsed_seconds": round(elapsed, 2),
        },
        "selected_quality": {
            "physics": selected_physics_stats,
            "entities": selected_entity_stats,
            "avg_directions": round(sum(r["directions_count"] for r in selected) / selected_count, 2) if selected_count else 0,
            "avg_ingredients": round(sum(r["ingredients_count"] for r in selected) / selected_count, 2) if selected_count else 0,
        },
        "marker_coverage": {
            "distinct_physics_keys_used": len(physics_freq),
            "distinct_entity_keys_used": len(entity_freq),
            "top_15_physics_markers": physics_freq.most_common(15),
            "top_15_entity_markers": entity_freq.most_common(15),
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

    print("\n" + "="*80)
    print("RecipeNLG Sieve - COMPLETE")
    print("="*80)
    print(f"[done] Total input rows: {total_rows:,}")
    print(f"[done] F1A (5-8 directions): {after_f1a:,} ({100.0*after_f1a/total_rows:.2f}%)")
    print(f"[done] F1B (5-10 ingredients): {after_f1b:,} ({100.0*after_f1b/total_rows:.2f}%)")
    print(f"[done] F1C (max 40 words/step): {after_f1c:,} ({100.0*after_f1c/total_rows:.2f}%)")
    print(f"[done] F2 (>=2 physics markers): {after_f2_physics:,} ({100.0*after_f2_physics/total_rows:.2f}%)")
    print(f"[done] F3 (>=1 entity marker): {after_f3_entity:,} ({100.0*after_f3_entity/total_rows:.2f}%)")
    print(f"[done] F4 (<=50 marker occ): {after_f4_density:,} ({100.0*after_f4_density/total_rows:.2f}%)")
    print(f"[done] F5 (after dedup): {after_f5_dedup:,} ({100.0*after_f5_dedup/total_rows:.2f}%)")
    print(f"[done] F6 (final selected): {selected_count:,} ({100.0*selected_count/total_rows:.2f}%)")
    print(f"[done] Elapsed: {elapsed:.2f}s")
    print("="*80)


def parse_args():
    """Parse CLI arguments for RecipeNLG sieve execution."""
    parser = argparse.ArgumentParser(description="RecipeNLG cultural adaptation sieve")
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "03_data" / "raw" / "recipenlg" / "recipenlg.jsonl",
        help="Path to RecipeNLG JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "04_experiments" / "analysis" / "recipe_sieve_summary.json",
        help="Path to write summary JSON.",
    )
    parser.add_argument(
        "--export-jsonl",
        type=Path,
        default=_ROOT / "03_data" / "processed" / "recipenlg" / "recipenlg_cultural_adaptation_50k_v3.jsonl",
        help="Path to write selected recipes JSONL.",
    )
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=300000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.output, args.export_jsonl, args.sample_size, args.random_seed, args.log_every)
