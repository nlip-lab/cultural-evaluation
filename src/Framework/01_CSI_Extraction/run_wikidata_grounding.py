"""Phase 1.3 lightweight Wikidata grounding on deduplicated CSI spans."""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR = EVAL_DIR / "01_CSI_Extraction"
IN_PATH = PHASE_DIR / "source_csi_with_functions_google.jsonl"
OUT_PATH = PHASE_DIR / "source_csi_final_google.jsonl"

TARGET_CATEGORIES = {
    "place_institution_brand",
    "holiday_event_ritual",
    "pop_culture_sport_media",
}

NULL_GROUND = {
    "wikidata_qid": None,
    "wikidata_desc": None,
    "wikidata_type": None,
}

USER_AGENT = "CulturalEvaluationFramework/1.0"
MIN_DELAY_SEC = 0.10
ACCEPT_THRESHOLD = 0.88

_last_api_ts = 0.0
_p31_cache: dict[str, str | None] = {}
_label_cache: dict[str, str | None] = {}


def _sleep_for_rate_limit():
    """Throttle API calls to a fixed minimum delay."""
    global _last_api_ts
    now = time.time()
    delta = now - _last_api_ts
    if delta < MIN_DELAY_SEC:
        time.sleep(MIN_DELAY_SEC - delta)
    _last_api_ts = time.time()


def _http_get_json(url: str) -> dict | None:
    """Fetch JSON with retry logic for transient failures."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        _sleep_for_rate_limit()
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            if exc.code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1 + attempt)
    return None


def _normalize(text: str) -> str:
    """Normalize text for deduplication and fuzzy matching."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _candidate_score(span: str, candidate: dict) -> float:
    """Compute confidence score for candidate entity match."""
    source = _normalize(span)
    label = _normalize(candidate.get("label", ""))
    if not source or not label:
        return 0.0

    if source == label:
        score = 1.0
    else:
        ratio = SequenceMatcher(None, source, label).ratio()
        overlap = len(set(source.split()) & set(label.split())) / max(len(set(source.split())), 1)
        near_match = source in label or label in source
        if near_match and min(len(source), len(label)) / max(len(source), len(label)) >= 0.75:
            score = max(0.92, ratio)
        else:
            score = 0.7 * ratio + 0.3 * overlap

    if candidate.get("match", {}).get("type") == "alias":
        score -= 0.03
    return max(0.0, min(1.0, score))


def _search_wikidata(span: str) -> list[dict]:
    """Search candidate entities for a span."""
    query = urllib.parse.quote(span)
    url = (
        "https://www.wikidata.org/w/api.php"
        "?action=wbsearchentities"
        f"&search={query}"
        "&language=en"
        "&format=json"
        "&limit=5"
    )
    data = _http_get_json(url)
    if not data:
        return []
    return data.get("search", [])


def _get_p31_qid(entity_qid: str) -> str | None:
    """Fetch first P31 target QID for an entity."""
    if entity_qid in _p31_cache:
        return _p31_cache[entity_qid]

    url = (
        "https://www.wikidata.org/w/api.php"
        "?action=wbgetentities"
        f"&ids={urllib.parse.quote(entity_qid)}"
        "&props=claims"
        "&format=json"
    )
    data = _http_get_json(url) or {}
    entity = (data.get("entities") or {}).get(entity_qid) or {}
    claims = entity.get("claims") or {}
    p31_claims = claims.get("P31") or []

    p31_qid = None
    for claim in p31_claims:
        value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
        if isinstance(value, dict) and value.get("id"):
            p31_qid = value["id"]
            break

    _p31_cache[entity_qid] = p31_qid
    return p31_qid


def _get_label(qid: str | None) -> str | None:
    """Fetch English label for a QID."""
    if not qid:
        return None
    if qid in _label_cache:
        return _label_cache[qid]

    url = (
        "https://www.wikidata.org/w/api.php"
        "?action=wbgetentities"
        f"&ids={urllib.parse.quote(qid)}"
        "&props=labels"
        "&languages=en"
        "&format=json"
    )
    data = _http_get_json(url) or {}
    entity = (data.get("entities") or {}).get(qid) or {}
    labels = entity.get("labels") or {}
    label = ((labels.get("en") or {}).get("value")) or None
    _label_cache[qid] = label
    return label


def ground_span(span: str) -> dict:
    """Ground one span to Wikidata and return normalized grounding fields."""
    candidates = _search_wikidata(span)
    if not candidates:
        return dict(NULL_GROUND)

    best = None
    best_score = -1.0
    for candidate in candidates:
        score = _candidate_score(span, candidate)
        if score > best_score:
            best_score = score
            best = candidate

    if not best or best_score < ACCEPT_THRESHOLD:
        return dict(NULL_GROUND)

    qid = best.get("id")
    desc = best.get("description")
    p31_qid = _get_p31_qid(qid) if qid else None
    p31_label = _get_label(p31_qid)

    return {"wikidata_qid": qid, "wikidata_desc": desc, "wikidata_type": p31_label}


def main():
    """Run deduplicated Wikidata grounding for Phase 1 outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(IN_PATH), help="Input source CSI JSONL file.")
    parser.add_argument("--limit-records", type=int, default=None, help="Process first N records only.")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output grounded JSONL file.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    print("=" * 72)
    print("Phase 1.3 Wikidata Grounding")
    print("=" * 72)

    print(f"\n[1] Loading {in_path.name}...")
    records: list[dict] = []
    with open(in_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit_records is not None:
        records = records[: max(args.limit_records, 0)]
        print(f"    LIMIT MODE active: {len(records):,} records.")
    print(f"    Records loaded: {len(records):,}")

    total_target_csis = 0
    unique_spans: dict[str, str] = {}
    for record in records:
        for csi in record.get("csis", []):
            if csi.get("category") in TARGET_CATEGORIES:
                total_target_csis += 1
                span = (csi.get("span") or "").strip()
                key = _normalize(span)
                if key and key not in unique_spans:
                    unique_spans[key] = span

    print(f"    Target CSI occurrences: {total_target_csis:,}")
    print(f"    Unique target spans: {len(unique_spans):,}")

    print("\n[2] Grounding unique spans...")
    span_grounding: dict[str, dict] = {}
    matched_unique = 0
    span_items = list(unique_spans.items())
    start_time = time.time()

    for index, (key, span) in enumerate(span_items, 1):
        grounded = ground_span(span)
        span_grounding[key] = grounded
        if grounded["wikidata_qid"]:
            matched_unique += 1
        if index % 200 == 0 or index == len(span_items):
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (len(span_items) - index) / rate if rate > 0 else 0.0
            print(f"    {index:>5}/{len(span_items)} | {rate:>5.1f} spans/s | ETA {eta/60:>4.1f}m")

    print("\n[3] Applying grounding to all records...")
    checked_by_category: dict[str, int] = defaultdict(int)
    matched_by_category: dict[str, int] = defaultdict(int)
    example_matched: list[dict] = []
    example_unmatched: list[dict] = []

    for record in records:
        for csi in record.get("csis", []):
            category = csi.get("category")
            if category not in TARGET_CATEGORIES:
                continue
            span = (csi.get("span") or "").strip()
            key = _normalize(span)
            grounded = span_grounding.get(key, dict(NULL_GROUND))
            csi.update(grounded)
            checked_by_category[category] += 1
            if grounded["wikidata_qid"]:
                matched_by_category[category] += 1
                if len(example_matched) < 6:
                    example_matched.append(
                        {
                            "span": span,
                            "category": category,
                            "qid": grounded["wikidata_qid"],
                            "type": grounded["wikidata_type"],
                        }
                    )
            elif len(example_unmatched) < 6:
                example_unmatched.append({"span": span, "category": category})

    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_checked = sum(checked_by_category.values())
    total_matched = sum(matched_by_category.values())

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  Output file: {out_path}")
    print(f"  Total CSIs checked: {total_checked:,}")
    print(f"  Unique spans grounded: {len(unique_spans):,}")
    print(f"  Unique span matches: {matched_unique:,}/{len(unique_spans):,}")
    pct = 100 * total_matched / total_checked if total_checked else 0
    print(f"  Occurrence match rate: {total_matched:,}/{total_checked:,} ({pct:.1f}%)")

    print("\n  Match rate by category:")
    for category in sorted(TARGET_CATEGORIES):
        checked = checked_by_category.get(category, 0)
        matched = matched_by_category.get(category, 0)
        category_pct = 100 * matched / checked if checked else 0
        print(f"    {category:<35} {matched:>6,}/{checked:<6,} ({category_pct:>5.1f}%)")

    print("\n  Example matched:")
    for item in example_matched:
        print(f"    [{item['category']}] {item['span']!r} -> {item['qid']} ({item['type']})")

    print("\n  Example unmatched:")
    for item in example_unmatched:
        print(f"    [{item['category']}] {item['span']!r}")


if __name__ == "__main__":
    main()
