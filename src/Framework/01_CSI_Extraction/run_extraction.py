"""Phase 1 source CSI extraction with Google and OpenAI backends."""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR = EVAL_DIR / "01_CSI_Extraction"
CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BENCHMARK = PROC_DIR / "benchmark.jsonl"
OUT_PATH = None
STATS_PATH = None
CKPT_PATH = None

PROMPT_TEMPLATE = (EVAL_DIR / "prompts" / "source_csi_extraction.txt").read_text(encoding="utf-8")

GOOGLE_MODEL = "gemini-2.5-flash-lite"
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
CHECKPOINT_N = 500
CONCURRENCY = 20
RETRY_DELAY = 60

VALID_CATEGORIES = {
    "food_drink",
    "place_institution_brand",
    "unit_currency_scale",
    "holiday_event_ritual",
    "personal_name_identity",
    "social_norm_address",
    "pop_culture_sport_media",
    "idiom_figurative",
}
VALID_NECESSITY = {"must_adapt", "may_adapt", "should_retain"}

_google_creds = None


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["google", "openai"], default="google")
    parser.add_argument("--test", action="store_true", help="Run on 3 items only.")
    return parser.parse_args()


def _load_google_creds():
    """Load Google service-account credentials for Generative Language API."""
    if not CREDS_PATH:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_file(
        CREDS_PATH,
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _refresh_if_needed(creds):
    """Refresh Google access token if near expiry."""
    from datetime import datetime, timezone
    import google.auth.transport.requests

    if creds.expiry:
        expiry = creds.expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if (expiry - datetime.now(timezone.utc)).total_seconds() < 300:
            creds.refresh(google.auth.transport.requests.Request())
    return creds


async def call_google(prompt: str, *, loop, lock) -> str:
    """Call Gemini generateContent endpoint and return text payload."""
    global _google_creds
    async with lock:
        if _google_creds is None:
            _google_creds = await loop.run_in_executor(None, _load_google_creds)
        _google_creds = await loop.run_in_executor(None, _refresh_if_needed, _google_creds)
        token = _google_creds.token

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
                "responseMimeType": "application/json",
            },
        }
    ).encode()

    for attempt in range(3):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            data = json.loads(raw)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode()
            if exc.code in (429, 503):
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Backend busy ({exc.code}), waiting {wait}s.")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Google API {exc.code}: {err_body[:200]}") from exc
    raise RuntimeError("Google API: max retries exceeded")


async def call_openai(prompt: str, *, client) -> str:
    """Call OpenAI chat completions and return response text."""
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as exc:
            if "rate" in str(exc).lower() or "429" in str(exc):
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    raise RuntimeError("OpenAI: max retries exceeded")


def _extract_json_array(text: str):
    """Parse a JSON array response, including fenced and wrapped variants."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text.startswith("{"):
        obj = json.loads(text)
        for value in obj.values():
            if isinstance(value, list):
                return value
        raise ValueError("JSON object has no list value.")
    return json.loads(text)


def _fuzzy_fix_offsets(csi: dict, source_text: str) -> dict:
    """Repair char offsets when span text is present but offsets are inconsistent."""
    span = csi.get("span", "")
    if not span:
        return csi
    char_start, char_end = csi.get("char_start", -1), csi.get("char_end", -1)
    if 0 <= char_start < char_end <= len(source_text) and source_text[char_start:char_end] == span:
        return csi
    idx = source_text.find(span)
    if idx != -1:
        csi["char_start"] = idx
        csi["char_end"] = idx + len(span)
        csi["_offset_corrected"] = True
        return csi
    idx = source_text.lower().find(span.lower())
    if idx != -1:
        csi["char_start"] = idx
        csi["char_end"] = idx + len(span)
        csi["_offset_corrected"] = True
        csi["_case_mismatch"] = True
        return csi
    csi["_span_not_found"] = True
    return csi


def validate_and_fix(csis: list, source_text: str) -> list:
    """Normalize and validate extracted CSI entries."""
    clean = []
    for csi in csis:
        if not isinstance(csi, dict):
            continue
        csi.setdefault("dependencies", [])
        csi.setdefault("scale_sensitive", False)
        csi.setdefault("figurative", False)
        csi.setdefault("salience", 0.5)
        csi.setdefault("local_role", "")
        if csi.get("category") not in VALID_CATEGORIES:
            csi["category"] = "idiom_figurative"
        if csi.get("adaptation_necessity") not in VALID_NECESSITY:
            csi["adaptation_necessity"] = "may_adapt"
        csi = _fuzzy_fix_offsets(csi, source_text)
        clean.append(csi)
    return clean


async def process_item(item: dict, *, caller, sem) -> dict:
    """Run extraction for a single benchmark item."""
    source_id = item["id"]
    domain = item["domain"]
    source_text = item["original_text"]
    prompt = PROMPT_TEMPLATE.replace("{source_text}", source_text).replace("{domain}", domain)

    status = "success"
    raw = ""
    csis = []

    async with sem:
        try:
            raw = await caller(prompt)
        except Exception as exc:
            return {
                "source_id": source_id,
                "domain": domain,
                "source_text": source_text,
                "csi_count": 0,
                "csis": [],
                "extraction_status": "api_error",
                "_error": str(exc),
            }
        try:
            csis = _extract_json_array(raw)
        except Exception:
            nudge = f"{prompt}\n\nReturn ONLY valid JSON array."
            try:
                raw_2 = await caller(nudge)
                csis = _extract_json_array(raw_2)
                status = "parse_retry"
            except Exception:
                return {
                    "source_id": source_id,
                    "domain": domain,
                    "source_text": source_text,
                    "csi_count": 0,
                    "csis": [],
                    "extraction_status": "parse_failed",
                    "_raw": raw,
                }

    if not csis:
        status = "no_csis"
    csis = validate_and_fix(csis, source_text)
    return {
        "source_id": source_id,
        "domain": domain,
        "source_text": source_text,
        "csi_count": len(csis),
        "csis": csis,
        "extraction_status": status,
    }


def load_checkpoint() -> set[str]:
    """Load checkpoint file as set of completed source IDs."""
    if CKPT_PATH.exists():
        data = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
        return set(data.get("done_ids", []))
    return set()


def save_checkpoint(done_ids: set[str]):
    """Persist checkpoint file with completed source IDs."""
    CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}), encoding="utf-8")


def prepare_run_state(*, test_mode: bool) -> set[str]:
    """Align checkpoint and output state before processing."""
    if test_mode:
        return set()
    if CKPT_PATH.exists() and OUT_PATH.exists():
        return load_checkpoint()
    if CKPT_PATH.exists() or OUT_PATH.exists():
        print("    Found stale partial state; starting fresh.")
        CKPT_PATH.unlink(missing_ok=True)
        OUT_PATH.unlink(missing_ok=True)
        return set()
    return set()


def _set_paths(backend: str):
    """Set backend-specific output paths."""
    global OUT_PATH, STATS_PATH, CKPT_PATH
    tag = "google" if backend == "google" else "openai"
    OUT_PATH = PHASE_DIR / f"source_csi_inventory_{tag}.jsonl"
    STATS_PATH = PHASE_DIR / f"extraction_stats_{tag}.json"
    CKPT_PATH = PHASE_DIR / f".checkpoint_{tag}.json"


def compute_stats(records: list[dict]) -> dict:
    """Compute aggregate extraction metrics."""
    total = len(records)
    items_with_csis = sum(1 for record in records if record["csi_count"] > 0)
    items_no_csis = sum(1 for record in records if record["extraction_status"] == "no_csis")
    parse_failures = sum(1 for record in records if record["extraction_status"] == "parse_failed")
    total_csis = sum(record["csi_count"] for record in records)

    csis_per_domain: dict[str, int] = defaultdict(int)
    count_per_domain: dict[str, int] = defaultdict(int)
    salience_per_domain: dict[str, list] = defaultdict(list)
    category_distribution: dict[str, int] = defaultdict(int)
    necessity_distribution: dict[str, int] = defaultdict(int)

    for record in records:
        domain = record["domain"]
        count_per_domain[domain] += 1
        csis_per_domain[domain] += record["csi_count"]
        for csi in record["csis"]:
            category_distribution[csi.get("category", "unknown")] += 1
            necessity_distribution[csi.get("adaptation_necessity", "unknown")] += 1
            salience = csi.get("salience")
            if isinstance(salience, (int, float)):
                salience_per_domain[domain].append(salience)

    avg_csis_per_domain = {
        domain: round(csis_per_domain[domain] / count_per_domain[domain], 3)
        for domain in count_per_domain
    }
    avg_salience_by_domain = {
        domain: round(sum(values) / len(values), 3) if values else 0.0
        for domain, values in salience_per_domain.items()
    }

    return {
        "total_items": total,
        "items_with_csis": items_with_csis,
        "items_no_csis": items_no_csis,
        "parse_failures": parse_failures,
        "total_csis_extracted": total_csis,
        "csis_per_domain": dict(csis_per_domain),
        "avg_csis_per_item_by_domain": avg_csis_per_domain,
        "category_distribution": dict(category_distribution),
        "necessity_distribution": dict(necessity_distribution),
        "avg_salience_by_domain": avg_salience_by_domain,
    }


async def main():
    """Run Phase 1 extraction pipeline."""
    args = parse_args()
    backend = args.backend
    test_mode = args.test
    _set_paths(backend)

    print("=" * 65)
    print(f"Phase 1 Source CSI Extraction [backend={backend}]")
    print("=" * 65)

    loop = asyncio.get_event_loop()
    if backend == "google":
        lock = asyncio.Lock()
        caller = lambda prompt: call_google(prompt, loop=loop, lock=lock)
        try:
            response = await caller("Return ONLY this JSON: []")
            parsed = _extract_json_array(response)
            assert isinstance(parsed, list)
            print(f"    Google API OK, model={GOOGLE_MODEL}.")
        except Exception as exc:
            print(f"    FATAL: Google API test failed: {exc}")
            sys.exit(1)
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print("    FATAL: openai package is not installed.")
            sys.exit(1)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("    FATAL: OPENAI_API_KEY is not set.")
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key)
        caller = lambda prompt: call_openai(prompt, client=client)
        try:
            response = await caller("Return ONLY this JSON array: []")
            parsed = _extract_json_array(response)
            assert isinstance(parsed, list)
            print(f"    OpenAI API OK, model={OPENAI_MODEL}.")
        except Exception as exc:
            print(f"    FATAL: OpenAI API test failed: {exc}")
            sys.exit(1)

    print("\n[1] Loading benchmark...")
    items: list[dict] = []
    with open(BENCHMARK, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if test_mode:
        items = items[:3]
        print(f"    TEST MODE: {len(items)} items.")
    else:
        print(f"    Loaded {len(items):,} items.")

    done_ids = prepare_run_state(test_mode=test_mode)
    remaining = [item for item in items if item["id"] not in done_ids]
    print(f"    Already done: {len(done_ids):,} | Remaining: {len(remaining):,}")

    if remaining:
        print(f"\n[2] Extracting CSIs with concurrency={CONCURRENCY}...")
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [process_item(item, caller=caller, sem=sem) for item in remaining]
        completed: list[dict] = []
        start_time = time.time()

        for index, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            completed.append(result)
            done_ids.add(result["source_id"])
            if index % 100 == 0 or index == len(tasks):
                elapsed = time.time() - start_time
                rate = index / elapsed if elapsed > 0 else 0.0
                eta = (len(tasks) - index) / rate if rate > 0 else 0.0
                print(f"    {index:>5}/{len(tasks)} | {rate:.1f} items/s | ETA {eta/60:.1f}m")
            if index % CHECKPOINT_N == 0:
                with open(OUT_PATH, "a", encoding="utf-8") as handle:
                    for record in completed:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_checkpoint(done_ids)
                completed.clear()
                print(f"    Checkpoint saved at {index}.")

        if completed:
            with open(OUT_PATH, "a", encoding="utf-8") as handle:
                for record in completed:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            save_checkpoint(done_ids)
    else:
        print("    All items already processed.")

    print("\n[3] Computing summary stats...")
    all_records: list[dict] = []
    with open(OUT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    stats = compute_stats(all_records)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Total items processed: {stats['total_items']:,}")
    print(f"  Items with CSIs: {stats['items_with_csis']:,}")
    print(f"  Items with no CSIs: {stats['items_no_csis']:,}")
    print(f"  Parse failures: {stats['parse_failures']:,}")
    print(f"  Total CSIs extracted: {stats['total_csis_extracted']:,}")
    print("\n  CSIs per domain:")
    for domain, count in stats["csis_per_domain"].items():
        avg = stats["avg_csis_per_item_by_domain"].get(domain, 0)
        print(f"    {domain:<20} {count:>6,} (avg {avg:.2f}/item)")
    print("\n  Category distribution:")
    for category, count in sorted(stats["category_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"    {category:<35} {count:>6,}")
    print("\n  Necessity distribution:")
    for necessity, count in sorted(stats["necessity_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"    {necessity:<20} {count:>6,}")
    print(f"\n  Output: {OUT_PATH}")
    print(f"  Stats: {STATS_PATH}")

    if test_mode:
        print("\n  [TEST MODE] Sample records:")
        for record in all_records[:3]:
            print(f"\n  source_id: {record['source_id']}")
            print(f"  domain: {record['domain']}")
            print(f"  status: {record['extraction_status']}")
            print(f"  csi_count: {record['csi_count']}")
            if record["csis"]:
                print(f"  first_csi: {json.dumps(record['csis'][0], indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
