"""Phase 1.2 function-frame enrichment for extracted CSIs."""

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
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR = EVAL_DIR / "01_CSI_Extraction"
CREDS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

IN_PATH = None
OUT_PATH = None
CKPT_PATH = None

PROMPT_TEMPLATE = (EVAL_DIR / "prompts" / "function_frame.txt").read_text(encoding="utf-8")

GOOGLE_MODEL = "gemini-2.5-flash-lite"
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
CONCURRENCY = 20
RETRY_DELAY = 60
CHECKPOINT_N = 500

_google_creds = None


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
                raise RuntimeError(f"Google API {exc.code}: {err_body[:300]}") from exc
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


def _extract_json_array(text: str) -> list:
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


def _fill_missing_frame_fields(frame: dict) -> dict:
    """Populate frame defaults for missing fields."""
    frame.setdefault("denotation", "")
    frame.setdefault("cultural_role", "")
    frame.setdefault("contextual_function", "")
    frame.setdefault("required_affordances", [])
    frame.setdefault("numeric_constraints", None)
    frame.setdefault("register_constraints", None)
    return frame


def _build_prompt(record: dict) -> str:
    """Build function-frame prompt for a source record."""
    slim_csis = [
        {
            "span": csi.get("span", ""),
            "category": csi.get("category", ""),
            "scale_sensitive": csi.get("scale_sensitive", False),
            "figurative": csi.get("figurative", False),
            "local_role": csi.get("local_role", ""),
        }
        for csi in record["csis"]
    ]
    return (
        PROMPT_TEMPLATE.replace("{source_text}", record["source_text"])
        .replace("{domain}", record["domain"])
        .replace("{csi_list_json}", json.dumps(slim_csis, ensure_ascii=False, indent=2))
    )


async def process_item(record: dict, *, caller, sem) -> dict:
    """Enrich a single record by adding function-frame fields to CSIs."""
    if record.get("csi_count", 0) == 0:
        record["ff_status"] = "skipped_no_csis"
        return record

    prompt = _build_prompt(record)
    raw = ""
    async with sem:
        try:
            raw = await caller(prompt)
        except Exception as exc:
            record["ff_status"] = "api_error"
            record["_ff_error"] = str(exc)
            return record
        try:
            frames = _extract_json_array(raw)
        except Exception:
            nudge = f"{prompt}\n\nReturn ONLY a valid JSON array, one object per CSI."
            try:
                raw_2 = await caller(nudge)
                frames = _extract_json_array(raw_2)
                record["ff_status"] = "parse_retry"
            except Exception:
                record["ff_status"] = "parse_failed"
                record["_ff_raw"] = raw
                return record

    span_map: dict[str, dict] = {}
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict) and frame.get("span"):
                span_map[frame["span"]] = frame

    enriched = []
    for index, csi in enumerate(record["csis"]):
        if isinstance(frames, list) and index < len(frames) and isinstance(frames[index], dict):
            frame = frames[index]
        else:
            frame = span_map.get(csi.get("span", ""), {})
        frame = _fill_missing_frame_fields(frame)
        merged = {**frame, **csi}
        for field_key in (
            "denotation",
            "cultural_role",
            "contextual_function",
            "required_affordances",
            "numeric_constraints",
            "register_constraints",
        ):
            merged[field_key] = frame.get(field_key)
        enriched.append(merged)

    record["csis"] = enriched
    if "ff_status" not in record:
        record["ff_status"] = "success"
    return record


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


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["google", "openai"], default="google")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to source CSI inventory JSONL. Uses backend default when omitted.",
    )
    parser.add_argument("--test", action="store_true", help="Run on 3 items only.")
    return parser.parse_args()


def _set_paths(backend: str, input_override: str | None):
    """Set backend-specific input and output paths."""
    global IN_PATH, OUT_PATH, CKPT_PATH
    tag = "google" if backend == "google" else "openai"
    IN_PATH = Path(input_override) if input_override else PHASE_DIR / f"source_csi_inventory_{tag}.jsonl"
    OUT_PATH = PHASE_DIR / f"source_csi_with_functions_{tag}.jsonl"
    CKPT_PATH = PHASE_DIR / f".ff_checkpoint_{tag}.json"


async def main():
    """Run Phase 1.2 function-frame enrichment pipeline."""
    args = parse_args()
    backend = args.backend
    _set_paths(backend, args.input)

    print("=" * 65)
    print(f"Phase 1.2 Function Frame Generation [backend={backend}]")
    print("=" * 65)

    loop = asyncio.get_event_loop()
    if backend == "google":
        lock = asyncio.Lock()
        caller = lambda prompt: call_google(prompt, loop=loop, lock=lock)
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print("FATAL: openai package is not installed.")
            sys.exit(1)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("FATAL: OPENAI_API_KEY is not set.")
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key)
        caller = lambda prompt: call_openai(prompt, client=client)

    print(f"\n[0] Testing {backend} API...")
    try:
        response = await caller('Return ONLY this JSON array: [{"span":"test"}]')
        parsed = _extract_json_array(response)
        assert isinstance(parsed, list) and len(parsed) > 0
        model_label = GOOGLE_MODEL if backend == "google" else OPENAI_MODEL
        print(f"    API OK, model={model_label}, items={len(parsed)}.")
    except Exception as exc:
        print(f"    FATAL: API test failed: {exc}")
        sys.exit(1)

    print("\n[1] Loading source inventory...")
    records: list[dict] = []
    with open(IN_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.test:
        records = [record for record in records if record.get("csi_count", 0) > 0][:3]
        print(f"    TEST MODE: {len(records)} items.")
    else:
        print(f"    Loaded {len(records):,} records.")

    items_with_csis = sum(1 for record in records if record.get("csi_count", 0) > 0)
    print(f"    Items with CSIs: {items_with_csis:,}")

    done_ids = prepare_run_state(test_mode=args.test)
    remaining = [record for record in records if record["source_id"] not in done_ids]
    print(f"    Already done: {len(done_ids):,} | Remaining: {len(remaining):,}")

    if remaining:
        print(f"\n[2] Generating function frames with concurrency={CONCURRENCY}...")
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [process_item(record, caller=caller, sem=sem) for record in remaining]
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
                statuses = {rec.get("ff_status", "?") for rec in completed[-100:]}
                print(f"    {index:>5}/{len(tasks)} | {rate:.1f}/s | ETA {eta/60:.1f}m | recent={statuses}")

            if not args.test and index % CHECKPOINT_N == 0:
                with open(OUT_PATH, "a", encoding="utf-8") as handle:
                    for record in completed:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_checkpoint(done_ids)
                completed.clear()
                print(f"    Checkpoint saved at {index}.")

        mode = "w" if args.test else "a"
        with open(OUT_PATH, mode, encoding="utf-8") as handle:
            for record in completed:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not args.test:
            save_checkpoint(done_ids)

    print("\n[3] Computing stats...")
    all_records: list[dict] = []
    with open(OUT_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    total_csis = sum(record["csi_count"] for record in all_records)
    affordance_counts = []
    numeric_count = 0
    register_count = 0
    ff_status_counts: dict[str, int] = defaultdict(int)

    for record in all_records:
        ff_status_counts[record.get("ff_status", "unknown")] += 1
        for csi in record.get("csis", []):
            affordance_counts.append(len(csi.get("required_affordances") or []))
            if csi.get("numeric_constraints"):
                numeric_count += 1
            if csi.get("register_constraints"):
                register_count += 1

    avg_affordances = sum(affordance_counts) / len(affordance_counts) if affordance_counts else 0
    pct_numeric = 100 * numeric_count / total_csis if total_csis else 0
    pct_register = 100 * register_count / total_csis if total_csis else 0

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Records in output file: {len(all_records):,}")
    print(f"  Total CSIs enriched: {total_csis:,}")
    print(f"  Avg affordances per CSI: {avg_affordances:.2f}")
    print(f"  CSIs with numeric constraints: {numeric_count:,} ({pct_numeric:.1f}%)")
    print(f"  CSIs with register constraints: {register_count:,} ({pct_register:.1f}%)")
    print("\n  Status breakdown:")
    for status, count in sorted(ff_status_counts.items(), key=lambda item: -item[1]):
        print(f"    {status:<25} {count:>6,}")
    print(f"\n  Output: {OUT_PATH}")

    if args.test:
        print("\n  [TEST MODE] Sample CSI:")
        for record in all_records[:3]:
            print(f"\n  source_id: {record['source_id']} | status={record.get('ff_status')}")
            for csi in record["csis"][:1]:
                print(f"  span: {csi.get('span')}")
                print(f"  denotation: {csi.get('denotation')}")
                print(f"  contextual_function: {csi.get('contextual_function')}")
                print(f"  required_affordances: {csi.get('required_affordances')}")
                print(f"  numeric_constraints: {csi.get('numeric_constraints')}")
                print(f"  register_constraints: {csi.get('register_constraints')}")


if __name__ == "__main__":
    asyncio.run(main())
