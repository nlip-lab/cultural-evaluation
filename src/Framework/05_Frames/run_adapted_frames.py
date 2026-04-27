"""
Phase 5 — Generate adapted-side function frames for deduplicated CSI pairs.

Reads:
  - 05_Frames/dedup_pairs.jsonl

Writes:
  - 05_Frames/_checkpoint.jsonl
  - 05_Frames/05_frames.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
CREDS_PATH = None

IN_PATH = PHASE5_DIR / "dedup_pairs.jsonl"
OUT_PATH = PHASE5_DIR / "05_frames.jsonl"
CKPT_PATH = PHASE5_DIR / "_checkpoint.jsonl"

GOOGLE_MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
MAX_TOKENS = 256
CONCURRENCY = 30
CHECKPOINT_N = 500
RETRY_DELAY = 30

PROMPT_TEMPLATE = """Source text: {representative_source_text}
Adapted text: {representative_adapted_text}

"{adapted_span}" replaces "{source_span}" in the adaptation.

What is "{adapted_span}" and what function does it serve in the adapted text?

Return ONLY JSON:
{{
  "denotation": "<what it literally is>",
  "cultural_role": "<cultural anchor it serves>",
  "contextual_function": "<what job it does in THIS text, 1 sentence>",
  "affordances": ["<property 1>", "<property 2>", ...]
}}"""

RETRY_NUDGE = "\n\nReturn ONLY valid JSON, no markdown."

_google_creds = None

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for adapted frame generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=IN_PATH)
    parser.add_argument("--output-path", type=Path, default=OUT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CKPT_PATH)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_N)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a smoke run.")
    return parser.parse_args()

def _load_google_creds():
    """Load Google service-account credentials for Gemini calls."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_PATH),
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds

def _refresh_if_needed(creds):
    """Refresh credential token when close to expiry."""
    import google.auth.transport.requests

    if creds.expiry:
        expiry = creds.expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if (expiry - datetime.now(timezone.utc)).total_seconds() < 300:
            creds.refresh(google.auth.transport.requests.Request())
    return creds

async def call_gemini(prompt: str, *, loop, lock) -> str:
    """Call Gemini API and return model text response."""
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
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            data = json.loads(raw)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode()
            if exc.code in (429, 503):
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini API {exc.code}: {err_body[:300]}") from exc

    raise RuntimeError("Gemini API: max retries exceeded")

def extract_json_object(text: str) -> dict:
    """Parse and validate a JSON object from model output text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed)}")
    return parsed

def normalize_frame(frame: dict) -> dict:
    """Normalize frame fields into a consistent output schema."""
    affordances = frame.get("affordances", [])
    if not isinstance(affordances, list):
        affordances = [str(affordances)] if affordances else []
    affordances = [str(item).strip() for item in affordances if str(item).strip()]
    return {
        "denotation": str(frame.get("denotation", "")).strip(),
        "cultural_role": str(frame.get("cultural_role", "")).strip(),
        "contextual_function": str(frame.get("contextual_function", "")).strip(),
        "affordances": affordances,
    }

def build_prompt(record: dict) -> str:
    """Render one prompt from a deduplicated pair record."""
    return PROMPT_TEMPLATE.format(
        representative_source_text=record.get("representative_source_text", ""),
        representative_adapted_text=record.get("representative_adapted_text", ""),
        adapted_span=record.get("adapted_span_representative", ""),
        source_span=record.get("source_span", ""),
    )

def build_output_record(record: dict, frame: dict, api_status: str) -> dict:
    """Build one output row for a dedup_id with normalized frame fields."""
    normalized = normalize_frame(frame)
    return {
        "dedup_id": record["dedup_id"],
        "source_span": record.get("source_span", ""),
        "adapted_span": record.get("adapted_span_representative", ""),
        "adapted_denotation": normalized["denotation"],
        "adapted_cultural_role": normalized["cultural_role"],
        "adapted_contextual_function": normalized["contextual_function"],
        "adapted_affordances": normalized["affordances"],
        "api_status": api_status,
    }

async def process_one(record: dict, *, sem: asyncio.Semaphore, loop, lock) -> dict:
    """Generate one adapted frame record with retry fallback."""
    prompt = build_prompt(record)

    async with sem:
        try:
            raw = await call_gemini(prompt, loop=loop, lock=lock)
            parsed = extract_json_object(raw)
            return build_output_record(record, parsed, "success")
        except Exception:
            pass

        try:
            raw = await call_gemini(prompt + RETRY_NUDGE, loop=loop, lock=lock)
            parsed = extract_json_object(raw)
            return build_output_record(record, parsed, "retry_success")
        except Exception:
            return build_output_record(record, {}, "parse_failed")

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL rows from a path when it exists."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to JSONL file, creating parent directories when needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def rewrite_output_from_checkpoint(checkpoint_rows: list[dict], output_path: Path) -> None:
    """Rewrite output file from checkpoint rows for consistent resume state."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in checkpoint_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_pending_records(input_path: Path, completed_ids: set[str], limit: int | None) -> list[dict]:
    """Load input records that are not yet completed in checkpoint."""
    pending: list[dict] = []
    with input_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["dedup_id"] in completed_ids:
                continue
            pending.append(record)
            if limit is not None and len(pending) >= limit:
                break
    return pending

def print_summary(all_rows: list[dict]) -> None:
    """Print aggregate API status summary for generated rows."""
    total = len(all_rows)
    successes = sum(row["api_status"] in {"success", "retry_success"} for row in all_rows)
    failures = sum(row["api_status"] == "parse_failed" for row in all_rows)
    success_rate = (successes / total) if total else 0.0
    print(f"Total processed: {total}")
    print(f"Success rate: {success_rate:.4f}")
    print(f"Failures logged: {failures}")

def print_progress(processed: int, total_pending: int, started_at: float, *, label: str) -> None:
    """Print runtime progress with throughput and ETA."""
    elapsed = max(time.time() - started_at, 1e-6)
    rate = processed / elapsed
    remaining = total_pending - processed
    eta_seconds = (remaining / rate) if rate > 0 else 0.0
    print(
        f"[{label}] processed={processed}/{total_pending} "
        f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m"
    )

async def run(args: argparse.Namespace) -> None:
    """Run asynchronous frame generation with checkpointed flushing."""
    checkpoint_rows = load_jsonl(args.checkpoint_path)
    completed_ids = {row["dedup_id"] for row in checkpoint_rows}
    rewrite_output_from_checkpoint(checkpoint_rows, args.output_path)

    pending_records = load_pending_records(args.input_path, completed_ids, args.limit)
    print(f"Loaded {len(checkpoint_rows)} completed rows from checkpoint: {args.checkpoint_path}")
    print(f"Pending records to process: {len(pending_records)}")
    print(f"Checkpoint/output will flush every {args.checkpoint_every} items")
    if not pending_records:
        print_summary(checkpoint_rows)
        return

    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    pending_flush: list[dict] = []
    all_rows = list(checkpoint_rows)
    tasks = [asyncio.create_task(process_one(record, sem=sem, loop=loop, lock=lock)) for record in pending_records]
    started_at = time.time()
    processed = 0

    for task in asyncio.as_completed(tasks):
        result = await task
        pending_flush.append(result)
        all_rows.append(result)
        processed += 1

        if processed == 1 or processed % 100 == 0:
            print_progress(processed, len(pending_records), started_at, label="progress")

        if len(pending_flush) >= args.checkpoint_every:
            append_jsonl(args.checkpoint_path, pending_flush)
            append_jsonl(args.output_path, pending_flush)
            print(
                f"[checkpoint] wrote {len(pending_flush)} rows "
                f"to {args.checkpoint_path} and {args.output_path}"
            )
            print_progress(processed, len(pending_records), started_at, label="checkpoint")
            pending_flush.clear()

    if pending_flush:
        append_jsonl(args.checkpoint_path, pending_flush)
        append_jsonl(args.output_path, pending_flush)
        print(
            f"[checkpoint] wrote final {len(pending_flush)} rows "
            f"to {args.checkpoint_path} and {args.output_path}"
        )
        print_progress(processed, len(pending_records), started_at, label="final_flush")

    print_summary(all_rows)

def main() -> None:
    """Entrypoint for phase-5 adapted frame generation."""
    args = parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
