"""
Phase 6 - Function Preservation Score (FPS) with Gemini.

Reads:
  - 05_Frames/paired_frames.jsonl
  - sample_ids_16k.json

Writes:
  - 06_Analysis/_checkpoint.jsonl
  - 06_Analysis/fps_failures.jsonl
  - 06_Analysis/fps_judge_scores.jsonl
  - 06_Analysis/fps_per_adaptation.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE5_DIR = EVAL_DIR / "05_Frames"
PHASE6_DIR = EVAL_DIR / "06_Analysis"
CREDS_PATH = None

PAIRED_FRAMES_PATH = PHASE5_DIR / "paired_frames.jsonl"
SAMPLE_IDS_PATH = EVAL_DIR / "sample_ids_16k.json"
CHECKPOINT_PATH = PHASE6_DIR / "_checkpoint.jsonl"
FAILURES_PATH = PHASE6_DIR / "fps_failures.jsonl"
SCORES_PATH = PHASE6_DIR / "fps_judge_scores.jsonl"
ADAPTATION_PATH = PHASE6_DIR / "fps_per_adaptation.jsonl"

GOOGLE_MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
MAX_TOKENS = 512
CONCURRENCY = 40
MAX_BATCH_SIZE = 4
CHECKPOINT_N = 500
RETRY_DELAY = 30

SCORE_KEYS = (
    "denotation_match",
    "functional_equivalence",
    "affordance_coverage",
    "register_match",
    "fps_overall",
)

RETRY_NUDGE = (
    "\n\nReturn ONLY a valid JSON array with exactly one object per replacement "
    "and exactly these keys: denotation_match, functional_equivalence, "
    "affordance_coverage, register_match, fps_overall, rationale. Keep each "
    "rationale to one very short sentence."
)

_google_creds = None

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FPS judging pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-frames-path", type=Path, default=PAIRED_FRAMES_PATH)
    parser.add_argument("--sample-ids-path", type=Path, default=SAMPLE_IDS_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--failure-log-path", type=Path, default=FAILURES_PATH)
    parser.add_argument("--scores-out", type=Path, default=SCORES_PATH)
    parser.add_argument("--adaptation-out", type=Path, default=ADAPTATION_PATH)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_N)
    parser.add_argument(
        "--limit-adaptations",
        type=int,
        default=None,
        help="Optional limit for a smoke run after filtering to the 16K sample.",
    )
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
    """Refresh Google credential token when close to expiry."""
    import google.auth.transport.requests

    if creds.expiry:
        expiry = creds.expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if (expiry - datetime.now(timezone.utc)).total_seconds() < 300:
            creds.refresh(google.auth.transport.requests.Request())
    return creds

async def call_gemini(prompt: str, *, loop, lock) -> str:
    """Call Gemini API and return model response text."""
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

    for attempt in range(4):
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
        except Exception:
            await asyncio.sleep(3 * (attempt + 1))

    raise RuntimeError("Gemini API: max retries exceeded")

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL rows from path when available."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to JSONL file, creating parent directory if needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to JSONL output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def mean(values: list[float]) -> float:
    """Return mean for a list of float values."""
    return (sum(values) / len(values)) if values else 0.0

def load_sample_eval_ids(path: Path) -> set[str]:
    """Load sample eval IDs from supported JSON schemas."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict):
        if isinstance(data.get("eval_ids"), list):
            return {str(item) for item in data["eval_ids"]}
        if isinstance(data.get("by_cell"), dict):
            eval_ids: set[str] = set()
            for values in data["by_cell"].values():
                if isinstance(values, list):
                    eval_ids.update(str(item) for item in values)
            return eval_ids
    raise ValueError(f"Unsupported sample ID schema in {path}")

def load_grouped_rows(path: Path, sample_eval_ids: set[str]) -> dict[str, list[dict]]:
    """Load paired-frame rows grouped by eval_id for sampled adaptations."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            eval_id = row.get("eval_id")
            if eval_id in sample_eval_ids:
                grouped[eval_id].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda item: item["csi_index"])
    return dict(sorted(grouped.items()))

def build_pair_meta(grouped_rows: dict[str, list[dict]]) -> dict[tuple[str, int], dict]:
    """Build metadata lookup keyed by eval_id and csi_index."""
    meta: dict[tuple[str, int], dict] = {}
    for eval_id, rows in grouped_rows.items():
        for row in rows:
            meta[(eval_id, int(row["csi_index"]))] = row
    return meta

def normalize_affordances(values) -> list[str]:
    """Normalize affordance values to a clean string list."""
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized

def build_batch_prompt(rows: list[dict]) -> str:
    """Build FPS judge prompt for one batch of replacement rows."""
    parts = [
        "Evaluate these replacements. Return a JSON array, one object per replacement, in the same order."
    ]
    for idx, row in enumerate(rows, start=1):
        source_function = row.get("source_function") or {}
        adapted_function = row.get("adapted_function") or {}
        parts.append(
            "\n".join(
                [
                    f"",
                    f"Replacement {idx}:",
                    f'Source item: "{row.get("source_span", "")}" ({row.get("source_category", "")})',
                    f'Source function: {str(source_function.get("contextual_function", "")).strip()}',
                    f"Source affordances: {json.dumps(normalize_affordances(source_function.get('affordances')), ensure_ascii=False)}",
                    f'Replacement: "{row.get("adapted_span", "")}"',
                    f'Replacement function: {str(adapted_function.get("contextual_function", "")).strip()}',
                    f"Replacement affordances: {json.dumps(normalize_affordances(adapted_function.get('affordances')), ensure_ascii=False)}",
                    f'Strategy: {row.get("strategy", "")}',
                ]
            ).strip()
        )
    parts.append(
        "\n".join(
            [
                "",
                "Score how well each replacement preserves the source item's function.",
                "Keep each rationale to one very short sentence.",
                "Return ONLY a JSON array where each object has exactly these keys:",
                "[",
                "  {",
                '    "denotation_match": <0.0-1.0>,',
                '    "functional_equivalence": <0.0-1.0>,',
                '    "affordance_coverage": <0.0-1.0>,',
                '    "register_match": <0.0-1.0>,',
                '    "fps_overall": <0.0-1.0>,',
                '    "rationale": "<1 very short sentence>"',
                "  }",
                "]",
            ]
        ).strip()
    )
    return "\n\n".join(parts)

def extract_json_array(text: str):
    """Extract JSON array from model output with fence and fallback handling."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError(f"Expected list response, got {type(parsed)}")
    return parsed

def validate_scores(items, expected_len: int) -> list[dict]:
    """Validate and normalize judge score items for one batch."""
    if len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} results, got {len(items)}")

    normalized: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict result, got {type(item)}")

        result = {}
        for key in SCORE_KEYS:
            value = item.get(key)
            if isinstance(value, bool):
                raise ValueError(f"{key} must be numeric, got bool")
            try:
                numeric = float(value)
            except Exception as exc:
                raise ValueError(f"Invalid numeric value for {key}: {value!r}") from exc
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{key} out of range: {numeric}")
            result[key] = round(numeric, 6)

        rationale = str(item.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Missing rationale")
        result["rationale"] = rationale
        normalized.append(result)

    return normalized

def build_score_row(row: dict, score: dict) -> dict:
    """Build one per-pair FPS score row."""
    return {
        "eval_id": row["eval_id"],
        "csi_index": int(row["csi_index"]),
        "source_span": row.get("source_span", ""),
        "adapted_span": row.get("adapted_span", ""),
        "denotation_match": score["denotation_match"],
        "functional_equivalence": score["functional_equivalence"],
        "affordance_coverage": score["affordance_coverage"],
        "register_match": score["register_match"],
        "fps_overall": score["fps_overall"],
        "rationale": score["rationale"],
    }

def build_adaptation_row(rows: list[dict], score_rows: list[dict]) -> dict | None:
    """Build one per-adaptation aggregate FPS row."""
    if not score_rows:
        return None

    return {
        "eval_id": rows[0]["eval_id"],
        "model": rows[0].get("model", ""),
        "prompt_strategy": rows[0].get("prompt_strategy", ""),
        "domain": rows[0].get("domain", ""),
        "mean_fps_overall": round(mean([row["fps_overall"] for row in score_rows]), 6),
        "mean_functional_equivalence": round(
            mean([row["functional_equivalence"] for row in score_rows]), 6
        ),
        "mean_affordance_coverage": round(
            mean([row["affordance_coverage"] for row in score_rows]), 6
        ),
        "min_fps_overall": round(min(row["fps_overall"] for row in score_rows), 6),
        "num_pairs": len(score_rows),
    }

def flatten_checkpoint_rows(checkpoint_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flatten checkpoint payload into score and adaptation row lists."""
    score_rows: list[dict] = []
    adaptation_rows: list[dict] = []
    for row in checkpoint_rows:
        score_rows.extend(row.get("score_rows") or [])
        adaptation_row = row.get("adaptation_row")
        if adaptation_row:
            adaptation_rows.append(adaptation_row)
    score_rows.sort(key=lambda item: (item["eval_id"], item["csi_index"]))
    adaptation_rows.sort(key=lambda item: item["eval_id"])
    return score_rows, adaptation_rows

def rewrite_outputs_from_checkpoint(
    checkpoint_rows: list[dict],
    scores_out: Path,
    adaptation_out: Path,
) -> tuple[list[dict], list[dict]]:
    """Rewrite output files from checkpoint state and return flattened rows."""
    score_rows, adaptation_rows = flatten_checkpoint_rows(checkpoint_rows)
    write_jsonl(scores_out, score_rows)
    write_jsonl(adaptation_out, adaptation_rows)
    return score_rows, adaptation_rows

def print_progress(processed: int, total: int, started_at: float) -> None:
    """Print runtime progress with throughput and ETA."""
    elapsed = max(time.time() - started_at, 1e-6)
    rate = processed / elapsed
    remaining = total - processed
    eta_seconds = (remaining / rate) if rate > 0 else 0.0
    print(
        f"[fps] processed={processed}/{total} "
        f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m"
    )

def print_summary(
    score_rows: list[dict],
    failure_rows: list[dict],
    pair_meta: dict[tuple[str, int], dict],
) -> None:
    """Print aggregate FPS summary across scored pairs."""
    print(f"Scored CSI pairs: {len(score_rows)}")
    print(f"Logged failures: {len(failure_rows)}")
    print(f"Overall mean FPS: {mean([row['fps_overall'] for row in score_rows]):.6f}")

    by_model: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_strategy: dict[str, list[float]] = defaultdict(list)

    for row in score_rows:
        meta = pair_meta.get((row["eval_id"], int(row["csi_index"])))
        if meta is None:
            continue
        score = float(row["fps_overall"])
        by_model[str(meta.get("model", ""))].append(score)
        by_domain[str(meta.get("domain", ""))].append(score)
        by_strategy[str(meta.get("strategy", ""))].append(score)

    print("FPS by model:")
    for key in sorted(by_model):
        print(f"  {key}: {mean(by_model[key]):.6f}")

    print("FPS by domain:")
    for key in sorted(by_domain):
        print(f"  {key}: {mean(by_domain[key]):.6f}")

    print("FPS by strategy type:")
    for key in sorted(by_strategy):
        print(f"  {key}: {mean(by_strategy[key]):.6f}")

async def score_batch(rows: list[dict], *, loop, lock) -> tuple[list[dict] | None, dict | None]:
    """Score one batch and return validated scores or a failure record."""
    prompt = build_batch_prompt(rows)
    attempts: list[dict] = []

    for attempt_index, prompt_text in enumerate((prompt, prompt + RETRY_NUDGE), start=1):
        raw = ""
        try:
            raw = await call_gemini(prompt_text, loop=loop, lock=lock)
            scores = validate_scores(extract_json_array(raw), len(rows))
            return scores, None
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "error": str(exc),
                    "raw": raw,
                }
            )

    failure = {
        "eval_id": rows[0]["eval_id"],
        "csi_indices": [int(row["csi_index"]) for row in rows],
        "source_spans": [row.get("source_span", "") for row in rows],
        "adapted_spans": [row.get("adapted_span", "") for row in rows],
        "batch_size": len(rows),
        "status": "parse_failed",
        "prompt": prompt,
        "attempts": attempts,
    }
    return None, failure

async def process_adaptation(rows: list[dict], *, batch_size: int, loop, lock) -> dict:
    """Process all batches for one adaptation and return checkpoint payload."""
    score_rows: list[dict] = []
    failures: list[dict] = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        scores, failure = await score_batch(chunk, loop=loop, lock=lock)
        if scores is None:
            failures.append(failure)
            continue

        for row, score in zip(chunk, scores, strict=True):
            score_rows.append(build_score_row(row, score))

    adaptation_row = build_adaptation_row(rows, score_rows)
    return {
        "eval_id": rows[0]["eval_id"],
        "score_rows": score_rows,
        "adaptation_row": adaptation_row,
        "failures": failures,
        "checkpoint_row": {
            "eval_id": rows[0]["eval_id"],
            "model": rows[0].get("model", ""),
            "prompt_strategy": rows[0].get("prompt_strategy", ""),
            "domain": rows[0].get("domain", ""),
            "num_input_pairs": len(rows),
            "num_scored_pairs": len(score_rows),
            "score_rows": score_rows,
            "adaptation_row": adaptation_row,
            "failed_batches": len(failures),
        },
    }

async def worker(
    work_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    *,
    batch_size: int,
    loop,
    lock,
) -> None:
    """Consume adaptation rows from queue and emit processed results."""
    while True:
        rows = await work_queue.get()
        if rows is None:
            break

        try:
            result = await process_adaptation(rows, batch_size=batch_size, loop=loop, lock=lock)
        except Exception as exc:
            eval_id = rows[0]["eval_id"]
            result = {
                "eval_id": eval_id,
                "score_rows": [],
                "adaptation_row": None,
                "failures": [
                    {
                        "eval_id": eval_id,
                        "csi_indices": [int(row["csi_index"]) for row in rows],
                        "source_spans": [row.get("source_span", "") for row in rows],
                        "adapted_spans": [row.get("adapted_span", "") for row in rows],
                        "batch_size": len(rows),
                        "status": "adaptation_error",
                        "error": str(exc),
                    }
                ],
                "checkpoint_row": {
                    "eval_id": eval_id,
                    "model": rows[0].get("model", ""),
                    "prompt_strategy": rows[0].get("prompt_strategy", ""),
                    "domain": rows[0].get("domain", ""),
                    "num_input_pairs": len(rows),
                    "num_scored_pairs": 0,
                    "score_rows": [],
                    "adaptation_row": None,
                    "failed_batches": 1,
                },
            }

        await result_queue.put(result)

async def run_async(
    grouped_rows: dict[str, list[dict]],
    pending_eval_ids: list[str],
    *,
    batch_size: int,
    concurrency: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    scores_out: Path,
    adaptation_out: Path,
    failure_log_path: Path,
) -> None:
    """Run asynchronous worker pool and checkpointed flushing."""
    loop = asyncio.get_event_loop()
    lock = asyncio.Lock()
    work_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()

    for eval_id in pending_eval_ids:
        work_queue.put_nowait(grouped_rows[eval_id])
    for _ in range(concurrency):
        work_queue.put_nowait(None)

    workers = [
        asyncio.create_task(
            worker(
                work_queue,
                result_queue,
                batch_size=batch_size,
                loop=loop,
                lock=lock,
            )
        )
        for _ in range(concurrency)
    ]

    checkpoint_buffer: list[dict] = []
    score_buffer: list[dict] = []
    adaptation_buffer: list[dict] = []
    failure_buffer: list[dict] = []
    started_at = time.time()
    total = len(pending_eval_ids)

    for processed in range(1, total + 1):
        result = await result_queue.get()
        checkpoint_buffer.append(result["checkpoint_row"])
        score_buffer.extend(result["score_rows"])
        if result["adaptation_row"] is not None:
            adaptation_buffer.append(result["adaptation_row"])
        failure_buffer.extend(result["failures"])

        if processed % checkpoint_every == 0 or processed == total:
            append_jsonl(checkpoint_path, checkpoint_buffer)
            append_jsonl(scores_out, score_buffer)
            append_jsonl(adaptation_out, adaptation_buffer)
            append_jsonl(failure_log_path, failure_buffer)
            checkpoint_buffer.clear()
            score_buffer.clear()
            adaptation_buffer.clear()
            failure_buffer.clear()
            print_progress(processed, total, started_at)

    await asyncio.gather(*workers)

def main() -> None:
    """Entrypoint for FPS judging over sampled adaptations."""
    args = parse_args()
    sample_eval_ids = load_sample_eval_ids(args.sample_ids_path)
    grouped_rows = load_grouped_rows(args.paired_frames_path, sample_eval_ids)
    pair_meta = build_pair_meta(grouped_rows)

    checkpoint_rows = load_jsonl(args.checkpoint_path)
    completed_ids = {row["eval_id"] for row in checkpoint_rows}
    score_rows, adaptation_rows = rewrite_outputs_from_checkpoint(
        checkpoint_rows,
        args.scores_out,
        args.adaptation_out,
    )

    pending_eval_ids = [eval_id for eval_id in grouped_rows if eval_id not in completed_ids]
    if args.limit_adaptations is not None:
        pending_eval_ids = pending_eval_ids[: args.limit_adaptations]

    print(f"Sample eval IDs loaded: {len(sample_eval_ids)}")
    print(f"Sample adaptations with paired frames: {len(grouped_rows)}")
    print(f"Completed from checkpoint: {len(completed_ids & set(grouped_rows))}")
    print(f"Pending adaptations: {len(pending_eval_ids)}")
    print(
        f"Paired CSI rows in sample: "
        f"{sum(len(rows) for rows in grouped_rows.values())}"
    )

    if pending_eval_ids:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        args.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(
            run_async(
                grouped_rows,
                pending_eval_ids,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                checkpoint_every=args.checkpoint_every,
                checkpoint_path=args.checkpoint_path,
                scores_out=args.scores_out,
                adaptation_out=args.adaptation_out,
                failure_log_path=args.failure_log_path,
            )
        )

        checkpoint_rows = load_jsonl(args.checkpoint_path)
        score_rows, adaptation_rows = rewrite_outputs_from_checkpoint(
            checkpoint_rows,
            args.scores_out,
            args.adaptation_out,
        )

    failure_rows = load_jsonl(args.failure_log_path)

    print(f"Wrote per-pair scores: {args.scores_out}")
    print(f"Wrote per-adaptation scores: {args.adaptation_out}")
    print(f"Total adaptation summaries: {len(adaptation_rows)}")
    print_summary(score_rows, failure_rows, pair_meta)

if __name__ == "__main__":
    main()
