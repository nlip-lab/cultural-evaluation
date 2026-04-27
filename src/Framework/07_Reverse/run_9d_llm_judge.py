"""
Phase 9d - LLM semantic equivalence judge.

For each item in the full round-trip dataset, the judge scores whether the
re-Americanized text is semantically equivalent to the original American
reference.
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


class _Tee:
    """Write output to both stdout and a log file."""

    def __init__(self, log_path: Path):
        """Create a tee stream bound to the given log path."""
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.__stdout__

    def write(self, s):
        """Write to both sinks."""
        self._stdout.write(s)
        self._log.write(s)

    def flush(self):
        """Flush both sinks."""
        self._stdout.flush()
        self._log.flush()

    def fileno(self):
        """Expose the stdout file descriptor."""
        return self._stdout.fileno()


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = PROJECT_ROOT / "03_data" / "processed" / "reverse_adapted_data" / "runs"
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR = EVAL_DIR / "07_Reverse"
CREDS_PATH = None
_creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds_env:
    CREDS_PATH = Path(_creds_env)
OUT_PATH = PHASE_DIR / "phase9d_llm_judge.jsonl"
STATS_PATH = PHASE_DIR / "phase9d_llm_judge_stats.json"
CKPT_PATH = PHASE_DIR / ".phase9d_checkpoint.json"
LOG_PATH = PHASE_DIR / "logs" / "phase9d_llm_judge.log"

DOMAINS = {"dialogue", "educational", "procedural", "stories"}
PROMPT_LEVELS = {"cot", "instructed", "oneshot", "simple"}
VALID_MODELS = {"gemma2", "llama", "llama8b", "mistral"}

GOOGLE_MODEL = "gemini-2.5-flash-lite"
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 512
CONCURRENCY = 80
CONCURRENCY_PROD = 100
RETRY_DELAY = 30
CHECKPOINT_N = 200
OPENAI_BATCH_SIZE = 45_000
POLL_INTERVAL = 30

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether two texts in an American cultural context convey the same meaning after a cultural round-trip (American -> Indian -> American).

TEXT A (original American):
{text_a}

TEXT B (re-Americanized after Indian cultural adaptation):
{text_b}

Score Text B relative to Text A on these dimensions (integer 0-10 each):

  meaning_preservation   : Same facts, events, logical relationships?
  structural_preservation: Same sentence and discourse flow?
  detail_preservation    : Specific numbers, names, brands, dates retained?
  overall                : Holistic round-trip fidelity (may be non-integer)

Also give a one-sentence rationale describing the main difference, if any.

Return ONLY a JSON object with exactly these keys:
{
  "meaning_preservation": <int 0-10>,
  "structural_preservation": <int 0-10>,
  "detail_preservation": <int 0-10>,
  "overall": <float 0-10>,
  "rationale": "<one sentence>"
}
"""


def parse_args():
    """Parse command-line arguments for the judge runner."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["google", "openai"], default="google")
    p.add_argument("--test", action="store_true", help="10 items only")
    p.add_argument("--domain", choices=list(DOMAINS), default=None)
    p.add_argument("--prompt-level", choices=list(PROMPT_LEVELS), default=None, dest="prompt_level")
    p.add_argument("--model", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-sample", type=int, default=None, dest="n_sample", help="Randomly sample N items from the full set (default: all)")
    return p.parse_args()


def _load_google_creds():
    """Load and refresh Google service-account credentials."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_PATH),
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _refresh_if_needed(creds):
    """Refresh credentials when they are close to expiry."""
    from datetime import timezone, datetime
    import google.auth.transport.requests

    if creds.expiry:
        expiry = creds.expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if (expiry - datetime.now(timezone.utc)).total_seconds() < 300:
            creds.refresh(google.auth.transport.requests.Request())
    return creds


_google_creds = None


async def call_gemini(prompt: str, *, loop, lock) -> str:
    """Call Gemini with retry/backoff handling."""
    global _google_creds
    async with lock:
        if _google_creds is None:
            _google_creds = await loop.run_in_executor(None, _load_google_creds)
        _google_creds = await loop.run_in_executor(None, _refresh_if_needed, _google_creds)
        token = _google_creds.token

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_MODEL}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
            "responseMimeType": "application/json",
        },
    }).encode()

    for attempt in range(4):
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        try:
            raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            data = json.loads(raw)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode()
            if e.code in (429, 503):
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"\n    [9d {e.code} attempt {attempt + 1}] backoff {wait:.0f}s ...", flush=True)
                await asyncio.sleep(wait)
            elif e.code == 500:
                await asyncio.sleep(5 + random.uniform(0, 5))
            else:
                raise RuntimeError(f"Gemini API {e.code}: {body_txt[:200]}")
    raise RuntimeError("Gemini: max retries exceeded")


def _build_prompt(item: dict) -> str:
    """Build the judge prompt for one item."""
    return JUDGE_PROMPT_TEMPLATE.format(
        text_a=item["reference"],
        text_b=item["hypothesis"],
    )


def run_async_google(items: list[dict], out_path: Path) -> None:
    """Run high-concurrency Gemini judging and checkpoint progress."""
    async def _run():
        loop = asyncio.get_event_loop()
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(CONCURRENCY_PROD)
        caller = lambda p: call_gemini(p, loop=loop, lock=lock)
        tasks = [judge_item(it, caller=caller, sem=sem) for it in items]
        pending: list[dict] = []
        t0 = time.time()
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            pending.append(await coro)
            if i % CHECKPOINT_N == 0 or i == len(tasks):
                with open(out_path, "a", encoding="utf-8") as fout:
                    for r in pending:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                pending.clear()
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(tasks) - i) / rate if rate > 0 else 0
                print(f"  {i:>6}/{len(tasks)}  {rate:.1f}/s  ETA {eta/60:.1f}m", flush=True)

    print(f"  Gemini async: {len(items):,} items  concurrency={CONCURRENCY_PROD}")
    asyncio.run(_run())


def run_batch_openai(items: list[dict]) -> list[dict]:
    """Run the judge through the OpenAI batch API and return parsed rows."""
    import openai

    client = openai.OpenAI()
    chunks = [items[i:i + OPENAI_BATCH_SIZE] for i in range(0, len(items), OPENAI_BATCH_SIZE)]
    print(f"  OpenAI batch: {len(items):,} items -> {len(chunks)} job(s)")

    batches: list[tuple[list[dict], object]] = []
    for ci, chunk in enumerate(chunks):
        lines = [json.dumps({
            "custom_id": item["rev_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": _build_prompt(item)}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        }) for item in chunk]
        jsonl_bytes = "\n".join(lines).encode()
        print(f"  [chunk {ci+1}/{len(chunks)}] uploading {len(jsonl_bytes)/1e6:.1f} MB ...", flush=True)
        bf = client.files.create(file=("batch.jsonl", jsonl_bytes), purpose="batch")
        batch = client.batches.create(
            input_file_id=bf.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"    -> batch_id={batch.id}  status={batch.status}")
        batches.append((chunk, batch))

    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        batches = [(chunk, client.batches.retrieve(b.id)) for chunk, b in batches]
        states = [b.status for _, b in batches]
        summary = "  |  ".join(
            f"job{i+1}: {b.status} ({b.request_counts.completed}/{b.request_counts.total})"
            for i, (_, b) in enumerate(batches)
        )
        print(f"\r  {summary}", end="", flush=True)
        if all(st in terminal for st in states):
            print()
            break
        time.sleep(POLL_INTERVAL)

    all_results: list[dict] = []
    for chunk, batch in batches:
        if batch.status != "completed":
            print(f"  WARNING: batch {batch.id} ended with {batch.status}")
            for item in chunk:
                all_results.append(_make_row(item, scores=None, status="batch_error"))
            continue
        content = client.files.content(batch.output_file_id).text
        result_map: dict[str, dict | None] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            if obj.get("error"):
                result_map[custom_id] = None
            else:
                try:
                    raw = obj["response"]["body"]["choices"][0]["message"]["content"]
                    result_map[custom_id] = _parse_response(raw)
                except Exception:
                    result_map[custom_id] = None
        for item in chunk:
            scores = result_map.get(item["rev_id"])
            all_results.append(_make_row(item, scores, status="success" if scores else "parse_failed"))
    return all_results


def _make_row(item: dict, scores: dict | None, status: str) -> dict:
    """Build one output row for a judged item."""
    def _s(key):
        return scores.get(key) if scores else None

    return {
        "rev_id": item["rev_id"],
        "original_id": item["original_id"],
        "domain": item["domain"],
        "rev_model": item["model"],
        "prompt_level": item["prompt_level"],
        "fwd_prompt_level": item["fwd_prompt_level"],
        "fwd_model": item.get("fwd_model", ""),
        "ref_len": len(item["reference"].split()),
        "hyp_len": len(item["hypothesis"].split()),
        "meaning_preservation": _s("meaning_preservation"),
        "structural_preservation": _s("structural_preservation"),
        "detail_preservation": _s("detail_preservation"),
        "overall": _s("overall"),
        "rationale": _s("rationale") or "",
        "judge_status": status,
    }


def _parse_response(raw: str) -> dict | None:
    """Parse the judge response and clamp numeric scores to the expected range."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except Exception:
                return None
        else:
            return None

    required = {"meaning_preservation", "structural_preservation", "detail_preservation", "overall", "rationale"}
    if not required.issubset(obj.keys()):
        return None

    for key in ("meaning_preservation", "structural_preservation", "detail_preservation", "overall"):
        try:
            val = float(obj[key])
            obj[key] = round(min(10.0, max(0.0, val)), 2)
        except (TypeError, ValueError):
            obj[key] = None
    return obj


async def judge_item(item: dict, *, caller, sem) -> dict:
    """Judge one item with a retry nudge if the first response is not parseable."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        text_a=item["reference"],
        text_b=item["hypothesis"],
    )
    scores = None
    status = "success"
    raw = ""

    async with sem:
        try:
            raw = await caller(prompt)
            scores = _parse_response(raw)
            if scores is None:
                nudge = prompt + "\n\nReturn ONLY the JSON object, no other text."
                raw2 = await caller(nudge)
                scores = _parse_response(raw2)
                if scores is None:
                    status = "parse_failed"
        except Exception:
            status = "api_error"
            scores = None

    def _s(key):
        return scores.get(key) if scores else None

    return {
        "rev_id": item["rev_id"],
        "original_id": item["original_id"],
        "domain": item["domain"],
        "rev_model": item["model"],
        "prompt_level": item["prompt_level"],
        "fwd_prompt_level": item["fwd_prompt_level"],
        "fwd_model": item.get("fwd_model", ""),
        "ref_len": len(item["reference"].split()),
        "hyp_len": len(item["hypothesis"].split()),
        "meaning_preservation": _s("meaning_preservation"),
        "structural_preservation": _s("structural_preservation"),
        "detail_preservation": _s("detail_preservation"),
        "overall": _s("overall"),
        "rationale": _s("rationale") or "",
        "judge_status": status,
    }


def load_items(domain_filter=None, prompt_level_filter=None, model_filter=None) -> list[dict]:
    """Load reverse-adapted evaluation items from the processed runs tree."""
    items = []
    for pl in sorted(PROMPT_LEVELS):
        if prompt_level_filter and pl != prompt_level_filter:
            continue
        pl_dir = RUNS_DIR / pl
        if not pl_dir.is_dir():
            continue
        for fpath in sorted(pl_dir.glob("*.jsonl")):
            parts = fpath.stem.split("_", 1)
            if len(parts) != 2:
                continue
            file_model, file_domain = parts
            if file_model not in VALID_MODELS:
                continue
            if domain_filter and file_domain != domain_filter:
                continue
            if model_filter and file_model != model_filter:
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("is_config_header"):
                        continue
                    adapted = rec.get("adapted_text", "")
                    ref = rec.get("original_american_text", "")
                    if not adapted or not adapted.strip() or not ref or not ref.strip():
                        continue
                    model = rec.get("model", file_model)
                    items.append({
                        "rev_id": rec.get("rev_id", ""),
                        "original_id": rec.get("id", ""),
                        "domain": rec.get("domain", file_domain),
                        "model": model,
                        "prompt_level": rec.get("prompt_level", pl),
                        "fwd_prompt_level": rec.get("fwd_prompt_level", ""),
                        "fwd_model": model,
                        "hypothesis": adapted,
                        "reference": ref,
                    })
    print(f"  Loaded {len(items):,} items")
    return items


def load_done_ids() -> set[str]:
    """Load rev_ids already written to the checkpoint file."""
    if CKPT_PATH.exists():
        return set(json.loads(CKPT_PATH.read_text()).get("done_ids", []))
    return set()


def save_checkpoint(done_ids: set[str]):
    """Persist the checkpoint set."""
    CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))


def compute_stats(records: list[dict]) -> dict:
    """Compute aggregate judge metrics over all records."""
    def _avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    def _grp(recs):
        ok = [r for r in recs if r["judge_status"] == "success"]
        return {
            "n": len(recs),
            "n_judged_ok": len(ok),
            "judge_ok_rate": round(len(ok) / len(recs), 4) if recs else 0,
            "avg_meaning": _avg([r.get("meaning_preservation") for r in ok]),
            "avg_structural": _avg([r.get("structural_preservation") for r in ok]),
            "avg_detail": _avg([r.get("detail_preservation") for r in ok]),
            "avg_overall": _avg([r.get("overall") for r in ok]),
        }

    def _bucket(key, recs):
        grps: dict[str, list] = defaultdict(list)
        for r in recs:
            grps[r.get(key, "")].append(r)
        return grps

    return {
        "total": len(records),
        "model": GOOGLE_MODEL,
        "global": _grp(records),
        "by_domain": {k: _grp(v) for k, v in _bucket("domain", records).items()},
        "by_rev_model": {k: _grp(v) for k, v in _bucket("rev_model", records).items()},
        "by_prompt_level": {k: _grp(v) for k, v in _bucket("prompt_level", records).items()},
        "by_fwd_prompt_level": {k: _grp(v) for k, v in _bucket("fwd_prompt_level", records).items()},
    }


def main():
    """Run the full judge workflow and emit records plus aggregate stats."""
    args = parse_args()

    if not args.test:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(LOG_PATH)

    print("=" * 70)
    print("Phase 9d - LLM Semantic Equivalence Judge  [backend=google/openai]")
    print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not CREDS_PATH.exists():
        print(f"FATAL: {CREDS_PATH} not found.")
        sys.exit(1)

    print("\n[1] Loading items ...")
    all_items = load_items(
        domain_filter=args.domain,
        prompt_level_filter=args.prompt_level,
        model_filter=args.model,
    )

    if args.test:
        items = all_items[:10]
        print(f"  TEST MODE: {len(items)} items")
    elif args.n_sample and args.n_sample < len(all_items):
        import random as _rng

        _rng.seed(42)
        items = _rng.sample(all_items, args.n_sample)
        print(f"  Sampled {len(items):,} / {len(all_items):,} items (--n-sample)")
    else:
        items = all_items
        print(f"  Full set: {len(items):,} items")

    done_ids = load_done_ids() if (args.resume and not args.test) else set()
    items = [it for it in items if it["rev_id"] not in done_ids]
    print(f"  To judge: {len(items):,}  (already done: {len(done_ids):,})")

    if not items:
        print("  Nothing left to judge.")
        return

    print("\n[2] Judging ...")

    mode = "a" if done_ids else "w"
    if mode == "w":
        open(OUT_PATH, "w").close()

    if args.test:
        async def _realtime():
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY + 20)
            loop = asyncio.get_event_loop()
            loop.set_default_executor(_executor)
            lock = asyncio.Lock()
            sem = asyncio.Semaphore(CONCURRENCY)
            caller = lambda p: call_gemini(p, loop=loop, lock=lock)
            tasks = [judge_item(it, caller=caller, sem=sem) for it in items]
            rows = []
            t0 = time.time()
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                r = await coro
                rows.append(r)
                if i % 5 == 0 or i == len(tasks):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(tasks)}  {rate:.1f}/s  overall={r.get('overall')}  status={r['judge_status']}")
            return rows

        results = asyncio.run(_realtime())
        with open(OUT_PATH, "a", encoding="utf-8") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        if args.backend == "google":
            if CREDS_PATH is None:
                print("FATAL: GOOGLE_APPLICATION_CREDENTIALS is not set")
                sys.exit(1)
            run_async_google(items, OUT_PATH)
        else:
            results = run_batch_openai(items)
            with open(OUT_PATH, "a", encoding="utf-8") as fout:
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n[3] Computing stats ...")
    all_records = []
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    stats = compute_stats(all_records)
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    g = stats["global"]
    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    print(f"  Items judged          : {stats['total']:,}")
    print(f"  Judge OK rate         : {g['judge_ok_rate']:.1%}")
    print(f"  Avg meaning (0-10)    : {g['avg_meaning']}")
    print(f"  Avg structural (0-10) : {g['avg_structural']}")
    print(f"  Avg detail (0-10)     : {g['avg_detail']}")
    print(f"  Avg overall (0-10)    : {g['avg_overall']}")
    print("\n  By domain:")
    for d, v in sorted(stats["by_domain"].items()):
        print(f"    {d:<20} overall={v['avg_overall']}  meaning={v['avg_meaning']}  detail={v['avg_detail']}")
    print("\n  By rev model:")
    for m, v in sorted(stats["by_rev_model"].items()):
        print(f"    {m:<20} overall={v['avg_overall']}")
    print("\n  By fwd model:")
    for m, v in sorted(stats["by_fwd_model"].items()):
        print(f"    {m:<20} overall={v['avg_overall']}")
    print(f"\n  Output : {OUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")

    if args.test:
        print("\n  [TEST] Sample records:")
        for r in all_records[:3]:
            print(f"\n  rev_id={r['rev_id']}")
            print(f"  overall={r['overall']}  status={r['judge_status']}")
            print(f"  rationale: {r['rationale']}")

    if not args.test and CKPT_PATH.exists():
        CKPT_PATH.unlink()

    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not args.test:
        print(f"  Log     : {LOG_PATH}")


if __name__ == "__main__":
    main()
