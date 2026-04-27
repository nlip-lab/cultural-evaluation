"""
Phase 9 - Round-trip CSI extraction for re-Americanized texts.

This script extracts American-context CSIs from each reverse-adapted text so
downstream phases can compare recovered, substituted, and leaked items.
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR = EVAL_DIR / "07_Reverse"
RUNS_DIR = PROC_DIR / "reverse_adapted_data" / "runs"
CREDS_PATH = None
_creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds_env:
    CREDS_PATH = Path(_creds_env)
PROMPT_TEMPLATE = (EVAL_DIR / "prompts" / "source_csi_extraction.txt").read_text()

OUT_PATH = None
STATS_PATH = None
CKPT_PATH = None

GOOGLE_MODEL = "gemini-2.5-flash-lite"
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
CHECKPOINT_N = 200
CONCURRENCY = 100
RETRY_DELAY = 30

REQUIRED_FIELDS = {
    "span", "char_start", "char_end", "category",
    "adaptation_necessity", "salience", "local_role",
    "scale_sensitive", "figurative", "dependencies",
}
VALID_CATEGORIES = {
    "food_drink", "place_institution_brand", "unit_currency_scale",
    "holiday_event_ritual", "personal_name_identity", "social_norm_address",
    "pop_culture_sport_media", "idiom_figurative",
}
VALID_NECESSITY = {"must_adapt", "may_adapt", "should_retain"}

PROMPT_LEVELS = {"cot", "instructed", "oneshot", "simple"}
DOMAINS = {"dialogue", "educational", "procedural", "stories"}
VALID_MODELS = {"gemma2", "llama", "llama8b", "mistral"}


def parse_args():
    """Parse command-line arguments for CSI extraction."""
    p = argparse.ArgumentParser(description="Phase 9: CSI extraction on re-Americanized reverse-adapted texts")
    p.add_argument("--backend", choices=["google", "openai"], default="google", help="LLM backend for CSI extraction")
    p.add_argument("--test", action="store_true", help="Run on 5 items only to verify the pipeline end-to-end")
    p.add_argument("--domain", choices=list(DOMAINS), default=None, help="Restrict to a single domain (optional)")
    p.add_argument("--prompt-level", choices=list(PROMPT_LEVELS), default=None, dest="prompt_level", help="Restrict to a single prompt level (optional)")
    p.add_argument("--model", default=None, help="Restrict to a single reverse-adaptation model (optional)")
    return p.parse_args()


def _load_google_creds():
    """Load Google service-account credentials."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_PATH),
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _refresh_if_needed(creds):
    """Refresh a Google credential if it is near expiry."""
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


async def call_google(prompt: str, *, loop, lock) -> str:
    """Call the Google backend with retries and token refresh handling."""
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

    for attempt in range(6):
        if attempt > 0:
            async with lock:
                _google_creds = await loop.run_in_executor(None, _refresh_if_needed, _google_creds)
                token = _google_creds.token
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        try:
            raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            data = json.loads(raw)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            if e.code in (429, 503):
                base = RETRY_DELAY * (2 ** attempt)
                jitter = random.uniform(0, base * 0.3)
                wait = min(base + jitter, 300)
                print(f"\n    [429/503 attempt {attempt + 1}] waiting {wait:.0f}s ...", flush=True)
                await asyncio.sleep(wait)
            elif e.code == 500:
                await asyncio.sleep(5 + random.uniform(0, 5))
            else:
                raise RuntimeError(f"Google API {e.code}: {err_body[:200]}")
    raise RuntimeError("Google API: max retries exceeded")


async def call_openai(prompt: str, *, client) -> str:
    """Call the OpenAI backend with retries."""
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
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                base = RETRY_DELAY * (2 ** attempt)
                jitter = random.uniform(0, base * 0.3)
                await asyncio.sleep(min(base + jitter, 300))
            else:
                raise
    raise RuntimeError("OpenAI: max retries exceeded")


def _extract_json_array(text: str):
    """Extract a JSON array from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text.startswith("{"):
        obj = json.loads(text)
        for v in obj.values():
            if isinstance(v, list):
                return v
        raise ValueError("JSON object has no list value")
    return json.loads(text)


def _fuzzy_fix_offsets(csi: dict, source_text: str) -> dict:
    """Correct the offsets when the returned span still exists in the source."""
    span = csi.get("span", "")
    if not span:
        return csi
    cs, ce = csi.get("char_start", -1), csi.get("char_end", -1)
    if 0 <= cs < ce <= len(source_text) and source_text[cs:ce] == span:
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
    """Fill defaults, normalize categories, and repair spans when possible."""
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
    """Extract CSIs for one reverse-adapted item."""
    rev_id = item["rev_id"]
    original_id = item["original_id"]
    domain = item["domain"]
    adapted_text = item["adapted_text"]

    prompt = PROMPT_TEMPLATE.replace("{source_text}", adapted_text).replace("{domain}", domain)

    status = "success"
    csis = []

    async with sem:
        try:
            raw = await caller(prompt)
        except Exception as e:
            return _error_record(item, "api_error", str(e))

        try:
            csis = _extract_json_array(raw)
        except Exception:
            nudge = prompt + "\n\nReturn ONLY valid JSON array."
            try:
                raw2 = await caller(nudge)
                csis = _extract_json_array(raw2)
                status = "parse_retry"
            except Exception:
                return _error_record(item, "parse_failed", raw)

    if not csis:
        status = "no_csis"

    csis = validate_and_fix(csis, adapted_text)

    return {
        "rev_id": rev_id,
        "original_id": original_id,
        "domain": domain,
        "rev_model": item["model"],
        "prompt_level": item["prompt_level"],
        "fwd_prompt_level": item.get("fwd_prompt_level", ""),
        "fwd_model": item["model"],
        "adapted_text": adapted_text,
        "original_american_text": item.get("original_american_text", ""),
        "indian_text": item.get("indian_text", ""),
        "csi_count": len(csis),
        "csis": csis,
        "extraction_status": status,
    }


def _error_record(item: dict, status: str, detail: str) -> dict:
    """Return a normalized error record for failed extraction."""
    return {
        "rev_id": item["rev_id"],
        "original_id": item["original_id"],
        "domain": item["domain"],
        "rev_model": item["model"],
        "prompt_level": item["prompt_level"],
        "adapted_text": item["adapted_text"],
        "original_american_text": item.get("original_american_text", ""),
        "indian_text": item.get("indian_text", ""),
        "csi_count": 0,
        "csis": [],
        "extraction_status": status,
        "_error_detail": detail[:500],
    }


def load_reverse_items(*, domain_filter: str | None = None, prompt_level_filter: str | None = None, model_filter: str | None = None) -> list[dict]:
    """Load reverse-adapted items from the runs directory."""
    items: list[dict] = []
    skipped_empty = 0
    skipped_header = 0

    prompt_levels = [prompt_level_filter] if prompt_level_filter else sorted(PROMPT_LEVELS)

    for pl in prompt_levels:
        pl_dir = RUNS_DIR / pl
        if not pl_dir.is_dir():
            print(f"    WARNING: prompt-level dir not found: {pl_dir}")
            continue

        for fpath in sorted(pl_dir.glob("*.jsonl")):
            stem_parts = fpath.stem.split("_", 1)
            if len(stem_parts) != 2:
                continue
            file_model, file_domain = stem_parts

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
                        skipped_header += 1
                        continue
                    adapted = rec.get("adapted_text", "")
                    if not adapted or not adapted.strip():
                        skipped_empty += 1
                        continue

                    items.append({
                        "rev_id": rec.get("rev_id", ""),
                        "original_id": rec.get("id", ""),
                        "domain": rec.get("domain", file_domain),
                        "model": rec.get("model", file_model),
                        "prompt_level": rec.get("prompt_level", pl),
                        "fwd_prompt_level": rec.get("fwd_prompt_level", ""),
                        "adapted_text": adapted,
                        "original_american_text": rec.get("original_american_text", ""),
                        "indian_text": rec.get("indian_text", ""),
                    })

    print(f"    Skipped {skipped_header:,} header lines, {skipped_empty:,} empty adapted_text records")
    return items


def load_checkpoint() -> set[str]:
    """Load the saved checkpoint set."""
    if CKPT_PATH.exists():
        data = json.loads(CKPT_PATH.read_text())
        return set(data.get("done_ids", []))
    return set()


def save_checkpoint(done_ids: set[str]):
    """Write the checkpoint set to disk."""
    CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))


def prepare_run_state(*, test_mode: bool) -> set[str]:
    """Resolve the starting checkpoint state."""
    if test_mode:
        return set()
    if CKPT_PATH.exists() and OUT_PATH.exists():
        return load_checkpoint()
    if CKPT_PATH.exists() or OUT_PATH.exists():
        print("    Partial stale state found - starting fresh.")
        CKPT_PATH.unlink(missing_ok=True)
        OUT_PATH.unlink(missing_ok=True)
        return set()
    return set()


def _set_paths(backend: str):
    """Set backend-specific output, stats, and checkpoint paths."""
    global OUT_PATH, STATS_PATH, CKPT_PATH
    tag = "google" if backend == "google" else "openai"
    OUT_PATH = PHASE_DIR / f"phase9_csi_inventory_{tag}.jsonl"
    STATS_PATH = PHASE_DIR / f"phase9_csi_stats_{tag}.json"
    CKPT_PATH = PHASE_DIR / f".phase9_checkpoint_{tag}.json"


def compute_stats(records: list[dict]) -> dict:
    """Compute summary statistics for the extracted CSIs."""
    total = len(records)
    items_with_csis = sum(1 for r in records if r["csi_count"] > 0)
    items_no_csis = sum(1 for r in records if r["extraction_status"] == "no_csis")
    parse_failures = sum(1 for r in records if r["extraction_status"] == "parse_failed")
    api_errors = sum(1 for r in records if r["extraction_status"] == "api_error")
    total_csis = sum(r["csi_count"] for r in records)

    by_domain: dict[str, dict] = defaultdict(lambda: {"count": 0, "csis": 0})
    by_model: dict[str, dict] = defaultdict(lambda: {"count": 0, "csis": 0})
    by_prompt_level: dict[str, dict] = defaultdict(lambda: {"count": 0, "csis": 0})
    cat_dist: dict[str, int] = defaultdict(int)
    nec_dist: dict[str, int] = defaultdict(int)
    salience_all: list[float] = []

    for r in records:
        d = r["domain"]
        m = r["rev_model"]
        pl = r["prompt_level"]
        by_domain[d]["count"] += 1
        by_domain[d]["csis"] += r["csi_count"]
        by_model[m]["count"] += 1
        by_model[m]["csis"] += r["csi_count"]
        by_prompt_level[pl]["count"] += 1
        by_prompt_level[pl]["csis"] += r["csi_count"]
        for csi in r["csis"]:
            cat_dist[csi.get("category", "unknown")] += 1
            nec_dist[csi.get("adaptation_necessity", "unknown")] += 1
            sal = csi.get("salience")
            if isinstance(sal, (int, float)):
                salience_all.append(sal)

    def avg_csi(grp: dict) -> float:
        return round(grp["csis"] / grp["count"], 3) if grp["count"] else 0.0

    return {
        "total_items": total,
        "items_with_csis": items_with_csis,
        "items_no_csis": items_no_csis,
        "parse_failures": parse_failures,
        "api_errors": api_errors,
        "total_csis_extracted": total_csis,
        "avg_csis_per_item": round(total_csis / total, 3) if total else 0.0,
        "avg_salience_global": round(sum(salience_all) / len(salience_all), 3) if salience_all else 0.0,
        "by_domain": {
            d: {"count": v["count"], "total_csis": v["csis"], "avg_csis": avg_csi(v)}
            for d, v in by_domain.items()
        },
        "by_rev_model": {
            m: {"count": v["count"], "total_csis": v["csis"], "avg_csis": avg_csi(v)}
            for m, v in by_model.items()
        },
        "by_prompt_level": {
            pl: {"count": v["count"], "total_csis": v["csis"], "avg_csis": avg_csi(v)}
            for pl, v in by_prompt_level.items()
        },
        "category_distribution": dict(cat_dist),
        "necessity_distribution": dict(nec_dist),
    }


async def main():
    """Run the full extraction pipeline and write outputs plus stats."""
    args = parse_args()
    backend = args.backend
    test_mode = args.test

    _set_paths(backend)

    print(f"{'=' * 70}")
    print(f"Phase 9 - Round-Trip CSI Extraction  [backend={backend}]")
    print(f"{'=' * 70}")

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY + 20)
    loop = asyncio.get_event_loop()
    loop.set_default_executor(_executor)

    print(f"\n[0] Testing {backend} API ...")

    if backend == "google":
        if CREDS_PATH is None:
            print("    FATAL: GOOGLE_APPLICATION_CREDENTIALS is not set")
            sys.exit(1)
        lock = asyncio.Lock()
        caller = lambda p: call_google(p, loop=loop, lock=lock)
        try:
            resp = await caller("Return ONLY this JSON: []")
            parsed = _extract_json_array(resp)
            assert isinstance(parsed, list)
            print(f"    Google API OK - model={GOOGLE_MODEL}")
        except Exception as e:
            print(f"    FATAL: Google API test failed: {e}")
            sys.exit(1)
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print("    FATAL: 'openai' package not installed. Run: pip install openai")
            sys.exit(1)
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            env_file = EVAL_DIR / "open_ai.env"
            if env_file.exists():
                for raw_line in env_file.read_text().splitlines():
                    raw_line = raw_line.strip()
                    if raw_line.startswith("OPENAI_API_KEY"):
                        _, _, val = raw_line.partition("=")
                        os.environ["OPENAI_API_KEY"] = val.strip().strip('"').strip("'")
                        api_key = os.environ["OPENAI_API_KEY"]
                        break
        if not api_key:
            print("    FATAL: OPENAI_API_KEY not set and not found in open_ai.env")
            sys.exit(1)
        oai_client = AsyncOpenAI(api_key=api_key)
        caller = lambda p: call_openai(p, client=oai_client)
        try:
            resp = await caller("Return ONLY this JSON array: []")
            parsed = _extract_json_array(resp)
            assert isinstance(parsed, list)
            print(f"    OpenAI API OK - model={OPENAI_MODEL}")
        except Exception as e:
            print(f"    FATAL: OpenAI API test failed: {e}")
            sys.exit(1)

    print("\n[1] Loading reverse-adapted items ...")
    items = load_reverse_items(
        domain_filter=args.domain,
        prompt_level_filter=args.prompt_level,
        model_filter=args.model,
    )
    print(f"    {len(items):,} reverse-adapted items loaded")

    if test_mode:
        items = items[:5]
        print(f"    TEST MODE: running on {len(items)} items only")

    done_ids = prepare_run_state(test_mode=test_mode)
    remaining = [it for it in items if it["rev_id"] not in done_ids]
    print(f"    Already done: {len(done_ids):,} | Remaining: {len(remaining):,}")

    if not remaining:
        print("    All items already processed.  Computing stats ...")
    else:
        print(f"\n[2] Extracting CSIs (concurrency={CONCURRENCY}) ...")
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [process_item(it, caller=caller, sem=sem) for it in remaining]

        completed: list[dict] = []
        t0 = time.time()
        t_win = t0
        i_win = 0
        total = len(tasks)
        already = len(done_ids)

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            completed.append(result)
            done_ids.add(result["rev_id"])

            if i % 200 == 0:
                t_win = time.time()
                i_win = i

            win_secs = (time.time() - t_win) or 0.001
            win_items = (i - i_win) or i
            rate = win_items / win_secs
            eta_s = (total - i) / rate if rate > 0 else 0
            total_done = already + i
            total_all = already + total
            pct = total_done / total_all * 100

            line = (
                f"  {total_done:>6}/{total_all} ({pct:.1f}%) | "
                f"{rate:>5.1f} it/s | ETA {eta_s/60:.1f}m | "
                f"{result['extraction_status']}"
            )
            if i % 500 == 0 or i == total:
                print(f"\r{line}", flush=True)
            else:
                print(f"\r{line}", end="", flush=True)

            if i % CHECKPOINT_N == 0:
                with open(OUT_PATH, "a", encoding="utf-8") as fout:
                    for r in completed:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                save_checkpoint(done_ids)
                completed.clear()

        if completed:
            with open(OUT_PATH, "a", encoding="utf-8") as fout:
                for r in completed:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            save_checkpoint(done_ids)

    print("\n[3] Computing summary stats ...")
    all_records: list[dict] = []
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    stats = compute_stats(all_records)
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total items processed    : {stats['total_items']:,}")
    print(f"  Items with CSIs          : {stats['items_with_csis']:,}")
    print(f"  Items with no CSIs       : {stats['items_no_csis']:,}")
    print(f"  Parse failures           : {stats['parse_failures']:,}")
    print(f"  API errors               : {stats['api_errors']:,}")
    print(f"  Total CSIs extracted     : {stats['total_csis_extracted']:,}")
    print(f"  Avg CSIs / item          : {stats['avg_csis_per_item']:.2f}")

    print("\n  By domain:")
    for d, v in sorted(stats["by_domain"].items()):
        print(f"    {d:<20} {v['count']:>7,} items   {v['avg_csis']:.2f} CSIs/item")

    print("\n  By reverse model:")
    for m, v in sorted(stats["by_rev_model"].items()):
        print(f"    {m:<20} {v['count']:>7,} items   {v['avg_csis']:.2f} CSIs/item")

    print("\n  By prompt level:")
    for pl, v in sorted(stats["by_prompt_level"].items()):
        print(f"    {pl:<20} {v['count']:>7,} items   {v['avg_csis']:.2f} CSIs/item")

    print("\n  Category distribution:")
    for cat, cnt in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<35} {cnt:>7,}")

    print("\n  Necessity distribution:")
    for nec, cnt in sorted(stats["necessity_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {nec:<20} {cnt:>7,}")

    print(f"\n  Output   : {OUT_PATH}")
    print(f"  Stats    : {STATS_PATH}")

    if test_mode:
        print("\n  [TEST MODE] Sample output records:")
        for r in all_records[:5]:
            print(f"\n  rev_id      : {r['rev_id']}")
            print(f"  original_id : {r['original_id']}")
            print(f"  domain      : {r['domain']}")
            print(f"  rev_model   : {r['rev_model']}")
            print(f"  prompt_level: {r['prompt_level']}")
            print(f"  status      : {r['extraction_status']}")
            print(f"  csi_count   : {r['csi_count']}")
            if r["csis"]:
                print(f"  first CSI   : {json.dumps(r['csis'][0], indent=4)}")

    if not test_mode and CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print("\n  Checkpoint file removed (clean run).")


if __name__ == "__main__":
    asyncio.run(main())
