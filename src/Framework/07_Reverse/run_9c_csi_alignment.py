"""
Phase 9c - CSI round-trip alignment using an LLM judge.

For each re-Americanized item, the judge aligns its extracted CSIs from Phase 9
back to the original American source CSIs from Phase 1.
"""

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
PHASE1_DIR = EVAL_DIR / "01_CSI_Extraction"
PHASE9_DIR = EVAL_DIR / "07_Reverse"
CREDS_PATH = None
_creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _creds_env:
    CREDS_PATH = Path(_creds_env)

DOMAINS = {"dialogue", "educational", "procedural", "stories"}
PROMPT_LEVELS = {"cot", "instructed", "oneshot", "simple"}
VALID_MODELS = {"gemma2", "llama", "llama8b", "mistral"}

GOOGLE_MODEL = "gemini-2.5-flash-lite"
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
CONCURRENCY = 20
CONCURRENCY_PROD = 60
RETRY_DELAY = 60
CHECKPOINT_N = 500
OPENAI_BATCH_SIZE = 45_000
POLL_INTERVAL = 30

ALIGN_PROMPT = """\
You are analyzing a cultural round-trip: American text -> Indian adaptation -> re-Americanized text.

ORIGINAL AMERICAN TEXT:
{source_text}

RE-AMERICANIZED TEXT:
{adapted_text}

SOURCE CSIs (culture-specific items from the original American text, indexed 0-based):
{source_csis}

RE-AMERICANIZED CSIs (culture-specific items found in the re-Americanized text, indexed 0-based):
{reamer_csis}

For each re-Americanized CSI classify it as exactly one of:
  "recovered" - semantically the same cultural concept as a source CSI
  "substituted" - a different American cultural item replaced a source CSI
  "leaked" - an Indian cultural marker that survived into the re-Americanized text

For recovered items, give the index of the matching source CSI.
List source CSI indices that were not recovered by any re-Americanized CSI as missing.

Return ONLY valid JSON with this exact structure (no other text):
{
  "alignments": [
    {
      "reamer_idx": <int>,
      "classification": "recovered" | "substituted" | "leaked",
      "source_idx": <int or null>,
      "reasoning": "<one short phrase>"
    }
  ],
  "missing_source_idxs": [<int>, ...]
}

If there are no re-Americanized CSIs, return {"alignments": [], "missing_source_idxs": [<all source indices>] }.
If there are no source CSIs, return {"alignments": [<classify each reamer CSI normally>], "missing_source_idxs": []}.
"""


def parse_args():
    """Parse command-line arguments for the alignment runner."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["google", "openai"], default="google")
    p.add_argument("--test", action="store_true", help="10 items only")
    p.add_argument("--domain", choices=list(DOMAINS), default=None)
    p.add_argument("--prompt-level", choices=list(PROMPT_LEVELS), default=None, dest="prompt_level")
    p.add_argument("--model", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _load_google_creds():
    """Load Google service-account credentials and refresh them."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_PATH),
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def _refresh_if_needed(creds):
    """Refresh Google credentials when they are close to expiry."""
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
    """Call Gemini using service-account auth and retry on transient errors."""
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
                print(f"    Gemini {e.code}, backing off {wait}s ...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Gemini {e.code}: {body_txt[:200]}")
    raise RuntimeError("Gemini: max retries exceeded")


def _csi_summary(csis: list[dict]) -> str:
    """Render CSIs as compact JSON for the prompt."""
    slim = []
    for i, c in enumerate(csis):
        slim.append({
            "idx": i,
            "span": c.get("span", ""),
            "category": c.get("category", ""),
            "salience": c.get("salience", ""),
        })
    return json.dumps(slim, ensure_ascii=False)


def _build_prompt(item: dict) -> str:
    """Build the alignment prompt for one item."""
    return ALIGN_PROMPT.format(
        source_text=item["source_text"],
        adapted_text=item["adapted_text"],
        source_csis=_csi_summary(item["source_csis"]),
        reamer_csis=_csi_summary(item["reamer_csis"]),
    )


def run_async_google(items: list[dict], out_path: Path) -> None:
    """Run Gemini alignment asynchronously and checkpoint intermediate rows."""
    async def _run():
        loop = asyncio.get_event_loop()
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(CONCURRENCY_PROD)
        caller = lambda p: call_gemini(p, loop=loop, lock=lock)
        tasks = [process_item(it, caller=caller, sem=sem) for it in items]
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
    """Run alignment with the OpenAI batch API and return parsed rows."""
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
                    result_map[custom_id] = _parse_alignment(raw)
                except Exception:
                    result_map[custom_id] = None
        for item in chunk:
            result = result_map.get(item["rev_id"])
            status = "success" if result else "parse_failed"
            if result is None:
                result = {"alignments": [], "missing_source_idxs": list(range(len(item["source_csis"])))}
            all_results.append(_make_row(item, result, status))
    return all_results


VALID_CLASSES = {"recovered", "substituted", "leaked"}


def _parse_alignment(raw: str) -> dict | None:
    """Parse and normalize the judge response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except Exception:
                return None
        else:
            return None

    if "alignments" not in obj or "missing_source_idxs" not in obj:
        return None

    clean_alignments = []
    for a in obj.get("alignments", []):
        if not isinstance(a, dict):
            continue
        cls = a.get("classification", "")
        if cls not in VALID_CLASSES:
            cls = "substituted"
        clean_alignments.append({
            "reamer_idx": int(a.get("reamer_idx", 0)),
            "classification": cls,
            "source_idx": int(a["source_idx"]) if a.get("source_idx") is not None else None,
            "reasoning": str(a.get("reasoning", ""))[:200],
        })

    missing = [int(x) for x in obj.get("missing_source_idxs", []) if isinstance(x, (int, float))]
    return {"alignments": clean_alignments, "missing_source_idxs": missing}


async def process_item(item: dict, *, caller, sem) -> dict:
    """Process one item through the LLM judge and return a normalized row."""
    source_csis = item["source_csis"]
    reamer_csis = item["reamer_csis"]

    if not source_csis and not reamer_csis:
        return _build_row(item, {"alignments": [], "missing_source_idxs": []}, status="no_csis_either_side")

    prompt = ALIGN_PROMPT.format(
        source_text=item["source_text"],
        adapted_text=item["adapted_text"],
        source_csis=_csi_summary(source_csis),
        reamer_csis=_csi_summary(reamer_csis),
    )

    status = "success"
    result = None

    async with sem:
        try:
            raw = await caller(prompt)
            result = _parse_alignment(raw)
            if result is None:
                nudge = prompt + "\n\nReturn ONLY the JSON object."
                raw2 = await caller(nudge)
                result = _parse_alignment(raw2)
                if result is None:
                    status = "parse_failed"
        except Exception:
            status = "api_error"
            result = None

    if result is None:
        result = {"alignments": [], "missing_source_idxs": list(range(len(source_csis)))}

    return _build_row(item, result, status=status)


def _build_row(item: dict, result: dict, status: str) -> dict:
    """Build a normalized row with summary counts and detail spans."""
    source_csis = item["source_csis"]
    reamer_csis = item["reamer_csis"]
    alignments = result["alignments"]
    missing_src = result["missing_source_idxs"]

    n_recovered = sum(1 for a in alignments if a["classification"] == "recovered")
    n_substituted = sum(1 for a in alignments if a["classification"] == "substituted")
    n_leaked = sum(1 for a in alignments if a["classification"] == "leaked")
    n_missing = len(missing_src)
    n_src = len(source_csis)
    n_reamer = len(reamer_csis)

    return {
        "rev_id": item["rev_id"],
        "original_id": item["original_id"],
        "domain": item["domain"],
        "rev_model": item["rev_model"],
        "prompt_level": item["prompt_level"],
        "fwd_prompt_level": item.get("fwd_prompt_level", ""),
        "align_status": status,
        "n_source_csis": n_src,
        "n_reamer_csis": n_reamer,
        "n_recovered": n_recovered,
        "n_substituted": n_substituted,
        "n_leaked": n_leaked,
        "n_missing": n_missing,
        "recovery_rate": round(n_recovered / n_src, 4) if n_src else None,
        "substitution_rate": round(n_substituted / n_src, 4) if n_src else None,
        "leakage_rate": round(n_leaked / n_reamer, 4) if n_reamer else None,
        "missing_rate": round(n_missing / n_src, 4) if n_src else None,
        "alignments": alignments,
        "missing_source_idxs": missing_src,
        "recovered_spans": [
            {
                "reamer_span": reamer_csis[a["reamer_idx"]].get("span", "") if a["reamer_idx"] < n_reamer else "",
                "source_span": source_csis[a["source_idx"]].get("span", "") if a.get("source_idx") is not None and a["source_idx"] < n_src else "",
                "category": reamer_csis[a["reamer_idx"]].get("category", "") if a["reamer_idx"] < n_reamer else "",
            }
            for a in alignments if a["classification"] == "recovered"
        ],
        "leaked_spans": [
            {
                "span": reamer_csis[a["reamer_idx"]].get("span", "") if a["reamer_idx"] < n_reamer else "",
                "category": reamer_csis[a["reamer_idx"]].get("category", "") if a["reamer_idx"] < n_reamer else "",
                "reasoning": a.get("reasoning", ""),
            }
            for a in alignments if a["classification"] == "leaked"
        ],
        "missing_source_spans": [
            {
                "span": source_csis[i].get("span", "") if i < n_src else "",
                "category": source_csis[i].get("category", "") if i < n_src else "",
                "salience": source_csis[i].get("salience", "") if i < n_src else "",
            }
            for i in missing_src if i < n_src
        ],
    }


def load_source_csis(backend: str) -> dict[str, dict]:
    """Load Phase 1 source CSIs keyed by source_id."""
    candidates = [
        PHASE1_DIR / f"source_csi_inventory_{backend}.jsonl",
        PHASE1_DIR / "source_csi_inventory_google.jsonl",
        PHASE1_DIR / "source_csi_inventory_openai.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("ERROR: Phase 1 source CSI inventory not found.")
        sys.exit(1)
    print(f"  Loading source CSIs from {path.name} ...")
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["source_id"]] = {
                "source_text": r.get("source_text", ""),
                "csis": r.get("csis", []),
            }
    print(f"  {len(idx):,} source items")
    return idx


def load_reamer_csis(backend: str, domain_filter=None, prompt_level_filter=None, model_filter=None) -> list[dict]:
    """Load Phase 9 CSI records filtered by optional domain, prompt level, and model."""
    inv_path = PHASE9_DIR / f"phase9_csi_inventory_{backend}.jsonl"
    if not inv_path.exists():
        print(f"ERROR: {inv_path} not found. Run run_csi_extraction.py first.")
        sys.exit(1)
    print(f"  Loading re-Americanized CSIs from {inv_path.name} ...")
    items = []
    with open(inv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("rev_model", "") not in VALID_MODELS:
                continue
            if domain_filter and r.get("domain") != domain_filter:
                continue
            if prompt_level_filter and r.get("prompt_level") != prompt_level_filter:
                continue
            if model_filter and r.get("rev_model") != model_filter:
                continue
            items.append(r)
    print(f"  {len(items):,} re-Americanized CSI records")
    return items


def load_done_ids(out_path: Path) -> set[str]:
    """Load rev_ids already written to the output file."""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["rev_id"])
                    except Exception:
                        pass
    return done


def compute_stats(records: list[dict]) -> dict:
    """Compute aggregate alignment statistics."""
    def _avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    def _grp(recs):
        ok = [r for r in recs if r["align_status"] in ("success", "no_csis_either_side")]
        return {
            "n": len(recs),
            "n_aligned_ok": len(ok),
            "avg_source_csis": _avg([r["n_source_csis"] for r in recs]),
            "avg_reamer_csis": _avg([r["n_reamer_csis"] for r in recs]),
            "avg_recovery_rate": _avg([r["recovery_rate"] for r in ok]),
            "avg_substitution_rate": _avg([r["substitution_rate"] for r in ok]),
            "avg_leakage_rate": _avg([r["leakage_rate"] for r in ok]),
            "avg_missing_rate": _avg([r["missing_rate"] for r in ok]),
            "total_leaked_csis": sum(r["n_leaked"] for r in recs),
            "total_recovered_csis": sum(r["n_recovered"] for r in recs),
            "total_missing_csis": sum(r["n_missing"] for r in recs),
        }

    def _bucket(key, recs):
        grps: dict[str, list] = defaultdict(list)
        for r in recs:
            grps[r.get(key, "")].append(r)
        return grps

    leaked_cats: dict[str, int] = defaultdict(int)
    missing_cats: dict[str, int] = defaultdict(int)
    for r in records:
        for s in r.get("leaked_spans", []):
            leaked_cats[s.get("category", "unknown")] += 1
        for s in r.get("missing_source_spans", []):
            missing_cats[s.get("category", "unknown")] += 1

    return {
        "total": len(records),
        "align_status_counts": dict(
            (k, sum(1 for r in records if r["align_status"] == k))
            for k in ("success", "parse_failed", "api_error", "no_csis_either_side")
        ),
        "global": _grp(records),
        "by_domain": {k: _grp(v) for k, v in _bucket("domain", records).items()},
        "by_rev_model": {k: _grp(v) for k, v in _bucket("rev_model", records).items()},
        "by_prompt_level": {k: _grp(v) for k, v in _bucket("prompt_level", records).items()},
        "by_fwd_prompt_level": {k: _grp(v) for k, v in _bucket("fwd_prompt_level", records).items()},
        "leaked_category_dist": dict(sorted(leaked_cats.items(), key=lambda x: -x[1])),
        "missing_category_dist": dict(sorted(missing_cats.items(), key=lambda x: -x[1])),
    }


def main():
    """Run the full CSI alignment workflow."""
    args = parse_args()
    backend = args.backend

    out_path = PHASE9_DIR / f"phase9c_csi_alignment_{backend}.jsonl"
    stats_path = PHASE9_DIR / f"phase9c_csi_alignment_stats_{backend}.json"

    print("=" * 70)
    print(f"Phase 9c - LLM CSI Round-Trip Alignment  [backend={backend}]")
    print("=" * 70)

    print("\n[1] Loading source CSIs (Phase 1) ...")
    source_index = load_source_csis(backend)

    print("\n[2] Loading re-Americanized CSIs (Phase 9-CSI) ...")
    reamer_records = load_reamer_csis(
        backend,
        domain_filter=args.domain,
        prompt_level_filter=args.prompt_level,
        model_filter=args.model,
    )

    if args.test:
        reamer_records = reamer_records[:10]
        print(f"  TEST MODE: {len(reamer_records)} items")

    done_ids = load_done_ids(out_path) if (args.resume and not args.test) else set()
    reamer_records = [r for r in reamer_records if r["rev_id"] not in done_ids]
    print(f"  To process: {len(reamer_records):,}  (already done: {len(done_ids):,})")

    if not reamer_records:
        print("  Nothing to process.")
        return

    if backend == "google" and CREDS_PATH is None:
        print("FATAL: GOOGLE_APPLICATION_CREDENTIALS is not set")
        sys.exit(1)

    items = []
    missing_source = 0
    for rec in reamer_records:
        oid = rec.get("original_id", "")
        src = source_index.get(oid)
        if src is None:
            missing_source += 1
            source_csis = []
            source_text = rec.get("original_american_text", "")
        else:
            source_csis = src["csis"]
            source_text = src["source_text"]
        items.append({
            "rev_id": rec["rev_id"],
            "original_id": oid,
            "domain": rec.get("domain", ""),
            "rev_model": rec.get("rev_model", ""),
            "prompt_level": rec.get("prompt_level", ""),
            "fwd_prompt_level": rec.get("fwd_prompt_level", ""),
            "source_csis": source_csis,
            "source_text": source_text,
            "reamer_csis": rec.get("csis", []),
            "adapted_text": rec.get("adapted_text", ""),
        })
    if missing_source:
        print(f"  WARNING: {missing_source:,} items had no Phase 1 source CSI record")

    trivial = [it for it in items if not it["source_csis"] and not it["reamer_csis"]]
    nontrivial = [it for it in items if it["source_csis"] or it["reamer_csis"]]
    trivial_rows = [
        _build_row(it, {"alignments": [], "missing_source_idxs": []}, status="no_csis_either_side")
        for it in trivial
    ]
    print(f"\n[3] Aligning {len(nontrivial):,} items via LLM (+ {len(trivial):,} trivial zero-CSI skipped) ...")

    mode = "a" if done_ids else "w"
    with open(out_path, mode, encoding="utf-8") as fout:
        for r in trivial_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.test:
        async def _realtime():
            loop = asyncio.get_event_loop()
            lock = asyncio.Lock()
            sem = asyncio.Semaphore(CONCURRENCY)
            caller = lambda p: call_gemini(p, loop=loop, lock=lock)
            tasks = [process_item(it, caller=caller, sem=sem) for it in nontrivial]
            rows = []
            t0 = time.time()
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                r = await coro
                rows.append(r)
                if i % 5 == 0 or i == len(tasks):
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(tasks)}  {rate:.1f}/s  rec={r['n_recovered']} sub={r['n_substituted']} leak={r['n_leaked']} miss={r['n_missing']}")
            return rows

        llm_rows = asyncio.run(_realtime())
        with open(out_path, "a", encoding="utf-8") as fout:
            for r in llm_rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        if backend == "google":
            run_async_google(nontrivial, out_path)
        else:
            llm_rows = run_batch_openai(nontrivial)
            with open(out_path, "a", encoding="utf-8") as fout:
                for r in llm_rows:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n[4] Computing stats ...")
    all_records = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    stats = compute_stats(all_records)
    stats_path.write_text(json.dumps(stats, indent=2))

    g = stats["global"]
    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    print(f"  Items aligned          : {stats['total']:,}")
    print(f"  Status                 : {stats['align_status_counts']}")
    print(f"  Avg source CSIs/item   : {g['avg_source_csis']:.2f}")
    print(f"  Avg reamer CSIs/item   : {g['avg_reamer_csis']:.2f}")
    print(f"  Avg recovery rate      : {g['avg_recovery_rate']:.3f}")
    print(f"  Avg substitution rate  : {g['avg_substitution_rate']:.3f}")
    print(f"  Avg leakage rate       : {g['avg_leakage_rate']:.3f}")
    print(f"  Avg missing rate       : {g['avg_missing_rate']:.3f}")
    print("\n  By domain:")
    for d, v in sorted(stats["by_domain"].items()):
        print(f"    {d:<20} rec={v['avg_recovery_rate']:.3f}  leak={v['avg_leakage_rate']:.3f}  miss={v['avg_missing_rate']:.3f}")
    print("\n  By rev model:")
    for m, v in sorted(stats["by_rev_model"].items()):
        print(f"    {m:<20} rec={v['avg_recovery_rate']:.3f}  leak={v['avg_leakage_rate']:.3f}  miss={v['avg_missing_rate']:.3f}")
    print("\n  By prompt level:")
    for pl, v in sorted(stats["by_prompt_level"].items()):
        print(f"    {pl:<20} rec={v['avg_recovery_rate']:.3f}  leak={v['avg_leakage_rate']:.3f}  miss={v['avg_missing_rate']:.3f}")
    print("\n  Top leaked categories:")
    for cat, cnt in list(stats["leaked_category_dist"].items())[:6]:
        print(f"    {cat:<35} {cnt:>6,}")
    print("\n  Top missing categories:")
    for cat, cnt in list(stats["missing_category_dist"].items())[:6]:
        print(f"    {cat:<35} {cnt:>6,}")
    print(f"\n  Output : {out_path}")
    print(f"  Stats  : {stats_path}")

    if args.test:
        print("\n  [TEST] Sample records:")
        for r in all_records[:3]:
            print(f"\n  rev_id={r['rev_id']}")
            print(f"  n_source={r['n_source_csis']}  n_reamer={r['n_reamer_csis']}")
            print(f"  recovered={r['n_recovered']}  sub={r['n_substituted']}  leaked={r['n_leaked']}  missing={r['n_missing']}")
            if r["recovered_spans"]:
                print(f"  recovered: {r['recovered_spans'][:2]}")
            if r["leaked_spans"]:
                print(f"  leaked: {r['leaked_spans'][:2]}")


if __name__ == "__main__":
    main()
