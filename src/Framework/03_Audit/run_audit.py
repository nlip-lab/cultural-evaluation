"""
Phase 3 — Adaptation Audit
For every adaptation that passed hygiene, audits each source CSI:
what strategy did the model use? (retained / localized / globalized /
transformed / omitted / distorted)

Backend: Gemini 2.5 Flash Lite (same service-account credentials as Phase 1)

Usage:
  python run_audit.py [--pilot N] [--resume] [--test]

  --pilot N   Run on first N eligible records (validates quality + estimates cost)
  --resume    Skip records already written (reads checkpoint)
  --test      Run on 5 records and print sample output
"""

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE_DIR    = EVAL_DIR / "03_Audit"
CREDS_PATH   = None

HYGIENE_PATH = EVAL_DIR / "02_Hygiene" / "adaptations_with_hygiene.jsonl"
CSI_PATH     = EVAL_DIR / "01_CSI_Extraction" / "source_csi_final_google.jsonl"
OUT_PATH     = PHASE_DIR / "audit_results.jsonl"
CKPT_PATH    = PHASE_DIR / ".audit_checkpoint.json"
STATS_PATH   = PHASE_DIR / "audit_stats.json"

PROMPT_TEMPLATE = (EVAL_DIR / "prompts" / "03_audit.txt").read_text()

GOOGLE_MODEL  = "gemini-2.5-flash-lite"
TEMPERATURE   = 0.0
MAX_TOKENS    = 4096
CONCURRENCY   = 50
RETRY_DELAY   = 60
CHECKPOINT_N  = 1000

VALID_STRATEGIES = {
    "retained", "localized", "globalized",
    "transformed", "omitted", "distorted",
}

def _load_google_creds():
    """Load Google service-account credentials for Gemini API calls."""
    from google.oauth2 import service_account
    import google.auth.transport.requests
    creds = service_account.Credentials.from_service_account_file(
        str(CREDS_PATH),
        scopes=["https://www.googleapis.com/auth/generative-language"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds

def _refresh_if_needed(creds):
    """Refresh Google access token when expiry is near."""
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
    """Call Gemini generateContent endpoint and return response text."""
    global _google_creds
    async with lock:
        if _google_creds is None:
            _google_creds = await loop.run_in_executor(None, _load_google_creds)
        _google_creds = await loop.run_in_executor(None, _refresh_if_needed, _google_creds)
        token = _google_creds.token

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GOOGLE_MODEL}:generateContent")
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
            raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req).read()
            )
            data = json.loads(raw)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            if e.code in (429, 503):
                wait = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Gemini API {e.code}: {err_body[:300]}")
    raise RuntimeError("Gemini API: max retries exceeded")

def _extract_json_object(text: str) -> dict:
    """
    Extract JSON object from model response.
    Handles: markdown fences, array wrappers, and truncated JSON.

    Truncation recovery: if strict parse fails, attempt to salvage a partial
    csi_alignments list from the prefix.  This happens when MAX_TOKENS is hit
    mid-response; we can still use the alignments that were fully written.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Unexpected top-level type: {type(obj)}")
    except json.JSONDecodeError:
        pass

    alignment_re = re.compile(
        r'\{[^{}]*"source_span"\s*:[^{}]*"strategy"\s*:\s*"[^"]+?"[^{}]*\}',
        re.DOTALL,
    )
    alignments = []
    for m in alignment_re.finditer(text):
        try:
            a = json.loads(m.group())
            alignments.append(a)
        except json.JSONDecodeError:
            pass

    if alignments:
        return {
            "csi_alignments":       alignments,
            "unsupported_additions": [],
            "_truncated":           True,
        }

    raise ValueError("Could not extract JSON object or partial alignments")

def _validate_and_repair(result: dict, source_csis: list, adapted_text: str) -> dict:
    """
    Validate the parsed audit result and repair common model errors:
    - Wrong count of csi_alignments (pad or truncate to match source)
    - Invalid strategy labels (fallback to 'omitted')
    - Mismatched char offsets (fuzzy-fix from span)
    - Recompute counts to match actual alignments
    """
    alignments = result.get("csi_alignments") or []
    n_source   = len(source_csis)

    if len(alignments) < n_source:
        existing_spans = {a.get("source_span", ""): a for a in alignments}
        for csi in source_csis:
            sp = csi.get("span", "")
            if sp not in existing_spans:
                alignments.append({
                    "source_span": sp,
                    "adapted_span": None,
                    "adapted_char_start": None,
                    "adapted_char_end": None,
                    "strategy": "omitted",
                    "validity_note": "No alignment returned by model; marked omitted.",
                })
    alignments = alignments[:n_source]

    for a in alignments:
        if a.get("strategy") not in VALID_STRATEGIES:
            a["strategy"] = "omitted"

        span = a.get("adapted_span")
        if span:
            cs = a.get("adapted_char_start")
            ce = a.get("adapted_char_end")
            if not (isinstance(cs, int) and isinstance(ce, int)
                    and 0 <= cs < ce <= len(adapted_text)
                    and adapted_text[cs:ce] == span):
                idx = adapted_text.find(span)
                if idx != -1:
                    a["adapted_char_start"] = idx
                    a["adapted_char_end"]   = idx + len(span)
                else:
                    idx = adapted_text.lower().find(span.lower())
                    if idx != -1:
                        a["adapted_char_start"] = idx
                        a["adapted_char_end"]   = idx + len(span)
                    else:
                        a["adapted_char_start"] = None
                        a["adapted_char_end"]   = None
        else:
            a["adapted_char_start"] = None
            a["adapted_char_end"]   = None

    result["csi_alignments"] = alignments

    counts: dict[str, int] = {s: 0 for s in VALID_STRATEGIES}
    for a in alignments:
        s = a.get("strategy", "omitted")
        counts[s] = counts.get(s, 0) + 1

    result["source_csis_total"] = len(alignments)
    for s in VALID_STRATEGIES:
        result[f"{s}_count"] = counts[s]

    if not isinstance(result.get("unsupported_additions"), list):
        result["unsupported_additions"] = []

    return result

def _build_prompt(record: dict, source_csis: list) -> str:
    """Build the audit prompt for one adaptation record."""
    slim_csis = [
        {
            "span":                 c.get("span", ""),
            "category":             c.get("category", ""),
            "adaptation_necessity": c.get("adaptation_necessity", ""),
            "local_role":           c.get("local_role", ""),
            "scale_sensitive":      c.get("scale_sensitive", False),
            "figurative":           c.get("figurative", False),
        }
        for c in source_csis
    ]
    return (PROMPT_TEMPLATE
            .replace("{source_text}",    record["source_text"])
            .replace("{adapted_text}",   record.get("adapted_text", ""))
            .replace("{domain}",         record.get("domain", ""))
            .replace("{source_csis_json}", json.dumps(slim_csis, ensure_ascii=False, indent=2)))

async def process_record(record: dict, source_csis: list, *, caller, sem) -> dict:
    """Audit one adaptation record and return normalized audit output."""
    eval_id   = record["eval_id"]
    source_id = record.get("source_id", "")
    domain    = record.get("domain", "")
    model     = record.get("model", "")
    strategy  = record.get("strategy", "")

    base = {
        "eval_id":             eval_id,
        "source_id":           source_id,
        "domain":              domain,
        "model":               model,
        "strategy":            strategy,
        "csi_alignments":      [],
        "unsupported_additions": [],
        "counts": {s: 0 for s in ("source_csis_total", "retained", "localized",
                                   "globalized", "transformed", "omitted", "distorted")},
    }

    if not source_csis:
        base["audit_status"] = "no_csis"
        return base

    prompt = _build_prompt(record, source_csis)

    async with sem:
        try:
            raw = await caller(prompt)
        except Exception as e:
            base["audit_status"] = "api_error"
            base["_error"] = str(e)
            return base

        try:
            parsed = _extract_json_object(raw)
        except Exception:
            nudge = prompt + "\n\nReturn ONLY a valid JSON object as specified."
            try:
                raw2   = await caller(nudge)
                parsed = _extract_json_object(raw2)
                parse_status = "parse_retry"
            except Exception:
                base["audit_status"] = "parse_failed"
                base["_raw"]         = raw[:500]
                return base
        else:
            parse_status = "truncated_recovered" if parsed.get("_truncated") else "success"

    adapted_text = record.get("adapted_text", "")
    try:
        parsed = _validate_and_repair(parsed, source_csis, adapted_text)
    except Exception as e:
        base["audit_status"] = "validation_error"
        base["_error"] = str(e)
        return base

    counts = {
        "source_csis_total": parsed.get("source_csis_total", len(source_csis)),
    }
    for s in VALID_STRATEGIES:
        counts[s] = parsed.get(f"{s}_count", 0)

    base["audit_status"]          = parse_status
    base["csi_alignments"]        = parsed.get("csi_alignments", [])
    base["unsupported_additions"] = parsed.get("unsupported_additions", [])
    base["counts"]                = counts
    return base

def load_checkpoint() -> set[str]:
    """Load completed eval IDs from checkpoint file."""
    if CKPT_PATH.exists():
        return set(json.loads(CKPT_PATH.read_text()).get("done_ids", []))
    return set()

def save_checkpoint(done_ids: set[str]):
    """Persist completed eval IDs to checkpoint file."""
    CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))

def prepare_run_state(*, fresh: bool) -> set[str]:
    """Fresh start clears stale files; resume mode loads checkpoint."""
    if fresh:
        CKPT_PATH.unlink(missing_ok=True)
        OUT_PATH.unlink(missing_ok=True)
        return set()
    if CKPT_PATH.exists() and OUT_PATH.exists():
        return load_checkpoint()
    if CKPT_PATH.exists() or OUT_PATH.exists():
        print("    Stale partial state without matching pair — starting fresh.")
        CKPT_PATH.unlink(missing_ok=True)
        OUT_PATH.unlink(missing_ok=True)
        return set()
    return set()

def build_stats(results: list[dict]) -> dict:
    """Aggregate status and strategy statistics from audit results."""
    total = len(results)
    status_dist: dict[str, int]              = defaultdict(int)
    strategy_dist: dict[str, int]            = defaultdict(int)
    by_model: dict[str, dict[str, int]]      = defaultdict(lambda: defaultdict(int))
    by_strategy: dict[str, dict[str, int]]   = defaultdict(lambda: defaultdict(int))
    by_domain: dict[str, dict[str, int]]     = defaultdict(lambda: defaultdict(int))
    csi_strategy_totals: dict[str, int]      = defaultdict(int)

    for r in results:
        status = r.get("audit_status", "unknown")
        status_dist[status] += 1
        m = r.get("model", "unknown")
        s = r.get("strategy", "unknown")
        d = r.get("domain", "unknown")

        for strat in VALID_STRATEGIES:
            cnt = r.get("counts", {}).get(strat, 0)
            if cnt:
                strategy_dist[strat]   += cnt
                by_model[m][strat]     += cnt
                by_strategy[s][strat]  += cnt
                by_domain[d][strat]    += cnt
                csi_strategy_totals[strat] += cnt

    total_csis = sum(csi_strategy_totals.values())

    return {
        "total_audited":       total,
        "status_distribution": dict(status_dist),
        "csi_strategy_distribution": dict(strategy_dist),
        "total_csis_audited":  total_csis,
        "by_model":   {m: dict(v) for m, v in sorted(by_model.items())},
        "by_strategy": {s: dict(v) for s, v in sorted(by_strategy.items())},
        "by_domain":  {d: dict(v) for d, v in sorted(by_domain.items())},
    }

def print_summary(stats: dict):
    """Print a concise terminal summary for audit outcomes."""
    print(f"\n{'='*72}")
    print("PHASE 3 AUDIT RESULTS")
    print(f"{'='*72}")
    print(f"  Total audited        : {stats['total_audited']:,}")
    print(f"  Total CSIs evaluated : {stats['total_csis_audited']:,}")

    print(f"\n  Audit status:")
    for k, v in sorted(stats["status_distribution"].items(), key=lambda x: -x[1]):
        pct = 100 * v / max(stats["total_audited"], 1)
        print(f"    {k:<25} {v:>8,}  ({pct:>5.1f}%)")

    total_csis = max(stats["total_csis_audited"], 1)
    print(f"\n  CSI strategy distribution (across all records):")
    for strat in ("localized", "globalized", "retained", "transformed", "omitted", "distorted"):
        cnt = stats["csi_strategy_distribution"].get(strat, 0)
        pct = 100 * cnt / total_csis
        print(f"    {strat:<15} {cnt:>8,}  ({pct:>5.1f}%)")

    print(f"\n  Localization rate by model (localized / total CSIs):")
    models = sorted(stats["by_model"].keys())
    for m in models:
        mv   = stats["by_model"][m]
        tot  = sum(mv.values())
        loc  = mv.get("localized", 0)
        ret  = mv.get("retained", 0)
        omit = mv.get("omitted", 0)
        dist = mv.get("distorted", 0)
        if tot:
            print(f"    {m:<12} loc={100*loc/tot:5.1f}%  ret={100*ret/tot:5.1f}%  "
                  f"omit={100*omit/tot:5.1f}%  dist={100*dist/tot:5.1f}%  (n={tot:,})")

    print(f"\n  Localization rate by prompting strategy:")
    for s in sorted(stats["by_strategy"].keys()):
        sv  = stats["by_strategy"][s]
        tot = sum(sv.values())
        loc = sv.get("localized", 0)
        if tot:
            print(f"    {s:<14} {100*loc/tot:5.1f}%  (n={tot:,})")

    print(f"\n  Output : {OUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")
    print(f"{'='*72}")

def parse_args():
    """Parse CLI arguments for audit run modes."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=int, default=None,
                   help="Run on first N eligible records (pilot mode)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint (skip already-processed records)")
    p.add_argument("--test", action="store_true",
                   help="Run on 5 records and print sample output")
    return p.parse_args()

async def main():
    """Run the phase 3 audit pipeline and export results and stats."""
    args = parse_args()

    print(f"{'='*65}")
    print(f"Phase 3 — Adaptation Audit  [backend=gemini/{GOOGLE_MODEL}]")
    print(f"{'='*65}")

    loop = asyncio.get_event_loop()
    lock = asyncio.Lock()
    caller = lambda p: call_gemini(p, loop=loop, lock=lock)

    print("\n[0] Testing Gemini API …")
    try:
        resp = await caller('Return ONLY this JSON: {"ok": true}')
        parsed = _extract_json_object(resp)
        assert isinstance(parsed, dict)
        print(f"    Gemini API OK — model={GOOGLE_MODEL}")
    except Exception as e:
        print(f"    FATAL: Gemini API test failed: {e}")
        sys.exit(1)

    print(f"\n[1] Loading CSI inventory from {CSI_PATH.name} …")
    csi_index: dict[str, list] = {}
    with open(CSI_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            csi_index[r["source_id"]] = r.get("csis", [])
    print(f"    {len(csi_index):,} source items indexed")

    print(f"\n[2] Loading hygiene-filtered adaptations …")
    eligible: list[dict] = []
    with open(HYGIENE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("02_hygiene", {}).get("overall") != "fail_hard":
                eligible.append(r)
    print(f"    {len(eligible):,} eligible records (hygiene != fail_hard)")

    if args.test:
        eligible = eligible[:5]
        print(f"    TEST MODE: {len(eligible)} records")
    elif args.pilot:
        eligible = eligible[: args.pilot]
        print(f"    PILOT MODE: first {len(eligible):,} records")

    done_ids = prepare_run_state(fresh=not (args.resume or args.test or args.pilot))
    remaining = [r for r in eligible if r["eval_id"] not in done_ids]
    write_mode = "a" if (args.resume and not args.test) else "w"
    print(f"    Already done: {len(done_ids):,} | Remaining: {len(remaining):,}")

    if not remaining:
        print("    Nothing to process — loading existing results for stats.")
    else:
        print(f"\n[3] Auditing {len(remaining):,} records (concurrency={CONCURRENCY}) …")
        sem       = asyncio.Semaphore(CONCURRENCY)
        tasks     = [
            process_record(r, csi_index.get(r.get("source_id", ""), []),
                           caller=caller, sem=sem)
            for r in remaining
        ]
        buffer: list[dict] = []
        t0 = time.time()

        with open(OUT_PATH, write_mode) as out_f:
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                buffer.append(result)
                done_ids.add(result["eval_id"])

                if i % 500 == 0 or i == len(tasks):
                    elapsed = time.time() - t0
                    rate    = i / elapsed if elapsed > 0 else 0
                    eta     = (len(tasks) - i) / rate if rate > 0 else 0
                    statuses = {}
                    for r in buffer[-200:]:
                        s = r.get("audit_status", "?")
                        statuses[s] = statuses.get(s, 0) + 1
                    print(f"    {i:>7}/{len(tasks)} | {rate:>5.0f}/s | "
                          f"ETA {eta/60:>4.1f}m | {statuses}")

                if i % CHECKPOINT_N == 0:
                    for r in buffer:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if not args.test:
                        save_checkpoint(done_ids)
                    buffer.clear()
                    print(f"    ✓ Checkpoint at {i}")

            for r in buffer:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not args.test:
                save_checkpoint(done_ids)

    print(f"\n[4] Computing stats …")
    all_results: list[dict] = []
    with open(OUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    stats = build_stats(all_results)
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print_summary(stats)

    if args.test:
        print("\n  [TEST] Sample results:")
        for r in all_results[:3]:
            print(f"\n  eval_id   : {r['eval_id']}")
            print(f"  status    : {r['audit_status']}")
            print(f"  counts    : {r['counts']}")
            for a in r.get("csi_alignments", [])[:2]:
                print(f"  alignment : [{a.get('strategy')}] "
                      f"{a.get('source_span')!r} → {a.get('adapted_span')!r}  "
                      f"| {a.get('validity_note','')[:60]}")
            if r.get("unsupported_additions"):
                print(f"  additions : {r['unsupported_additions'][:1]}")

if __name__ == "__main__":
    asyncio.run(main())
