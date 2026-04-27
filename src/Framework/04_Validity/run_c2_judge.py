"""
Phase 4.2 — Substitution Validity Judge (C2)
============================================
Uses Gemini 2.5 Flash Lite to score each CSI substitution on 5 dimensions:
affordance, scale, dependency, identity, figurative.

Batches up to 3 substitutions per API call to reduce cost.
Deterministic ScaleChecker overrides LLM scale_transfer scores for currency
conversions that are provably wrong (value unchanged or implausible ratio).

Usage:
    python run_c2_judge.py [--pilot N] [--resume] [--test]

Outputs:
    04_Validity/c2_scores.jsonl
    04_Validity/c2_stats.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parents[1]
PHASE3_DIR   = EVAL_DIR / "03_Audit"
PHASE1_DIR   = EVAL_DIR / "01_CSI_Extraction"
PHASE4_DIR   = EVAL_DIR / "04_Validity"
PHASE2_DIR   = EVAL_DIR / "02_Hygiene"
PROMPTS_DIR  = EVAL_DIR / "prompts"

AUDIT_PATH   = PHASE3_DIR / "audit_results.jsonl"
CSI_PATH     = PHASE1_DIR / "source_csi_final_google.jsonl"
HYGIENE_PATH = PHASE2_DIR / "adaptations_with_hygiene.jsonl"
PROMPT_PATH  = PROMPTS_DIR / "04_validity.txt"
OUT_PATH     = PHASE4_DIR / "c2_scores.jsonl"
STATS_PATH   = PHASE4_DIR / "c2_stats.json"
CKPT_PATH    = PHASE4_DIR / ".c2_checkpoint.json"

sys.path.insert(0, str(PHASE4_DIR))
from scale_checker import ScaleChecker

_SCALE_CHECKER = ScaleChecker()

try:
    import google.auth
    import google.auth.transport.requests
    _GOOGLE_CREDS = None
    _TOKEN_EXPIRY = 0.0

    def _get_token() -> str:
        """Return a valid Google OAuth token for Vertex requests."""
        global _GOOGLE_CREDS, _TOKEN_EXPIRY
        if _GOOGLE_CREDS is None:
            _GOOGLE_CREDS, _ = google.auth.load_credentials_from_file(
                str(EVAL_DIR / "google.json"),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        if time.time() > _TOKEN_EXPIRY - 60:
            req = google.auth.transport.requests.Request()
            _GOOGLE_CREDS.refresh(req)
            _TOKEN_EXPIRY = (
                _GOOGLE_CREDS.expiry.timestamp()
                if _GOOGLE_CREDS.expiry
                else time.time() + 3600
            )
        return _GOOGLE_CREDS.token

except ImportError:
    def _get_token():
        """Raise a dependency error when google-auth is unavailable."""
        raise RuntimeError("google-auth not installed. Run: pip install google-auth")

import urllib.request
import urllib.error

GEMINI_PROJECT  = os.getenv("GEMINI_PROJECT", "cultural-eval-project")
GEMINI_LOCATION = "us-central1"
GEMINI_MODEL    = "gemini-2.5-flash-lite-preview-06-17"
GEMINI_ENDPOINT = (
    f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1/projects/"
    f"{GEMINI_PROJECT}/locations/{GEMINI_LOCATION}/publishers/google/models/"
    f"{GEMINI_MODEL}:generateContent"
)
MAX_TOKENS    = 2048
TEMPERATURE   = 0.0
CONCURRENCY   = 30
BATCH_SIZE    = 3
CHECKPOINT_N  = 500

_SEVERITY_RANK = {
    "none":                             0,
    "C2.5_figurative_equivalence_failure": 1,
    "C2.4_identity_marker_mismatch":    2,
    "C2.3_dependency_propagation_failure": 3,
    "C2.1_affordance_violation":        4,
    "C2.2_scale_transfer_error":        5,
}

def _worst_subtype(subtypes: list[str]) -> str:
    """Return the subtype with the highest severity rank."""
    return max(subtypes, key=lambda s: _SEVERITY_RANK.get(s, 0), default="none")

_PROMPT_TEMPLATE: str | None = None

def _load_prompt() -> str:
    """Load and cache the validity prompt template."""
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = PROMPT_PATH.read_text()
    return _PROMPT_TEMPLATE

async def _call_gemini(prompt: str, sem: asyncio.Semaphore) -> str | None:
    """Call Gemini with retries and return response text when available."""
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS},
    }).encode()

    async with sem:
        for attempt in range(4):
            try:
                token = _get_token()
                req = urllib.request.Request(
                    GEMINI_ENDPOINT,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                loop = asyncio.get_event_loop()
                def _do_request():
                    """Execute the blocking HTTP request in executor context."""
                    with urllib.request.urlopen(req, timeout=60) as r:
                        return json.loads(r.read())
                data = await loop.run_in_executor(None, _do_request)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    await asyncio.sleep(20 * (attempt + 1))
                elif e.code >= 500:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    return None
            except Exception:
                await asyncio.sleep(3 * (attempt + 1))
        return None

def _extract_json_array(text: str) -> list | None:
    """Parse model output into a JSON array or return None."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None

def load_audit_index(path: Path) -> dict[str, dict]:
    """Load audit_results indexed by eval_id (success records with alignments only)."""
    idx = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("audit_status") == "success" and r.get("csi_alignments"):
                    idx[r["eval_id"]] = r
    return idx

def load_csi_index(path: Path) -> dict[str, dict]:
    """Load source_csi_final indexed by source_id → {span → csi_dict}."""
    idx = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("source_id") or r.get("id")
            if not sid:
                continue
            span_map = {}
            for csi in r.get("csis", []):
                span = (csi.get("span") or "").strip()
                if span:
                    span_map[span] = csi
            idx[sid] = span_map
    return idx

def load_hygiene_index(path: Path) -> dict[str, dict]:
    """Load adaptations_with_hygiene indexed by eval_id (for source/adapted text)."""
    idx = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["eval_id"]] = r
    return idx

def _currency_override(sub_info: dict, score_rec: dict) -> dict | None:
    """
    If this CSI is a USD→INR currency substitution, run ScaleChecker
    deterministically. Returns override dict if a problem is found, else None.
    """
    if not sub_info.get("scale_sensitive"):
        return None
    if sub_info.get("category") != "unit_currency_scale":
        return None

    source_span  = sub_info.get("source_span", "")
    adapted_span = sub_info.get("adapted_span") or ""

    if "$" not in source_span and "USD" not in source_span.upper():
        return None

    src_nums = _SCALE_CHECKER.extract_numbers(source_span)
    ada_nums = _SCALE_CHECKER.extract_numbers(adapted_span)

    src_usd = next((n["value"] for n in src_nums if n["value"] > 0), None)
    ada_inr = next((n["value"] for n in ada_nums if n["value"] > 0), None)

    if src_usd is None or ada_inr is None:
        return None

    result = _SCALE_CHECKER.check_currency_plausibility(src_usd, ada_inr, adapted_span)
    if result["plausible"]:
        return None

    return {
        "check": "currency_plausibility",
        "source_usd": src_usd,
        "adapted_inr": ada_inr,
        "ratio": result["ratio"],
        "error": result["error"],
        "scale_transfer_overridden_to": 0.0,
    }

def build_substitutions(
    audit_record: dict,
    csi_span_map: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (to_judge, auto_scored).

    to_judge:     needs LLM scoring (localized / globalized / transformed / distorted)
    auto_scored:  deterministic result — no API call needed
      - retained + must_adapt → c2 = 0.0  (failed to substitute)
      - omitted               → c2 = 1.0  (omission is a scope choice; RF penalises it)
      - retained + other      → c2 = 1.0  (intentional retain, not a validity failure)
    """
    to_judge = []
    auto     = []

    for aln in audit_record.get("csi_alignments", []):
        strategy      = aln.get("strategy", "omitted")
        source_span   = (aln.get("source_span") or "").strip()
        adapted_span  = aln.get("adapted_span")
        csi_info      = csi_span_map.get(source_span, {})
        necessity     = csi_info.get("adaptation_necessity", "may_adapt")

        sub = {
            "source_span":          source_span,
            "adapted_span":         adapted_span,
            "strategy":             strategy,
            "necessity":            necessity,
            "category":             csi_info.get("category", ""),
            "contextual_function":  csi_info.get("contextual_function", ""),
            "required_affordances": csi_info.get("required_affordances", []),
            "figurative":           csi_info.get("figurative", False),
            "scale_sensitive":      csi_info.get("scale_sensitive", False),
            "numeric_constraints":  csi_info.get("numeric_constraints"),
        }

        if strategy == "retained" and necessity == "must_adapt":
            auto.append({**sub, "auto_c2": 0.0, "auto_reason": "retained_must_adapt"})
        elif strategy in ("omitted", "retained"):
            auto.append({**sub, "auto_c2": 1.0, "auto_reason": strategy})
        else:
            to_judge.append(sub)

    return to_judge, auto

def format_prompt(
    source_text: str,
    adapted_text: str,
    domain: str,
    subs: list[dict],
) -> str:
    """Format one prompt for a batch of substitutions."""
    subs_for_prompt = [
        {
            "source_span":          s["source_span"],
            "adapted_span":         s["adapted_span"],
            "strategy":             s["strategy"],
            "category":             s["category"],
            "contextual_function":  s["contextual_function"],
            "required_affordances": s["required_affordances"],
        }
        for s in subs
    ]
    tmpl = _load_prompt()
    tmpl = tmpl.replace("{source_text}",      source_text)
    tmpl = tmpl.replace("{adapted_text}",     adapted_text)
    tmpl = tmpl.replace("{domain}",           domain)
    tmpl = tmpl.replace("{substitutions_json}",
                         json.dumps(subs_for_prompt, indent=2, ensure_ascii=False))
    return tmpl

def assemble_output(
    eval_id: str,
    audit_record: dict,
    judged: list[tuple[dict, dict]],
    auto_scored: list[dict],
) -> dict:
    """Assemble one normalized output record with C2 scores."""
    csi_scores = []

    for sub_info, llm_result in judged:
        scale_score = _safe_score(llm_result.get("scale_transfer"), default=1.0)
        override    = _currency_override(sub_info, {"scale_transfer": scale_score})

        if override:
            scale_score = 0.0

        aff = _safe_score(llm_result.get("affordance_validity"), default=0.5)
        dep = _safe_score(llm_result.get("dependency_propagation"), default=1.0)
        idf = _safe_score(llm_result.get("identity_marker_fit"), default=1.0)
        fig = _safe_score(llm_result.get("figurative_equivalence"), default=1.0)
        overall = (
            aff * 0.35 + scale_score * 0.30 + dep * 0.15 + idf * 0.10 + fig * 0.10
            if override
            else _safe_score(llm_result.get("overall_c2_score"), default=0.5)
        )

        score_rec = {
            "source_span":              sub_info["source_span"],
            "adapted_span":             sub_info["adapted_span"],
            "strategy":                 sub_info["strategy"],
            "affordance_validity":      aff,
            "scale_transfer":           scale_score,
            "dependency_propagation":   dep,
            "identity_marker_fit":      idf,
            "figurative_equivalence":   fig,
            "overall_c2_score":         round(overall, 4),
            "c2_subtype":               llm_result.get("c2_subtype", "none"),
            "c2_severity":              llm_result.get("c2_severity", "none"),
            "deterministic_override":   override,
        }

        if override and score_rec["c2_subtype"] == "none":
            score_rec["c2_subtype"]  = "C2.2_scale_transfer_error"
            score_rec["c2_severity"] = "critical" if overall < 0.3 else "substantive"

        csi_scores.append(score_rec)

    for auto in auto_scored:
        c2       = auto["auto_c2"]
        severity = "critical" if c2 == 0.0 else "none"
        subtype  = "C2.1_affordance_violation" if c2 == 0.0 else "none"
        csi_scores.append({
            "source_span":              auto["source_span"],
            "adapted_span":             auto["adapted_span"],
            "strategy":                 auto["strategy"],
            "affordance_validity":      c2,
            "scale_transfer":           1.0,
            "dependency_propagation":   1.0,
            "identity_marker_fit":      1.0,
            "figurative_equivalence":   1.0,
            "overall_c2_score":         c2,
            "c2_subtype":               subtype,
            "c2_severity":              severity,
            "deterministic_override":   {"reason": auto["auto_reason"]},
        })

    all_c2 = [s["overall_c2_score"] for s in csi_scores]
    aggregate_c2 = round(sum(all_c2) / len(all_c2), 4) if all_c2 else 1.0

    all_subtypes  = [s["c2_subtype"] for s in csi_scores]
    worst_subtype = _worst_subtype(all_subtypes)

    return {
        "eval_id":                 eval_id,
        "source_id":               audit_record.get("source_id"),
        "domain":                  audit_record.get("domain"),
        "model":                   audit_record.get("model"),
        "prompt_strategy":         audit_record.get("strategy"),
        "csi_substitution_scores": csi_scores,
        "aggregate_c2":            aggregate_c2,
        "worst_c2_subtype":        worst_subtype,
    }

def _safe_score(val, default: float = 0.5) -> float:
    """Extract float score from an LLM result field (handles dict or raw float)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        s = val.get("score", default)
        return float(s) if s is not None else default
    return default

async def process_one(
    eval_id: str,
    audit_record: dict,
    csi_index: dict,
    hygiene_index: dict,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Process one eval record and return C2 scoring output."""
    source_id    = audit_record["source_id"]
    csi_span_map = csi_index.get(source_id, {})

    hyg_rec      = hygiene_index.get(eval_id, {})
    source_text  = hyg_rec.get("source_text", "")
    adapted_text = hyg_rec.get("adapted_text", "")
    domain       = audit_record.get("domain", "")

    to_judge, auto_scored = build_substitutions(audit_record, csi_span_map)

    if not to_judge:
        return assemble_output(eval_id, audit_record, [], auto_scored)

    judged: list[tuple[dict, dict]] = []
    for i in range(0, len(to_judge), BATCH_SIZE):
        batch  = to_judge[i : i + BATCH_SIZE]
        prompt = format_prompt(source_text, adapted_text, domain, batch)
        raw    = await _call_gemini(prompt, sem)

        if raw is None:
            for sub in batch:
                judged.append((sub, _default_llm_result()))
            continue

        parsed = _extract_json_array(raw)
        if parsed is None or len(parsed) < len(batch):
            for sub in batch:
                judged.append((sub, _default_llm_result()))
            continue

        for sub, result in zip(batch, parsed):
            judged.append((sub, result))

    return assemble_output(eval_id, audit_record, judged, auto_scored)

def _default_llm_result() -> dict:
    """Return fallback scoring payload when LLM output is unavailable."""
    return {
        "affordance_validity":    {"score": 0.5},
        "scale_transfer":         {"score": 1.0},
        "dependency_propagation": {"score": 1.0},
        "identity_marker_fit":    {"score": 1.0},
        "figurative_equivalence": {"score": 1.0},
        "overall_c2_score":       0.5,
        "c2_subtype":             "none",
        "c2_severity":            "none",
    }

def compute_stats(out_path: Path) -> dict:
    """Compute aggregate summary metrics from output JSONL."""
    by_model:    dict[str, list[float]] = defaultdict(list)
    by_strategy: dict[str, list[float]] = defaultdict(list)
    by_domain:   dict[str, list[float]] = defaultdict(list)
    subtypes:    dict[str, int]          = defaultdict(int)
    total = 0
    all_c2: list[float] = []

    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c2 = r.get("aggregate_c2", 1.0)
            all_c2.append(c2)
            by_model[r.get("model", "?")].append(c2)
            by_strategy[r.get("prompt_strategy", "?")].append(c2)
            by_domain[r.get("domain", "?")].append(c2)
            subtypes[r.get("worst_c2_subtype", "none")] += 1
            total += 1

    def _mean(vals):
        """Return rounded mean for non-empty numeric values."""
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "total":           total,
        "mean_c2":         _mean(all_c2),
        "by_model":        {k: {"mean_c2": _mean(v), "n": len(v)} for k, v in by_model.items()},
        "by_strategy":     {k: {"mean_c2": _mean(v), "n": len(v)} for k, v in by_strategy.items()},
        "by_domain":       {k: {"mean_c2": _mean(v), "n": len(v)} for k, v in by_domain.items()},
        "worst_subtype_distribution": dict(subtypes),
    }

def parse_args():
    """Parse CLI arguments for validity judging run modes."""
    p = argparse.ArgumentParser()
    p.add_argument("--pilot",  type=int,  default=None, help="Process only N eval_ids")
    p.add_argument("--resume", action="store_true",     help="Continue from checkpoint")
    p.add_argument("--test",   action="store_true",     help="Print one prompt and exit")
    return p.parse_args()

async def main_async(args):
    """Run asynchronous phase 4 validity judging workflow."""
    print("=" * 72)
    print("Phase 4.2 — Substitution Validity Judge (C2)")
    print("=" * 72)

    print("\n[1] Loading audit index …")
    audit_index = load_audit_index(AUDIT_PATH)
    print(f"    Eligible records: {len(audit_index):,}")

    print("[2] Loading CSI index …")
    csi_index = load_csi_index(CSI_PATH)
    print(f"    Source items: {len(csi_index):,}")

    print("[3] Loading hygiene index …")
    hygiene_index = load_hygiene_index(HYGIENE_PATH)
    print(f"    Hygiene records: {len(hygiene_index):,}")

    done_ids: set[str] = set()
    if args.resume and CKPT_PATH.exists():
        ckpt     = json.loads(CKPT_PATH.read_text())
        done_ids = set(ckpt.get("done_ids", []))
        print(f"    Resuming: {len(done_ids):,} already done.")

    eval_ids = [eid for eid in audit_index if eid not in done_ids]
    if args.pilot:
        eval_ids = eval_ids[: args.pilot]
        print(f"    PILOT MODE: {args.pilot} items")

    if args.test:
        eid = eval_ids[0] if eval_ids else list(audit_index.keys())[0]
        rec = audit_index[eid]
        csi_map = csi_index.get(rec["source_id"], {})
        to_judge, _ = build_substitutions(rec, csi_map)
        hyg = hygiene_index.get(eid, {})
        if to_judge:
            prompt = format_prompt(
                hyg.get("source_text", ""),
                hyg.get("adapted_text", ""),
                rec.get("domain", ""),
                to_judge[:BATCH_SIZE],
            )
            print(f"\n── SAMPLE PROMPT (eval_id={eid}) ──\n")
            print(prompt[:3000])
        else:
            print(f"eval_id {eid} has no LLM-judged substitutions.")
        return

    print(f"\n[4] Processing {len(eval_ids):,} eval_ids …")
    sem  = asyncio.Semaphore(CONCURRENCY)
    t0   = time.time()
    processed = 0

    fout = open(OUT_PATH, "a")
    try:
        tasks = {
            asyncio.create_task(
                process_one(eid, audit_index[eid], csi_index, hygiene_index, sem)
            ): eid
            for eid in eval_ids
        }

        for coro in asyncio.as_completed(tasks):
            result = await coro
            processed += 1
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                done_ids.add(result["eval_id"])

            if processed % 100 == 0:
                fout.flush()
                elapsed = time.time() - t0
                rate    = processed / max(elapsed, 1)
                eta     = (len(eval_ids) - processed) / max(rate, 0.01)
                print(f"  {processed:>6}/{len(eval_ids)} | {rate:.1f}/s | ETA {eta/60:.1f}m")

            if processed % CHECKPOINT_N == 0:
                CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))

    finally:
        fout.close()
        CKPT_PATH.write_text(json.dumps({"done_ids": list(done_ids)}))

    elapsed = time.time() - t0
    print(f"\n  Done: {processed:,} in {elapsed:.1f}s ({processed/elapsed:.1f}/s)")
    print(f"  Output → {OUT_PATH}")

    print("  Computing stats …")
    stats = compute_stats(OUT_PATH)
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"  Mean aggregate_c2 : {stats['mean_c2']}")
    print(f"  Total records     : {stats['total']:,}")
    print(f"  Stats → {STATS_PATH}")

def main():
    """Entrypoint that parses CLI args and runs async pipeline."""
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
