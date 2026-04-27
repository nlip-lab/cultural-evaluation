"""
Phase 3.3 — Compute CSI-level metrics: RF, TCT, SAT
=====================================================
Reads audit_results.jsonl + source_csi_final_google.jsonl and computes:
  - RF  (Residual Foreignness)   : weighted leakage of un-adapted source CSIs
  - TCT (Target Cultural Transfer): weighted validity of transfer
  - SAT (Cultural Saturation)    : unsupported Indian-culture additions

Usage:
    python compute_csi_metrics.py [--audit AUDIT_JSONL] [--csi CSI_JSONL] [--out OUT_JSONL]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PHASE3_DIR   = PROJECT_ROOT / "04_evaluation" / "03_Audit"
PHASE1_DIR   = PROJECT_ROOT / "04_evaluation" / "01_CSI_Extraction"

DEFAULT_AUDIT = PHASE3_DIR / "audit_results.jsonl"
DEFAULT_CSI   = PHASE1_DIR / "source_csi_final_google.jsonl"
DEFAULT_OUT   = PHASE3_DIR / "csi_metrics.jsonl"
DEFAULT_SUMMARY = PHASE3_DIR / "csi_metrics_summary.json"

EPSILON = 0.01

def _leakage(strategy: str, necessity: str) -> float:
    """Map strategy and necessity to residual foreignness leakage weight."""
    if strategy == "retained":
        if necessity == "must_adapt":   return 1.0
        if necessity == "may_adapt":    return 0.5
        if necessity == "should_retain": return 0.0
        return 0.5
    if strategy == "distorted":
        return 0.7
    return 0.0

def _transfer(strategy: str, necessity: str) -> float:
    """Map strategy and necessity to target-transfer validity weight."""
    if strategy == "localized":  return 1.0
    if strategy == "transformed": return 0.8
    if strategy == "globalized":  return 0.5
    if strategy == "retained":
        if necessity == "should_retain": return 1.0
        return 0.0
    if strategy == "omitted":   return 0.0
    if strategy == "distorted": return 0.0
    return 0.0

STRATEGIES = ["retained", "localized", "globalized", "transformed", "omitted", "distorted"]

def _strategy_dist(alignments: list[dict]) -> dict:
    """Compute normalized strategy distribution for one alignment list."""
    total = max(len(alignments), 1)
    counts = defaultdict(int)
    for a in alignments:
        counts[a.get("strategy", "omitted")] += 1
    return {f"pct_{s}": round(counts[s] / total, 4) for s in STRATEGIES}

def load_csi_index(csi_path: Path) -> dict[str, dict]:
    """Returns {source_id: {span_text: {salience, necessity}}}."""
    idx: dict[str, dict] = {}
    with open(csi_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("source_id") or rec.get("id")
            if not sid:
                continue
            span_map: dict[str, dict] = {}
            for csi in rec.get("csis", []):
                span = (csi.get("span") or "").strip()
                if span:
                    span_map[span] = {
                        "salience":   float(csi.get("salience", 0.5)),
                        "necessity":  csi.get("adaptation_necessity", "may_adapt"),
                    }
            idx[sid] = span_map
    return idx

def compute_metrics(record: dict, csi_index: dict) -> dict | None:
    """Compute RF, TCT, and SAT for a successful audit record."""
    if record.get("audit_status") != "success":
        return None

    source_id  = record["source_id"]
    alignments = record.get("csi_alignments", [])
    additions  = record.get("unsupported_additions", [])
    span_map   = csi_index.get(source_id, {})

    sum_w   = 0.0
    sum_wl  = 0.0
    sum_wv  = 0.0

    for aln in alignments:
        span      = (aln.get("source_span") or "").strip()
        strategy  = aln.get("strategy", "omitted")
        csi_info  = span_map.get(span, {})
        necessity = csi_info.get("necessity", "may_adapt")
        salience  = csi_info.get("salience", 0.5)

        w = salience
        sum_w  += w
        sum_wl += w * _leakage(strategy, necessity)
        sum_wv += w * _transfer(strategy, necessity)

    rf  = sum_wl / sum_w if sum_w > 0 else 0.0
    tct = sum_wv / sum_w if sum_w > 0 else 0.0

    sat_num = sum(
        0.3 if a.get("justified") else 1.0
        for a in additions
    )
    sat = sat_num / (sum_w + EPSILON)

    strategy_dist = _strategy_dist(alignments)

    return {
        "eval_id":                  record["eval_id"],
        "source_id":                source_id,
        "domain":                   record.get("domain", ""),
        "model":                    record.get("model", ""),
        "prompt_strategy":          record.get("strategy", ""),
        "rf":                       round(rf, 4),
        "tct":                      round(tct, 4),
        "sat":                      round(sat, 4),
        "strategy_dist":            strategy_dist,
        "source_csi_count":         len(alignments),
        "unsupported_addition_count": len(additions),
    }

def _mean(vals: list[float]) -> float | None:
    """Return rounded mean for non-empty numeric list."""
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)

def _agg_by(records: list[dict], key: str) -> dict:
    """Aggregate metrics grouped by the requested key."""
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        g = r.get(key, "unknown")
        groups[g]["rf"].append(r["rf"])
        groups[g]["tct"].append(r["tct"])
        groups[g]["sat"].append(r["sat"])
        for s in STRATEGIES:
            groups[g][f"pct_{s}"].append(r["strategy_dist"].get(f"pct_{s}", 0.0))
    out = {}
    for g, metrics in groups.items():
        out[g] = {
            "rf":  _mean(metrics["rf"]),
            "tct": _mean(metrics["tct"]),
            "sat": _mean(metrics["sat"]),
            "count": len(metrics["rf"]),
            "strategy_dist": {s: _mean(metrics[f"pct_{s}"]) for s in STRATEGIES},
        }
    return out

def build_summary(records: list[dict]) -> dict:
    """Build aggregated stats dict."""
    by_model    = _agg_by(records, "model")
    by_strategy = _agg_by(records, "prompt_strategy")
    by_domain   = _agg_by(records, "domain")

    by_model_strategy: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = f"{r['model']}__{r['prompt_strategy']}"
        by_model_strategy[key]["rf"].append(r["rf"])
        by_model_strategy[key]["tct"].append(r["tct"])
        by_model_strategy[key]["sat"].append(r["sat"])
    mxs = {}
    for k, m in by_model_strategy.items():
        mxs[k] = {"rf": _mean(m["rf"]), "tct": _mean(m["tct"]), "sat": _mean(m["sat"]), "count": len(m["rf"])}

    return {
        "total_records":    len(records),
        "by_model":         by_model,
        "by_strategy":      by_strategy,
        "by_domain":        by_domain,
        "by_model_strategy": mxs,
    }

def main():
    """Run metric computation pipeline and write summary outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit",   default=str(DEFAULT_AUDIT))
    parser.add_argument("--csi",     default=str(DEFAULT_CSI))
    parser.add_argument("--out",     default=str(DEFAULT_OUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    audit_path   = Path(args.audit)
    csi_path     = Path(args.csi)
    out_path     = Path(args.out)
    summary_path = Path(args.summary)

    print(f"Loading CSI index from {csi_path.name} …")
    csi_index = load_csi_index(csi_path)
    print(f"  {len(csi_index):,} source items indexed.")

    print(f"Processing {audit_path.name} …")
    total = success = skipped = 0
    metric_records = []

    with open(audit_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            m = compute_metrics(record, csi_index)
            if m is None:
                skipped += 1
                continue
            success += 1
            metric_records.append(m)
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"  Total: {total:,}  |  Success: {success:,}  |  Skipped: {skipped:,}")
    print(f"  Output → {out_path}")

    print("Building summary …")
    summary = build_summary(metric_records)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary → {summary_path}")

    print("\n── Mean RF / TCT / SAT by model ──")
    for model, s in sorted(summary["by_model"].items()):
        print(f"  {model:<14} RF={s['rf']:.3f}  TCT={s['tct']:.3f}  SAT={s['sat']:.3f}  n={s['count']:,}")

    print("\n── Mean RF / TCT / SAT by strategy ──")
    for strat, s in sorted(summary["by_strategy"].items()):
        print(f"  {strat:<12} RF={s['rf']:.3f}  TCT={s['tct']:.3f}  SAT={s['sat']:.3f}  n={s['count']:,}")

    print("\n── Mean RF / TCT / SAT by domain ──")
    for domain, s in sorted(summary["by_domain"].items()):
        print(f"  {domain:<14} RF={s['rf']:.3f}  TCT={s['tct']:.3f}  SAT={s['sat']:.3f}  n={s['count']:,}")

if __name__ == "__main__":
    main()
