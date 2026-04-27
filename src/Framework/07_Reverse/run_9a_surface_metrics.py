"""
Phase 9a - Surface metrics: BLEU, ROUGE-L, BERTScore

Compares each re-Americanized text (adapted_text) to the original American
reference (original_american_text) using three surface and embedding metrics:

BLEU - n-gram precision with brevity penalty
ROUGE-L - longest common subsequence recall
BERTScore - contextual embedding cosine similarity

The comparison is valid because both texts are in the same cultural frame:
American to American. Surface overlap is meaningful here because lexical
divergence can signal information loss.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNS_DIR = PROJECT_ROOT / "03_data" / "processed" / "reverse_adapted_data" / "runs"
PHASE_DIR = PROJECT_ROOT / "04_evaluation" / "07_Reverse"
OUT_PATH = PHASE_DIR / "phase9a_surface_metrics.jsonl"
STATS_PATH = PHASE_DIR / "phase9a_surface_stats.json"

DOMAINS = {"dialogue", "educational", "procedural", "stories"}
PROMPT_LEVELS = {"cot", "instructed", "oneshot", "simple"}
VALID_MODELS = {"gemma2", "llama", "llama8b", "mistral"}
BERTSCORE_MODEL_DEFAULT = "bert-base-uncased"
BERTSCORE_BATCH_SIZE = 128


def _check_deps(need_bertscore: bool) -> tuple:
    """Import optional metric dependencies and exit with install guidance if missing."""
    missing = []
    try:
        import sacrebleu as sb
    except ImportError:
        sb = None
        missing.append("sacrebleu")
    try:
        from rouge_score import rouge_scorer as _rs_mod
        rouge_scorer = _rs_mod.RougeScorer
    except ImportError:
        rouge_scorer = None
        missing.append("rouge-score")
    if need_bertscore:
        try:
            import bert_score as bs
        except ImportError:
            bs = None
            missing.append("bert-score")
    else:
        bs = None

    if missing:
        print("Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    return sb, rouge_scorer, bs


def parse_args():
    """Parse command-line arguments for the surface-metrics runner."""
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="Run on 20 items only")
    p.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore")
    p.add_argument("--domain", choices=list(DOMAINS), default=None)
    p.add_argument("--prompt-level", choices=list(PROMPT_LEVELS), default=None, dest="prompt_level")
    p.add_argument("--model", default=None, help="Restrict to one reverse-adaptation model")
    p.add_argument("--bertscore-model", default=BERTSCORE_MODEL_DEFAULT, help=f"HuggingFace model for BERTScore (default: {BERTSCORE_MODEL_DEFAULT})")
    p.add_argument("--resume", action="store_true", help="Skip rev_ids already in output file")
    return p.parse_args()


def load_items(domain_filter=None, prompt_level_filter=None, model_filter=None) -> list[dict]:
    """Load reverse-adapted items filtered by optional domain and model."""
    items = []
    skipped = 0
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
                        skipped += 1
                        continue
                    items.append({
                        "rev_id": rec.get("rev_id", ""),
                        "original_id": rec.get("id", ""),
                        "domain": rec.get("domain", file_domain),
                        "model": rec.get("model", file_model),
                        "prompt_level": rec.get("prompt_level", pl),
                        "fwd_prompt_level": rec.get("fwd_prompt_level", ""),
                        "hypothesis": adapted,
                        "reference": ref,
                        "hyp_len": len(adapted.split()),
                        "ref_len": len(ref.split()),
                    })
    print(f"  Loaded {len(items):,} items  ({skipped:,} skipped - empty text)")
    return items


def load_done_ids() -> set[str]:
    """Load rev_ids already present in the output file."""
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["rev_id"])
                    except Exception:
                        pass
    return done


def compute_bleu_rouge(items: list[dict], sb, rouge_scorer_cls) -> list[dict]:
    """Compute BLEU and ROUGE-L in place for each item."""
    scorer = rouge_scorer_cls(["rougeL"], use_stemmer=False)
    total = len(items)
    t0 = time.time()
    t_win = t0
    i_win = 0

    for i, item in enumerate(items, 1):
        hyp = item["hypothesis"]
        ref = item["reference"]
        try:
            b = sb.sentence_bleu(hyp, [ref])
            item["bleu"] = round(b.score / 100.0, 6)
        except Exception:
            item["bleu"] = 0.0
        try:
            sc = scorer.score(ref, hyp)
            item["rouge_l_p"] = round(sc["rougeL"].precision, 6)
            item["rouge_l_r"] = round(sc["rougeL"].recall, 6)
            item["rouge_l_f"] = round(sc["rougeL"].fmeasure, 6)
        except Exception:
            item["rouge_l_p"] = item["rouge_l_r"] = item["rouge_l_f"] = 0.0

        if i % 2000 == 0:
            t_win = time.time()
            i_win = i

        win_secs = (time.time() - t_win) or 0.001
        win_items = (i - i_win) or i
        rate = win_items / win_secs
        eta_s = (total - i) / rate if rate > 0 else 0
        pct = i / total * 100

        line = (
            f"  {i:>6}/{total} ({pct:.1f}%) | "
            f"{rate:>7.0f} it/s | ETA {eta_s/60:.1f}m | "
            f"bleu={item['bleu']:.3f}  rouge_l={item['rouge_l_f']:.3f}"
        )
        if i % 5000 == 0 or i == total:
            print(f"\r{line}", flush=True)
        else:
            print(f"\r{line}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"  BLEU+ROUGE-L done in {elapsed:.1f}s  ({total/elapsed:.0f} it/s)")
    return items


def compute_bertscore_batched(
    items: list[dict],
    bs_module,
    model_name: str,
    batch_size: int = BERTSCORE_BATCH_SIZE,
) -> list[dict]:
    """Compute BERTScore in batches and attach the scores in place."""
    hyps = [it["hypothesis"] for it in items]
    refs = [it["reference"] for it in items]

    t0 = time.time()
    print(f"  Running BERTScore [{model_name}] on {len(items):,} items ...")

    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            print(f"  BERTScore device: {device} ({gpu_name}, {free_gb:.1f} GB free)")
        else:
            device = "cpu"
            print("  BERTScore device: cpu")
        p, r, f = bs_module.score(
            hyps,
            refs,
            model_type=model_name,
            batch_size=batch_size,
            device=device,
            verbose=True,
            rescale_with_baseline=False,
        )
        p = p.tolist()
        r = r.tolist()
        f = f.tolist()
    except Exception as e:
        print(f"  WARNING: BERTScore failed ({e}); setting scores to -1")
        p = r = f = [-1.0] * len(items)

    elapsed = time.time() - t0
    for i, item in enumerate(items):
        item["bertscore_p"] = round(p[i], 6)
        item["bertscore_r"] = round(r[i], 6)
        item["bertscore_f1"] = round(f[i], 6)

    print(f"  BERTScore done in {elapsed/60:.1f}m")
    return items


def compute_stats(records: list[dict], has_bertscore: bool) -> dict:
    """Compute aggregate statistics for the output records."""
    def _avg(vals):
        v = [x for x in vals if x is not None and x >= 0]
        return round(sum(v) / len(v), 5) if v else None

    def _bucket(grp_key, records):
        grps = defaultdict(list)
        for r in records:
            grps[r[grp_key]].append(r)
        return grps

    def _grp_stats(recs):
        out = {
            "n": len(recs),
            "avg_bleu": _avg([r.get("bleu") for r in recs]),
            "avg_rouge_l_f": _avg([r.get("rouge_l_f") for r in recs]),
            "avg_ref_len": round(sum(r["ref_len"] for r in recs) / len(recs), 1) if recs else 0,
            "avg_hyp_len": round(sum(r["hyp_len"] for r in recs) / len(recs), 1) if recs else 0,
        }
        if has_bertscore:
            out["avg_bertscore_f1"] = _avg([r.get("bertscore_f1") for r in recs])
        return out

    return {
        "total": len(records),
        "has_bertscore": has_bertscore,
        "global": _grp_stats(records),
        "by_domain": {k: _grp_stats(v) for k, v in _bucket("domain", records).items()},
        "by_rev_model": {k: _grp_stats(v) for k, v in _bucket("model", records).items()},
        "by_prompt_level": {k: _grp_stats(v) for k, v in _bucket("prompt_level", records).items()},
        "by_fwd_prompt_level": {k: _grp_stats(v) for k, v in _bucket("fwd_prompt_level", records).items()},
    }


def main():
    """Run the full surface-metrics pipeline."""
    args = parse_args()
    need_bs = not args.no_bertscore

    print("=" * 70)
    print("Phase 9a - Surface Metrics (BLEU / ROUGE-L / BERTScore)")
    print("=" * 70)

    sb, rouge_scorer_cls, bs_module = _check_deps(need_bs)

    print("\n[1] Loading items ...")
    items = load_items(
        domain_filter=args.domain,
        prompt_level_filter=args.prompt_level,
        model_filter=args.model,
    )

    if args.test:
        items = items[:20]
        print(f"  TEST MODE: {len(items)} items")

    if args.resume and not args.test:
        done = load_done_ids()
        items = [it for it in items if it["rev_id"] not in done]
        print(f"  Resuming: {len(items):,} remaining after skipping {len(done):,} done")

    if not items:
        print("  Nothing to process.")
        return

    print(f"\n[2] Computing BLEU + ROUGE-L for {len(items):,} items ...")
    t0 = time.time()
    items = compute_bleu_rouge(items, sb, rouge_scorer_cls)
    print(f"  Done in {time.time() - t0:.1f}s")

    if need_bs:
        print("\n[3] Computing BERTScore ...")
        items = compute_bertscore_batched(items, bs_module, args.bertscore_model)
    else:
        for it in items:
            it["bertscore_p"] = it["bertscore_r"] = it["bertscore_f1"] = None

    print(f"\n[4] Writing {len(items):,} records ...")
    mode = "a" if args.resume else "w"
    with open(OUT_PATH, mode, encoding="utf-8") as fout:
        for it in items:
            row = {k: v for k, v in it.items() if k not in ("hypothesis", "reference")}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n[5] Computing aggregate stats ...")
    all_records = []
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    stats = compute_stats(all_records, has_bertscore=need_bs)
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    g = stats["global"]
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Items processed:   {stats['total']:,}")
    print(f"  Avg BLEU:          {g['avg_bleu']:.4f}")
    print(f"  Avg ROUGE-L F1:    {g['avg_rouge_l_f']:.4f}")
    if need_bs:
        print(f"  Avg BERTScore F1:  {g.get('avg_bertscore_f1', 'N/A')}")
    print("\n  By domain:")
    for d, v in sorted(stats["by_domain"].items()):
        bs_str = f"  BERTScore={v.get('avg_bertscore_f1', ''):.4f}" if need_bs and v.get("avg_bertscore_f1") else ""
        print(f"    {d:<20} BLEU={v['avg_bleu']:.4f}  ROUGE-L={v['avg_rouge_l_f']:.4f}{bs_str}")
    print("\n  By rev model:")
    for m, v in sorted(stats["by_rev_model"].items()):
        print(f"    {m:<20} BLEU={v['avg_bleu']:.4f}  ROUGE-L={v['avg_rouge_l_f']:.4f}")
    print("\n  By prompt level:")
    for pl, v in sorted(stats["by_prompt_level"].items()):
        print(f"    {pl:<20} BLEU={v['avg_bleu']:.4f}  ROUGE-L={v['avg_rouge_l_f']:.4f}")
    print(f"\n  Output : {OUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")


if __name__ == "__main__":
    main()
