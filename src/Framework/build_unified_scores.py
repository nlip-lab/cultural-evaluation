"""
Build unified evaluation file by joining all available phase signals.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = Path(__file__).resolve().parent

def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def _resolve_source_csi_path() -> Path:
    p1 = EVAL_DIR / "01_CSI_Extraction" / "source_csi_final.jsonl"
    if p1.exists():
        return p1
    return EVAL_DIR / "01_CSI_Extraction" / "source_csi_final_google.jsonl"

def _find_existing(candidates):
    for c in candidates:
        if c.exists():
            return c
    return None

def build_source_dependency_index(source_csi_path: Path):
    out = {}
    for rec in _load_jsonl(source_csi_path):
        source_id = rec.get("source_id")
        if not source_id:
            continue
        csis = rec.get("csis", [])
        span_map = defaultdict(list)
        for idx, csi in enumerate(csis):
            span = _norm_text(csi.get("span", ""))
            if not span:
                continue
            deps = [d for d in csi.get("dependencies", []) if isinstance(d, str) and d.strip()]
            span_map[span].append({"idx": idx, "dependencies": deps})
        out[source_id] = span_map
    return out

def compute_dependency_completion_by_eval(audit_results_path: Path, source_dep_index):
    out = {}
    for rec in _load_jsonl(audit_results_path):
        eval_id = rec.get("eval_id")
        source_id = rec.get("source_id")
        alignments = rec.get("csi_alignments", [])
        span_map = source_dep_index.get(source_id, {})

        seen_counter = Counter()
        span_strategy_lists = defaultdict(list)
        for aln in alignments:
            span_strategy_lists[_norm_text(aln.get("source_span", ""))].append(aln.get("strategy", "omitted"))

        completions = []
        for aln in alignments:
            span_key = _norm_text(aln.get("source_span", ""))
            entries = span_map.get(span_key, [])
            if not entries:
                continue
            occ = seen_counter[span_key]
            seen_counter[span_key] += 1
            entry = entries[occ] if occ < len(entries) else entries[-1]
            deps = entry.get("dependencies", [])
            if not deps:
                continue
            dep_adapted = 0
            for dep in deps:
                dep_strats = span_strategy_lists.get(_norm_text(dep), [])
                if any(s != "retained" for s in dep_strats):
                    dep_adapted += 1
            completions.append(dep_adapted / len(deps))

        out[eval_id] = (sum(completions) / len(completions)) if completions else None
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(EVAL_DIR / "unified_scores.jsonl"))
    args = parser.parse_args()

    phase0_path = EVAL_DIR / "all_adaptations_indexed.jsonl"
    phase2_path = EVAL_DIR / "02_Hygiene" / "adaptations_with_hygiene.jsonl"
    phase3_metrics_path = EVAL_DIR / "03_Audit" / "csi_metrics.jsonl"
    phase3_audit_path = EVAL_DIR / "03_Audit" / "audit_results.jsonl"
    phase4_c2_path = EVAL_DIR / "04_Validity" / "c2_scores.jsonl"
    source_enrichment_path = EVAL_DIR / "" / "source_enrichment.jsonl"
    adaptation_consistency_path = EVAL_DIR / "" / "adaptation_consistency.jsonl"
    salience_weighted_c2_path = EVAL_DIR / "" / "salience_weighted_c2.jsonl"
    source_csi_path = _resolve_source_csi_path()

    fes_path = _find_existing(
        [
            EVAL_DIR / "06_Analysis" / "fes_results.jsonl",
            EVAL_DIR / "06_Analysis" / "fes_per_adaptation.jsonl",
            EVAL_DIR / "06_Analysis" / "fes_scores.jsonl",
        ]
    )
    fbs_path = _find_existing(
        [
            EVAL_DIR / "06_Analysis" / "fbs_results.jsonl",
            EVAL_DIR / "06_Analysis" / "fbs_per_adaptation.jsonl",
            EVAL_DIR / "06_Analysis" / "fbs_scores.jsonl",
        ]
    )
    fps_path = _find_existing(
        [
            EVAL_DIR / "06_Analysis" / "fps_results.jsonl",
            EVAL_DIR / "06_Analysis" / "fps_per_adaptation.jsonl",
            EVAL_DIR / "06_Analysis" / "fps_judge_scores.jsonl",
        ]
    )

    rt_surface_path = _find_existing([EVAL_DIR / "07_Reverse" / "phase9a_surface_metrics.jsonl"])
    rt_judge_path = _find_existing([EVAL_DIR / "07_Reverse" / "phase9d_llm_judge.jsonl"])
    rt_csi_path = _find_existing(
        [
            EVAL_DIR / "07_Reverse" / "phase9c_csi_alignment.jsonl",
            EVAL_DIR / "07_Reverse" / "phase9c_csi_alignment_google.jsonl",
        ]
    )

    base_records = list(_load_jsonl(phase0_path))
    base_by_eval = {r["eval_id"]: r for r in base_records}
    eval_by_key = {
        (r.get("model"), r.get("strategy"), r.get("source_id")): r.get("eval_id")
        for r in base_records
    }

    hygiene_by_eval = {}
    for r in _load_jsonl(phase2_path):
        h = r.get("02_hygiene", {})
        hygiene_by_eval[r["eval_id"]] = {
            "hygiene_overall": h.get("overall"),
            "hygiene_craft_label": h.get("craft_label"),
        }

    phase3_by_eval = {
        r["eval_id"]: {
            "rf": r.get("rf"),
            "tct": r.get("tct"),
            "sat": r.get("sat"),
            "strategy_dist": r.get("strategy_dist"),
        }
        for r in _load_jsonl(phase3_metrics_path)
    }
    audit_status_by_eval = {r["eval_id"]: r.get("audit_status") for r in _load_jsonl(phase3_audit_path)}

    c2_by_eval = {
        r["eval_id"]: {
            "c2_aggregate": r.get("aggregate_c2"),
            "worst_c2_subtype": r.get("worst_c2_subtype"),
        }
        for r in _load_jsonl(phase4_c2_path)
    }

    source_enrichment_by_source = {
        r["source_id"]: {
            "source_csi_count": r.get("csi_count"),
            "source_csi_density": r.get("csi_density"),
            "source_adaptation_load": r.get("adaptation_load"),
        }
        for r in _load_jsonl(source_enrichment_path)
    }

    consistency_by_eval = {
        r["eval_id"]: {
            "adaptation_consistency_entropy": r.get("strategy_entropy"),
            "adaptation_dominant_strategy": r.get("dominant_strategy"),
        }
        for r in _load_jsonl(adaptation_consistency_path)
    }

    salience_by_eval = {
        r["eval_id"]: {"c2_salience_weighted": r.get("salience_weighted_c2")}
        for r in _load_jsonl(salience_weighted_c2_path)
    }

    dep_index = build_source_dependency_index(source_csi_path)
    dependency_completion_by_eval = compute_dependency_completion_by_eval(phase3_audit_path, dep_index)

    fes_by_eval = {}
    if fes_path:
        for r in _load_jsonl(fes_path):
            eval_id = r.get("eval_id")
            if not eval_id:
                continue
            fes_by_eval[eval_id] = {
                "fes_contextual": r.get("mean_fes_contextual", r.get("fes_contextual")),
                "fes_holistic": r.get("mean_fes_holistic", r.get("fes_holistic")),
            }

    fbs_by_eval = {}
    if fbs_path:
        for r in _load_jsonl(fbs_path):
            eval_id = r.get("eval_id")
            if not eval_id:
                continue
            fbs_by_eval[eval_id] = {
                "fbs_contextual": r.get("mean_fbs_f1", r.get("fbs_contextual")),
                "fbs_holistic": r.get("mean_fbs_holistic_f1", r.get("fbs_holistic")),
            }

    fps_by_eval = {}
    if fps_path:
        for r in _load_jsonl(fps_path):
            eval_id = r.get("eval_id")
            if not eval_id:
                continue
            fps_by_eval[eval_id] = {"fps_overall": r.get("mean_fps_overall", r.get("fps_overall"))}

    rt_surface_by_eval = {}
    if rt_surface_path:
        for r in _load_jsonl(rt_surface_path):
            model = r.get("fwd_model", r.get("model"))
            prompt = r.get("fwd_prompt_level", r.get("prompt_level"))
            source_id = r.get("original_id")
            eval_id = eval_by_key.get((model, prompt, source_id))
            if not eval_id:
                continue
            rt_surface_by_eval[eval_id] = {
                "rt_bleu": r.get("bleu"),
                "rt_rouge_l": r.get("rouge_l_f"),
                "rt_bertscore": r.get("bertscore_f1"),
            }

    rt_judge_by_eval = {}
    if rt_judge_path:
        for r in _load_jsonl(rt_judge_path):
            model = r.get("fwd_model", r.get("model"))
            prompt = r.get("fwd_prompt_level", r.get("prompt_level"))
            source_id = r.get("original_id")
            eval_id = eval_by_key.get((model, prompt, source_id))
            if not eval_id:
                continue
            rt_judge_by_eval[eval_id] = {
                "rt_judge_overall": r.get("overall"),
                "rt_judge_meaning": r.get("meaning_preservation"),
                "rt_judge_structural": r.get("structural_preservation"),
                "rt_judge_detail": r.get("detail_preservation"),
            }

    rt_csi_by_eval = {}
    if rt_csi_path:
        for r in _load_jsonl(rt_csi_path):
            model = r.get("fwd_model", r.get("model"))
            prompt = r.get("fwd_prompt_level", r.get("prompt_level"))
            source_id = r.get("original_id")
            eval_id = eval_by_key.get((model, prompt, source_id))
            if not eval_id:
                continue
            rt_csi_by_eval[eval_id] = {
                "rt_csi_recovery": r.get("recovery_rate"),
                "rt_csi_leakage": r.get("leakage_rate"),
            }

    total_records = 0
    count_phase3 = 0
    count_phase4 = 0
    count_phase6_fes_fbs = 0
    count_phase6_fps = 0
    count_phase9 = 0
    count_core = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for base in base_records:
            eval_id = base["eval_id"]
            source_id = base.get("source_id")

            hygiene = hygiene_by_eval.get(eval_id, {})
            p3 = phase3_by_eval.get(eval_id, {})
            c2 = c2_by_eval.get(eval_id, {})
            source_enrich = source_enrichment_by_source.get(source_id, {})
            consistency = consistency_by_eval.get(eval_id, {})
            sal = salience_by_eval.get(eval_id, {})
            fes = fes_by_eval.get(eval_id, {})
            fbs = fbs_by_eval.get(eval_id, {})
            fps = fps_by_eval.get(eval_id, {})
            rts = rt_surface_by_eval.get(eval_id, {})
            rtj = rt_judge_by_eval.get(eval_id, {})
            rtc = rt_csi_by_eval.get(eval_id, {})

            unified = {
                "eval_id": eval_id,
                "source_id": source_id,
                "domain": base.get("domain"),
                "model": base.get("model"),
                "prompt_strategy": base.get("strategy"),
                "source_text": base.get("source_text"),
                "adapted_text": base.get("adapted_text"),
                "hygiene_overall": hygiene.get("hygiene_overall"),
                "hygiene_craft_label": hygiene.get("hygiene_craft_label"),
                "rf": p3.get("rf"),
                "tct": p3.get("tct"),
                "sat": p3.get("sat"),
                "strategy_dist": p3.get("strategy_dist"),
                "audit_status": audit_status_by_eval.get(eval_id),
                "c2_aggregate": c2.get("c2_aggregate"),
                "c2_salience_weighted": sal.get("c2_salience_weighted"),
                "worst_c2_subtype": c2.get("worst_c2_subtype"),
                "source_csi_count": source_enrich.get("source_csi_count", 0),
                "source_csi_density": source_enrich.get("source_csi_density", 0.0),
                "source_adaptation_load": source_enrich.get("source_adaptation_load", 0.0),
                "adaptation_consistency_entropy": consistency.get("adaptation_consistency_entropy"),
                "adaptation_dominant_strategy": consistency.get("adaptation_dominant_strategy"),
                "dependency_chain_completion": dependency_completion_by_eval.get(eval_id),
                "fes_contextual": fes.get("fes_contextual"),
                "fes_holistic": fes.get("fes_holistic"),
                "fbs_contextual": fbs.get("fbs_contextual"),
                "fbs_holistic": fbs.get("fbs_holistic"),
                "fps_overall": fps.get("fps_overall"),
                "rt_bleu": rts.get("rt_bleu"),
                "rt_rouge_l": rts.get("rt_rouge_l"),
                "rt_bertscore": rts.get("rt_bertscore"),
                "rt_judge_overall": rtj.get("rt_judge_overall"),
                "rt_judge_meaning": rtj.get("rt_judge_meaning"),
                "rt_judge_structural": rtj.get("rt_judge_structural"),
                "rt_judge_detail": rtj.get("rt_judge_detail"),
                "rt_csi_recovery": rtc.get("rt_csi_recovery"),
                "rt_csi_leakage": rtc.get("rt_csi_leakage"),
                "c3_overall": None,
                "c4_overall": None,
                "c1_score": None,
                "c5_score": None,
                "inca_score": None,
                "primary_craft_label": None,
            }
            fout.write(json.dumps(unified, ensure_ascii=False) + "\n")

            total_records += 1
            has_phase3 = all(unified.get(k) is not None for k in ("rf", "tct", "sat"))
            has_phase4 = unified.get("c2_aggregate") is not None
            has_phase6_fes_fbs = (unified.get("fes_contextual") is not None) and (
                unified.get("fbs_contextual") is not None
            )
            has_phase6_fps = unified.get("fps_overall") is not None
            has_phase9 = any(
                unified.get(k) is not None
                for k in (
                    "rt_bleu",
                    "rt_rouge_l",
                    "rt_bertscore",
                    "rt_judge_overall",
                    "rt_judge_meaning",
                    "rt_judge_structural",
                    "rt_judge_detail",
                    "rt_csi_recovery",
                    "rt_csi_leakage",
                )
            )
            has_core = (
                unified.get("hygiene_overall") is not None
                and has_phase3
                and has_phase4
            )

            count_phase3 += int(has_phase3)
            count_phase4 += int(has_phase4)
            count_phase6_fes_fbs += int(has_phase6_fes_fbs)
            count_phase6_fps += int(has_phase6_fps)
            count_phase9 += int(has_phase9)
            count_core += int(has_core)

    print(f"Unified file written: {out_path}")
    print(f"Total records: {total_records:,}")
    print(f"Records with Phase 3 data: {count_phase3:,}")
    print(f"Records with Phase 4 data: {count_phase4:,}")
    print(f"Records with Phase 6 FES/FBS data: {count_phase6_fes_fbs:,}")
    print(f"Records with Phase 6 FPS data: {count_phase6_fps:,}")
    print(f"Records with Phase 9 data: {count_phase9:,}")
    print(f"Records with all non-null core fields (hygiene + RF/TCT/SAT + C2): {count_core:,}")

if __name__ == "__main__":
    main()
