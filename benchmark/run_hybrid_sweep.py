#!/usr/bin/env python3
"""
Extends run_benchmark.py's own lexical-vs-semantic comparison (Part A) with
the hybrid-ranking semantic-weight sweep and the robustness/recommendation
report (Part B/C) needed to actually tune production's SEMANTIC_RANKING_WEIGHT
— one run, over the same dataset and the same per-query lexical/semantic
signals run_benchmark.py itself computes, rather than a second pipeline that
could drift from it.

python run_hybrid_sweep.py [--model-path PATH] [--dataset-dir DIR] [--out-dir DIR]

Only the semantic weight is swept (hybrid.py's LEXICAL_WEIGHT stays fixed at
its real production value) — no new models, no dataset changes; see this
directory's README for the full scope and non-goals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lexical
from bootstrap import bootstrap_mrr
from download_model import download
from embedder import E5Embedder
from hybrid import hybrid_rank
from metrics import QueryResult, aggregate, aggregate_by_category, cosine_similarity
from recommend import recommend_weight
from run_benchmark import DEFAULT_DATASET_DIR, load_jsonl, print_table

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_OUT_DIR = BENCHMARK_DIR / "results"

# This task's own explicit sweep.
SEMANTIC_WEIGHTS = [0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

# Categories the task asked to watch explicitly in every weight's breakdown.
WATCH_CATEGORIES = ["exact", "identifier", "low_overlap", "synonym", "morphology", "paraphrase"]


def compute_raw(queries: list[dict], memories: list[dict], embedder: E5Embedder) -> list[dict]:
    """
    One pass over every query: real lexical.rank() hits (MemoryRanking's own
    algorithm) and the full corpus scored by cosine similarity — computed
    once and reused for every swept weight, so the sweep is a pure
    re-ranking exercise, not eight repeated embedding passes.
    """
    memory_vectors = {m["id"]: embedder.embed_passage(m["text"]) for m in memories}
    raw = []
    for q in queries:
        lexical_hits = lexical.rank(q["text"], memories)
        qvec = embedder.embed_query(q["text"])
        scored = [(mid, cosine_similarity(qvec, vec)) for mid, vec in memory_vectors.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        raw.append({"query": q, "lexical_hits": lexical_hits, "semantic_scored": scored})
    return raw


def run_hybrid_for_weight(raw: list[dict], semantic_weight: float) -> list[QueryResult]:
    results = []
    for item in raw:
        q = item["query"]
        ranked_ids = hybrid_rank(item["lexical_hits"], item["semantic_scored"], semantic_weight)
        results.append(
            QueryResult(
                query_id=q["id"],
                category=q["category"],
                ranked_ids=ranked_ids,
                ground_truth_ids=q["ground_truth_ids"],
            )
        )
    return results


def print_sweep_table(sweep: dict):
    print(f"\n{'weight':>8s} {'Recall@1':>9s} {'Recall@5':>9s} {'Recall@10':>10s} {'MRR':>8s} {'MRR std':>9s}")
    for w in sorted(sweep.keys()):
        o = sweep[w]["overall"]
        b = sweep[w]["bootstrap"]
        print(
            f"{w:8.2f} {o['recall_at_1']:9.3f} {o['recall_at_5']:9.3f} "
            f"{o['recall_at_10']:10.3f} {o['mrr']:8.3f} {b['mrr_std']:9.4f}"
        )


def print_category_tables(sweep: dict, categories: list[str]):
    for cat in categories:
        print(f"\nCategory: {cat}")
        print(f"{'weight':>8s} {'Recall@1':>9s} {'Recall@5':>9s} {'Recall@10':>10s} {'MRR':>8s}")
        for w in sorted(sweep.keys()):
            m = sweep[w]["by_category"].get(cat)
            if m is None:
                continue
            print(f"{w:8.2f} {m['recall_at_1']:9.3f} {m['recall_at_5']:9.3f} {m['recall_at_10']:10.3f} {m['mrr']:8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=None, help="Local GGUF path; downloads from Hugging Face if omitted")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    memories = load_jsonl(dataset_dir / "memories.jsonl")
    queries = load_jsonl(dataset_dir / "queries.jsonl")
    print(f"Loaded {len(memories)} memories, {len(queries)} queries from {dataset_dir}")

    model_path = args.model_path or download()
    print(f"Using model: {model_path}")
    embedder = E5Embedder(model_path)
    print(f"Embedding dimension: {embedder.dimension}")

    # Part A: re-confirm the existing lexical-vs-semantic baseline.
    lexical_results = [
        QueryResult(
            query_id=q["id"],
            category=q["category"],
            ranked_ids=[h.memory_id for h in lexical.rank(q["text"], memories)],
            ground_truth_ids=q["ground_truth_ids"],
        )
        for q in queries
    ]
    lex_overall = aggregate(lexical_results)
    lex_by_cat = aggregate_by_category(lexical_results)
    print_table("Part A -- Overall (lexical baseline, re-confirmed)", {"lexical": lex_overall})

    # Part B/C: hybrid sweep, computed once, reused across every weight.
    raw = compute_raw(queries, memories, embedder)

    sweep = {}
    for w in SEMANTIC_WEIGHTS:
        hybrid_results = run_hybrid_for_weight(raw, w)
        overall = aggregate(hybrid_results)
        by_cat = aggregate_by_category(hybrid_results)
        boot = bootstrap_mrr(hybrid_results)
        sweep[w] = {
            "overall": overall.as_dict(),
            "by_category": {cat: m.as_dict() for cat, m in by_cat.items()},
            "bootstrap": boot,
        }
        print(
            f"weight={w:.2f}: MRR={overall.mrr:.3f} (bootstrap std={boot['mrr_std']:.4f}), "
            f"Recall@1={overall.recall_at_1:.3f}"
        )

    print("\nPart B -- Hybrid sweep, overall")
    print_sweep_table(sweep)

    print("\nPart B/C -- Category breakdown per weight")
    print_category_tables(sweep, WATCH_CATEGORIES)

    recommendation = recommend_weight(
        lexical_overall=lex_overall.as_dict(),
        lexical_by_category={cat: m.as_dict() for cat, m in lex_by_cat.items()},
        sweep=sweep,
    )
    print(f"\n{recommendation['reason']}")
    print(f"RECOMMENDED_SEMANTIC_WEIGHT = {recommendation['weight']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "lexical_overall": lex_overall.as_dict(),
        "lexical_by_category": {cat: m.as_dict() for cat, m in lex_by_cat.items()},
        "sweep": {str(w): v for w, v in sweep.items()},
        "recommendation": recommendation,
    }
    out_path = out_dir / "hybrid_sweep_results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
