#!/usr/bin/env python3
"""
Entry point: python run_benchmark.py [--model-path PATH] [--dataset-dir DIR] [--out-dir DIR]

Loads the benchmark dataset, downloads (or reuses a cached)
multilingual-e5-base GGUF, runs both lexical and semantic retrieval over
every query, prints the headline comparison table plus a per-category
breakdown, and saves the full results to JSON and CSV.

Standalone: doesn't import anything from the app's Android/Kotlin code, and
doesn't change Mobile_mem0's own production sources — see this benchmark
directory's own README for why that separation is deliberate. Semantic
ranking here is a direct cosine-similarity sort for measurement purposes
only; it is not RRF, not a learned fusion, and nothing about it is meant to
be copied into Mobile_mem0's or the app's commercial ranking layer as-is.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lexical
from download_model import download
from embedder import E5Embedder
from metrics import AggregateMetrics, QueryResult, aggregate, aggregate_by_category, cosine_similarity

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_DATASET_DIR = BENCHMARK_DIR / "dataset"
DEFAULT_OUT_DIR = BENCHMARK_DIR / "results"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_lexical(queries: list[dict], memories: list[dict]) -> list[QueryResult]:
    results = []
    for q in queries:
        hits = lexical.rank(q["text"], memories)
        results.append(
            QueryResult(
                query_id=q["id"],
                category=q["category"],
                ranked_ids=[h.memory_id for h in hits],
                ground_truth_ids=q["ground_truth_ids"],
            )
        )
    return results


def run_semantic(queries: list[dict], memories: list[dict], embedder: E5Embedder) -> list[QueryResult]:
    memory_vectors = {m["id"]: embedder.embed_passage(m["text"]) for m in memories}

    results = []
    for q in queries:
        qvec = embedder.embed_query(q["text"])
        scored = [(mid, cosine_similarity(qvec, vec)) for mid, vec in memory_vectors.items()]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        results.append(
            QueryResult(
                query_id=q["id"],
                category=q["category"],
                ranked_ids=[mid for mid, _ in scored],
                ground_truth_ids=q["ground_truth_ids"],
                scores_by_id=dict(scored),
            )
        )
    return results


def print_table(title: str, rows: dict[str, AggregateMetrics]):
    print(f"\n{title}")
    print(f"{'':16s} {'Recall@1':>9s} {'Recall@5':>9s} {'Recall@10':>10s} {'MRR':>8s} {'n':>5s}")
    for name, m in rows.items():
        print(f"{name:16s} {m.recall_at_1:9.3f} {m.recall_at_5:9.3f} {m.recall_at_10:10.3f} {m.mrr:8.3f} {m.n_queries:5d}")


def save_results(out_dir: Path, overall: dict, by_category: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"overall": overall, "by_category": by_category}
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "retriever", "category", "recall_at_1", "recall_at_5", "recall_at_10", "mrr",
            "avg_cosine_positive", "avg_cosine_negative", "n_queries",
        ])
        for retriever, cats in by_category.items():
            writer.writerow([retriever, "overall", *overall[retriever].values()])
            for category, m in cats.items():
                writer.writerow([retriever, category, *m.values()])
    print(f"\nSaved {out_dir / 'results.json'} and {csv_path}")


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
    if embedder.dimension != 768:
        print(f"WARNING: expected 768-dimensional embeddings for multilingual-e5-base, got {embedder.dimension}")

    lexical_results = run_lexical(queries, memories)
    semantic_results = run_semantic(queries, memories, embedder)

    lex_overall = aggregate(lexical_results)
    sem_overall = aggregate(semantic_results)
    lex_by_cat = aggregate_by_category(lexical_results)
    sem_by_cat = aggregate_by_category(semantic_results)

    print_table("Overall", {"lexical": lex_overall, "semantic (base)": sem_overall})
    print(
        f"\nSemantic (base) avg cosine — positive matches: {sem_overall.avg_cosine_positive:.3f}, "
        f"negative (non-matches): {sem_overall.avg_cosine_negative:.3f}"
    )

    for cat in sorted(lex_by_cat.keys()):
        print_table(f"Category: {cat}", {"lexical": lex_by_cat[cat], "semantic (base)": sem_by_cat[cat]})

    save_results(
        Path(args.out_dir),
        overall={"lexical": lex_overall.as_dict(), "semantic_base": sem_overall.as_dict()},
        by_category={
            "lexical": {cat: m.as_dict() for cat, m in lex_by_cat.items()},
            "semantic_base": {cat: m.as_dict() for cat, m in sem_by_cat.items()},
        },
    )


if __name__ == "__main__":
    main()
