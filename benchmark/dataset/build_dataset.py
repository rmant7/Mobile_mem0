#!/usr/bin/env python3
"""
Expands topics.py into memories.jsonl and queries.jsonl.

Deterministic by construction: ids are derived purely from each topic's
position in TOPICS and each memory/query's index within its topic, never
from anything like a random seed or wall-clock time — running this twice
against an unchanged topics.py produces byte-identical output. That is what
lets a benchmark run today be compared against one run next month: the
dataset itself never silently drifts.

Run from anywhere:
    python benchmark/dataset/build_dataset.py

Re-run this after editing topics.py; the two .jsonl files are checked in
(so the dataset can be read without running Python at all) but are always
regenerated from topics.py, never hand-edited directly.
"""
import json
from pathlib import Path

from topics import NEGATIVE_QUERIES, TOPICS

OUT_DIR = Path(__file__).parent


def build():
    memories = []
    queries = []
    memory_id_by_topic_index = {}

    for topic in TOPICS:
        topic_id = topic["id"]
        ids_for_topic = []
        for i, text in enumerate(topic["memories"]):
            mem_id = f"mem-{topic_id}-{i:02d}"
            ids_for_topic.append(mem_id)
            memories.append({"id": mem_id, "text": text, "topic": topic_id})
        memory_id_by_topic_index[topic_id] = ids_for_topic

        for i, q in enumerate(topic["queries"]):
            ground_truth = [ids_for_topic[idx] for idx in q["memory_indices"]]
            queries.append(
                {
                    "id": f"q-{topic_id}-{i:02d}",
                    "text": q["text"],
                    "category": q["category"],
                    "ground_truth_ids": ground_truth,
                }
            )

    for i, text in enumerate(NEGATIVE_QUERIES):
        queries.append(
            {
                "id": f"q-negative-{i:02d}",
                "text": text,
                "category": "negative",
                "ground_truth_ids": [],
            }
        )

    # Sanity checks — a mistyped memory_indices in topics.py should fail this
    # script loudly, not silently produce a query pointing at nothing (or, an
    # off-by-one away, at the wrong memory entirely).
    all_memory_ids = {m["id"] for m in memories}
    for q in queries:
        for gt in q["ground_truth_ids"]:
            assert gt in all_memory_ids, f"{q['id']} references unknown memory id {gt}"
        if q["category"] != "negative":
            assert q["ground_truth_ids"], f"{q['id']} is not category=negative but has no ground truth"
    duplicate_mem_ids = [m["id"] for m in memories if memories.count(m) > 1]
    assert len(all_memory_ids) == len(memories), f"duplicate memory ids: {duplicate_mem_ids}"
    duplicate_query_ids = {q["id"] for q in queries}
    assert len(duplicate_query_ids) == len(queries), "duplicate query ids"

    return memories, queries


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    memories, queries = build()
    write_jsonl(OUT_DIR / "memories.jsonl", memories)
    write_jsonl(OUT_DIR / "queries.jsonl", queries)

    by_category = {}
    for q in queries:
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1

    print(f"Wrote {len(memories)} memories and {len(queries)} queries to {OUT_DIR}")
    print("Queries by category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category:12s} {count}")


if __name__ == "__main__":
    main()
