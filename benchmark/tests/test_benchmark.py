"""
No GGUF download here — a small hand-written stand-in for E5Embedder covers
the prefix/dimension wiring; the rest of these tests are dataset integrity
and the metrics math itself, both checkable with nothing but the stdlib.
"""
import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "dataset"))

import lexical  # noqa: E402
import metrics  # noqa: E402
from build_dataset import build  # noqa: E402
from topics import NEGATIVE_QUERIES  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- dataset ---------------------------------------------------------------


def test_dataset_build_is_deterministic():
    memories_a, queries_a = build()
    memories_b, queries_b = build()
    assert memories_a == memories_b
    assert queries_a == queries_b


def test_committed_dataset_matches_current_topics_py():
    """Guards against editing topics.py without regenerating the checked-in .jsonl files."""
    memories, queries = build()
    committed_memories = _read_jsonl(BENCHMARK_DIR / "dataset" / "memories.jsonl")
    committed_queries = _read_jsonl(BENCHMARK_DIR / "dataset" / "queries.jsonl")
    assert memories == committed_memories, "run `python dataset/build_dataset.py` after editing topics.py"
    assert queries == committed_queries, "run `python dataset/build_dataset.py` after editing topics.py"


def test_dataset_size_is_in_the_requested_range():
    memories, queries = build()
    assert 100 <= len(memories) <= 150
    assert 50 <= len(queries) <= 110  # a little headroom over 100 for category balance


def test_every_ground_truth_id_exists():
    memories, queries = build()
    ids = {m["id"] for m in memories}
    for q in queries:
        for gt in q["ground_truth_ids"]:
            assert gt in ids


def test_negative_queries_have_no_ground_truth():
    _, queries = build()
    negatives = [q for q in queries if q["category"] == "negative"]
    assert len(negatives) == len(NEGATIVE_QUERIES)
    assert all(q["ground_truth_ids"] == [] for q in negatives)


def test_non_negative_queries_have_ground_truth():
    _, queries = build()
    for q in queries:
        if q["category"] != "negative":
            assert q["ground_truth_ids"], f"{q['id']} has no ground truth"


def test_all_seven_categories_present():
    _, queries = build()
    categories = {q["category"] for q in queries}
    assert categories == {"exact", "morphology", "synonym", "paraphrase", "low_overlap", "identifier", "negative"}


# --- lexical.py (ported from MemoryRanking.kt) ------------------------------


def test_lexical_tokenize_lowercases_and_drops_short_terms():
    assert lexical.tokenize("Пользователь ест Мясо") == {"пользователь", "ест", "мясо"}
    # "не" and "у" are below MIN_TERM_LENGTH (3) — dropped, same as the Kotlin original.
    assert lexical.tokenize("не у меня") == {"меня"}


def test_lexical_overlap_is_jaccard_normalised_by_query_term_count():
    query_terms = {"мясо", "ест", "пользователь"}
    item_terms = {"мясо", "рыба"}
    assert lexical.overlap(query_terms, item_terms) == 1 / 3


def test_lexical_overlap_empty_inputs_are_zero_not_nan():
    assert lexical.overlap(set(), {"мясо"}) == 0.0
    assert lexical.overlap({"мясо"}, set()) == 0.0


def test_lexical_rank_finds_overlap_and_skips_zero_overlap_items():
    memories = [
        {"id": "a", "text": "Пользователь не ест мясо"},
        {"id": "b", "text": "Совершенно другой текст без общих слов"},
    ]
    hits = lexical.rank("ест ли он мясо", memories)
    assert [h.memory_id for h in hits] == ["a"]


def test_lexical_rank_query_with_no_real_tokens_finds_nothing():
    memories = [{"id": "a", "text": "Пользователь ест мясо"}]
    assert lexical.rank("но и от", memories) == []  # every token below MIN_TERM_LENGTH


# --- metrics.py --------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one():
    assert abs(metrics.cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert abs(metrics.cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert metrics.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_recall_and_mrr_on_a_hand_computed_example():
    results = [
        metrics.QueryResult("q1", "exact", ranked_ids=["a", "b", "c"], ground_truth_ids=["a"]),
        metrics.QueryResult("q2", "exact", ranked_ids=["x", "y", "b"], ground_truth_ids=["b"]),
        metrics.QueryResult("q3", "exact", ranked_ids=["z"], ground_truth_ids=["not-found"]),
    ]
    agg = metrics.aggregate(results)
    # q1: hit at rank 1 -> recall@1=1, rr=1
    # q2: hit at rank 3 -> recall@1=0, recall@5=1, rr=1/3
    # q3: never found -> all zero
    assert agg.recall_at_1 == 1 / 3
    assert agg.recall_at_5 == 2 / 3
    assert agg.recall_at_10 == 2 / 3
    assert abs(agg.mrr - ((1.0 + 1 / 3 + 0.0) / 3)) < 1e-9
    assert agg.n_queries == 3


def test_negative_queries_excluded_from_recall_but_feed_avg_cosine_negative():
    results = [
        metrics.QueryResult("q1", "exact", ranked_ids=["a"], ground_truth_ids=["a"], scores_by_id={"a": 0.9, "b": 0.1}),
        metrics.QueryResult("q2", "negative", ranked_ids=["b"], ground_truth_ids=[], scores_by_id={"a": 0.05, "b": 0.02}),
    ]
    agg = metrics.aggregate(results)
    assert agg.n_queries == 1  # only q1 has ground truth, so only it counts toward Recall/MRR
    assert agg.recall_at_1 == 1.0
    # positive: q1's "a" (its own ground truth, 0.9).
    # negative: q1's "b" (0.1) plus both of q2's scores (0.05, 0.02) - q2 has
    # no ground truth at all, so neither of its ids can be a positive.
    assert abs(agg.avg_cosine_positive - 0.9) < 1e-9
    assert abs(agg.avg_cosine_negative - ((0.1 + 0.05 + 0.02) / 3)) < 1e-9


def test_aggregate_by_category_splits_correctly():
    results = [
        metrics.QueryResult("q1", "exact", ranked_ids=["a"], ground_truth_ids=["a"]),
        metrics.QueryResult("q2", "low_overlap", ranked_ids=["z"], ground_truth_ids=["a"]),
    ]
    by_cat = metrics.aggregate_by_category(results)
    assert set(by_cat.keys()) == {"exact", "low_overlap"}
    assert by_cat["exact"].recall_at_1 == 1.0
    assert by_cat["low_overlap"].recall_at_1 == 0.0


# --- embedder.py: prefix + dimension wiring (no real GGUF, no download) ----


def test_embed_query_and_embed_passage_use_the_documented_prefixes(monkeypatch):
    pytest.importorskip("llama_cpp")
    import embedder as embedder_module

    calls = []

    class _RecordingLlama:
        def __init__(self, **kwargs):
            pass

        def n_embd(self):
            return 8

        def embed(self, text):
            calls.append(text)
            return [0.1] * 8

    monkeypatch.setattr(embedder_module, "Llama", _RecordingLlama)

    e = embedder_module.E5Embedder("unused-path")
    e.embed_query("привет")
    e.embed_passage("факт о пользователе")

    assert calls == [f"{embedder_module.QUERY_PREFIX}привет", f"{embedder_module.PASSAGE_PREFIX}факт о пользователе"]


def test_embedder_reports_the_loaded_models_own_dimension(monkeypatch):
    pytest.importorskip("llama_cpp")
    import embedder as embedder_module

    class _FixedDimLlama:
        def __init__(self, **kwargs):
            pass

        def n_embd(self):
            return 768

        def embed(self, text):
            return [0.0] * 768

    monkeypatch.setattr(embedder_module, "Llama", _FixedDimLlama)

    e = embedder_module.E5Embedder("unused-path")
    assert e.dimension == 768


def test_embed_output_is_l2_normalized(monkeypatch):
    pytest.importorskip("llama_cpp")
    import embedder as embedder_module

    class _UnnormalizedLlama:
        def __init__(self, **kwargs):
            pass

        def n_embd(self):
            return 3

        def embed(self, text):
            return [3.0, 4.0, 0.0]  # norm 5, deliberately not already unit length

    monkeypatch.setattr(embedder_module, "Llama", _UnnormalizedLlama)

    e = embedder_module.E5Embedder("unused-path")
    vector = e.embed_query("test")
    norm = sum(x * x for x in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9
