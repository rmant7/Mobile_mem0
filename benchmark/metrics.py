"""
Retrieval metrics — deliberately tiny and dependency-free (plain lists of
ids in, plain numbers out) so lexical.py's and embedder.py's outputs can be
scored identically, and so test_benchmark.py can check them against
hand-computed values without needing numpy/pandas at all.
"""
from dataclasses import dataclass, field


def recall_at_k(ranked_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    """1.0 if any ground-truth id appears in the top k ranked ids, else 0.0 (per-query; average across queries for the usual Recall@k)."""
    if not ground_truth_ids:
        return 0.0
    truth = set(ground_truth_ids)
    return 1.0 if any(rid in truth for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], ground_truth_ids: list[str]) -> float:
    """1/rank of the first correct hit (1-indexed), 0.0 if none of ranked_ids is a ground-truth id."""
    truth = set(ground_truth_ids)
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in truth:
            return 1.0 / i
    return 0.0


@dataclass
class QueryResult:
    """One query's outcome for one retriever, kept around so per-category breakdowns can be computed after the fact without re-running retrieval."""

    query_id: str
    category: str
    ranked_ids: list[str]
    ground_truth_ids: list[str]
    # Cosine similarity between the query and each ranked id, when the
    # retriever computed one (semantic) — empty for lexical, whose score
    # isn't a cosine similarity and shouldn't be reported as if it were one.
    scores_by_id: dict = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    avg_cosine_positive: float | None
    avg_cosine_negative: float | None
    n_queries: int

    def as_dict(self) -> dict:
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "avg_cosine_positive": self.avg_cosine_positive,
            "avg_cosine_negative": self.avg_cosine_negative,
            "n_queries": self.n_queries,
        }


def aggregate(results: list[QueryResult]) -> AggregateMetrics:
    """
    Negative (no-ground-truth) queries are excluded from Recall/MRR — those
    metrics are undefined for a query with nothing to find, not zero (zero
    would just be "the retriever failed," which is backwards: finding
    nothing for a negative is the *correct* outcome). They still contribute
    to avg_cosine_negative, which is exactly the number that answers "how
    confidently does this retriever score something that isn't there."
    """
    scored = [r for r in results if r.ground_truth_ids]
    n = len(scored)

    def avg(fn) -> float:
        return sum(fn(r) for r in scored) / n if n else 0.0

    positive_scores = []
    negative_scores = []
    for r in results:
        truth = set(r.ground_truth_ids)
        for mem_id, score in r.scores_by_id.items():
            (positive_scores if mem_id in truth else negative_scores).append(score)

    return AggregateMetrics(
        recall_at_1=avg(lambda r: recall_at_k(r.ranked_ids, r.ground_truth_ids, 1)),
        recall_at_5=avg(lambda r: recall_at_k(r.ranked_ids, r.ground_truth_ids, 5)),
        recall_at_10=avg(lambda r: recall_at_k(r.ranked_ids, r.ground_truth_ids, 10)),
        mrr=avg(lambda r: reciprocal_rank(r.ranked_ids, r.ground_truth_ids)),
        avg_cosine_positive=(sum(positive_scores) / len(positive_scores)) if positive_scores else None,
        avg_cosine_negative=(sum(negative_scores) / len(negative_scores)) if negative_scores else None,
        n_queries=n,
    )


def aggregate_by_category(results: list[QueryResult]) -> dict[str, AggregateMetrics]:
    categories = sorted({r.category for r in results})
    return {cat: aggregate([r for r in results if r.category == cat]) for cat in categories}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
