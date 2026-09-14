"""
hybrid.py, bootstrap.py and recommend.py are all pure functions over plain
lists/dicts — no GGUF, no llama_cpp needed to test the actual sweep and
recommendation logic that decides RECOMMENDED_SEMANTIC_WEIGHT.
"""
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

import bootstrap  # noqa: E402
import hybrid  # noqa: E402
import recommend  # noqa: E402
from lexical import LexicalHit  # noqa: E402
from metrics import QueryResult, aggregate  # noqa: E402

# --- hybrid.hybrid_rank -----------------------------------------------------


def test_hybrid_rank_merges_and_dedups_lexical_and_semantic_candidates():
    lexical_hits = [LexicalHit("a", 1.0), LexicalHit("b", 0.5)]
    semantic_scored = [("b", 0.8), ("c", 0.6)]
    ranked = hybrid.hybrid_rank(lexical_hits, semantic_scored, semantic_weight=0.5, lexical_weight=0.5)
    # a: 1.0*0.5 = 0.5 ; b: 0.5*0.5 + 0.8*0.5 = 0.65 ; c: 0.6*0.5 = 0.3
    assert ranked == ["b", "a", "c"]


def test_hybrid_rank_zero_semantic_weight_still_includes_semantic_only_items_with_zero_score():
    lexical_hits = [LexicalHit("a", 1.0)]
    semantic_scored = [("b", 0.99)]
    ranked = hybrid.hybrid_rank(lexical_hits, semantic_scored, semantic_weight=0.0, lexical_weight=0.45)
    # b has no lexical hit and semantic_weight=0 contributes nothing -> score 0, still present (union), just last.
    assert ranked[0] == "a"
    assert "b" in ranked


def test_hybrid_rank_semantic_candidate_limit_excludes_low_ranked_semantic_items():
    lexical_hits: list = []
    semantic_scored = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    ranked = hybrid.hybrid_rank(lexical_hits, semantic_scored, semantic_weight=1.0, semantic_candidate_limit=2)
    assert ranked == ["a", "b"]  # "c" fell outside the top-2 semantic candidates, never enters the union


def test_hybrid_rank_ties_break_by_memory_id():
    lexical_hits = [LexicalHit("z", 0.5), LexicalHit("a", 0.5)]
    ranked = hybrid.hybrid_rank(lexical_hits, [], semantic_weight=0.3, lexical_weight=1.0)
    assert ranked == ["a", "z"]


# --- bootstrap.bootstrap_mrr -------------------------------------------------


def test_bootstrap_is_deterministic_for_the_same_seed():
    results = [
        QueryResult("q1", "exact", ranked_ids=["a", "b"], ground_truth_ids=["a"]),
        QueryResult("q2", "exact", ranked_ids=["b", "a"], ground_truth_ids=["a"]),
        QueryResult("q3", "exact", ranked_ids=["c"], ground_truth_ids=["a"]),
    ]
    first = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=42)
    second = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=42)
    assert first == second


def test_bootstrap_different_seeds_can_differ():
    results = [
        QueryResult("q1", "exact", ranked_ids=["a", "b"], ground_truth_ids=["a"]),
        QueryResult("q2", "exact", ranked_ids=["b"], ground_truth_ids=["a"]),
        QueryResult("q3", "exact", ranked_ids=["c"], ground_truth_ids=["a"]),
        QueryResult("q4", "exact", ranked_ids=["d", "a"], ground_truth_ids=["a"]),
    ]
    a = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=1)
    b = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=2)
    assert a != b


def test_bootstrap_all_perfect_hits_has_zero_std():
    results = [QueryResult(f"q{i}", "exact", ranked_ids=["a"], ground_truth_ids=["a"]) for i in range(5)]
    out = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=42)
    assert out["mrr_mean"] == 1.0
    assert out["mrr_std"] == 0.0


def test_bootstrap_no_ground_truth_queries_returns_zeros_without_crashing():
    results = [QueryResult("q1", "negative", ranked_ids=["a"], ground_truth_ids=[])]
    out = bootstrap.bootstrap_mrr(results, n_iterations=50, seed=42)
    assert out == {
        "mrr_mean": 0.0,
        "mrr_std": 0.0,
        "recall_at_1_mean": 0.0,
        "recall_at_1_std": 0.0,
        "n_iterations": 50,
    }


# --- recommend.recommend_weight ---------------------------------------------


def _agg_dict(recall_at_1: float, mrr: float, recall_at_5: float = None, recall_at_10: float = None) -> dict:
    r5 = recall_at_5 if recall_at_5 is not None else recall_at_1
    r10 = recall_at_10 if recall_at_10 is not None else r5
    return {
        "recall_at_1": recall_at_1,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "mrr": mrr,
        "avg_cosine_positive": None,
        "avg_cosine_negative": None,
        "n_queries": 10,
    }


def _boot(std: float) -> dict:
    return {"mrr_mean": 0.0, "mrr_std": std, "recall_at_1_mean": 0.0, "recall_at_1_std": std, "n_iterations": 200}


def test_recommend_picks_a_clearly_winning_weight():
    lexical_overall = _agg_dict(recall_at_1=0.50, mrr=0.60)
    lexical_by_cat = {
        "identifier": _agg_dict(0.90, 0.90),
        "exact": _agg_dict(0.95, 0.95),
    }
    sweep = {
        0.0: {"overall": lexical_overall, "by_category": lexical_by_cat, "bootstrap": _boot(0.01)},
        0.2: {
            "overall": _agg_dict(0.55, 0.70),
            "by_category": {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)},
            "bootstrap": _boot(0.01),
        },
        0.5: {
            "overall": _agg_dict(0.56, 0.705),  # only a marginal (0.005) extra gain over 0.2's 0.70
            "by_category": {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)},
            "bootstrap": _boot(0.01),
        },
    }
    result = recommend.recommend_weight(lexical_overall, lexical_by_cat, sweep)
    assert result["ambiguous"] is False
    assert result["weight"] == 0.2  # 0.5's extra gain over 0.2 (0.005) doesn't clear noise (std 0.01)


def test_recommend_falls_back_to_0_20_when_every_weight_regresses_identifier():
    lexical_overall = _agg_dict(recall_at_1=0.50, mrr=0.60)
    lexical_by_cat = {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)}
    sweep = {
        0.3: {
            "overall": _agg_dict(0.60, 0.70),
            "by_category": {"identifier": _agg_dict(0.70, 0.70), "exact": _agg_dict(0.95, 0.95)},  # identifier tanked
            "bootstrap": _boot(0.01),
        },
    }
    result = recommend.recommend_weight(lexical_overall, lexical_by_cat, sweep)
    assert result["ambiguous"] is True
    assert result["weight"] == recommend.FALLBACK_WEIGHT
    assert result["candidates"] == []


def test_recommend_falls_back_to_0_20_when_gain_is_within_bootstrap_noise():
    lexical_overall = _agg_dict(recall_at_1=0.50, mrr=0.60)
    lexical_by_cat = {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)}
    sweep = {
        0.3: {
            "overall": _agg_dict(0.505, 0.605),  # technically higher, but tiny
            "by_category": {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)},
            "bootstrap": _boot(0.05),  # noise band (0.05) swamps the 0.005 gain
        },
    }
    result = recommend.recommend_weight(lexical_overall, lexical_by_cat, sweep)
    assert result["ambiguous"] is True
    assert result["weight"] == recommend.FALLBACK_WEIGHT
    assert result["candidates"] == [0.3]  # it passed the regression gate, but not the noise gate


def test_recommend_falls_back_when_no_weight_beats_lexical_mrr_or_recall1():
    lexical_overall = _agg_dict(recall_at_1=0.50, mrr=0.60)
    lexical_by_cat = {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)}
    sweep = {
        0.3: {
            "overall": _agg_dict(0.40, 0.50),  # worse than lexical on both
            "by_category": {"identifier": _agg_dict(0.90, 0.90), "exact": _agg_dict(0.95, 0.95)},
            "bootstrap": _boot(0.01),
        },
    }
    result = recommend.recommend_weight(lexical_overall, lexical_by_cat, sweep)
    assert result["ambiguous"] is True
    assert result["weight"] == recommend.FALLBACK_WEIGHT


# --- sanity: hybrid_rank output feeds straight into metrics.aggregate ------


def test_hybrid_rank_output_is_directly_usable_by_metrics_aggregate():
    lexical_hits = [LexicalHit("a", 1.0)]
    semantic_scored = [("a", 0.9), ("b", 0.1)]
    ranked = hybrid.hybrid_rank(lexical_hits, semantic_scored, semantic_weight=0.3)
    result = QueryResult("q1", "exact", ranked_ids=ranked, ground_truth_ids=["a"])
    agg = aggregate([result])
    assert agg.recall_at_1 == 1.0
