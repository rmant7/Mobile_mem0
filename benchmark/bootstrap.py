"""
Deterministic bootstrap resampling over the query pool — same seed every
run, so a stability report is itself reproducible, not a fresh random
result each time someone runs the notebook. This is not classical
statistical inference; it's a concrete, repeatable answer to "if this had
been a slightly different day's 92 queries, would the same weight still
look best," which a single point estimate can't answer on its own.
"""
from __future__ import annotations

import random
import statistics

from metrics import QueryResult, aggregate

DEFAULT_SEED = 42
DEFAULT_ITERATIONS = 200


def bootstrap_mrr(
    results: list[QueryResult],
    n_iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Resamples `results` (one weight's per-query outcomes) with replacement
    n_iterations times, recomputes MRR and Recall@1 on each resample, and
    reports their distribution. `seed` fixes the exact sequence of resamples
    — rerunning this with the same inputs always produces the same numbers.
    """
    scored = [r for r in results if r.ground_truth_ids]
    if not scored:
        return {
            "mrr_mean": 0.0,
            "mrr_std": 0.0,
            "recall_at_1_mean": 0.0,
            "recall_at_1_std": 0.0,
            "n_iterations": n_iterations,
        }

    rng = random.Random(seed)
    mrrs = []
    recall1s = []
    for _ in range(n_iterations):
        sample = rng.choices(scored, k=len(scored))
        agg = aggregate(sample)
        mrrs.append(agg.mrr)
        recall1s.append(agg.recall_at_1)

    return {
        "mrr_mean": statistics.mean(mrrs),
        "mrr_std": statistics.pstdev(mrrs),
        "recall_at_1_mean": statistics.mean(recall1s),
        "recall_at_1_std": statistics.pstdev(recall1s),
        "n_iterations": n_iterations,
    }
