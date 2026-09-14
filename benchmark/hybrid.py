"""
Hybrid ranking for the semantic-weight sweep — merges lexical and semantic
candidates into one deduplicated, weighted-sum ranking.

Mirrors the *shape* of ai.localstudio.commercialmemory.HeuristicContextRanker
(a weighted sum of independent signals, sorted descending, no threshold, no
RRF) using this benchmark's own two available signals — lexical overlap and
cosine similarity — not the full production formula, which also needs
taskRelevance/recency/sourcePriority signals this synthetic dataset has no
meaningful values for. Per this benchmark's own explicit scope, only the
semantic weight varies; every other production weight (LEXICAL_WEIGHT here)
stays fixed at its real, shipped value.

This is a measurement tool, not a candidate implementation to copy into
commercial-memory or Mobile_mem0 as-is — see this benchmark's own README.
"""
from __future__ import annotations

# RankingWeights.lexical's own production value (commercial-memory,
# ai.localstudio.commercialmemory.RankingWeights) — held fixed across the
# whole semantic-weight sweep; never swept itself, per this benchmark's own
# explicit scope ("остальные production weights оставить неизменными").
LEXICAL_WEIGHT = 0.45

# Matches ai.localstudio.commercialmemory.AppMemory.DEFAULT_CANDIDATE_LIMIT —
# what a real MemorySemanticIndex.search(vector, limit) actually returns,
# not the whole corpus. Scoring every memory in the corpus by cosine
# similarity (this benchmark's own run_semantic() does, deliberately, to
# compute Recall@k fairly) and then treating all of them as "semantic
# candidates" for hybrid ranking would let a low-ranked item's small but
# nonzero cosine score leak into its hybrid score — something a real
# semantic search that never retrieved that item at all would never do
# (production's own CandidateMerge leaves semanticScore null, which
# HeuristicContextRanker treats as exactly zero, not "whatever cosine
# happened to be").
SEMANTIC_CANDIDATE_LIMIT = 40


def hybrid_rank(
    lexical_hits: list,
    semantic_scored: list[tuple[str, float]],
    semantic_weight: float,
    lexical_weight: float = LEXICAL_WEIGHT,
    semantic_candidate_limit: int = SEMANTIC_CANDIDATE_LIMIT,
) -> list[str]:
    """
    lexical_hits: lexical.rank()'s own output — List[LexicalHit], already
        only the memories with nonzero lexical overlap (MemoryRanking's own
        requireOverlap contract), each with its real Jaccard score.
    semantic_scored: the *full* corpus scored by cosine similarity against
        one query, sorted descending — e.g. run_semantic()'s own per-query
        `scored` list. Capped here to semantic_candidate_limit before
        combining, so this function decides what "found by semantic search"
        means, not the caller.

    Returns memory ids ranked by hybrid_score = lexical_score * lexical_weight
    + semantic_score * semantic_weight, descending (ties broken by id for
    determinism) — a plain weighted sum, exactly HeuristicContextRanker's own
    shape, over the union of what either retriever actually found.
    """
    lexical_by_id = {h.memory_id: h.score for h in lexical_hits}
    semantic_top = dict(semantic_scored[:semantic_candidate_limit])

    all_ids = set(lexical_by_id) | set(semantic_top)
    scored = [
        (mid, lexical_by_id.get(mid, 0.0) * lexical_weight + semantic_top.get(mid, 0.0) * semantic_weight)
        for mid in all_ids
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [mid for mid, _ in scored]
