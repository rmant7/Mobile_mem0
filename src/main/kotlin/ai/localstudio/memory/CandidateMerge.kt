package ai.localstudio.memory

/**
 * Merges lexical and semantic retrieval into one [MemoryCandidate] list —
 * factored out so [FileMemoryStore] and [InMemoryMemoryProvider] answer
 * [MemoryProvider.candidates] identically, the same reasoning [MemoryRanking]
 * already exists for [MemoryProvider.search].
 *
 * Deliberately does no fusion of its own: a candidate found by only one
 * retriever keeps `null` for the other's rank, and one found by both keeps
 * both. Which of them matters more is the caller's decision — see
 * `SEMANTIC_RETRIEVAL_DESIGN.md`'s boundary section.
 */
internal object CandidateMerge {
    fun merge(
        query: MemoryQuery,
        lexicalHits: List<MemoryItem>,
        semanticHits: List<ScoredId>,
        allItemsById: Map<String, MemoryItem>,
    ): List<MemoryCandidate> {
        val lexicalRankById = lexicalHits.withIndex().associate { (i, item) -> item.id to i }
        val semanticRankById = semanticHits.withIndex().associate { (i, hit) -> hit.id to i }
        val semanticScoreById = semanticHits.associate { it.id to it.score }

        val ids = LinkedHashSet<String>().apply {
            addAll(lexicalHits.map { it.id })
            addAll(semanticHits.map { it.id })
        }

        return ids.mapNotNull { id ->
            // A semantic hit can outlive its record between an index write
            // and the record's own removal (see FileMemoryStore.forget's own
            // best-effort cleanup ordering), and neither retriever knows
            // about the other's notion of scope/metadata — both are
            // re-checked here against the one MemoryItem that is still the
            // source of truth, rather than trusting either retriever alone.
            val item = allItemsById[id]?.takeIf { it.scope in query.scopes } ?: return@mapNotNull null
            if (query.metadataFilter.any { (k, v) -> item.metadata[k] != v }) return@mapNotNull null

            MemoryCandidate(
                item = item,
                lexicalRank = lexicalRankById[id],
                semanticRank = semanticRankById[id],
                semanticScore = semanticScoreById[id]?.toFloat(),
            )
        }
    }
}
