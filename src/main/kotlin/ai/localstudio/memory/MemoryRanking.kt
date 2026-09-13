package ai.localstudio.memory

/**
 * The lexical-overlap-plus-recency scoring [FileMemoryStore] used before
 * [InMemoryMemoryProvider] existed too — factored out so both stores answer
 * [MemoryProvider.search] identically instead of one drifting from the
 * other's copy of the same logic. Pure and storage-agnostic on purpose: it
 * takes whatever snapshot of items its caller already holds under its own
 * lock and returns a ranked list, with no opinion on how those items are
 * kept or persisted.
 */
internal object MemoryRanking {

    private const val RECENCY_WEIGHT = 0.25
    private const val MIN_TERM_LENGTH = 3
    private val NON_WORD = Regex("[^\\p{L}\\p{N}]+")

    fun search(all: List<MemoryItem>, query: MemoryQuery): List<MemoryItem> {
        if (query.limit == 0) return emptyList()
        val terms = tokenize(query.text)
        // A caller that scopes by metadata, or asks for [MemoryQuery.matchAll],
        // is stating relevance explicitly and needs no lexical overlap on top
        // of that. See FileMemoryStore's own history: without this, a
        // meta-question about a just-attached document (or about memory
        // itself) found nothing, since it shares no vocabulary with the
        // actual content by definition, no matter how relevant it obviously is.
        val requireOverlap = query.metadataFilter.isEmpty() && !query.matchAll
        if (requireOverlap && terms.isEmpty()) return emptyList()

        val newest = all.maxOfOrNull { it.createdAt } ?: return emptyList()
        val oldest = all.minOfOrNull { it.createdAt } ?: newest
        val span = (newest - oldest).coerceAtLeast(1)

        return all
            .filter { it.scope in query.scopes }
            .filter { item -> query.metadataFilter.all { (k, v) -> item.metadata[k] == v } }
            .mapNotNull { item ->
                val overlap = overlap(terms, tokenize(item.text))
                if (requireOverlap && overlap == 0.0) return@mapNotNull null
                // Recency applies only to time-bound memories: a preference
                // stated a month ago is exactly as true as one stated today.
                val recency = if (item.scope == MemoryScope.SEMANTIC) {
                    0.0
                } else {
                    (item.createdAt - oldest).toDouble() / span
                }
                item.copy(relevance = (overlap + recency * RECENCY_WEIGHT) * scopeWeight(item.scope))
            }
            .sortedWith(compareByDescending<MemoryItem> { it.relevance ?: 0.0 }.thenBy { it.id })
            .take(query.limit)
    }

    private fun tokenize(text: String): Set<String> =
        text.lowercase()
            .split(NON_WORD)
            .filter { it.length >= MIN_TERM_LENGTH }
            .toSet()

    /**
     * Jaccard-style overlap, normalised by the query so long memories are not
     * favoured. An empty query has nothing to overlap with — dividing by a
     * query-term count of zero without this guard produced NaN, which then
     * poisoned every downstream relevance score for that item; every caller
     * with an empty [MemoryQuery.text] (metadata-scoped or [MemoryQuery.matchAll])
     * relied on [requireOverlap] alone to skip the *filter*, but still fed
     * this NaN into ranking, silently collapsing it to id order.
     */
    private fun overlap(queryTerms: Set<String>, itemTerms: Set<String>): Double {
        if (itemTerms.isEmpty() || queryTerms.isEmpty()) return 0.0
        val shared = queryTerms.count { it in itemTerms }
        return shared.toDouble() / queryTerms.size
    }

    /**
     * A stable fact beats an episode that matches equally well: semantic memory
     * is the distilled form, and the episode it was distilled from is
     * redundant. Between episodes, recency still decides.
     */
    private fun scopeWeight(scope: MemoryScope): Double = when (scope) {
        MemoryScope.SEMANTIC -> 1.4
        MemoryScope.EPISODIC -> 1.0
        MemoryScope.WORKING -> 1.0
    }
}
