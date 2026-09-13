package ai.localstudio.memory

/**
 * The [MemorySemanticIndex] + [MemoryEmbedder] half of a [MemoryProvider]
 * backend — factored out for the same reason [MemoryRanking] and
 * [CandidateMerge] already are: [FileMemoryStore] and [InMemoryMemoryProvider]
 * both compose the same way with an optional semantic index, and this is the
 * one place that composition is implemented, so neither can drift from the
 * other's copy of it.
 */
internal object SemanticRetrieval {

    /**
     * Embeds every item in [allItems] that [index] doesn't have a vector for
     * yet, up to [limitPerCall] of them. See [FileMemoryStore.embedPending]'s
     * own doc comment for why this exists as an explicit, caller-invoked
     * operation rather than something [MemoryProvider.remember] does itself.
     */
    suspend fun embedPending(allItems: List<MemoryItem>, index: MemorySemanticIndex, embedder: MemoryEmbedder, limitPerCall: Int) {
        val byId = allItems.associateBy { it.id }
        val pendingIds = index.missing(allItems.map { it.id }).take(limitPerCall)
        val toEmbed = pendingIds.mapNotNull { id -> byId[id]?.let { id to it.text } }
        if (toEmbed.isEmpty()) return
        val vectors = embedder.embedForStorage(toEmbed.map { it.second })
        toEmbed.forEachIndexed { i, (id, _) -> index.upsert(id, vectors[i]) }
    }

    /**
     * [MemoryProvider.candidates] for a backend that may or may not have
     * [index] and [embedder] configured — falls back to lexical-only,
     * exactly matching [MemoryProvider.candidates]'s own default
     * implementation, whenever either is missing, [query] has blank text, or
     * embedding the query itself fails to produce a usable vector.
     */
    suspend fun candidates(
        query: MemoryQuery,
        lexicalHits: List<MemoryItem>,
        allItems: List<MemoryItem>,
        index: MemorySemanticIndex?,
        embedder: MemoryEmbedder?,
    ): List<MemoryCandidate> {
        val lexicalOnly = {
            lexicalHits.mapIndexed { i, item -> MemoryCandidate(item, lexicalRank = i, semanticRank = null, semanticScore = null) }
        }
        if (index == null || embedder == null || query.text.isBlank()) return lexicalOnly()

        val queryVector = embedder.embedForQuery(query.text)
        if (queryVector.isEmpty()) return lexicalOnly()

        val semanticHits = index.search(queryVector, query.limit)
        return CandidateMerge.merge(query, lexicalHits, semanticHits, allItems.associateBy { it.id })
    }
}
