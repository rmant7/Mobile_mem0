package ai.localstudio.memory

/**
 * A [MemorySemanticIndex] backed by nothing but a map — the semantic-index
 * counterpart to [InMemoryMemoryProvider]: for tests that want real vector
 * search behavior without touching disk, and proven against the same
 * [MemorySemanticIndexContractTest] suite [FileSemanticIndex] is.
 */
class InMemorySemanticIndex(
    override val modelId: String,
    override val dimension: Int,
) : MemorySemanticIndex {

    private val lock = Any()
    private val vectors = LinkedHashMap<String, FloatArray>()

    override suspend fun upsert(id: String, embedding: FloatArray) {
        require(embedding.size == dimension) { "expected a $dimension-dimensional vector, got ${embedding.size}" }
        synchronized(lock) { vectors[id] = embedding }
    }

    override suspend fun remove(id: String) {
        synchronized(lock) { vectors.remove(id) }
    }

    override suspend fun search(embedding: FloatArray, limit: Int): List<ScoredId> {
        require(embedding.size == dimension) { "expected a $dimension-dimensional vector, got ${embedding.size}" }
        if (limit == 0) return emptyList()
        val snapshot = synchronized(lock) { vectors.entries.map { it.key to it.value } }
        return snapshot
            .map { (id, vector) -> ScoredId(id, Cosine.similarity(embedding, vector)) }
            .sortedWith(compareByDescending<ScoredId> { it.score }.thenBy { it.id })
            .take(limit)
    }

    override suspend fun missing(ids: Collection<String>): List<String> =
        synchronized(lock) { ids.filter { it !in vectors } }
}
