package ai.localstudio.memory

import kotlin.math.sqrt

/** One vector-search hit: a [MemoryItem.id] and its cosine similarity to the query, in -1..1. */
data class ScoredId(val id: String, val score: Double)

/**
 * Vector storage and search for [MemoryItem] embeddings, kept separate from
 * [MemoryProvider] itself: not every memory has an embedding yet (embedding
 * happens off the hot path — see [SEMANTIC_RETRIEVAL_DESIGN.md] at the repo
 * root), and not every deployment configures a [MemoryEmbedder] at all. A
 * [MemoryProvider] backend composes with an index rather than being one.
 *
 * [modelId] and [dimension] identify the embedding space this index actually
 * holds vectors for — a caller compares these against its current
 * [MemoryEmbedder]'s own `modelId`/`dimension` to decide whether the stored
 * vectors are still usable, or whether this id needs [missing] treatment
 * until it is re-embedded. This index does not make that decision itself: it
 * stores whatever it is given and reports what it holds.
 */
interface MemorySemanticIndex {
    val modelId: String
    val dimension: Int

    /** Replaces [id]'s vector if it already has one. [embedding] must have [dimension] elements. */
    suspend fun upsert(id: String, embedding: FloatArray)

    /** A no-op if [id] has no vector — matches [MemoryProvider.forget] not erroring on an unknown id either. */
    suspend fun remove(id: String)

    /** The [limit] closest vectors to [embedding] by cosine similarity, descending. */
    suspend fun search(embedding: FloatArray, limit: Int): List<ScoredId>

    /** Which of [ids] this index has no vector for — the re-embedding work queue. */
    suspend fun missing(ids: Collection<String>): List<String>
}

internal object Cosine {
    /**
     * Cosine similarity in -1..1. A zero vector yields 0 rather than NaN: an
     * embedder that returns an all-zero vector is a bug to surface elsewhere,
     * not a crash to trigger in the middle of a search.
     */
    fun similarity(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "Dimension mismatch: ${a.size} vs ${b.size}" }
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i].toDouble() * b[i]
            normA += a[i].toDouble() * a[i]
            normB += b[i].toDouble() * b[i]
        }
        if (normA == 0.0 || normB == 0.0) return 0.0
        return dot / (sqrt(normA) * sqrt(normB))
    }
}
