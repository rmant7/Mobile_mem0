package ai.localstudio.memory

/**
 * The model-backed seam for turning text into vectors — same reasoning as
 * [MemoryExtractor]: this module has no model runtime of its own and must not
 * acquire one, so embedding is an interface the embedding application
 * provides, not a dependency this module pulls in.
 *
 * Split into [embedForStorage] and [embedForQuery] rather than one `embed`
 * method: many real embedding models — e5, bge, gte, and others — are
 * *asymmetric* dual encoders, trained with a different instruction or prefix
 * for what gets indexed than for what searches it (e5's own convention is a
 * literal `"query: "` versus `"passage: "` prefix). Getting this wrong
 * produces vectors that still look like valid embeddings — same dimension,
 * same rough magnitude — while retrieval quality quietly degrades, which is
 * exactly the kind of failure that is easy to ship unnoticed with a single
 * generic `embed()`. A symmetric model implements both methods identically;
 * an asymmetric one cannot be made correct after the fact if this module
 * only ever asked for one kind of embedding.
 */
interface MemoryEmbedder {
    /**
     * Identifies the model *and* its revision — changing the model, its
     * weights, or its quantization must change this. It is what
     * [MemorySemanticIndex] compares against its own stored `modelId` to
     * decide whether an existing vector is still valid: a different model's
     * vector space is not the old one, even at the same [dimension].
     */
    val modelId: String

    /** The length of every [FloatArray] this embedder returns. */
    val dimension: Int

    /**
     * Embeds [texts] for storage — the "document"/"passage" side of the
     * model. Batched rather than one string at a time: a real embedder is
     * several times faster per item run as a batch than called once per
     * item, and consolidation embeds a batch of newly-written memories at
     * once, not a single one. Returns one embedding per input, in the same
     * order.
     */
    suspend fun embedForStorage(texts: List<String>): List<FloatArray>

    /**
     * Embeds a single search query — the other side of the same pair as
     * [embedForStorage]. Not batched: [MemoryProvider.candidates] embeds
     * exactly one query per call, and a model's query-side encoding is
     * frequently a genuinely different computation (a different prefix, a
     * different pooling target), not just the same function called on
     * shorter input.
     */
    suspend fun embedForQuery(query: String): FloatArray
}
