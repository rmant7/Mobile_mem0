package ai.localstudio.memory

/**
 * The model-backed seam for turning text into vectors — same reasoning as
 * [MemoryExtractor]: this module has no model runtime of its own and must not
 * acquire one, so embedding is an interface the embedding application
 * provides, not a dependency this module pulls in.
 *
 * `suspend`, matching every other model-touching call in this module
 * ([MemoryExtractor.extract]): an on-device embedding call is slow and
 * blocking, exactly like extraction's model call.
 *
 * Batched rather than one string at a time: a real embedder is several times
 * faster per item run as a batch than called once per item, and consolidation
 * embeds a batch of newly-written memories at once, not a single one.
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

    /** Returns one embedding per input text, in the same order. */
    suspend fun embed(texts: List<String>): List<FloatArray>
}
