package ai.localstudio.memory

import kotlinx.serialization.Serializable

/**
 * A standalone local-memory engine — deliberately its own Gradle module with
 * zero dependency on this repository's `core`, `openai`, or `app` modules
 * (only Kotlin's stdlib, coroutines, and kotlinx-serialization). Everything
 * in this module is meant to be lift-out-able into its own repository
 * without untangling anything: `core` depends on this module, never the
 * other way around.
 *
 * What's *not* here on purpose: anything that decides what a conversation
 * is worth remembering. That decision needs a language model, and which
 * model (or whether there even is a local one loaded right now) is entirely
 * an embedding application's concern — see [MemoryExtractor], the seam this
 * module hands that decision through, and `ai.localstudio.core.memory` for
 * this app's own model-backed implementation of it.
 */

/**
 * Memory is split by lifetime and meaning, not by storage engine.
 *
 * - [WORKING]  — the current conversation, dropped when it ends.
 * - [EPISODIC] — what happened before: decisions, events, sessions.
 * - [SEMANTIC] — stable facts: preferences, project properties, entities.
 */
@Serializable
enum class MemoryScope { WORKING, EPISODIC, SEMANTIC }

data class MemoryItem(
    val id: String,
    val text: String,
    val scope: MemoryScope,
    val createdAt: Long,
    val relevance: Double? = null,
    val metadata: Map<String, String> = emptyMap(),
)

data class MemoryQuery(
    val text: String,
    val scopes: Set<MemoryScope> = setOf(MemoryScope.EPISODIC, MemoryScope.SEMANTIC),
    val limit: Int = 8,
    val metadataFilter: Map<String, String> = emptyMap(),
    /**
     * Bypasses the requirement that [text] share a term with a stored item —
     * the same bypass a non-empty [metadataFilter] already grants implicitly
     * (see [MemoryRanking]'s own comment), made available on its own for a
     * caller with no metadata to scope by, just a real reason to want
     * everything in [scopes] back: a question about memory itself ("what do
     * you know about me?") shares no vocabulary with what's actually stored,
     * by definition, no matter how relevant every item obviously is.
     * [text] is still used for ranking when it isn't blank — this only
     * removes it as a *requirement* for a result to appear at all.
     */
    val matchAll: Boolean = false,
) {
    init {
        require(limit >= 0) { "limit must not be negative: $limit" }
    }
}

/**
 * Decides what from a finished exchange is worth keeping.
 *
 * Extraction is a model call in any serious implementation, which is why it
 * is an interface: on a phone that call is background work, not something
 * that runs synchronously after every answer, and it is squarely an
 * embedding application's concern — this module has no model runtime to
 * call one with.
 */
fun interface MemoryExtractor {
    suspend fun extract(conversationId: String, workingMemory: List<MemoryItem>): List<MemoryItem>
}

/**
 * One [search] hit reported alongside where each retrieval method that found
 * it ranked it — not a final relevance decision. See
 * [SEMANTIC_RETRIEVAL_DESIGN.md] (repo root): this module reports signals and
 * positions; which of them matters more is deliberately left to the
 * embedding application's own ranking layer, never decided here.
 *
 * [lexicalRank] and [semanticRank] are independently nullable — a candidate
 * found by only one retrieval method has `null` for the other, which is
 * itself information (this method didn't find it at all), not a zero to
 * average away. [semanticScore] is the raw cosine similarity backing
 * [semanticRank]; kept alongside the rank because a caller doing
 * score-based fusion needs the score, and one doing rank-based fusion
 * (e.g. Reciprocal Rank Fusion) needs the rank, and this module does not
 * get to decide which one the caller wants.
 */
data class MemoryCandidate(
    val item: MemoryItem,
    val lexicalRank: Int?,
    val semanticRank: Int?,
    val semanticScore: Float?,
)

/**
 * Long-term memory, deliberately kept behind an interface.
 *
 * The application must be able to move between backends — a local store,
 * Mem0, anything else — without touching whatever assembles a model's
 * context out of what this returns.
 */
interface MemoryProvider {
    suspend fun search(query: MemoryQuery): List<MemoryItem>

    /** Returns the id of the stored memory. */
    suspend fun remember(text: String, scope: MemoryScope, metadata: Map<String, String> = emptyMap()): String

    suspend fun forget(id: String)

    /** Distils a finished exchange into durable memories. Returns what was written. */
    suspend fun consolidate(conversationId: String): List<MemoryItem>

    /**
     * [search]'s results as [MemoryCandidate]s, ready for a caller that wants
     * to fuse them with a second retrieval method's own candidates (see
     * [MemorySemanticIndex]) rather than trust [search]'s ranking alone.
     *
     * Defaulted rather than declared abstract: every existing [MemoryProvider]
     * — this module's own two backends, and any other implementation already
     * written against this interface — keeps compiling and behaves exactly as
     * it does today, with [search]'s own order reported as [lexicalRank] and
     * no semantic signal at all. A backend that actually maintains a semantic
     * index overrides this to merge both result sets instead.
     */
    suspend fun candidates(query: MemoryQuery): List<MemoryCandidate> =
        search(query).mapIndexed { index, item ->
            MemoryCandidate(item, lexicalRank = index, semanticRank = null, semanticScore = null)
        }

    /**
     * Embeds up to [limitPerCall] stored items a backend's own semantic index
     * doesn't have a vector for yet — see [FileMemoryStore.embedPending]'s
     * own doc comment for why this exists as an explicit, caller-invoked
     * operation, never something [remember] triggers itself.
     *
     * A no-op default, same reasoning as [candidates]'s own default: a
     * backend with no semantic index configured has nothing to catch up.
     */
    suspend fun embedPending(limitPerCall: Int = 64) {}
}
