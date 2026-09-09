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
)

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
}
