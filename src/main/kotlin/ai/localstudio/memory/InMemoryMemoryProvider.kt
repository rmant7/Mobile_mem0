package ai.localstudio.memory

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.ConcurrentHashMap

/**
 * A [MemoryProvider] with nothing behind it but a map — no file, no
 * serialization, nothing to clean up between test runs beyond letting the
 * instance go out of scope. Ranking is identical to [FileMemoryStore]'s
 * (both delegate to [MemoryRanking]), so a test written against this store
 * exercises the same retrieval behaviour a real, persistent one would;
 * only survivability across a process restart is different, which is
 * exactly the one thing a unit test never needs.
 */
class InMemoryMemoryProvider(
    private val extractor: MemoryExtractor = FileMemoryStore.PromoteWorkingMemory,
    // See FileMemoryStore's own doc comment on these two, including why they
    // sit before clock rather than after it — identical reasoning, kept in
    // sync deliberately.
    private val semanticIndex: MemorySemanticIndex? = null,
    private val embedder: MemoryEmbedder? = null,
    private val clock: () -> Long = System::currentTimeMillis,
) : MemoryProvider {

    private val lock = Any()
    private val items = LinkedHashMap<String, MemoryItem>()
    private var counter = 0L

    // See FileMemoryStore's own doc comment on this same field — identical
    // reasoning, kept in sync deliberately.
    private val consolidateLocks = ConcurrentHashMap<String, Mutex>()

    private fun snapshot(): List<MemoryItem> = synchronized(lock) { items.values.toList() }

    override suspend fun search(query: MemoryQuery): List<MemoryItem> = MemoryRanking.search(snapshot(), query)

    override suspend fun remember(text: String, scope: MemoryScope, metadata: Map<String, String>): String {
        require(text.isNotBlank()) { "cannot remember blank text" }
        return synchronized(lock) {
            val id = "mem-%010d".format(++counter)
            items[id] = MemoryItem(id, text.trim(), scope, clock(), metadata = metadata)
            id
        }
    }

    override suspend fun forget(id: String) {
        synchronized(lock) { items.remove(id) }
        // Outside the lock — see FileMemoryStore.forget's own comment.
        semanticIndex?.remove(id)
    }

    override suspend fun consolidate(conversationId: String): List<MemoryItem> =
        consolidateLocks.computeIfAbsent(conversationId) { Mutex() }.withLock {
            val working = snapshot().filter {
                it.scope == MemoryScope.WORKING && it.metadata[FileMemoryStore.CONVERSATION_KEY] == conversationId
            }
            if (working.isEmpty()) return@withLock emptyList()

            // Same reasoning as FileMemoryStore: extraction is a model call in
            // a real implementation, deliberately run outside [lock] (though
            // still inside this conversation's own consolidate mutex,
            // above). If it throws, working memory is left untouched.
            val extracted = extractor.extract(conversationId, working)

            val result = synchronized(lock) {
                working.forEach { items.remove(it.id) }
                extracted
                    // See FileMemoryStore's own doc comment on this filter —
                    // identical reasoning, kept in sync deliberately.
                    .filter { it.scope != MemoryScope.WORKING }
                    .map { item ->
                        val id = "mem-%010d".format(++counter)
                        val stored = item.copy(id = id, createdAt = clock())
                        items[id] = stored
                        stored
                    }
            }
            // Outside the lock — see FileMemoryStore.consolidate's own comment.
            semanticIndex?.let { index -> working.forEach { index.remove(it.id) } }
            result
        }

    /** See [FileMemoryStore.embedPending] — identical reasoning, kept in sync deliberately. */
    override suspend fun embedPending(limitPerCall: Int) {
        val index = semanticIndex ?: return
        val embed = embedder ?: return
        SemanticRetrieval.embedPending(snapshot(), index, embed, limitPerCall)
    }

    override suspend fun candidates(query: MemoryQuery): List<MemoryCandidate> {
        val lexicalHits = search(query)
        return SemanticRetrieval.candidates(query, lexicalHits, snapshot(), semanticIndex, embedder)
    }

    fun all(): List<MemoryItem> = snapshot()
}
