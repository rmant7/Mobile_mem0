package ai.localstudio.memory

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
    private val clock: () -> Long = System::currentTimeMillis,
) : MemoryProvider {

    private val lock = Any()
    private val items = LinkedHashMap<String, MemoryItem>()
    private var counter = 0L

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
    }

    override suspend fun consolidate(conversationId: String): List<MemoryItem> {
        val working = snapshot().filter {
            it.scope == MemoryScope.WORKING && it.metadata[FileMemoryStore.CONVERSATION_KEY] == conversationId
        }
        if (working.isEmpty()) return emptyList()

        // Same reasoning as FileMemoryStore: extraction is a model call in a
        // real implementation, deliberately run outside the lock.
        val extracted = extractor.extract(conversationId, working)

        return synchronized(lock) {
            working.forEach { items.remove(it.id) }
            extracted.map { item ->
                val id = "mem-%010d".format(++counter)
                val stored = item.copy(id = id, createdAt = clock())
                items[id] = stored
                stored
            }
        }
    }

    fun all(): List<MemoryItem> = snapshot()
}
