package ai.localstudio.memory

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
private data class StoredItem(
    val id: String,
    val text: String,
    val scope: MemoryScope,
    val createdAt: Long,
    val metadata: Map<String, String> = emptyMap(),
)

@Serializable
private data class StoredState(val items: List<StoredItem> = emptyList(), val counter: Long = 0L)

/**
 * A persistent [MemoryProvider] — the one thing an in-memory map fundamentally
 * cannot be: every memory here survives the process dying, not just the
 * conversation that produced it. The whole file is read once at construction
 * and rewritten whole on every mutation; for a personal local memory store
 * (hundreds to low thousands of items, not a shared multi-user backend) that
 * is simple and correct, and it is the same pattern this app already uses
 * for chat history and attached documents — proven, not novel.
 *
 * Retrieval is lexical: shared-term overlap, weighted by scope and recency.
 * It is not as good as embeddings, and it is not meant to be — swapping in a
 * real embedding-backed implementation later is exactly the kind of change
 * this module's own [MemoryProvider] interface exists to make painless.
 */
class FileMemoryStore(
    private val file: File,
    private val extractor: MemoryExtractor = PromoteWorkingMemory,
    private val clock: () -> Long = System::currentTimeMillis,
) : MemoryProvider {

    private val json = Json { ignoreUnknownKeys = true }

    // Same reasoning as the in-memory version this replaced: a generation
    // reads this from a background thread for the whole length of a turn
    // while attaching a document writes to it from the UI thread, and
    // without a lock that pair is a ConcurrentModificationException waiting
    // to happen. Every read and write — including the file I/O itself —
    // goes through this.
    private val lock = Any()
    private val items = LinkedHashMap<String, MemoryItem>()
    private var counter = 0L

    init {
        synchronized(lock) { load() }
    }

    private fun load() {
        if (!file.isFile) return
        runCatching {
            val state = json.decodeFromString(StoredState.serializer(), file.readText())
            state.items.forEach { stored ->
                items[stored.id] = MemoryItem(stored.id, stored.text, stored.scope, stored.createdAt, metadata = stored.metadata)
            }
            counter = state.counter
        }
        // A corrupt or unreadable file degrades to "starts empty," same as a
        // fresh install — not a crash on every future launch over one bad write.
    }

    /** Caller must already hold [lock]. */
    private fun persist() {
        val state = StoredState(
            items = items.values.map { StoredItem(it.id, it.text, it.scope, it.createdAt, it.metadata) },
            counter = counter,
        )
        runCatching { file.writeText(json.encodeToString(StoredState.serializer(), state)) }
    }

    /** Caller must already hold [lock]. */
    private fun nextId(): String = "mem-%010d".format(++counter)

    private fun snapshot(): List<MemoryItem> = synchronized(lock) { items.values.toList() }

    override suspend fun search(query: MemoryQuery): List<MemoryItem> {
        val terms = tokenize(query.text)
        // A caller that scopes by metadata is stating relevance explicitly
        // — "exactly this attached document's chunks" — and needs no
        // lexical overlap on top of that. Requiring one anyway is what made
        // a plain "what does this file say" find nothing: a meta-question
        // about a just-attached document shares no vocabulary with that
        // document's actual content, by definition, no matter how relevant
        // it obviously is. Without a metadata filter this is an ordinary
        // free-text query, unchanged: an empty or entirely stop-word query
        // still returns nothing, and every result still needs a shared term.
        val requireOverlap = query.metadataFilter.isEmpty()
        if (requireOverlap && terms.isEmpty()) return emptyList()

        val all = snapshot()
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

    override suspend fun remember(text: String, scope: MemoryScope, metadata: Map<String, String>): String =
        synchronized(lock) {
            val id = nextId()
            items[id] = MemoryItem(id, text.trim(), scope, clock(), metadata = metadata)
            persist()
            id
        }

    override suspend fun forget(id: String) {
        synchronized(lock) {
            items.remove(id)
            persist()
        }
    }

    /**
     * Turns this conversation's working memory into durable memories and
     * clears the working set — working memory that outlives its
     * conversation is just a leak with a nicer name.
     */
    override suspend fun consolidate(conversationId: String): List<MemoryItem> {
        val working = snapshot().filter {
            it.scope == MemoryScope.WORKING && it.metadata[CONVERSATION_KEY] == conversationId
        }
        if (working.isEmpty()) return emptyList()

        // Extraction is a model call in a real implementation — deliberately
        // outside the lock, so a slow extractor cannot block every other
        // read and write for its whole duration.
        val extracted = extractor.extract(conversationId, working)

        return synchronized(lock) {
            working.forEach { items.remove(it.id) }
            val result = extracted.map { item ->
                val stored = item.copy(id = nextId(), createdAt = clock())
                items[stored.id] = stored
                stored
            }
            persist()
            result
        }
    }

    fun all(): List<MemoryItem> = snapshot()

    private fun tokenize(text: String): Set<String> =
        text.lowercase()
            .split(NON_WORD)
            .filter { it.length >= MIN_TERM_LENGTH }
            .toSet()

    /** Jaccard-style overlap, normalised by the query so long memories are not favoured. */
    private fun overlap(queryTerms: Set<String>, itemTerms: Set<String>): Double {
        if (itemTerms.isEmpty()) return 0.0
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

    companion object {
        const val CONVERSATION_KEY = "conversationId"
        private const val RECENCY_WEIGHT = 0.25
        private const val MIN_TERM_LENGTH = 3
        private val NON_WORD = Regex("[^\\p{L}\\p{N}]+")

        /**
         * Default extraction: keep working-memory entries verbatim as episodic
         * memories. Crude but honest — it never invents a fact the user did not
         * say, which a weak extraction model absolutely will.
         */
        val PromoteWorkingMemory = MemoryExtractor { _, working ->
            working.map { it.copy(scope = MemoryScope.EPISODIC) }
        }
    }
}
