package ai.localstudio.memory

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.ConcurrentHashMap

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
    /**
     * Both null by default: every existing caller keeps working exactly as
     * before, lexical-only. Passing both switches [candidates] to also
     * retrieve semantically — see [embedPending] for how vectors actually
     * get into [semanticIndex] in the first place, deliberately not from
     * inside [remember] itself.
     */
    private val semanticIndex: MemorySemanticIndex? = null,
    private val embedder: MemoryEmbedder? = null,
    // Kept last: existing callers pass this positionally as a trailing
    // lambda (`FileMemoryStore(file, extractor) { ... }`), which binds to
    // whichever parameter is actually last — inserting semanticIndex/embedder
    // after this one instead of before it would silently break every one of
    // those call sites with a confusing type mismatch at the call site, not
    // a hint anywhere near this constructor.
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

    // One Mutex per conversation, not a single global one: consolidate()
    // itself already runs its (possibly slow, model-calling) extractor
    // outside [lock] specifically so it doesn't block search/remember/forget
    // for unrelated conversations — a single shared consolidate-wide lock
    // would defeat that. But two concurrent consolidate() calls for the
    // *same* conversation both read the same working-memory snapshot before
    // either has removed anything, both extract from it, and both then
    // write their own copy of the result — the same working memory
    // consolidated twice into duplicated durable memories. This is what
    // actually closes that race, without serializing unrelated conversations
    // against each other. Grows by one entry per conversationId ever
    // consolidated and is never pruned — acceptable for a personal local
    // store; a Mutex is a handful of bytes, and this matches how the rest of
    // this app's own long-lived caches are already treated.
    private val consolidateLocks = ConcurrentHashMap<String, Mutex>()

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

    override suspend fun search(query: MemoryQuery): List<MemoryItem> = MemoryRanking.search(snapshot(), query)

    override suspend fun remember(text: String, scope: MemoryScope, metadata: Map<String, String>): String {
        require(text.isNotBlank()) { "cannot remember blank text" }
        return synchronized(lock) {
            val id = nextId()
            items[id] = MemoryItem(id, text.trim(), scope, clock(), metadata = metadata)
            persist()
            id
        }
    }

    override suspend fun forget(id: String) {
        synchronized(lock) {
            items.remove(id)
            persist()
        }
        // Outside the lock: a suspend call must never run while holding a
        // plain synchronized monitor — if it ever genuinely suspended, the
        // thread would sit blocked on the lock across that suspension point.
        semanticIndex?.remove(id)
    }

    /**
     * Turns this conversation's working memory into durable memories and
     * clears the working set — working memory that outlives its
     * conversation is just a leak with a nicer name.
     */
    override suspend fun consolidate(conversationId: String): List<MemoryItem> =
        consolidateLocks.computeIfAbsent(conversationId) { Mutex() }.withLock {
            val working = snapshot().filter {
                it.scope == MemoryScope.WORKING && it.metadata[CONVERSATION_KEY] == conversationId
            }
            if (working.isEmpty()) return@withLock emptyList()

            // Extraction is a model call in a real implementation — deliberately
            // outside [lock] (though still inside this conversation's own
            // consolidate mutex, above), so a slow extractor cannot block
            // every other read and write for its whole duration. If it
            // throws, this whole call throws too and nothing below runs —
            // working memory is left exactly as it was, not half-removed, so
            // a failed consolidation can simply be retried later rather than
            // silently losing the conversation it was trying to distil.
            val extracted = extractor.extract(conversationId, working)

            val result = synchronized(lock) {
                working.forEach { items.remove(it.id) }
                val promoted = extracted
                    // A durable memory that's still WORKING-scoped is a
                    // contradiction consolidate() itself would act on: it
                    // would satisfy this exact filter again on the *next*
                    // call for this conversation, silently re-entering
                    // extraction a second time while also being invisible to
                    // a normal search() (whose default scopes exclude
                    // WORKING). Dropped rather than coerced to EPISODIC —
                    // promoting content the extractor explicitly marked
                    // provisional is a bigger, more surprising liberty than
                    // just not storing it.
                    .filter { it.scope != MemoryScope.WORKING }
                    .map { item ->
                        val stored = item.copy(id = nextId(), createdAt = clock())
                        items[stored.id] = stored
                        stored
                    }
                persist()
                promoted
            }
            // Outside the lock, same reasoning as forget()'s own comment.
            // The WORKING items just consolidated may already have been
            // embedded by an earlier embedPending() call; their ids are gone
            // from items now, and a vector with no matching record left
            // behind is exactly the kind of orphan CandidateMerge otherwise
            // has to filter out on every single search from here on.
            semanticIndex?.let { index -> working.forEach { index.remove(it.id) } }
            result
        }

    /**
     * Embeds every stored item [semanticIndex] doesn't have a vector for yet
     * — the catch-up half of [candidates] being able to retrieve
     * semantically at all. Deliberately not called from [remember] or
     * [consolidate] themselves: embedding is a model call, and [remember]
     * runs on every turn of every conversation — see
     * `SEMANTIC_RETRIEVAL_DESIGN.md`'s own reasoning for keeping that off
     * the hot path. The caller decides when catching up is convenient (after
     * a consolidation, on a periodic background job, ...); this only does
     * the work once actually asked.
     *
     * A no-op, not an error, when [semanticIndex] or [embedder] was never
     * configured — every existing caller of this class is unaffected by
     * this method existing at all.
     */
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

    companion object {
        const val CONVERSATION_KEY = "conversationId"

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
