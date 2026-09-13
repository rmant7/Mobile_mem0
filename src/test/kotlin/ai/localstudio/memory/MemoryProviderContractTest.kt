package ai.localstudio.memory

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The behavioral contract every [MemoryProvider] implementation must satisfy
 * identically — run once per backend (see the two subclasses of this class)
 * so a future backend cannot drift from what [FileMemoryStore] and
 * [InMemoryMemoryProvider] already do without either implementation's own,
 * separate test file ever catching it. Backend-specific behavior
 * (persistence across process restarts, corrupt-file recovery, and the
 * like) stays in each implementation's own test file — this is only what
 * [MemoryProvider] itself promises, regardless of what's actually behind it.
 *
 * `consolidate()` gets the bulk of this file's attention on purpose: it is
 * the one place a [MemoryProvider] makes a decision on the extractor's
 * behalf (what to do with whatever it returns) rather than just storing or
 * returning whatever the caller handed it directly — which is exactly where
 * an implementation's own edge-case handling can silently diverge from
 * another's.
 */
abstract class MemoryProviderContractTest {

    /** A fresh, empty provider — one call, one test, no state shared between tests. */
    protected abstract fun provider(extractor: MemoryExtractor = FileMemoryStore.PromoteWorkingMemory): MemoryProvider

    @Test
    fun `remember then search finds it by shared terms`() = runBlocking {
        val memory = provider()
        memory.remember("the user prefers dark mode", MemoryScope.SEMANTIC)

        val hits = memory.search(MemoryQuery("what theme does the user prefer"))

        assertTrue(hits.any { it.text.contains("dark mode") })
    }

    @Test
    fun `forget removes an item so it no longer matches`() = runBlocking {
        val memory = provider()
        val id = memory.remember("the sky is blue", MemoryScope.SEMANTIC)

        memory.forget(id)

        assertTrue(memory.search(MemoryQuery("sky blue")).none { it.id == id })
    }

    @Test
    fun `remembering blank text is rejected`() = runBlocking {
        assertFailsWith<IllegalArgumentException> { provider().remember("   ", MemoryScope.SEMANTIC) }
    }

    @Test
    fun `a negative query limit is rejected`() {
        assertFailsWith<IllegalArgumentException> { MemoryQuery("x", limit = -1) }
    }

    @Test
    fun `matchAll finds items sharing no vocabulary with the query text`() = runBlocking {
        val memory = provider()
        memory.remember("user prefers dark mode", MemoryScope.SEMANTIC)
        memory.remember("decided to use Kotlin for the new module", MemoryScope.EPISODIC)

        // Deliberately shares not one real word with either stored item —
        // exactly a "what do you know about me" style meta-question, which a
        // plain lexical search (matchAll = false, the default) would find
        // nothing for, no matter how relevant everything stored obviously is.
        val hits = memory.search(MemoryQuery(text = "tell me everything", matchAll = true))

        assertEquals(2, hits.size)
    }

    @Test
    fun `a metadata-scoped search with blank text also avoids the NaN relevance collapse`() = runBlocking {
        // The exact shape a real caller already uses in production (see
        // AppContainer.rememberDocument / NodeExecutors' attached-document
        // retrieval): text is blank because a meta-question about a document
        // shares no vocabulary with its content, and metadataFilter alone
        // states relevance explicitly. This predates matchAll entirely and
        // was silently broken by the same NaN before that fix touched the
        // shared overlap() helper too.
        val memory = provider()
        memory.remember("first chunk of the file", MemoryScope.SEMANTIC, metadata = mapOf("source" to "report.pdf"))
        memory.remember("second chunk of the file", MemoryScope.SEMANTIC, metadata = mapOf("source" to "report.pdf"))

        val hits = memory.search(MemoryQuery(text = "", metadataFilter = mapOf("source" to "report.pdf")))

        assertEquals(2, hits.size)
        assertTrue(hits.all { it.relevance != null && !it.relevance!!.isNaN() })
    }

    @Test
    fun `matchAll with a blank query still ranks results, not just returns them in id order`() = runBlocking {
        val memory = provider()
        memory.remember("a semantic fact", MemoryScope.SEMANTIC)
        memory.remember("an episodic memory", MemoryScope.EPISODIC)

        val hits = memory.search(MemoryQuery(text = "", scopes = MemoryScope.entries.toSet(), matchAll = true))

        assertTrue(
            hits.all { it.relevance != null && !it.relevance!!.isNaN() },
            "a blank query's zero query-terms must not divide the overlap score into NaN " +
                "(NaN silently collapses every item to the same 'relevance', discarding scope/recency weighting)",
        )
    }

    @Test
    fun `consolidate promotes working memory out of WORKING and clears it`() = runBlocking {
        val memory = provider()
        memory.remember(
            "decided to use Kotlin",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        val promoted = memory.consolidate("c1")

        assertEquals(1, promoted.size)
        assertTrue(promoted.all { it.scope != MemoryScope.WORKING })
        assertTrue(
            memory.search(MemoryQuery("decided to use Kotlin", scopes = setOf(MemoryScope.EPISODIC))).isNotEmpty(),
            "the promoted memory must actually be findable afterward, not just removed from WORKING",
        )
    }

    @Test
    fun `consolidate with no working memory for that conversation is not an error`() = runBlocking {
        assertTrue(provider().consolidate("no-such-conversation").isEmpty())
    }

    @Test
    fun `calling consolidate twice in a row only promotes once`() = runBlocking {
        val memory = provider()
        memory.remember(
            "only said once",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        val first = memory.consolidate("c1")
        val second = memory.consolidate("c1")

        assertEquals(1, first.size)
        assertTrue(second.isEmpty(), "nothing is left in WORKING for this conversation the second time")
    }

    @Test
    fun `an extractor that throws leaves working memory untouched`() = runBlocking {
        val memory = provider(extractor = MemoryExtractor { _, _ -> throw IllegalStateException("boom") })
        memory.remember(
            "something said mid-conversation",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        assertFailsWith<IllegalStateException> { memory.consolidate("c1") }

        val stillWorking = memory.search(
            MemoryQuery(
                text = "",
                scopes = setOf(MemoryScope.WORKING),
                metadataFilter = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
            ),
        )
        assertEquals(1, stillWorking.size, "a failed extraction must not remove the working memory it failed to consolidate")
    }

    @Test
    fun `an extractor returning duplicate content stores both, without deduping`() = runBlocking {
        val memory = provider(
            extractor = MemoryExtractor { _, working ->
                val distilled = working.first().copy(scope = MemoryScope.EPISODIC)
                listOf(distilled, distilled)
            },
        )
        memory.remember(
            "said twice worth of content",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        val promoted = memory.consolidate("c1")

        assertEquals(
            2,
            promoted.size,
            "storage stores whatever the extractor decided to keep — deduping its output is not this layer's job",
        )
        assertEquals(2, promoted.map { it.id }.distinct().size, "each stored copy still gets its own identity")
    }

    @Test
    fun `an extractor that returns a WORKING-scoped item has it dropped, not stored`() = runBlocking {
        val memory = provider(
            // Simulates a buggy extractor that forgot to promote scope —
            // exactly the shape of bug this filter exists to catch.
            extractor = MemoryExtractor { _, working -> working.map { it.copy(scope = MemoryScope.WORKING) } },
        )
        memory.remember(
            "never actually distilled",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        val promoted = memory.consolidate("c1")

        assertTrue(promoted.isEmpty(), "a WORKING-scoped 'durable' memory is a contradiction consolidate() must not store")
        val stillWorking = memory.search(
            MemoryQuery(
                text = "",
                scopes = setOf(MemoryScope.WORKING),
                metadataFilter = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
            ),
        )
        assertTrue(stillWorking.isEmpty(), "the original working memory was still consumed, even though nothing durable came of it")
    }

    @Test
    fun `concurrent consolidate calls for the same conversation do not duplicate the result`() = runBlocking {
        // The default extractor never suspends, so under a single-threaded
        // test dispatcher every consolidate() call would otherwise just run
        // to completion before the next one starts — no actual interleaving,
        // and the race this test exists to catch would never manifest even
        // with the bug still present (verified directly: this test kept
        // passing after temporarily removing the per-conversation mutex,
        // using the default extractor). A real suspension point is what
        // forces the scheduler to actually interleave these five calls —
        // exactly what a real, slow, model-calling extractor would do too.
        val slowExtractor = MemoryExtractor { _, working ->
            delay(10)
            working.map { it.copy(scope = MemoryScope.EPISODIC) }
        }
        val memory = provider(extractor = slowExtractor)
        memory.remember(
            "one fact stated once",
            MemoryScope.WORKING,
            metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"),
        )

        val results = (1..5).map { async { memory.consolidate("c1") } }.awaitAll()

        assertEquals(
            1,
            results.sumOf { it.size },
            "the same working memory must be consolidated exactly once even when several callers race to do it",
        )
    }

    @Test
    fun `concurrent consolidate calls for DIFFERENT conversations both proceed`() = runBlocking {
        val slowExtractor = MemoryExtractor { _, working ->
            delay(10)
            working.map { it.copy(scope = MemoryScope.EPISODIC) }
        }
        val memory = provider(extractor = slowExtractor)
        memory.remember("c1's fact", MemoryScope.WORKING, metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c1"))
        memory.remember("c2's fact", MemoryScope.WORKING, metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to "c2"))

        val (a, b) = listOf(async { memory.consolidate("c1") }, async { memory.consolidate("c2") }).awaitAll()

        assertEquals(1, a.size)
        assertEquals(1, b.size)
    }
}
