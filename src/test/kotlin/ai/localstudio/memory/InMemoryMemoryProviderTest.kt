package ai.localstudio.memory

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Not a re-test of [MemoryRanking] (that's [FileMemoryStoreTest]'s job) —
 * this only checks that [InMemoryMemoryProvider] wires the shared ranking
 * and its own lifecycle correctly, and that it behaves like a
 * [MemoryProvider] a caller can swap in for tests without noticing.
 */
class InMemoryMemoryProviderTest {

    @Test
    fun `remember then search finds it by shared terms`() = runBlocking {
        val memory = InMemoryMemoryProvider()
        memory.remember("Пользователь строит локальный AI runtime", MemoryScope.SEMANTIC)

        val hits = memory.search(MemoryQuery("локальный runtime"))

        assertEquals(1, hits.size)
    }

    @Test
    fun `forget removes an item`() = runBlocking {
        val memory = InMemoryMemoryProvider()
        val id = memory.remember("временный факт", MemoryScope.SEMANTIC)

        memory.forget(id)

        assertTrue(memory.all().isEmpty())
    }

    @Test
    fun `consolidate promotes working memory for one conversation and leaves others alone`() = runBlocking {
        val memory = InMemoryMemoryProvider()
        memory.remember("обсуждали capability router", MemoryScope.WORKING, mapOf("conversationId" to "c1"))
        memory.remember("чужой разговор", MemoryScope.WORKING, mapOf("conversationId" to "c2"))

        val promoted = memory.consolidate("c1")

        assertEquals(1, promoted.size)
        assertEquals(MemoryScope.EPISODIC, promoted.single().scope)
        assertEquals(1, memory.all().count { it.scope == MemoryScope.WORKING })
    }

    @Test
    fun `blank text is rejected`() = runBlocking {
        val memory = InMemoryMemoryProvider()
        assertFailsWith<IllegalArgumentException> { memory.remember("  ", MemoryScope.SEMANTIC) }
    }

    @Test
    fun `nothing survives a new instance, unlike FileMemoryStore`() = runBlocking {
        InMemoryMemoryProvider().remember("не переживёт новый экземпляр", MemoryScope.SEMANTIC)

        assertTrue(InMemoryMemoryProvider().all().isEmpty())
    }
}
