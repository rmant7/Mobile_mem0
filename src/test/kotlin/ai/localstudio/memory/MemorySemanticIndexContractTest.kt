package ai.localstudio.memory

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The behavioral contract every [MemorySemanticIndex] must satisfy
 * identically — run once per backend (see the two subclasses of this class),
 * matching [MemoryProviderContractTest]'s own reasoning for why a shared
 * abstract suite exists: so [FileSemanticIndex] and [InMemorySemanticIndex]
 * cannot drift from each other without a shared test catching it.
 */
abstract class MemorySemanticIndexContractTest {

    protected abstract fun index(modelId: String = "test-model", dimension: Int = 3): MemorySemanticIndex

    @Test
    fun `upsert then search finds it, ranked near the top by similarity`() = runBlocking {
        val idx = index()
        idx.upsert("a", floatArrayOf(1f, 0f, 0f))
        idx.upsert("b", floatArrayOf(0f, 1f, 0f))

        val hits = idx.search(floatArrayOf(1f, 0f, 0f), limit = 8)

        assertEquals("a", hits.first().id)
        assertTrue(hits.first().score > 0.99, "an identical vector should score ~1.0, got ${hits.first().score}")
    }

    @Test
    fun `search is ranked by similarity, most similar first`() = runBlocking {
        val idx = index()
        idx.upsert("close", floatArrayOf(1f, 0.1f, 0f))
        idx.upsert("far", floatArrayOf(0f, 1f, 0f))

        val hits = idx.search(floatArrayOf(1f, 0f, 0f), limit = 8)

        assertEquals(listOf("close", "far"), hits.map { it.id })
    }

    @Test
    fun `remove makes an entry disappear from search`() = runBlocking {
        val idx = index()
        idx.upsert("a", floatArrayOf(1f, 0f, 0f))

        idx.remove("a")

        assertTrue(idx.search(floatArrayOf(1f, 0f, 0f), limit = 8).isEmpty())
    }

    @Test
    fun `removing an id that was never upserted is not an error`() = runBlocking {
        index().remove("never-existed")
    }

    @Test
    fun `upserting the same id twice replaces the vector, not adds a second entry`() = runBlocking {
        val idx = index()
        idx.upsert("a", floatArrayOf(1f, 0f, 0f))

        idx.upsert("a", floatArrayOf(0f, 1f, 0f))

        val hits = idx.search(floatArrayOf(0f, 1f, 0f), limit = 8)
        assertEquals(1, hits.size)
        assertTrue(hits.first().score > 0.99)
    }

    @Test
    fun `missing reports ids with no vector and excludes ids that have one`() = runBlocking {
        val idx = index()
        idx.upsert("has-vector", floatArrayOf(1f, 0f, 0f))

        val result = idx.missing(listOf("has-vector", "no-vector"))

        assertEquals(listOf("no-vector"), result)
    }

    @Test
    fun `a vector of the wrong dimension is rejected on upsert`() = runBlocking {
        val idx = index(dimension = 3)
        assertFailsWith<IllegalArgumentException> { idx.upsert("a", floatArrayOf(1f, 0f)) }
    }

    @Test
    fun `a query vector of the wrong dimension is rejected on search`() = runBlocking {
        val idx = index(dimension = 3)
        assertFailsWith<IllegalArgumentException> { idx.search(floatArrayOf(1f, 0f), limit = 8) }
    }

    @Test
    fun `a limit of zero returns nothing, even with matching entries`() = runBlocking {
        val idx = index()
        idx.upsert("a", floatArrayOf(1f, 0f, 0f))

        assertTrue(idx.search(floatArrayOf(1f, 0f, 0f), limit = 0).isEmpty())
    }

    @Test
    fun `search on an empty index returns nothing`() = runBlocking {
        assertTrue(index().search(floatArrayOf(1f, 0f, 0f), limit = 8).isEmpty())
    }
}
