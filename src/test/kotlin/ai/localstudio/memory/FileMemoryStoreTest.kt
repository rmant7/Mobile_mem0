package ai.localstudio.memory

import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FileMemoryStoreTest {

    private var now = 1_000L
    private fun provider(extractor: MemoryExtractor = FileMemoryStore.PromoteWorkingMemory): FileMemoryStore {
        val file = File.createTempFile("memory-test", ".json").apply { deleteOnExit() }
        return FileMemoryStore(file, extractor) { now += 10; now }
    }

    @Test
    fun `search finds a memory by shared terms`() = runBlocking {
        val memory = provider()
        memory.remember("Пользователь строит локальный AI runtime на Kotlin", MemoryScope.SEMANTIC)
        memory.remember("Обсуждали рецепт борща", MemoryScope.EPISODIC)

        val hits = memory.search(MemoryQuery("что мы решили про локальный runtime"))

        assertEquals(1, hits.size)
        assertTrue(hits.single().text.contains("runtime"))
    }

    @Test
    fun `scopes are respected`() = runBlocking {
        val memory = provider()
        memory.remember("Проект использует Kotlin", MemoryScope.SEMANTIC)
        memory.remember("Вчера чинили пайплайн", MemoryScope.EPISODIC)

        val semantic = memory.search(MemoryQuery("Kotlin пайплайн", scopes = setOf(MemoryScope.SEMANTIC)))
        val episodic = memory.search(MemoryQuery("Kotlin пайплайн", scopes = setOf(MemoryScope.EPISODIC)))

        assertEquals(listOf("Проект использует Kotlin"), semantic.map { it.text })
        assertEquals(listOf("Вчера чинили пайплайн"), episodic.map { it.text })
    }

    @Test
    fun `a stable fact outranks an episode at equal overlap`() = runBlocking {
        val memory = provider()
        memory.remember("Проект использует Qdrant", MemoryScope.SEMANTIC)
        memory.remember("Проект использует Qdrant", MemoryScope.EPISODIC)

        val hits = memory.search(MemoryQuery("проект использует qdrant"))

        assertEquals(MemoryScope.SEMANTIC, hits.first().scope)
    }

    @Test
    fun `between episodes the more recent one wins`() = runBlocking {
        val memory = provider()
        memory.remember("Решили использовать llama.cpp", MemoryScope.EPISODIC)
        memory.remember("Решили использовать llama.cpp и MediaPipe", MemoryScope.EPISODIC)

        val hits = memory.search(MemoryQuery("что решили использовать llama"))

        assertEquals("Решили использовать llama.cpp и MediaPipe", hits.first().text)
    }

    @Test
    fun `metadata filters narrow the search`() = runBlocking {
        val memory = provider()
        memory.remember("Решили делать registry", MemoryScope.EPISODIC, mapOf("conversationId" to "c1"))
        memory.remember("Решили делать registry", MemoryScope.EPISODIC, mapOf("conversationId" to "c2"))

        val hits = memory.search(
            MemoryQuery("решили делать registry", metadataFilter = mapOf("conversationId" to "c2")),
        )

        assertEquals(1, hits.size)
        assertEquals("c2", hits.single().metadata["conversationId"])
    }

    @Test
    fun `a metadata filter bypasses the lexical overlap requirement`() = runBlocking {
        val memory = provider()
        memory.remember(
            "Содержимое приложенного файла, никак не связанное с вопросом",
            MemoryScope.SEMANTIC,
            mapOf("source" to "report.pdf"),
        )

        val hits = memory.search(MemoryQuery("что это", metadataFilter = mapOf("source" to "report.pdf")))

        assertEquals(1, hits.size)
    }

    @Test
    fun `unrelated memories are not returned at all`() = runBlocking {
        val memory = provider()
        memory.remember("Пользователь предпочитает тёмную тему", MemoryScope.SEMANTIC)

        assertTrue(memory.search(MemoryQuery("квантование моделей")).isEmpty())
    }

    @Test
    fun `forget removes a memory`() = runBlocking {
        val memory = provider()
        val id = memory.remember("Временный факт про runtime", MemoryScope.SEMANTIC)

        memory.forget(id)

        assertTrue(memory.search(MemoryQuery("runtime")).isEmpty())
        assertTrue(memory.all().isEmpty())
    }

    @Test
    fun `consolidate promotes working memory and clears it`() = runBlocking {
        val memory = provider()
        memory.remember("Обсуждали capability router", MemoryScope.WORKING, mapOf("conversationId" to "c1"))
        memory.remember("Чужой разговор", MemoryScope.WORKING, mapOf("conversationId" to "c2"))

        val promoted = memory.consolidate("c1")

        assertEquals(1, promoted.size)
        assertEquals(MemoryScope.EPISODIC, promoted.single().scope)
        assertEquals(
            listOf(MemoryScope.WORKING),
            memory.all().filter { it.metadata["conversationId"] == "c2" }.map { it.scope },
        )
        assertTrue(memory.all().none { it.scope == MemoryScope.WORKING && it.metadata["conversationId"] == "c1" })
    }

    @Test
    fun `consolidation is pluggable because extraction is a model call`() = runBlocking {
        val memory = provider { _, working ->
            listOf(MemoryItem("", "Итог: " + working.size + " реплик", MemoryScope.SEMANTIC, 0))
        }
        memory.remember("реплика один", MemoryScope.WORKING, mapOf("conversationId" to "c1"))
        memory.remember("реплика два", MemoryScope.WORKING, mapOf("conversationId" to "c1"))

        val promoted = memory.consolidate("c1")

        assertEquals(listOf("Итог: 2 реплик"), promoted.map { it.text })
        assertEquals(MemoryScope.SEMANTIC, promoted.single().scope)
    }

    @Test
    fun `nothing to consolidate is not an error`() = runBlocking {
        assertTrue(provider().consolidate("unknown").isEmpty())
    }

    @Test
    fun `memory survives a fresh instance over the same file`() = runBlocking {
        val file = File.createTempFile("memory-test", ".json").apply { deleteOnExit() }
        FileMemoryStore(file).remember("Факт, который должен пережить перезапуск", MemoryScope.SEMANTIC)

        // A new instance over the same file is what a process restart looks
        // like — the whole point of this store over the in-memory one it
        // replaced.
        val reloaded = FileMemoryStore(file)

        assertEquals(1, reloaded.all().size)
        assertTrue(reloaded.all().single().text.contains("перезапуск"))
    }

    @Test
    fun `a missing or corrupt file starts empty instead of crashing`() {
        val missing = File.createTempFile("memory-test", ".json").apply { delete() }
        assertTrue(FileMemoryStore(missing).all().isEmpty())

        val corrupt = File.createTempFile("memory-test", ".json").apply {
            deleteOnExit()
            writeText("{ not valid json")
        }
        assertTrue(FileMemoryStore(corrupt).all().isEmpty())
    }

    /**
     * Reading while writing is the app's normal case, not an edge case: a
     * generation searches memory on a background thread for the length of a
     * turn, while attaching a document writes to it from the UI thread. That
     * pair used to throw ConcurrentModificationException and take the whole
     * app down with it, back when this was a plain in-memory map.
     */
    @Test
    fun `searching while remembering from another thread does not blow up`() {
        val memory = provider()
        runBlocking { repeat(200) { memory.remember("исходный фрагмент $it", MemoryScope.SEMANTIC) } }

        val failure = java.util.concurrent.atomic.AtomicReference<Throwable>()
        val writer = Thread {
            runCatching {
                runBlocking { repeat(200) { memory.remember("новый фрагмент $it", MemoryScope.SEMANTIC) } }
            }.onFailure(failure::set)
        }
        val reader = Thread {
            runCatching {
                runBlocking { repeat(200) { memory.search(MemoryQuery("фрагмент")) } }
            }.onFailure(failure::set)
        }

        writer.start()
        reader.start()
        writer.join()
        reader.join()

        assertEquals(null, failure.get(), "concurrent read/write threw: ${failure.get()}")
    }
}
