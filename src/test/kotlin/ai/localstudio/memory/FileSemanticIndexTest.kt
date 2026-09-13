package ai.localstudio.memory

import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [FileSemanticIndex]'s own behavior beyond the shared contract
 * ([FileSemanticIndexContractTest]) — persistence across a process restart,
 * corrupt-file recovery, embedding-model changes, and compaction all only
 * make sense for a backend that actually has a file.
 */
class FileSemanticIndexTest {

    private fun tempFile(): File = File.createTempFile("semantic-index-test", ".bin").apply { deleteOnExit() }

    @Test
    fun `vectors survive reopening the same file`() = runBlocking {
        val file = tempFile()
        FileSemanticIndex(file, "model-a", 3).upsert("x", floatArrayOf(1f, 0f, 0f))

        val reopened = FileSemanticIndex(file, "model-a", 3)

        val hits = reopened.search(floatArrayOf(1f, 0f, 0f), limit = 8)
        assertEquals(listOf("x"), hits.map { it.id })
    }

    @Test
    fun `a removal survives reopening the same file too`() = runBlocking {
        val file = tempFile()
        val original = FileSemanticIndex(file, "model-a", 3)
        original.upsert("x", floatArrayOf(1f, 0f, 0f))
        original.remove("x")

        val reopened = FileSemanticIndex(file, "model-a", 3)

        assertTrue(reopened.search(floatArrayOf(1f, 0f, 0f), limit = 8).isEmpty())
    }

    @Test
    fun `a corrupt file degrades to starting empty, not a crash`() = runBlocking {
        val file = tempFile()
        file.writeBytes(byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8))

        val idx = FileSemanticIndex(file, "model-a", 3)

        assertTrue(idx.search(floatArrayOf(1f, 0f, 0f), limit = 8).isEmpty())
        // And it is actually usable afterward, not left in some half-broken state.
        idx.upsert("x", floatArrayOf(1f, 0f, 0f))
        assertEquals(1, idx.search(floatArrayOf(1f, 0f, 0f), limit = 8).size)
    }

    @Test
    fun `reopening with a different modelId discards the old vectors rather than mixing them in`() = runBlocking {
        val file = tempFile()
        FileSemanticIndex(file, "model-a", 3).upsert("x", floatArrayOf(1f, 0f, 0f))

        val reopened = FileSemanticIndex(file, "model-b", 3)

        assertTrue(
            reopened.search(floatArrayOf(1f, 0f, 0f), limit = 8).isEmpty(),
            "a different embedding space's vector must never be reported as this model's own",
        )
        assertEquals(listOf("x"), reopened.missing(listOf("x")), "the old vector must count as missing, ready to be re-embedded")
    }

    @Test
    fun `reopening with a different dimension also discards the old vectors`() = runBlocking {
        val file = tempFile()
        FileSemanticIndex(file, "model-a", 3).upsert("x", floatArrayOf(1f, 0f, 0f))

        val reopened = FileSemanticIndex(file, "model-a", 5)

        assertTrue(reopened.search(floatArrayOf(1f, 0f, 0f, 0f, 0f), limit = 8).isEmpty())
    }

    @Test
    fun `compact preserves every live vector`() = runBlocking {
        val file = tempFile()
        val idx = FileSemanticIndex(file, "model-a", 3)
        idx.upsert("a", floatArrayOf(1f, 0f, 0f))
        idx.upsert("b", floatArrayOf(0f, 1f, 0f))
        idx.remove("a")
        idx.upsert("a", floatArrayOf(0f, 0f, 1f))

        idx.compact()

        val reopened = FileSemanticIndex(file, "model-a", 3)
        val hits = reopened.search(floatArrayOf(0f, 0f, 1f), limit = 8).map { it.id }.toSet()
        assertEquals(setOf("a", "b"), hits.toSet())
    }

    @Test
    fun `compact shrinks a file bloated by churn on the same id`() = runBlocking {
        val file = tempFile()
        val idx = FileSemanticIndex(file, "model-a", 3)
        repeat(50) { i -> idx.upsert("same-id", floatArrayOf(i.toFloat(), 0f, 0f)) }
        val bloatedSize = file.length()

        idx.compact()

        assertTrue(file.length() < bloatedSize, "compacting away 49 superseded writes to the same id must shrink the file")
    }

    @Test
    fun `repeatedly overwriting a small set of ids compacts itself automatically`() = runBlocking {
        val file = tempFile()
        val idx = FileSemanticIndex(file, "model-a", 3)

        // 500 writes, but only 5 distinct ids ever superseding each other —
        // almost all of that log is waste a compaction should reclaim, and
        // this never calls compact() itself: the point is the automatic
        // trigger, not the manual one (already covered above).
        repeat(500) { i ->
            val id = "id-${i % 5}"
            idx.upsert(id, floatArrayOf(i.toFloat(), 0f, 0f))
        }

        assertTrue(idx.missing((0..4).map { "id-$it" }).isEmpty(), "all 5 ids must still be present after the churn")
        assertTrue(
            file.length() < recordSizeEstimate(dimension = 3) * 30,
            "500 writes to only 5 ids without automatic compaction kicking in at some point " +
                "would leave a file sized for hundreds of records instead of a handful",
        )
    }

    private fun recordSizeEstimate(dimension: Int): Int = 1 + 2 + 8 + dimension * 4 // generous upper bound per record
}
