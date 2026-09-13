package ai.localstudio.memory

import kotlinx.coroutines.runBlocking
import kotlin.random.Random
import kotlin.system.measureNanoTime
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * How [MemoryRanking.search] actually scales, at sizes a real, long-lived
 * personal memory store could plausibly reach — measured, not guessed.
 * Retrieval here is lexical (shared-term overlap), which means cost grows
 * linearly with how much is stored, unlike an indexed vector store; this is
 * exactly the tradeoff the README's own "What this deliberately isn't"
 * section names, made concrete as a number instead of a claim.
 *
 * Run with `--info` (or `-i`) to see the printed table — Gradle otherwise
 * swallows a passing test's stdout:
 * `./gradlew test --tests "*RetrievalBenchmarkTest*" --info`
 *
 * The assertions are a coarse regression tripwire (an order-of-magnitude
 * slowdown fails CI), not a performance contract — exact numbers depend on
 * the machine running them, which is why this prints instead of asserting
 * tight bounds.
 */
class RetrievalBenchmarkTest {

    @Test
    fun `search scales linearly, not catastrophically, up to 100k items`() = runBlocking {
        val sizes = listOf(1_000, 10_000, 100_000)
        val results = sizes.map { size -> benchmarkSize(size) }

        println("\n${"size".padEnd(10)}${"populate ms".padEnd(14)}${"avg search ms".padEnd(16)}${"p95 search ms"}")
        results.forEach { (size, populateMs, avgSearchMs, p95SearchMs) ->
            println("${size.toString().padEnd(10)}${"%.1f".format(populateMs).padEnd(14)}${"%.2f".format(avgSearchMs).padEnd(16)}${"%.2f".format(p95SearchMs)}")
        }

        // A coarse tripwire, not a tight bound — this is meant to catch "someone
        // accidentally made search() quadratic," not to enforce a specific
        // number on hardware this test has no idea about. Measured directly
        // (not guessed) at ~400ms on ordinary, shared CI-grade hardware —
        // each search tokenizes every stored item's text, so cost genuinely
        // is linear in store size, not free. 3s leaves roughly 7x headroom
        // above that measurement before this fails.
        val at100k = results.last()
        assertTrue(
            at100k.avgSearchMs < 3_000.0,
            "search() at 100k items averaged ${at100k.avgSearchMs}ms — that's " +
                "far slower than expected, look for an accidental quadratic pass " +
                "over the whole store (this used to average well under 1s)",
        )
    }

    private data class SizeResult(val size: Int, val populateMs: Double, val avgSearchMs: Double, val p95SearchMs: Double)

    private suspend fun benchmarkSize(size: Int): SizeResult {
        val random = Random(seed = size.toLong())
        val memory = InMemoryMemoryProvider()

        val populateNanos = measureNanoTime {
            repeat(size) { i ->
                runBlocking {
                    memory.remember(
                        text = randomSentence(random),
                        scope = if (i % 5 == 0) MemoryScope.SEMANTIC else MemoryScope.EPISODIC,
                    )
                }
            }
        }

        // Repeated real searches, not one — a single call is too noisy
        // (JIT warmup, GC pauses) to say anything about steady-state cost.
        val searchLatenciesMs = (1..50).map {
            val query = MemoryQuery(text = randomSentence(random), limit = 8)
            val nanos = measureNanoTime { runBlocking { memory.search(query) } }
            nanos / 1_000_000.0
        }.sorted()

        return SizeResult(
            size = size,
            populateMs = populateNanos / 1_000_000.0,
            avgSearchMs = searchLatenciesMs.average(),
            p95SearchMs = searchLatenciesMs[(searchLatenciesMs.size * 0.95).toInt().coerceAtMost(searchLatenciesMs.lastIndex)],
        )
    }

    /** Realistic-enough lexical overlap: a handful of words from a small, fixed vocabulary, not truly random noise. */
    private fun randomSentence(random: Random): String =
        (1..12).joinToString(" ") { VOCABULARY[random.nextInt(VOCABULARY.size)] }

    private companion object {
        val VOCABULARY = listOf(
            "user", "prefers", "dark", "mode", "project", "deadline", "friday", "kotlin",
            "meeting", "notes", "budget", "travel", "plan", "recipe", "reminder", "settings",
            "email", "client", "invoice", "feedback", "release", "bug", "feature", "design",
            "database", "server", "mobile", "android", "memory", "context", "conversation", "model",
        )
    }
}
