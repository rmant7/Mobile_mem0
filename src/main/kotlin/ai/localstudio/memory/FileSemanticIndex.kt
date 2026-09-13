package ai.localstudio.memory

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream

/**
 * A persistent [MemorySemanticIndex] — a binary, append-only sidecar file,
 * deliberately not JSON and not SQLite; see [SEMANTIC_RETRIEVAL_DESIGN.md]
 * (repo root) for the numbers behind that choice. A 384-dimensional vector is
 * ~1.5 KB binary versus ~5 KB as JSON text, and rewriting the whole file on
 * every write — [FileMemoryStore]'s own pattern — would mean rewriting
 * megabytes on every conversation turn. This appends one small record per
 * write instead, and [compact] bounds how large the log can grow between
 * writes that actually still matter.
 *
 * File layout: a header ([MAGIC], [FORMAT_VERSION], [modelId], [dimension]),
 * then a sequence of records — an upsert (id + [dimension] floats) or a
 * tombstone (id alone) — replayed in order at load time, last write per id
 * winning. The full current state is kept in memory at all times, same as
 * [FileMemoryStore]; the file is the persistence path, never the read path.
 *
 * A vector search here is brute-force cosine over every live entry — see the
 * design doc's own measurement of why that is fast enough (faster, in fact,
 * than the lexical scan [MemoryRanking] already does at the same size) — not
 * an approximation this class is pretending is exact.
 */
class FileSemanticIndex(
    private val file: File,
    override val modelId: String,
    override val dimension: Int,
) : MemorySemanticIndex {

    private val lock = Any()
    private val vectors = LinkedHashMap<String, FloatArray>()

    // Counts every record appended since the file was last written fresh
    // (construction or [compact]) — tombstones included, since a tombstone
    // is exactly as much log growth as an upsert. Compaction resets this.
    private var recordsSinceCompaction = 0

    init {
        synchronized(lock) { load() }
    }

    private fun load() {
        if (!file.isFile) {
            writeFreshFile(emptyMap())
            return
        }
        val loaded = runCatching { readFile() }.getOrNull()
        if (loaded == null || loaded.modelId != modelId || loaded.dimension != dimension) {
            // Either unreadable/corrupt, or a different embedding space than
            // this instance was constructed for — a model change invalidates
            // vectors, never the memory records they point at (that
            // invariant lives in MemoryProvider, not here). Either way there
            // is nothing usable to carry over; start empty under the new
            // header rather than half-trusting stale bytes.
            writeFreshFile(emptyMap())
            return
        }
        vectors.putAll(loaded.vectors)
    }

    private data class LoadedState(val modelId: String, val dimension: Int, val vectors: Map<String, FloatArray>)

    private fun readFile(): LoadedState {
        DataInputStream(file.inputStream().buffered()).use { input ->
            if (input.readInt() != MAGIC) throw IllegalStateException("not a semantic index file")
            if (input.readInt() != FORMAT_VERSION) throw IllegalStateException("unsupported semantic index format version")
            val fileModelId = input.readUTF()
            val fileDimension = input.readInt()
            val live = LinkedHashMap<String, FloatArray>()
            while (true) {
                val recordType = try {
                    input.readByte()
                } catch (_: EOFException) {
                    break
                }
                val id = input.readUTF()
                when (recordType) {
                    RECORD_UPSERT.toByte() -> {
                        val vector = FloatArray(fileDimension) { input.readFloat() }
                        live[id] = vector
                    }
                    RECORD_TOMBSTONE.toByte() -> live.remove(id)
                    else -> throw IllegalStateException("unknown semantic index record type: $recordType")
                }
            }
            return LoadedState(fileModelId, fileDimension, live)
        }
    }

    /** Caller must already hold [lock]. Replaces the file wholesale with just a header — the log restarts empty. */
    private fun writeFreshFile(initial: Map<String, FloatArray>) {
        DataOutputStream(FileOutputStream(file).buffered()).use { out ->
            writeHeader(out)
            initial.forEach { (id, vector) -> writeUpsert(out, id, vector) }
        }
        recordsSinceCompaction = initial.size
    }

    private fun writeHeader(out: DataOutputStream) {
        out.writeInt(MAGIC)
        out.writeInt(FORMAT_VERSION)
        out.writeUTF(modelId)
        out.writeInt(dimension)
    }

    private fun writeUpsert(out: DataOutputStream, id: String, vector: FloatArray) {
        out.writeByte(RECORD_UPSERT)
        out.writeUTF(id)
        vector.forEach { out.writeFloat(it) }
    }

    private fun appendUpsert(id: String, vector: FloatArray) {
        DataOutputStream(FileOutputStream(file, /* append = */ true).buffered()).use { out -> writeUpsert(out, id, vector) }
    }

    private fun appendTombstone(id: String) {
        DataOutputStream(FileOutputStream(file, /* append = */ true).buffered()).use { out ->
            out.writeByte(RECORD_TOMBSTONE)
            out.writeUTF(id)
        }
    }

    override suspend fun upsert(id: String, embedding: FloatArray) {
        require(embedding.size == dimension) { "expected a $dimension-dimensional vector, got ${embedding.size}" }
        synchronized(lock) {
            vectors[id] = embedding
            appendUpsert(id, embedding)
            recordsSinceCompaction++
            compactIfDue()
        }
    }

    override suspend fun remove(id: String) {
        synchronized(lock) {
            if (vectors.remove(id) == null) return
            appendTombstone(id)
            recordsSinceCompaction++
            compactIfDue()
        }
    }

    override suspend fun search(embedding: FloatArray, limit: Int): List<ScoredId> {
        require(embedding.size == dimension) { "expected a $dimension-dimensional vector, got ${embedding.size}" }
        if (limit == 0) return emptyList()
        val snapshot = synchronized(lock) { vectors.entries.map { it.key to it.value } }
        return snapshot
            .map { (id, vector) -> ScoredId(id, Cosine.similarity(embedding, vector)) }
            .sortedWith(compareByDescending<ScoredId> { it.score }.thenBy { it.id })
            .take(limit)
    }

    override suspend fun missing(ids: Collection<String>): List<String> =
        synchronized(lock) { ids.filter { it !in vectors } }

    /**
     * Rewrites the file from the current in-memory state alone — every
     * tombstone and every superseded upsert this log has ever accumulated is
     * dropped, leaving exactly one record per live entry. Safe to call at
     * any time; also called automatically (see [compactIfDue]) so the log
     * cannot grow without bound under steady churn.
     */
    fun compact() {
        synchronized(lock) { writeFreshFile(vectors) }
    }

    /** Caller must already hold [lock]. */
    private fun compactIfDue() {
        // Not a tight budget, just "the log stops being a small multiple of
        // the live set" — compaction itself is an O(live-set) file rewrite,
        // so it should stay rare relative to individual writes, not run on
        // every few of them.
        if (recordsSinceCompaction > COMPACTION_THRESHOLD && recordsSinceCompaction > vectors.size * 2) {
            writeFreshFile(vectors)
        }
    }

    private companion object {
        const val MAGIC = 0x4D534931 // "MSI1" — Memory Semantic Index, format 1
        const val FORMAT_VERSION = 1
        const val RECORD_UPSERT = 0
        const val RECORD_TOMBSTONE = 1
        const val COMPACTION_THRESHOLD = 64
    }
}
