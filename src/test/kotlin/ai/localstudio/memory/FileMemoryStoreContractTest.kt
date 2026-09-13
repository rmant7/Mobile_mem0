package ai.localstudio.memory

import java.io.File

class FileMemoryStoreContractTest : MemoryProviderContractTest() {
    // A fresh temp file per provider() call, not one shared across a whole
    // test class run — a test that (now or later) constructs more than one
    // provider must not have the second one see the first's data on disk.
    override fun provider(extractor: MemoryExtractor, semanticIndex: MemorySemanticIndex?, embedder: MemoryEmbedder?): MemoryProvider =
        FileMemoryStore(
            File.createTempFile("mobile-mem0-contract-", ".json").also { it.deleteOnExit() },
            extractor,
            semanticIndex,
            embedder,
        )
}
