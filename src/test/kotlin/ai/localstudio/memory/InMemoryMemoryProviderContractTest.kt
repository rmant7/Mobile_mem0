package ai.localstudio.memory

class InMemoryMemoryProviderContractTest : MemoryProviderContractTest() {
    override fun provider(extractor: MemoryExtractor, semanticIndex: MemorySemanticIndex?, embedder: MemoryEmbedder?): MemoryProvider =
        InMemoryMemoryProvider(extractor, semanticIndex, embedder)
}
