package ai.localstudio.memory

class InMemoryMemoryProviderContractTest : MemoryProviderContractTest() {
    override fun provider(extractor: MemoryExtractor): MemoryProvider = InMemoryMemoryProvider(extractor)
}
