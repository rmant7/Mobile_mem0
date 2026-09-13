package ai.localstudio.memory

class InMemorySemanticIndexContractTest : MemorySemanticIndexContractTest() {
    override fun index(modelId: String, dimension: Int): MemorySemanticIndex = InMemorySemanticIndex(modelId, dimension)
}
