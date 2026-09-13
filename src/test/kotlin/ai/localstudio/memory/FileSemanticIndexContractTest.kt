package ai.localstudio.memory

import java.io.File

class FileSemanticIndexContractTest : MemorySemanticIndexContractTest() {
    override fun index(modelId: String, dimension: Int): MemorySemanticIndex {
        val file = File.createTempFile("semantic-index-contract-test", ".bin").apply { deleteOnExit() }
        return FileSemanticIndex(file, modelId, dimension)
    }
}
