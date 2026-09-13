package com.example.chatapp

import ai.localstudio.memory.FileMemoryStore
import ai.localstudio.memory.MemoryExtractor
import ai.localstudio.memory.MemoryItem
import ai.localstudio.memory.MemoryProvider
import ai.localstudio.memory.MemoryQuery
import ai.localstudio.memory.MemoryScope
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import java.io.File

/**
 * A minimal, realistic wiring of Mobile mem0 into an Android chat screen —
 * not a library API, just the shape real usage takes. Copy what you need.
 *
 * Stand-in for whatever local or remote model call your app already has —
 * any `suspend fun generate(prompt: String): String` works as a
 * [MemoryExtractor], since that's a one-method interface.
 */
interface MyLlmRuntime {
    suspend fun generate(prompt: String): String
}

class ChatMemoryViewModel(
    application: Application,
    private val llm: MyLlmRuntime,
) : AndroidViewModel(application) {

    /**
     * One JSON file, in the app's own private storage — no server, no
     * permissions to request, survives the process being killed. The
     * extractor is the one piece that actually calls a model: it turns a
     * finished conversation's WORKING memory into a handful of durable
     * EPISODIC/SEMANTIC facts instead of keeping every line verbatim
     * forever.
     */
    private val memory: MemoryProvider = FileMemoryStore(
        file = File(application.filesDir, "memory.json"),
        extractor = MemoryExtractor { conversationId, workingMemory ->
            val transcript = workingMemory.joinToString("\n") { it.text }
            val prompt = "Summarize the durable facts worth remembering from " +
                "this conversation ($conversationId), one per line:\n\n$transcript"
            val summary = llm.generate(prompt)
            summary.lines()
                .filter { it.isNotBlank() }
                .map { line ->
                    MemoryItem(
                        id = "", // FileMemoryStore assigns the real id on write
                        text = line.trim(),
                        scope = MemoryScope.EPISODIC,
                        createdAt = 0L, // FileMemoryStore stamps the real time on write
                    )
                }
        },
    )

    /** Called once per user turn — stores what was said as WORKING memory for this conversation. */
    fun rememberUserTurn(conversationId: String, text: String) {
        viewModelScope.launch {
            memory.remember(
                text = text,
                scope = MemoryScope.WORKING,
                metadata = mapOf(FileMemoryStore.CONVERSATION_KEY to conversationId),
            )
        }
    }

    /**
     * Called before building the prompt for the model's next reply — pulls
     * whatever durable memory is actually relevant to what the user just
     * asked, so the model doesn't need the entire conversation history
     * re-sent (or worse, no memory of anything said in an earlier session).
     */
    fun relevantMemory(query: String, onResult: (List<MemoryItem>) -> Unit) {
        viewModelScope.launch {
            onResult(memory.search(MemoryQuery(text = query, limit = 8)))
        }
    }

    /** Called when the user starts a new conversation — distills the finished one, clears its working set. */
    fun consolidatePreviousConversation(conversationId: String) {
        viewModelScope.launch {
            memory.consolidate(conversationId)
        }
    }
}
