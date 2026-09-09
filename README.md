# Mobile mem0

A small, dependency-free local memory engine for LLM apps — a `mem0`-shaped
building block that runs entirely on-device, with no server, no API key,
and no network call.

Extracted from a real, shipping Android app ([Local AI Studio](https://github.com/rmant7/AI),
a fully local/offline-capable AI chat client), not written as a spec nobody's
used. The interface and its first implementation were live in that app's
codebase before being pulled out here — same code, same tests, now with zero
dependency on the app it came from.

## What it actually is

Three things:

- **`MemoryProvider`** — the contract: `remember`, `search`, `forget`,
  `consolidate`. Whatever backs it, the rest of your app talks to this and
  nothing else.
- **`FileMemoryStore`** — a persistent, lexical-search implementation of that
  contract. JSON file on disk, thread-safe, survives a process restart.
  Retrieval is shared-term overlap weighted by scope and recency — not
  embeddings, and not pretending to be.
- **`MemoryExtractor`** — the seam where a language model plugs in for
  *consolidation*: turning a finished conversation's raw working memory into
  a handful of durable facts, instead of keeping every line verbatim
  forever. This module doesn't call a model itself — bring your own
  (`suspend (String) -> String`, or whatever runtime you already have)
  — but the interface, and a plug-in point for it, are here.

## Why this exists

Building "memory" into an LLM app usually means one of two things: wiring up
a full vector-DB-backed service (Mem0, Zep, etc. — real network dependency,
real ongoing cost, real infrastructure to run), or hand-rolling something
one-off inside the app and never touching it again. This is the third
option: a real interface, worth keeping stable as your actual backend
changes, with a working implementation you can ship *today* without any of
that infrastructure — and swap out later without your app's code noticing.

Memory is split by lifetime, not by storage engine:

```
WORKING    the current conversation — dropped once it's consolidated
EPISODIC   what happened before — "decided to use Kotlin," "fixed the pipeline yesterday"
SEMANTIC   stable facts — preferences, project properties, entities
```

A stable (`SEMANTIC`) fact ranks above an episode at equal relevance; between
two episodes, the more recent one wins; semantic facts aren't discounted for
age at all — a preference stated a month ago is exactly as true as one
stated today.

## Quick start

```kotlin
val memory: MemoryProvider = FileMemoryStore(File(context.filesDir, "memory.json"))

// Remember something, tagged however you like.
val id = memory.remember(
    "User prefers dark mode",
    MemoryScope.SEMANTIC,
    metadata = mapOf("source" to "settings"),
)

// Search it back later — lexical, so the query needs to share real words
// with what you stored, UNLESS you scope by metadata instead (see below).
val hits = memory.search(MemoryQuery("what does the user prefer for theme"))

// A metadata filter bypasses the word-overlap requirement entirely — built
// for exactly the "I attached this specific document, just give it back to
// me regardless of how the question is phrased" case.
val thisFile = memory.search(
    MemoryQuery(text = "", metadataFilter = mapOf("source" to "report.pdf")),
)

memory.forget(id)

// Turn a conversation's working memory into durable facts, via whichever
// model you wire up as a MemoryExtractor. Clears the working set either way.
val kept: List<MemoryItem> = memory.consolidate(conversationId = "chat-42")
```

Wiring a real model in for consolidation is just implementing the one-method
interface with whatever you already call to generate text:

```kotlin
val extractor = MemoryExtractor { prompt -> myLlmRuntime.generate(prompt) }
val memory = FileMemoryStore(file, extractor = extractor)
```

## Installation

Not on Maven Central (yet) — the fastest path today is [JitPack](https://jitpack.io/#rmant7/Mobile_mem0),
which builds straight from this repo, no publishing setup required on either
side:

```kotlin
repositories {
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation("com.github.rmant7:Mobile_mem0:main-SNAPSHOT") // or a tag once one exists
}
```

Or just copy `src/main/kotlin/ai/localstudio/memory/` into your project —
it's two files and has no dependency beyond Kotlin coroutines and
kotlinx-serialization.

## What this deliberately isn't (yet)

- **No embeddings / semantic search.** Retrieval is lexical word-overlap.
  Good enough to ship, not competitive with a real vector index — swapping
  one in behind `MemoryProvider` is exactly the kind of change this
  interface exists to make painless, not a rewrite.
- **No bundled model-calling extractor.** `MemoryExtractor` is one method;
  you provide the model call. A reference implementation that calls a local
  GGUF model directly is a natural next addition.
- **No Mem0-compatible API.** The shape is inspired by Mem0's memory model
  (working/episodic/semantic scopes, consolidation as a distinct step), not
  a drop-in replacement for its client library.

Issues and PRs — including for any of the above — welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE).
