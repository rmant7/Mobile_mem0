# Semantic retrieval (v0.3) — design

**Status: v0.3.0-alpha1 implements steps 1–8 below.** `MemoryEmbedder`,
`MemorySemanticIndex`, `FileSemanticIndex`, `MemoryCandidate`, and the
per-backend `candidates()` wiring all exist and are covered by the shared
contract test suite, including forget()/consolidate() vector cleanup and a
model/dimension mismatch degrading to lexical-only rather than to nothing.
`semanticScore` reaches the consuming app's private ranking layer, at a
fusion weight of `0.0` — deliberately unconsidered, not a guess (see step 9).

**No concrete embedding model is chosen or downloadable yet, on purpose.**
Every abstraction here is built and tested against a fake, deterministic
`MemoryEmbedder` — nothing in this repo, or in the consuming app's build,
references a real Hugging Face repository, GGUF filename, or model id. That
choice needs verification this repository has no way to perform (fetching a
real model listing, confirming a specific GGUF quant exists at a specific
path, and eventually running it against real device hardware) and must not
be guessed at from training-data familiarity with model names that may have
been renamed, requantized, or taken down since. See "Choosing the concrete
model" near the end of this document for exactly what that verification
needs to cover before a `LocalModelSeed` entry and JNI wiring reference one.

This document otherwise exists to be argued with *before* any of the rest of
it is built, because v0.3 touches five things at once — the open-source/
commercial boundary, the persistence format, the embedding lifecycle, a
native model runtime, and the public API — and getting the API right after
discovering the storage requirements is the expensive order to do it in.

## The problem this solves

Retrieval today is lexical: shared whole-word overlap, no stemming, no
semantics (see [`MemoryRanking`](src/main/kotlin/ai/localstudio/memory/MemoryRanking.kt)).
That fails on a case that is not an edge case at all in a real conversation:

```
stored:  "Пользователь искал место для зимовки; рекомендовано Израиль, Турция, Египет…"
query:   "что ты мне предлагала в предыдущем чате кроме Израиля?"
result:  0 candidates
```

Not one shared token. `Израиль` and `Израиля` are different strings; the rest
of the query (`предлагала`, `предыдущем`, `чате`) never appears in the stored
text at all. The question is unmistakably about that memory, and lexical
retrieval cannot see it.

Russian makes this systemic rather than occasional — six cases means any
entity referenced in a follow-up usually appears in a different surface form
than it was stored in. Prefix-truncation stemming would paper over this
specific class of miss; it would do nothing for `зимовка` ↔ `где перезимовать`,
which is the same question one paraphrase away. Semantics is the actual fix,
so this is the fix being designed.

`MemoryQuery.matchAll` (v0.2) stays and is unaffected: it is a deliberate
"return everything in scope" switch for questions *about* memory, not a
relevance mechanism.

## Division of responsibility

The existing boundary (`commercial-memory/README.md` in the consuming app)
already settles most of this, and this design does not move it:

> **Mobile_mem0 stores and retrieves; this module decides what matters.**
> Do not add to Mobile_mem0: user-specific ranking formulas or weights.

So:

| Mobile_mem0 (this repo, open source) | The consuming app's private layer |
| --- | --- |
| store memory records | decide which memories matter for *this* request |
| lexical retrieval | fusion of lexical + semantic signals |
| semantic candidate retrieval | ranking weights, RRF constants, tuning |
| merge both candidate sets, deduplicated | context budget and selection |
| embedding/model compatibility + index lifecycle | feedback-driven adjustment |
| persistence of records and vectors | — |

**A weighted fusion formula living in this repo would violate that boundary.**
`0.45·semantic + 0.35·lexical + 0.20·recency` is precisely "ranking weights",
and the private layer is already built to hold exactly this: `ContextCandidate`
carries separate signals (`lexicalScore`, `taskRelevance`, `recencyScore`,
`sourcePriority`), `RankingWeights` holds their weights, and `ContextRanker` is
swappable. Adding `semanticScore` there extends a structure designed for it.

This repo therefore reports **signals and positions** and stops. Reciprocal
Rank Fusion — or any other fusion — is the consumer's decision, implemented on
its side, because fusion *is* the relevance decision.

## API

### `MemoryCandidate` — what retrieval returns

```kotlin
data class MemoryCandidate(
    val item: MemoryItem,
    /** Position in the lexical result set, 0-based; null if lexical did not match it. */
    val lexicalRank: Int?,
    /** Position in the semantic result set, 0-based; null if not retrieved semantically. */
    val semanticRank: Int?,
    /** Cosine similarity in -1..1; null when there is no usable embedding for this item. */
    val semanticScore: Float?,
)
```

Ranks as well as scores, because rank-based fusion (RRF) needs positions and
score-based fusion needs scores, and this repo does not get to decide which one
the consumer uses. A candidate retrieved by only one of the two retrievers has
`null` for the other — that is information, not a gap to fill in with zero.

`MemoryProvider` gains one method and keeps the old one working:

```kotlin
interface MemoryProvider {
    suspend fun search(query: MemoryQuery): List<MemoryItem>          // unchanged

    /** Both retrievers, merged and deduplicated, unranked beyond each retriever's own order. */
    suspend fun candidates(query: MemoryQuery): List<MemoryCandidate> =
        search(query).mapIndexed { i, item -> MemoryCandidate(item, lexicalRank = i, semanticRank = null, semanticScore = null) }
}
```

The default implementation means every existing `MemoryProvider` — including
anyone else's — keeps compiling and behaves exactly as it does today.

### `MemoryEmbedder` — the model seam

Same seam, same reasoning as `MemoryExtractor`: this module has no model
runtime and must not acquire one.

```kotlin
interface MemoryEmbedder {
    /** Identifies the model *and* its revision. Changing the model must change this. */
    val modelId: String
    val dimension: Int

    /** Batched and suspending: an on-device embedding call is slow and blocking. */
    suspend fun embed(texts: List<String>): List<FloatArray>
}
```

Two deliberate changes from the original sketch: `suspend`, because every other
model-touching call in this library is (`MemoryExtractor.extract`), and llama.cpp
blocks; and `List<String>` rather than one string, because batch embedding is
several times faster per item than one call each and consolidation embeds a
batch at a time. This also matches `EmbeddingFunction` in the consuming app's
`:core/knowledge`, so one adapter can serve both and they cannot drift apart.

Nothing about JNI, GGUF, or llama.cpp appears in this repo. The chain lives on
the consumer's side:

```
MemoryEmbedder  ←  LlamaCppMemoryEmbedder  →  JNI  →  llama.cpp
   (here)              (consuming app)
```

An HTTP embedding API, ONNX Runtime, or a future on-device NPU path all satisfy
the same interface without this repo changing.

### `MemorySemanticIndex` — vectors

```kotlin
interface MemorySemanticIndex {
    val modelId: String
    val dimension: Int

    suspend fun upsert(id: String, embedding: FloatArray)
    suspend fun remove(id: String)
    suspend fun search(embedding: FloatArray, limit: Int): List<ScoredId>
    /** Ids present in the record store but not here yet — the re-embedding work queue. */
    suspend fun missing(ids: Collection<String>): List<String>
}
```

## Persistence

**Decision: a separate binary sidecar file, not JSON, and not SQLite.**

Embeddings must not go into `memory.json`. A 384-dimensional float32 vector is
1.5 KB binary but roughly 5 KB as JSON text, and `FileMemoryStore` rewrites the
*entire* file on every `remember()` — which happens on every conversation turn.
At a thousand memories that is a multi-megabyte synchronous rewrite per turn.
This is not a tuning problem, it is the wrong container.

SQLite is the reflex answer and is not needed yet. Measured against realistic
sizes:

| memories | vectors (384d, f32) | brute-force cosine per search |
| --- | --- | --- |
| 1 000 | 1.5 MB | ~0.4M multiply-adds — sub-millisecond |
| 10 000 | 15 MB | ~3.8M — low single-digit ms |
| 100 000 | 153 MB | ~38M — tens of ms |

For comparison, [`RetrievalBenchmarkTest`](src/test/kotlin/ai/localstudio/memory/RetrievalBenchmarkTest.kt)
measured the *existing lexical* scan at ~400 ms average at 100k items, because
it re-tokenizes every stored item's text on every search. A brute-force vector
scan over the same 100k is expected to be **faster than what already ships**.
No ANN index, no vector database, no SQLite dependency is justified by these
numbers — and 100k personal memories is the benchmark's stress ceiling, not a
number a real user reaches.

So v0.3's index is a flat file: a small header (`modelId`, `dimension`,
`count`), then fixed-stride records of `id → float32[dimension]`, appended, with
tombstones for removals and periodic compaction. Sequential, memory-mappable,
no full rewrite on insert.

**The contract that matters more than the format:** `MemoryProvider` says
nothing about how any of this is stored. `FileMemoryStore` keeps its JSON
records and gains a sidecar index; a future `SqliteMemoryStore` (or
sqlite-vec, or ObjectBox) replaces both halves without any consumer changing a
line. That is the actual deliverable of this section — the flat file is v0.3's
implementation, not v0.3's API.

## Embedding lifecycle

**Memory records are never lost, altered, or hidden because of anything that
happens to an embedding.** Every rule below follows from that one.

**Embedding happens off the hot path.** `remember()` must not block on a model
call — it runs on every turn of every conversation, and a chat turn waiting on
an embedding is a regression a user feels. New records are written immediately
and enqueued; their vectors land afterwards. A record without a vector yet is
retrievable lexically in the meantime.

**Partial coverage is the normal state, not an error.** New items arrive while
a re-embedding pass is still running. Retrieval must therefore always tolerate
a mix of embedded and un-embedded records — which falls out naturally, since
`semanticRank`/`semanticScore` are nullable by design.

**A model change invalidates vectors, never records.**

```
index.modelId == embedder.modelId   →  semantic + lexical
index.modelId != embedder.modelId   →  lexical only, re-embed in background
no embedder configured              →  lexical only, forever, no degradation
```

`modelId` must change whenever the weights change — same architecture, different
quantization is a *different* model for this purpose, because its vector space
is not the old one. Dimension mismatch is a special case of this and is already
modelled in the consuming app's `VectorIndex` (`IndexDimensionMismatchException`),
whose comment names the exact failure this avoids: "the embedding model changed
and the knowledge base needs re-indexing."

Documents can be re-ingested from their source files. Personal memory cannot —
there is no source to re-read. That asymmetry is why the invariant at the top
of this section is absolute.

## Invariants worth a contract test

These belong in [`MemoryProviderContractTest`](src/test/kotlin/ai/localstudio/memory/MemoryProviderContractTest.kt),
so every backend proves them rather than each being trusted:

1. With no `MemoryEmbedder` configured, behaviour is identical to v0.2 — every
   existing test passes unchanged, `candidates()` returns lexical-only results.
2. `forget(id)` removes the vector as well as the record.
3. A record whose vector has not been computed yet is still returned by lexical
   retrieval, with `semanticRank == null`.
4. After a `modelId` change, every record is still retrievable lexically and
   nothing is deleted.
5. `candidates()` never returns the same `MemoryItem.id` twice, however many
   retrievers matched it.

## Costs to accept explicitly

- **A second resident model.** Embedding needs its own llama.cpp context
  (embeddings require a different pooling configuration than generation), so
  it cannot share the chat model's. On a phone already doing RAM-aware model
  eviction, that is another consumer competing for the same budget.
- **It must be multilingual.** The primary usage is Russian. `all-MiniLM-L6-v2`
  and similar English-only models are not usable here. A model in the
  `multilingual-e5` family (small/base) is the plausible shape of candidate on
  size grounds alone; `bge-m3` and LaBSE are likely too large for the target
  devices. Neither claim has been checked against a real, currently-existing
  GGUF listing — see "Choosing the concrete model" below.
- **Every search embeds the query.** One model call per turn, on the latency
  path, in addition to the generation call.
- **The native entry point exists in the consuming app (`llama_jni.cpp`),
  unverified.** `nativeLoadEmbeddingModel`/`nativeEmbed` were added and their
  llama.cpp API usage checked line-by-line against the actual vendored
  header and upstream reference implementation at the exact pinned commit —
  but with no real embedding GGUF to load, nothing has run them, on an
  emulator or a device. This is real, load-bearing native code that has only
  been checked by a compiler so far.

## What this deliberately is not

- **Not an ANN index.** Brute force is measurably sufficient at realistic
  sizes (see the table above). HNSW/IVF is justified by a number nobody has
  yet.
- **Not a reranker.** Cross-encoder reranking is a ranking decision, and
  ranking is not this repo's job.
- **Not semantic-only.** Lexical retrieval stays, and stays first-class: exact
  names, project identifiers, repository names, dates, and rare terms are
  precisely where embeddings are weakest and token matching is strongest.
  Dropping it to "simplify" would trade one class of miss for another.
- **Not a fusion algorithm.** See the boundary section. This repo's last word
  is a deduplicated candidate list with per-retriever ranks.

## Choosing the concrete model

Deliberately not done yet, and not a guess this document or its
implementation should ship ahead of. Whoever picks up this step needs to
actually verify — with real network and, eventually, real device or
emulator access, neither of which this design work had — at least:

- **The model exists, right now, as a GGUF.** A specific Hugging Face repo
  and exact filename, confirmed reachable (not renamed, not taken down,
  not gated behind a license click-through this app's plain download can't
  satisfy — see `LocalModelSeed.repoIds`'s own doc comment on why gated
  official repos need a community mirror listed first).
- **It is actually multilingual and actually small enough.** Confirm
  parameter count, output dimension, and quantized file size directly from
  that listing — not from a model name's family resemblance to one this
  document guessed might fit.
- **It embeds correctly through this repo's own `nativeEmbed` path.** Load
  it via `nativeLoadEmbeddingModel`, embed a few known sentence pairs, and
  check the results make sense (near-duplicates score high, unrelated
  sentences score low) — the JNI has been checked against llama.cpp's own
  API by reading, never by running.
- **Mean pooling is the right choice for this specific model.** This design
  hardcodes `LLAMA_POOLING_TYPE_MEAN` in `nativeLoadEmbeddingModel`; some
  embedding checkpoints expect CLS-token pooling instead, and using the
  wrong one silently produces *usable-looking but wrong* vectors rather than
  an error — exactly the failure mode hardest to notice without checking a
  known-similar/known-dissimilar pair by hand.
- **`Capability.EMBEDDING` and the existing model-download UI.** Whether the
  memory embedder reuses the same `Capability`/`LocalModelSeed` machinery
  the document-RAG path already declares, or needs its own — worth deciding
  once a real model is in hand, not before.

Only once these are confirmed does a `LocalModelSeed` entry, a
`LlamaCppMemoryEmbedder` construction site in `AppContainer`, and a non-zero
`RankingWeights.semantic` belong in either repo.

## Order of work

Each step is reviewable on its own, and the expensive, hard-to-reverse
decisions come before anything depends on them:

1. **This document, approved.** ✅
2. **Persistence contract** — the sidecar format and its compaction rule,
   written down before code. ✅
3. **`MemoryEmbedder`** — interface only, no implementation. ✅
4. **`MemorySemanticIndex`** — interface only. ✅
5. **Flat-file index implementation** + contract tests, driven by a fake
   embedder. No real model involved, fully testable on the JVM. ✅
6. **JNI entry point**, in the consuming app — `nativeLoadEmbeddingModel`/
   `nativeEmbed`, checked against llama.cpp's own API by reading its source
   at the pinned commit. ✅ code exists; ⏳ **unverified against any real
   model** — see "Choosing the concrete model" above, which is the actual
   remaining half of this step.
7. **Semantic candidate retrieval wired into `candidates()`.** ✅
8. **Fusion and ranking** — in the private layer: `semanticScore` reaches
   `HeuristicContextRanker` behind `RankingWeights.semantic`. ✅ plumbing;
   ⏳ **weight is 0.0** — see step 9.
9. **Benchmark and decide** — `ExperimentLogger`/`ExperimentRecord` already
   exist to compare modes on real queries. Which fusion, and what weights, is
   a measurement, not a guess. **Blocked on a real model** (step 6's
   remaining half) to have anything real to measure.

Steps 1–5 needed no model, no NDK, and no downloads, and were where the
irreversible API decisions lived — all landed in v0.3.0-alpha1. Everything
from here forward is blocked on a verified, concrete embedding model.
