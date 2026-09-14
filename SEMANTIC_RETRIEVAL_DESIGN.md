# Semantic retrieval (v0.3) — design

**Status: steps 1–8 below are done, and step 9 has a real first measurement.**
`MemoryEmbedder`, `MemorySemanticIndex`, `FileSemanticIndex`,
`MemoryCandidate`, and the per-backend `candidates()` wiring all exist and
are covered by the shared contract test suite, including
forget()/consolidate() vector cleanup and a model/dimension mismatch
degrading to lexical-only rather than to nothing. `semanticScore` reaches
the consuming app's private ranking layer, at a fusion weight of `0.30` —
a benchmark-informed starting value (see "Choosing the concrete model"
below and step 9), not a finally-tuned one: no A/B run against the app's
own real `ExperimentLogger` data exists yet, only the standalone retrieval
benchmark's own dataset.

As of alpha3, `MemoryEmbedder` is split into `embedForStorage`/
`embedForQuery` rather than one `embed` method — most real embedding models
(e5, bge, gte, and others) are asymmetric dual encoders trained with a
different prefix for what they index than for what searches it, and a
single generic method has no seam for that distinction. Changed while still
alpha, before anything outside this repo's own two backends implemented the
interface. The consuming app's `LlamaCppMemoryEmbedder` also gained a
required `pooling` parameter (no default) — a wrong pooling choice for a
given model, like a wrong prefix, does not error, it silently produces
plausible-looking but degraded vectors.

**A concrete embedding model is now chosen, verified, and loaded
automatically: `multilingual-e5-base` (`groonga/multilingual-e5-base-Q4_K_M-GGUF`,
768 dimensions, mean pooling, `"query: "`/`"passage: "` prefixes).** This
repo's own abstractions are still built and tested against a fake,
deterministic `MemoryEmbedder` — that hasn't changed — but the consuming
app's `AppContainer` now downloads this exact GGUF and loads it through
`LlamaCppMemoryEmbedder` automatically, off the main thread, the first time
memory is enabled. See "Choosing the concrete model" below for the
verification history (an on-device dimension/cosine check, then a
standalone Python retrieval benchmark) and for the one candidate that was
tried and rejected (`cstr/multilingual-e5-small-GGUF`: fails to load with
`bert model needs to define token type count`, a metadata defect in that
specific conversion — not something this app's own code can work around).
Nothing in this repository itself references a real Hugging Face
repository, GGUF filename, or model id; that stays entirely the consuming
app's concern, per the boundary this document already draws.

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

Done, in two verification passes — recorded here rather than left only in
commit messages, since this section's whole point was to say what "done"
would actually require.

**Candidates tried:**

- `cstr/multilingual-e5-small-GGUF` (`IQ4_XS`) — **rejected.** Fails to
  load on-device (`ExperimentalEmbeddingsActivity`'s Test button) with
  `llama_model_load: error loading model: bert model needs to define token
  type count` — a metadata field missing from this specific conversion.
  Not a pooling or prefix problem, not fixable from this app's own code.
  Kept in `ExperimentalEmbeddingModels.kt` as a record of what was tried,
  explicitly marked not to retry as-is.
- `cstr/multilingual-e5-base-GGUF` (`Q4_K`, `-imatrix` variant) — also
  **rejected**, for the same reason: fails to load (this time surfaced via
  `llama-cpp-python` in the standalone Colab benchmark below) with the
  same generic "failed to load model" signature. Two failures from the
  same uploader's conversions was treated as a real signal, not
  re-guessed at a third time.
- `groonga/multilingual-e5-base-Q4_K_M-GGUF` (`Q4_K_M`) — **chosen.**
  278M parameters, 768 dimensions, mean pooling, XLM-R base architecture.

**Verification pass 1 — on-device sanity check**
(`ExperimentalEmbeddingModelTest` / `ExperimentalEmbeddingsActivity`'s Test
button): loaded successfully, dimension reported as 768 (read from the
model itself via `nativeEmbeddingDimension`, never assumed), L2 norm ≈ 1.0,
`cosine(query, similar passage) = 0.897 > cosine(query, dissimilar passage)
= 0.754`. Pooling confirmed as `EmbeddingPooling.MEAN`, prefixes confirmed
as e5's own `"query: "`/`"passage: "` convention — both exactly as assumed,
but confirmed by a passing test against a real loaded model, not inherited
from family resemblance.

**Verification pass 2 — standalone retrieval benchmark**
(`benchmark/e5_base_benchmark.ipynb`, `benchmark/run_benchmark.py` — runs
on Google Colab's free CPU tier via `llama-cpp-python`, no Android/JNI, no
Firestore, downloads the GGUF itself and never commits it to this repo).
105 memories / 102 queries across seven categories (`exact`, `morphology`,
`synonym`, `paraphrase`, `low_overlap`, `identifier`, `negative`), lexical
retrieval scored by a direct Python port of this repo's own
`MemoryRanking.tokenize()`/`overlap()`, semantic scored by cosine
similarity — never a threshold, only relative ranking, the same way
`HeuristicContextRanker` itself works. Result, 92 queries with ground
truth:

|          | Recall@1 | Recall@5 | Recall@10 | MRR   |
|----------|----------|----------|-----------|-------|
| Lexical  | 0.500    | 0.598    | 0.620     | 0.543 |
| Semantic | 0.848    | 0.946    | 0.946     | 0.892 |

By category (Recall@1, lexical → semantic): exact 0.778→0.944, identifier
1.000→1.000 (tied, not worse — the one category semantic was expected to
possibly lose), low_overlap 0.176→0.471, morphology 0.500→0.950, synonym
0.200→0.933, paraphrase 0.600→0.867. Average cosine similarity: positive
matches 0.841, negative (unrelated) 0.753.

This is what set `RankingWeights.semantic` to `0.30` in the consuming
app's `AppContainer` (see status header above) — a benchmark-informed
starting value, deliberately still below `RankingWeights.taskRelevance`'s
0.30 rather than at or above it, since no A/B run against this app's own
real usage (`ExperimentLogger` data) exists yet. `Capability.EMBEDDING`
and the existing chat-model download UI were deliberately *not* reused —
the embedding model has its own small download/verification screen
(`ExperimentalEmbeddingsActivity`) rather than joining
`LocalModelSeed`/`LocalModels`, which stays a chat-model-only catalog.

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
   at the pinned commit. ✅ code exists; ✅ **verified against a real model**
   — see "Choosing the concrete model" above.
7. **Semantic candidate retrieval wired into `candidates()`.** ✅
8. **Fusion and ranking** — in the private layer: `semanticScore` reaches
   `HeuristicContextRanker` behind `RankingWeights.semantic`. ✅ plumbing;
   ✅ **weight is `0.30`**, a benchmark-informed starting value — see step 9.
9. **Benchmark and decide** — `ExperimentLogger`/`ExperimentRecord` exist to
   compare modes on real queries; the standalone
   `benchmark/e5_base_benchmark.ipynb` provided the first real measurement
   (see "Choosing the concrete model" above) and picked `0.30` as a starting
   point. ✅ The hybrid-ranking sweep itself now exists
   (`benchmark/hybrid.py`, `benchmark/bootstrap.py`, `benchmark/recommend.py`,
   `benchmark/run_hybrid_sweep.py`, notebook steps 10–13, all unit tested on
   synthetic data in `benchmark/tests/test_hybrid.py`): for each of
   `0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50` it merges lexical and
   semantic candidates into one hybrid ranking, reports Recall@1/5/10/MRR and
   the category breakdown, runs a deterministic bootstrap resample for
   stability, and applies a fixed decision rule to print one
   `RECOMMENDED_SEMANTIC_WEIGHT` (or explicitly fall back to `0.20` if the
   result is ambiguous, never guess between `0.30`/`0.35`). ⏳ **Not the
   final word**: this only becomes a real recommendation once the sweep is
   actually run against real embeddings in Colab; an eventual A/B run
   against this app's own real `ExperimentLogger` data is also still open —
   this step stays open until one of those produces a value confident
   enough to call tuned rather than informed.

Steps 1–5 needed no model, no NDK, and no downloads, and were where the
irreversible API decisions lived — all landed in v0.3.0-alpha1. Steps 6–8
are now done against a real, verified model; step 9 has a real first
answer but is not finished.
