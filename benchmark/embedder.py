"""
GGUF embedding inference via llama-cpp-python — a minimal GGUF-compatible
runtime with prebuilt wheels for Colab and most laptops, so nothing here
needs the Android/JNI build this app's own on-device path uses.
"""
from __future__ import annotations

from llama_cpp import Llama

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# llama.cpp's own llama_pooling_type enum (include/llama.h): NONE=0, MEAN=1,
# CLS=2, LAST=3 — the same ordinals Mobile_mem0's Android-side JNI bridge
# documents (ai.localstudio.app.llama.LlamaBridge.EmbeddingPooling). e5-family
# models are trained for mean pooling over token embeddings; this benchmark
# exists specifically to test that family, so it isn't a configurable option
# here the way it is on the JNI side — there's only one model family in scope.
POOLING_TYPE_MEAN = 1


class E5Embedder:
    """Wraps one loaded GGUF context. Not thread-safe — matches llama.cpp's own single-sequence-at-a-time contract."""

    def __init__(self, model_path: str, n_ctx: int = 512, n_threads: int | None = None, verbose: bool = False):
        self._llm = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=POOLING_TYPE_MEAN,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )
        self.dimension = self._llm.n_embd()

    def _embed(self, text: str) -> list[float]:
        vector = self._llm.embed(text)
        # embed() returns a flat vector when the context pools (pooling_type
        # != NONE, set above); some llama-cpp-python versions have returned a
        # one-element list of per-token vectors instead even when pooled.
        # Guarded rather than assumed: silently zipping the wrong shape into
        # cosine similarity produces a confidently wrong number, not an error.
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        return _l2_normalize(vector)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(QUERY_PREFIX + text)

    def embed_passage(self, text: str) -> list[float]:
        return self._embed(PASSAGE_PREFIX + text)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]
