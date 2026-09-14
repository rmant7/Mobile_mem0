"""
Python port of Mobile_mem0's own lexical matching primitive — see
src/main/kotlin/ai/localstudio/memory/MemoryRanking.kt's tokenize()/overlap().

Deliberately NOT a generic baseline like TF-IDF or BM25: the whole point of
this benchmark is to compare semantic retrieval against what this library
*actually* does today, not against some other lexical algorithm nobody ships.

Left out on purpose: MemoryRanking's own recency and scope weighting
(RECENCY_WEIGHT, scopeWeight). Those exist to break near-ties between
otherwise-similar memories in a real conversation history with real
timestamps; this dataset has neither meaningful timestamps nor
scope-durability distinctions; they would just inject arbitrary noise into
what should be a clean measurement of the actual matching signal (word
overlap) against the actual matching signal semantic search proposes to add
or replace (cosine similarity). This file's `overlap()` is the one piece of
MemoryRanking that measures "does this text match this query" at all — that
is what the comparison is about.
"""
import re
from dataclasses import dataclass

# Mirrors MemoryRanking.kt's `MIN_TERM_LENGTH` and `NON_WORD` regex. Kotlin's
# \p{L}\p{N} is full-Unicode; this dataset is Cyrillic/Latin/digits only, so
# an explicit character class is both a faithful and a simpler match than
# pulling in the `regex` package for \p{L}\p{N} support.
_MIN_TERM_LENGTH = 3
_NON_WORD = re.compile(r"[^A-Za-zА-Яа-яЁё0-9]+")


def tokenize(text: str) -> set[str]:
    return {t for t in _NON_WORD.split(text.lower()) if len(t) >= _MIN_TERM_LENGTH}


def overlap(query_terms: set[str], item_terms: set[str]) -> float:
    """Jaccard-style overlap normalised by the query, exactly as MemoryRanking.overlap()."""
    if not item_terms or not query_terms:
        return 0.0
    shared = sum(1 for t in query_terms if t in item_terms)
    return shared / len(query_terms)


@dataclass
class LexicalHit:
    memory_id: str
    score: float


def rank(query_text: str, memories: list[dict]) -> list[LexicalHit]:
    """
    All memories with nonzero lexical overlap, sorted by score descending
    (ties broken by memory id, matching MemoryRanking's own tie-break) — a
    query with no term-3+ tokens at all (MemoryRanking's own `requireOverlap`
    short-circuit) returns nothing, same as production.
    """
    query_terms = tokenize(query_text)
    if not query_terms:
        return []

    hits = []
    for memory in memories:
        score = overlap(query_terms, tokenize(memory["text"]))
        if score > 0.0:
            hits.append(LexicalHit(memory["id"], score))

    hits.sort(key=lambda h: (-h.score, h.memory_id))
    return hits
