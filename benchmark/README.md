# Semantic retrieval benchmark

An objective comparison of lexical vs. semantic retrieval quality for
Mobile_mem0's memory model, using `multilingual-e5-base` as the reference
embedding model. **Standalone**: no Android, no JNI, no Firestore, no
production code changes — see `SEMANTIC_RETRIEVAL_DESIGN.md` at the repo
root for how this fits into that design's step 9 (choosing real fusion
weights from measurement, not a guess).

Runs anywhere with Python 3.10+: a laptop, CI, or (the intended default) a
free Google Colab runtime.

## What it measures

Two retrievers, scored identically over the same dataset:

- **Lexical** — a direct Python port of Mobile_mem0's own
  `MemoryRanking.tokenize()`/`overlap()` (see `lexical.py`'s own doc comment
  for exactly what was and wasn't ported, and why).
- **Semantic (base)** — cosine similarity between `multilingual-e5-base`
  embeddings, `"query: "`/`"passage: "` prefixed per the e5 family's own
  convention, mean-pooled, 768-dimensional.

Metrics: Recall@1/5/10, MRR, and (semantic only — a lexical overlap score
isn't a cosine similarity) average cosine similarity for positive and
negative matches. All broken down by category:
`exact / morphology / synonym / paraphrase / low_overlap / identifier /
negative` — see `dataset/topics.py`'s own doc comment for what each one is
testing. `low_overlap` is the category that most directly answers "is
semantic retrieval adding anything lexical search can't already do."

A third retriever, **hybrid**, extends this with `run_hybrid_sweep.py`: for
every query it merges lexical hits and the top semantic candidates
(deduplicated) and ranks by `lexical_score * LEXICAL_WEIGHT +
semantic_score * semantic_weight` — the same weighted-sum shape as
`ai.localstudio.commercialmemory.HeuristicContextRanker`, but with only
this benchmark's two available signals. It sweeps semantic weight across
`0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50` (every other production
weight held fixed), reports Recall@1/5/10/MRR and the category breakdown
for each weight, runs a deterministic bootstrap resample per weight to
report how stable that weight's MRR/Recall@1 actually are, and applies a
fixed decision rule (`recommend.py`) to print a single
`RECOMMENDED_SEMANTIC_WEIGHT` — this is the tool that produces the number
that goes into `AppContainer.SEMANTIC_RANKING_WEIGHT`.

## Files

```
benchmark/
  dataset/
    topics.py          source data (edit this to extend the dataset)
    build_dataset.py   expands topics.py into the two files below
    memories.jsonl      105 memory items (generated, checked in)
    queries.jsonl        ~100 queries with ground_truth_ids (generated, checked in)
  lexical.py           lexical retrieval (ported from MemoryRanking.kt)
  embedder.py          GGUF embedding via llama-cpp-python
  download_model.py    resolves + downloads the GGUF from Hugging Face
  metrics.py           Recall@k, MRR, cosine similarity, aggregation
  hybrid.py            merges lexical + semantic candidates into one weighted-sum ranking
  bootstrap.py         deterministic bootstrap resampling (stability of MRR/Recall@1)
  recommend.py         fixed decision rule: sweep results -> RECOMMENDED_SEMANTIC_WEIGHT
  run_benchmark.py     entry point — Part A, lexical vs. semantic (base)
  run_hybrid_sweep.py  entry point — Part B/C, hybrid weight sweep + recommendation
  e5_base_benchmark.ipynb   the same steps, as a Colab notebook
  tests/
    test_benchmark.py  unit tests for dataset/lexical/metrics/embedder — no GGUF download needed
    test_hybrid.py     unit tests for hybrid/bootstrap/recommend — no GGUF download needed
```

## Running in Google Colab (recommended)

1. Upload this `benchmark/` directory to Colab (or clone the repo there),
   with `dataset/`, `lexical.py`, `embedder.py`, `download_model.py`,
   `metrics.py`, and `run_benchmark.py` all in the Colab working directory.
2. Open `e5_base_benchmark.ipynb` and run every cell top to bottom. It
   installs `llama-cpp-python`/`huggingface_hub`, downloads the GGUF
   straight from `groonga/multilingual-e5-base-Q4_K_M-GGUF` (never from this repo —
   nothing here commits a GGUF anywhere), builds embeddings, runs both
   retrievers, and prints/saves the same tables `run_benchmark.py` does.
   The final cells then run `run_hybrid_sweep.py` (Part B/C) using the
   already-downloaded model file, printing the full weight sweep, category
   breakdowns, and the `RECOMMENDED_SEMANTIC_WEIGHT` line.

No Hugging Face token is needed — this repo is public.

## Running locally

```bash
cd benchmark
pip install -r requirements.txt
python run_benchmark.py
python run_hybrid_sweep.py
```

First run downloads the GGUF (a few hundred MB) into `huggingface_hub`'s
own cache (`~/.cache/huggingface` by default); later runs (including
`run_hybrid_sweep.py`) reuse it. Pass `--model-path /path/to/file.gguf` to
either script to use an already-downloaded file instead.

`run_benchmark.py` prints/saves to `benchmark/results/results.json`/`.csv`.
`run_hybrid_sweep.py` prints the sweep tables, the category breakdown per
weight, and `RECOMMENDED_SEMANTIC_WEIGHT = X`, and saves the full sweep
(including every weight's bootstrap stats and the recommendation's own
reasoning) to `benchmark/results/hybrid_sweep_results.json`.

## Running the tests

```bash
cd benchmark
pip install -r requirements.txt
python -m pytest tests/
```

`test_benchmark.py` never downloads a model — a small hand-computed
embedder stands in for the real one, so it checks the dataset, the metrics
math, and the query/passage prefix wiring, not embedding quality itself.
`test_hybrid.py` covers `hybrid.py`/`bootstrap.py`/`recommend.py` the same
way: all three are pure functions over plain lists/dicts, so the actual
merge-ranking, resampling, and recommendation-decision logic is fully
tested without a GGUF or llama_cpp at all.

## Extending the dataset

Add a new topic dict to `TOPICS` in `dataset/topics.py` (or a string to
`NEGATIVE_QUERIES`), then run `python dataset/build_dataset.py` to
regenerate `memories.jsonl`/`queries.jsonl`. Ids are derived from each
topic's position and each memory/query's index within it — appending never
renumbers anything that already exists, so old results stay comparable to
new ones for every id that didn't change.

## What this benchmark is not

Per this task's own scope: no new embedding models, no RRF, no learned
fusion, and no changes to Mobile_mem0's own production code live here or
anywhere else as a result of this benchmark. The direct cosine-sort
"semantic ranking" in `run_benchmark.py`, and the weighted-sum
`hybrid_rank()` in `hybrid.py`, exist only to measure retrieval quality —
neither is a candidate implementation to copy into Mobile_mem0's own
sources. `hybrid.py` mirrors the *shape* of
`ai.localstudio.commercialmemory.HeuristicContextRanker` (a weighted sum,
sorted descending, no threshold) so the sweep measures something
structurally close to what production actually does, but the only thing
this benchmark ever changes in the app repo is the single
`SEMANTIC_RANKING_WEIGHT` number `run_hybrid_sweep.py` recommends — never
the ranking formula itself, and never anything inside Mobile_mem0.
