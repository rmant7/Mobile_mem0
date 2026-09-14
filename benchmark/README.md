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
  run_benchmark.py     entry point — run this
  e5_base_benchmark.ipynb   the same steps, as a Colab notebook
  tests/
    test_benchmark.py  unit tests — no GGUF download needed
```

## Running in Google Colab (recommended)

1. Upload this `benchmark/` directory to Colab (or clone the repo there),
   with `dataset/`, `lexical.py`, `embedder.py`, `download_model.py`,
   `metrics.py`, and `run_benchmark.py` all in the Colab working directory.
2. Open `e5_base_benchmark.ipynb` and run every cell top to bottom. It
   installs `llama-cpp-python`/`huggingface_hub`, downloads the GGUF
   straight from `cstr/multilingual-e5-base-GGUF` (never from this repo —
   nothing here commits a GGUF anywhere), builds embeddings, runs both
   retrievers, and prints/saves the same tables `run_benchmark.py` does.

No Hugging Face token is needed — this repo is public.

## Running locally

```bash
cd benchmark
pip install -r requirements.txt
python run_benchmark.py
```

First run downloads the GGUF (a few hundred MB) into `huggingface_hub`'s
own cache (`~/.cache/huggingface` by default); later runs reuse it. Pass
`--model-path /path/to/file.gguf` to use an already-downloaded file instead.

Results are printed to the console and saved to `benchmark/results/results.json`
and `results.csv`.

## Running the tests

```bash
cd benchmark
pip install -r requirements.txt
python -m pytest tests/
```

These never download a model — a small hand-computed embedder stands in
for the real one, so they check the dataset, the metrics math, and the
query/passage prefix wiring, not embedding quality itself.

## Extending the dataset

Add a new topic dict to `TOPICS` in `dataset/topics.py` (or a string to
`NEGATIVE_QUERIES`), then run `python dataset/build_dataset.py` to
regenerate `memories.jsonl`/`queries.jsonl`. Ids are derived from each
topic's position and each memory/query's index within it — appending never
renumbers anything that already exists, so old results stay comparable to
new ones for every id that didn't change.

## What this benchmark is not

Per this task's own scope: no user-specific ranking weights, no RRF, no
fusion formula, and no changes to Mobile_mem0's production code live here
or anywhere else as a result of this benchmark. The direct cosine-sort
"semantic ranking" in `run_benchmark.py` exists only to measure retrieval
quality — it is not a candidate implementation for
`ai.localstudio.commercialmemory`'s or Mobile_mem0's own ranking layer.
