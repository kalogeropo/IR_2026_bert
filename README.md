# Information retrieval with BERT-family encoders and LLMs

An explanatory notebook on **real IR benchmarks**: BM25 → dense bi-encoder → reciprocal rank fusion → an LLM answering from retrieved
passages with citations that are checked against the sources. Every number comes from actual runs against the official relevance judgments
of **BEIR SciFact** (Part A) and **BEIR NFCorpus** (Part B).

* `LLM_BERT_Information_Retrieval_Lab.ipynb`: the notebook (executed)
* `irlab/`: the helper package the notebook imports (data loading, BM25, dense retrieval, fusion, generation, measures, caching)
* [`DOCUMENTATION.md`](DOCUMENTATION.md): choices, pipeline, caching, results, runtimes, limitations, links

```bash
uv sync
uv run jupyter lab      # open the notebook and run all cells (the first run downloads data/models and fills cache/)
```
