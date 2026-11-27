# Information retrieval with BERT-family encoders and LLMs: documentation

This document explains the notebook `LLM_BERT_Information_Retrieval_Lab.ipynb`: how it was built, what each experiment shows, how to run and
adapt it, and what its results do and do not support. It is meant to be read **without** reading the code. Every number quoted here was
produced by the notebook and saved under `results/`. Where a number comes from elsewhere (a paper, a dataset page), the source is named.

Contents

1. [What is delivered](#1-what-is-delivered)
2. [Audit of the original notebook](#2-audit-of-the-original-notebook)
3. [Collections: choice, verification, licences, loading](#3-collections)
4. [Models: choice and caveats](#4-models)
5. [Evaluation measures and protocol](#5-evaluation-measures-and-protocol)
6. [How the pipeline works, step by step](#6-how-the-pipeline-works-step-by-step)
7. [Experiments and their results](#7-experiments-and-their-results)
8. [Runtime and hardware](#8-runtime-and-hardware)
9. [Installation, data, reproduction](#9-installation-data-reproduction)
10. [Adapting the notebook to another collection](#10-adapting-the-notebook-to-another-collection)
11. [Limitations, inconclusive results, what was not run](#11-limitations-inconclusive-results-what-was-not-run)
12. [References and official documentation](#12-references-and-official-documentation)

---

## 1. What is delivered

| Path | What it is |
|---|---|
| `LLM_BERT_Information_Retrieval_Lab.ipynb` | The revised notebook, in English, **executed** top to bottom. Part A (Sections 1–8) is the pipeline on SciFact; Part B (B1–B8) contains further experiments. The markdown explains each step, its inputs and outputs, and what the output shows. Section 1 introduces the main objects (`LabConfig` and its fields, `Lab` and its methods, the *run*, queries and qrels dictionaries, `Collection`), and every code cell is preceded by a "Code in the next cell" block that lists the classes, functions and parameters it uses. |
| `DOCUMENTATION.md` | This file. |
| `irlab/` | Small Python package with the reusable code the notebook calls: data loading with integrity checks (`data.py`), measures (`metrics.py`), BM25 (`lexical.py`), bi-encoders and the embedding cache (`dense.py`), fusion (`fusion.py`), LLM prompting and citation checks (`generation.py`), diagnostics (`analysis.py`), printing (`display.py`), and the memoising `Lab` object (`pipeline.py`). **All settings** are in `irlab/config.py`, and the notebook's first code cell overrides them in one place. |
| `results/` | CSV/JSON outputs written by the notebook when it runs: main tables per collection, every tuning table, bootstrap intervals, LLM answers, cost tables, and the software environment. |
| `pyproject.toml`, `uv.lock` | Exact dependency versions. |

The repository contains only this notebook, the `irlab/` package and the project files (`pyproject.toml`, `uv.lock`, `README.md`, this file). The LLM questions and the query reformulations are defined inside the notebook. `data/`, `cache/` and `results/` are created when the notebook runs.

**Scope decisions made during revision.** The notebook is written as an explanatory document, not as a guided lesson: it has no exercises,
time allocations or instructor notes. Chunking (splitting documents into windows) was evaluated and removed (§11). A cross-encoder reranking stage was implemented and evaluated in an earlier version and then
**removed at the request of the course owner**. Its code, cached scores and results were deleted. §11 summarises what that earlier evaluation
had shown.

---

## 2. Audit of the original notebook

The original notebook used 16 invented "student services" announcements, 8 queries with one relevant document each, and Hit@3/MRR@3.
Its retrieval cells were executed (with `rank-bm25`, as in the original) before revision. Problems found, with what execution showed:

| # | Problem | Evidence from execution / inspection | What the revision does |
|---|---|---|---|
| 1 | **Not a real test collection.** Documents, queries and judgments were all written by the notebook author. The documents are effectively written *for* the queries, so the metrics measure the author's choices, not system behaviour. | Dense retrieval scored Hit@3 = 1.0 and MRR@3 = 0.938 on 8 queries, numbers with no evidential value. | Two public benchmark collections with independent queries and qrels (§3). |
| 2 | **A document built to make BM25 fail.** D16 is an "orientation flyer" listing *password, book, deadline, Wi-Fi, library* with no content. | BM25 ranked D16 first for "How long can I keep a paper book?". | No manufactured documents. Examples are chosen by fixed rules from real runs (§7). |
| 3 | **Zero-score documents counted as retrieved, ordered by list position.** `order()` sorted all 16 documents, so documents with BM25 score 0 filled ranks 2–3 in index order (D01, D02). | For "Who can approve a late assignment?" only D05 had a non-zero score, and D01 and D02 followed "by score" 0.000. These ties feed Hit@3, MRR@3 and RRF. | BM25 returns only documents with score > 0. All ties are broken deterministically, and RRF ignores documents a system did not retrieve. |
| 4 | **Morphology presented as paraphrase.** No stemming, so "book" never matched "books". | The tokens of "How long can I keep a paper book?" were `long, keep, paper, book`, and D03 says "printed *books*". | Snowball stemming, with the preprocessing choice measured in B5 (+2.3 to +2.6 nDCG points on dev). |
| 5 | **No development/test separation.** The reranking depth, the prompt and the examples were all chosen and evaluated on the same 8 queries. | — | A dev split for every choice, a single test evaluation, and a stated protocol (§5). |
| 6 | **Unsupported claim** that "a plain pretrained BERT is not automatically a good retriever", asserted but never shown. | — | Measured: plain BERT nDCG@10 = 0.133 vs. 0.713 for a retrieval-trained BERT-family encoder on SciFact, with cosine statistics that explain why (Section 5). |
| 7 | **Synthetic unanswerable question.** The Sunday-hours question was answerable only because D02 was written to say "weekend hours are not listed". | — | Unsupported questions drawn from the collection: one whose judged abstract carries no rationale in SciFact, and one outside the corpus. Answers are checked against SciFact's own rationale labels. |
| 8 | **Generic notes** ("there is no guaranteed winner", "add more documents"). | — | Each step's markdown states what the output shows, with the specific values. |
| 9 | **Reproducibility.** Unpinned `%pip install`, no seeds, no recorded versions, no cost separation, results recomputed every run. | — | Locked dependencies, pinned dataset and model revisions, fixed seeds, greedy decoding, on-disk caches keyed by every setting, setup vs. query vs. generation timing. |
| 10 | Greek optional section with a multilingual model on 3 sentences, without evaluation. | — | Removed, since the brief asks for English and retrieval evaluation. |

Kept from the original: the separation of BM25, bi-encoder and LLM; RRF instead of adding incompatible scores; and the warning that a citation
or an abstention instruction is not a guarantee.

---

## 3. Collections

### 3.1 Why these two

The goal was two real collections that can be processed on a laptop and that **differ in the factors that drive retrieval behaviour**, while
keeping the document type comparable, so that differences can be attributed to queries and judgments:

| | SciFact (BEIR) | NFCorpus (BEIR) |
|---|---|---|
| Domain | biomedical research abstracts | nutrition and medical research abstracts |
| Documents | 5,183 (median 204 words, title + abstract) | 3,633 (median 237 words) |
| Query style | an expert-written **claim**, median 12 words, often with gene, drug and disease names | a NutritionFacts.org **page title or topic tag**, median 2 words (e.g. *Peoria*, *chickpeas*, *Unsafe at Any Feed*) |
| Judged relevant per query | 1.13 (test), binary | 38.2 (test), graded 1–2 |
| Splits used | dev = BEIR `train` (809), test = BEIR `test` (300) | dev = `dev` (324), test = `test` (323) |
| Role | Part A | B1 and all two-collection experiments |

SciFact rewards precise matching of technical terms and penalises vocabulary mismatch. NFCorpus has short, under-specified queries, many loosely
judged documents, and some queries that cannot be answered from content at all (B1). Document *length* is similar in the two, so length is
examined only through the effect of the 512-token truncation limit (§11). Larger BEIR collections (e.g. TREC-COVID with 171k documents, FiQA with 57k) were not
chosen, because encoding them is impractical on ordinary CPUs. ArguAna (8.7k documents, very long argument queries) would add a long-query
condition but has no dev split, so it could only be used without any tuning (§10 explains how to add it).

### 3.2 What was verified, and how

* **Sizes.** The loader checks the number of documents, queries, judged queries and judgment pairs per split against the BEIR documentation and
  stops if they differ. The BEIR README lists SciFact: 300 test queries, 5K documents, 1.1 relevant per query; NFCorpus: 323 test queries,
  3.6K documents, 38.2 relevant per query. The loaded counts are SciFact 5,183 docs, 1,109 queries, test 300/339 pairs, train 809/919; NFCorpus
  3,633 docs, 3,237 queries, dev 324/11,385, test 323/12,334, train 2,590/110,575. The label distributions are SciFact {1: 339} (test) and
  NFCorpus test {1: 11,758, 2: 576}.
* **What SciFact's BEIR splits are.** The original SciFact release (`data.tar.gz`, sha256 `11c62128…`) was downloaded and its ids compared: the
  300 BEIR test queries are exactly the 300 original **dev** claims, and the 809 BEIR train queries are the original train claims. The original
  SciFact README states that the test-set labels are not public.
* **What SciFact's qrels mean.** The 339 BEIR test pairs equal the claims' `cited_doc_ids`. Only 209 of them carry an evidence rationale
  (SUPPORT or CONTRADICT) in the original release. So BEIR "relevant" = cited, whether or not the abstract supports, contradicts or does not
  settle the claim. Of the 300 dev claims, 124 have SUPPORT evidence, 64 CONTRADICT and 112 none.
* **NFCorpus relevance levels.** The NFCorpus page defines three levels: a direct link from the NutritionFacts article to the document (highest),
  a link via another article (middle), and a shared topic tag (lowest). The BEIR qrels contain the labels 2 and 1 only. BEIR does not document
  how the three original levels were mapped, and the original archive could not be inspected (its download link returned an HTML page), so
  only 2 > 1 > unjudged is relied on. BEIR's NFCorpus has 3,633 documents, whereas the original page lists 9,964. The BEIR version is a subset,
  and its selection criterion was not verified.
* **Loading.** The canonical BEIR zip files (`public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{scifact,nfcorpus}.zip`, MD5s in the
  BEIR README) were **unreachable** (connection time-out) during preparation. The loader therefore uses the BEIR mirror on the Hugging Face Hub
  (`BeIR/scifact`, `BeIR/scifact-qrels`, `BeIR/nfcorpus`, `BeIR/nfcorpus-qrels`), **pinned by commit hash** in `irlab/config.py`. The size
  checks above match the BEIR documentation exactly. The files could not be compared byte-for-byte with the zips.

### 3.3 Licences

* **SciFact**: claims and annotations CC BY 4.0, abstracts (from S2ORC) ODC-By 1.0, per `LICENSE.md` in the SciFact repository. The Hugging Face
  mirror's card says CC BY-SA 4.0. That is the mirror's metadata, and the original licences are the ones to cite.
* **NFCorpus**: "free for academic purposes" according to its homepage. Other uses of NutritionFacts.org content are subject to that site's terms.
  Cite Boteva et al. (2016).
* **BEIR code**: Apache-2.0. The BEIR README states that users must check each dataset's licence themselves.

Use in a university course fits these terms. Redistributing the data in modified form should follow the attribution requirements above.

---

## 4. Models

| Role | Model (pinned revision in `irlab/config.py`) | Size | Licence | Why |
|---|---|---|---|---|
| main bi-encoder | `BAAI/bge-small-en-v1.5` | 33M, 384-d, BERT architecture | MIT | Best mean **dev** nDCG@10 over both collections among the five encoders tried (0.537), also the fastest. Chosen on dev only. |
| negative control | `google-bert/bert-base-uncased` + mean pooling | 110M, 768-d | Apache-2.0 | Shows that a pretrained BERT without retrieval training fails (Section 5). |
| comparison | `sentence-transformers/msmarco-distilbert-base-tas-b` | 66M, dot product | Apache-2.0 | MS MARCO-only training, so a clean zero-shot transfer. Published BEIR numbers allow a sanity check. Used for the similarity-function experiment. |
| comparison | `sentence-transformers/msmarco-distilbert-base-v4` | 66M, cosine | Apache-2.0 | The model of the original notebook. |
| comparison | `intfloat/e5-small-v2` | 33M, needs `query:`/`passage:` prefixes | MIT | Shows that prompts are part of a model's interface. |
| generator | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Apache-2.0 | Runs without an API key on a GPU (≈3.5 GB in bf16) or a CPU (≈7 GB RAM in fp32). It is the model of the original notebook, now pinned. Fallback: `Qwen/Qwen2.5-0.5B-Instruct` (revision in the notebook). |

**Training-data caveats.** The E5 paper states that its pre-training data (CCPairs) includes (title, abstract) pairs from S2ORC, the source of
SciFact's abstracts, so E5 may have seen these *documents* (not the claims or qrels). BGE's English training data is very large and mixed, and
it could not be verified at document level whether it excludes these abstracts. The two MS MARCO models were trained on web-search data only.

**Why not a larger model.** The models were chosen to run on ordinary hardware. Larger embedding models (e.g. 335M-parameter "large"
encoders) would raise encoding time substantially on CPUs. Their effect on these collections was not measured, and no claim is made about them.

---

## 5. Evaluation measures and protocol

**Measures** (implemented in `irlab/metrics.py`, checked against `ir_measures`/pytrec_eval in Section 3 of the notebook):

* **nDCG@10**, BEIR's main measure. Gain = qrels label (linear, as in trec_eval), discount log₂(rank + 1), ideal ranking from all judged documents
  of the query. It rewards putting the most relevant documents first and uses NFCorpus' grades.
* **MRR@10**, the reciprocal rank of the first relevant document (label ≥ 1). It answers "how soon is the first hit?".
* **Recall@10** and **Recall@100**. R@100 describes the candidate pool of a first stage, and R@10 what a reader of the first page sees.

Conventions: unjudged documents count as non-relevant, every judged query is averaged, and a query with an empty result scores 0.

**Protocol.**
1. A-priori defaults are fixed before looking at any result: BM25 k1 = 1.2, b = 0.75 (textbook/Elasticsearch defaults), stopwords + Snowball
   stemming, RRF k = 60 (Cormack et al., 2009), and 3 sources for the LLM.
2. The main dense model was chosen on **dev** (B4).
3. Every tuning experiment varies **one** setting, chooses on dev nDCG@10, and evaluates the a-priori and the chosen value **once** on test.
4. Test results of the a-priori configuration are therefore unbiased estimates, and test results of dev-chosen settings are honest estimates of
   what tuning gains on unseen queries.
5. Uncertainty: paired bootstrap over queries (10,000 resamples, seed 13), 95% percentile intervals, without multiple-comparison correction (B8).

**Why scores from different collections must not be compared as model rankings.** SciFact asks "find the abstract cited for this claim", with
about one judged document per query. NFCorpus asks "find the abstracts linked to this consumer-health page or tag", with dozens of judged
documents, some related only by a tag. An nDCG@10 of 0.36 on NFCorpus is not "worse" than 0.73 on SciFact: the two numbers measure different
tasks with different judgment processes. Only the **ordering of systems within a collection** is comparable.

**Answer quality is assessed separately.** Retrieval measures say nothing about whether the LLM's sentences are supported. The notebook reports
(a) mechanical checks: cited ids were provided, sentences without a citation, abstention string present; and (b) a reading of each answer against
its cited passages, using the SciFact rationale labels as a reference.

---

## 6. How the pipeline works, step by step

1. **Load and check data** (`irlab/data.py`). Pinned BEIR files are downloaded once to `data/beir/`. Documents become `title + " " + text`, as in
   BEIR's reference implementation. Counts are checked, and dev and test queries are checked to be disjoint.
2. **BM25** (`irlab/lexical.py`). Text is lowercased and split into `[a-z0-9]+` tokens (so `IL-6` → `il`, `6`, and Greek letters are dropped).
   Optionally 33 English stopwords are removed and Snowball stemming applied, and the result is indexed with `bm25s` (Lucene BM25 variant).
   Each query is scored against all documents. The top 100 with score > 0 are kept, with ties broken by document position.
3. **Dense retrieval** (`irlab/dense.py`). Every document is encoded once with the bi-encoder (prompt and similarity function as the model card
   specifies; texts truncated at 512 tokens), and the vectors are cached on disk under a key that contains the model revision, the maximum
   length and a hash of the corpus. A query is encoded with its query prompt and compared with every document vector (cosine for cosine models,
   dot product for TAS-B), and the top 100 are kept. On these corpora exhaustive search takes milliseconds. A larger collection would need an
   approximate nearest-neighbour index.
4. **Fusion** (`irlab/fusion.py`). Reciprocal rank fusion: score(d) = Σ 1/(k + rankᵢ(d)) over the systems that retrieved d in their top `depth`.
   Optionally min-max score fusion with weight α (B6).
5. **Generation** (`irlab/generation.py`). The top `n` RRF results for the question (the first 300 words of each) are placed in a prompt with
   their document ids. The system prompt asks for a citation after every sentence and an exact refusal string when the passages are
   insufficient. Decoding is greedy (`do_sample=False`) for repeatability. Citations are parsed, and the mechanical checks are reported next to
   the answer.
6. **Evaluation and logging** (`irlab/metrics.py`, `irlab/pipeline.py`). Every run is evaluated against the qrels of its split, tables are saved to
   `results/`, and every step records its time as *setup*, *query* or *generate*.

### 6.1 Caching

Expensive results are stored and cheap ones recomputed. There are three layers:

| Layer | Location | Contents | Reused when | Recomputed when |
|---|---|---|---|---|
| Downloads | `data/beir/<collection>/`, `data/scifact_release/`, `~/.cache/huggingface/` | collection files at a pinned commit; the original SciFact release (sha256-checked); model weights | the files are present | the folder is deleted |
| Corpus embeddings | `cache/embeddings/<model>__<revision>__len<max>__<corpus hash>.npy` + `.json` | one float32 vector per document; metadata (model, revision, device, encoding seconds, number of truncated documents) | the file name matches, i.e. same model, revision, length limit and corpus text | any of the four changes, because the name changes. Stale vectors are therefore never reused |
| In memory | the `Lab` object | every BM25 and dense run keyed by all its parameters (collection, hash of the query set, model, similarity, length, depth, k1, b, preprocessing); BM25 indexes; loaded models. RRF runs are not stored: they are recomputed from the stored BM25 and dense runs in milliseconds | for the rest of the Python session | after a kernel restart |

The corpus hash is a SHA-256 over all document ids and texts in order (`Collection.fingerprint()`). Query embeddings, evaluation measures and LLM
answers are not cached. Each is cheap or must be regenerated. Because runs are memoised in memory, a Part B cell never depends on a variable
created by another Part B cell, and re-running a cell returns the same result. The notebook explains each layer where it is first used
(Sections 1, 2, 5 and 7) and lists the embedding files with their metadata in Section 5. Deleting `cache/` is always safe: the next run recomputes
and stores the vectors again.

---

## 7. Experiments and their results

All tables are copied from the executed notebook (`results/*.csv`). nDCG = nDCG@10.

### 7.1 Part A (SciFact, test split, a-priori settings)

| System | nDCG@10 | MRR@10 | R@10 | R@100 |
|---|---|---|---|---|
| BM25 (stop+stem, k1 = 1.2, b = 0.75) | 0.686 | 0.651 | 0.817 | 0.921 |
| bert-base-uncased, mean pooling | 0.133 | 0.115 | 0.211 | 0.482 |
| bge-small-en-v1.5 | 0.713 | 0.682 | 0.836 | 0.942 |
| RRF(BM25, bge-small), k = 60 | **0.728** | **0.695** | **0.855** | **0.968** |

**What it shows.**
* *Plain BERT vs. a retrieval-trained encoder.* The architecture is similar and the result is not: 0.133 vs. 0.713. For a rule-picked claim (42),
  plain BERT's cosine to the relevant abstract is 0.819, but its *median* cosine to all 5,183 abstracts is 0.763, and 223 abstracts score higher.
  bge-small gives 0.844 against a median of 0.592, at rank 1. Mean-pooled pretrained embeddings make almost everything look similar (cf.
  Ethayarajh, 2019). Contrastive training on query–document pairs is what creates a usable similarity.
* *BM25 vs. dense, per query.* BM25 ranks the first relevant abstract higher on 58 claims, bge-small on 80, and 162 are tied. Rule-picked cases
  show the mechanisms: a single rare exact term (`onchocerciasis`, df = 1) wins for BM25, while a paraphrase (*does not vary across genetically
  identical cells* vs. *stochastic gene expression*) and a token-split gene name (`Bcl2` vs. `BCL-2` → `bcl`, `2`) win for the dense model.
* *Fusion.* RRF has the best value on every measure, and its R@100 exceeds both inputs'. Per claim it is better than the better input on 21
  claims, equal on 208 and worse on 71. Its cost is visible in rule-picked cases: a document found by only one system, even at rank 1, drops to
  rank 16 (claims 1363 and 452), because a single contribution 1/61 is outweighed by documents both systems rank in their top ~60.

### 7.2 Part B

**B1 · NFCorpus, same pipeline (test).** BM25 0.323, bge-small 0.344, RRF **0.363** (nDCG@10). MRR@10 is 0.53–0.57 and R@100 0.25–0.31, while the
best achievable R@100 is 0.965 because some queries have more than 100 relevant documents. The ordering of systems is the same as on SciFact.
Some queries cannot be solved from content at all: *Peoria* occurs in no abstract, and its relevant documents are linked only by the site's tag
structure.

**B2 · Query groups (SciFact test, diagnostic).**

| group | claims | BM25 | bge-small | RRF |
|---|---|---|---|---|
| overlap with relevant abstract < 0.5 | 98 | 0.339 | **0.454** | 0.445 |
| overlap 0.5–0.8 | 125 | 0.808 | 0.789 | **0.818** |
| overlap ≥ 0.8 | 77 | 0.931 | 0.918 | **0.941** |
| rarest term unknown to index (df = 0) | 57 | 0.649 | 0.710 | **0.730** |
| rarest term df 1–5 | 122 | **0.747** | 0.711 | 0.730 |
| rarest term df ≥ 6 | 121 | 0.642 | 0.716 | **0.725** |

The dense model's advantage is concentrated in vocabulary-mismatch claims, and BM25's in claims with rare terms. 27 claims are missed in the top
10 by all three systems. The gallery prints two rule-picked examples per group (lowest overlap, rare term, unknown term, longest, shortest,
all-fail). Negation pairs (e.g. 53 SUPPORT / 51 CONTRADICT, sharing abstract 45638119) get the shared abstract at the same rank: retrieval finds
topic, not truth.

**B3 · Controlled reformulations (23 variants of 8 claims, written before running).** bge-small kept the relevant abstract in its top 3 for all
23. BM25 dropped out of the top 10 for 4 (two paraphrases, a keyword query, and a long question that lost the rare word *microerythrocyte*).
Expected BM25 failures for the spelling correction, the abbreviation *CDCA* and the synonym *fractalkine receptor* did **not** happen, because
the remaining terms sufficed. RRF kept the abstract in the top 10 for all variants except one (rank 15). All variants inherit the source claim's
qrels, an assumption stated in the notebook.

**B4 · Encoders (dev for selection, test for report), nDCG@10.**

| encoder | SciFact dev | SciFact test | NFCorpus dev | NFCorpus test |
|---|---|---|---|---|
| BM25 (reference) | 0.692 | 0.686 | 0.290 | 0.323 |
| bert-base, mean pooling | 0.124 | 0.133 | 0.043 | 0.043 |
| msmarco-distilbert-v4 | 0.556 | 0.529 | 0.260 | 0.268 |
| TAS-B | 0.646 | 0.643 | 0.293 | 0.320 |
| e5-small-v2 | 0.708 | 0.688 | 0.322 | 0.326 |
| **bge-small-en-v1.5** | **0.747** | **0.713** | **0.328** | **0.344** |

The dev order equals the test order in both collections. Our TAS-B test scores reproduce BEIR Table 2 (0.643 and 0.319). In the similarity
experiment, TAS-B with cosine instead of its trained dot product loses 11 points on SciFact dev (0.646 → 0.536) and 4 on NFCorpus dev
(0.293 → 0.255). For bge-small, cosine and dot product are identical because its vectors are normalised.

**B5 · BM25.** Stemming: +0.026 (SciFact dev), +0.023 (NFCorpus dev). Removing stopwords slightly *hurts* (by 0.001–0.003). The k1 × b grid
(k1 ∈ {0.5, 0.9, 1.2, 1.5, 2.0}, b ∈ {0.3, 0.5, 0.75, 0.9}) is flat: SciFact dev 0.683–0.695, NFCorpus dev 0.283–0.292. Test effect of the
dev-chosen settings: SciFact −0.0015 [−0.009, +0.006] (no stopword removal, k1 = 1.2, b = 0.75), NFCorpus +0.004 [+0.0005, +0.008] (no stopword
removal, k1 = 1.5, b = 0.5). Tuning gained a fraction of a point at most, and the a-priori defaults were reasonable.

**B6 · Fusion.** RRF k: dev prefers k = 1 on SciFact (0.758 vs. 0.750 at 60) and k = 30 on NFCorpus (0.340 vs. 0.339). On test the chosen k changes
nDCG@10 by less than 0.001. Fusion depth 10 instead of 100 lowers R@100 (SciFact dev 0.902 vs. 0.970; NFCorpus 0.178 vs. 0.320). Min-max score
fusion: α = 0.3 is best on dev in both collections, and on test it scores 0.729 vs. RRF 0.728 (SciFact) and 0.361 vs. 0.363 (NFCorpus). RRF's merit
is robustness without tuning.

**B7 · Number of LLM sources (1, 3, 5).** S1 is correct in all three settings. S2 **abstains with 1 source**, although that source contradicts the
premise, and with 3 and 5 it gives the self-contradictory "Yes, open-access articles are … cited more". With 5 it adds correct figures from the
passage in an uncited second sentence, which the mechanical check flags. U1 says "Yes" in all three and cites an unjudged abstract on white-fat
browning, even when the judged abstract is among the 5 sources. U2 abstains in all three. More context did not improve faithfulness here. These
are 12 case studies, not an evaluation of the model.

**B8 · Paired bootstrap (test, nDCG@10 difference, 95% CI).**

| collection | comparison | mean diff | 95% CI | wins/losses/ties |
|---|---|---|---|---|
| SciFact | bge-small − BM25 | +0.027 | [−0.005, +0.058] | 71/50/179 |
| SciFact | RRF − bge-small | +0.015 | [−0.007, +0.036] | 57/38/205 |
| SciFact | RRF − BM25 | +0.042 | [+0.021, +0.063] | 74/23/203 |
| NFCorpus | bge-small − BM25 | +0.021 | [+0.001, +0.042] | 121/90/112 |
| NFCorpus | RRF − bge-small | +0.020 | [+0.006, +0.033] | 115/81/127 |
| NFCorpus | RRF − BM25 | +0.041 | [+0.027, +0.056] | 122/59/142 |

With 300 SciFact claims, the dense-vs-BM25 and RRF-vs-dense gains cannot be distinguished from zero. The hybrid's gain over BM25 is clear on
both collections.

### 7.3 LLM answers in Part A

| id | question type | retrieved (top 3, RRF) | answer | assessment from reading the cited passage |
|---|---|---|---|---|
| S1 | supported | judged abstract 45638119 at rank 1 (SUPPORT) | "Yes, ALDH1 expression is associated with a poorer prognosis in breast cancer. [45638119]" | supported |
| S2 | supported, premise contradicted | judged abstract 27768226 at rank 1 (CONTRADICT) | "[27768226] Yes, open-access articles are more immediately recognized and cited than non-open-access articles." | the paraphrase is supported, but "Yes" answers the question wrongly: self-contradictory |
| U1 | no rationale in the judged abstract | judged 970012 at rank 4, not shown; unjudged 29381091 first | "Yes [29381091]." | partly supported at most (the passage reports cold-induced browning of *white* fat) |
| U2 | outside the collection | FDA postmarket safety; CRISPR in rats | "INSUFFICIENT EVIDENCE IN SOURCES" | correct abstention |

All mechanical checks pass for S1–U2 (every cited id was provided). Only reading reveals the problems in S2 and U1, which is why retrieval
metrics and answer assessment are kept apart.

---

## 8. Runtime and hardware

**Preparation machine:** 24-core x86-64 CPU, 30 GB RAM, NVIDIA RTX 5070 Ti (16 GB), Linux, Python 3.13.12, PyTorch 2.15.0+cu132,
sentence-transformers 6.3.0, transformers 5.24.1, bm25s 0.3.16 (full list in `results/environment.json` and `uv.lock`).

**Measured costs** (SciFact, 5,183 abstracts, 300 test claims). GPU values come from the notebook runs in this directory. CPU values come from
a clean run with `IRLAB_DEVICE=cpu`: a fresh copy of the repository, a new virtual environment built from `uv.lock`, and an empty Hugging Face
cache, so every model was downloaded and every embedding computed on the CPU.

| step | kind | GPU (RTX 5070 Ti) | CPU (24 cores) |
|---|---|---|---|
| load collection (download on first use) | setup | ≈ 4 s | 4.9 s |
| build BM25 index (5,183 abstracts) | setup | 0.4–0.5 s | 0.5–0.7 s |
| BM25 search, all 300 claims | query | 0.05 s (0.2 ms/claim) | 0.05 s (0.2 ms/claim) |
| encode corpus, bge-small (5,183 abstracts) | setup, cached | 8.1 s (1.6 ms/doc) | 145 s (28 ms/doc) |
| encode corpus, bert-base (5,183 abstracts) | setup, cached | 21.1 s (4.1 ms/doc) | 424 s (82 ms/doc) |
| encode + search, bge-small, all 300 claims | query | 0.25 s (0.8 ms/claim) | 0.73 s (2.4 ms/claim) |
| RRF fusion, all 300 claims | query | 0.01 s | 0.01 s |
| load LLM, first time (includes the ≈3 GB download) | setup | 95 s | 96 s |
| load LLM, weights already downloaded | setup | 1–2 s | 1.2 s |
| one LLM answer (≈1,000–1,300 prompt tokens, 10–28 new tokens) | generate | 0.1–0.4 s | 4–8 s |

Totals:
* **CPU, empty caches:** the setup steps of Part A (downloads, both corpus encodings, LLM download) took **726 s (≈12 min)**. Afterwards Part A
  executed in **36 s**. (This was measured with a preparation script that ran exactly Part A's setup steps and was removed when the repository was
  cleaned. Running the notebook once from empty caches performs the same work.)
* **GPU, warm caches:** the whole notebook (Part A + Part B, 32 code cells) executes in **≈55 s**.
* Part B on a CPU from empty caches additionally encodes both corpora with the three remaining encoders. Extrapolating from the per-document
  times above (not measured as a whole), expect roughly 10–20 more minutes.

**CPU and GPU give the same results.** In the clean CPU run every embedding was computed on the CPU, yet the SciFact test table agreed with
the GPU run to all 4 printed decimals, and all four LLM answers of Section 7 were identical.

**Practical guidance.**
* The first full run of the notebook downloads everything and fills the embedding cache. Later runs reuse it. To avoid the first-run cost on a
  slow machine, copy `cache/` and the Hugging Face cache (`~/.cache/huggingface`) from a machine where the notebook already ran. Copying works
  because the cache keys contain only model revisions, lengths and corpus hashes, not paths.
* **Disk:** about 3.5 GB of model weights for Part A (Qwen2.5-1.5B 2.9 GB, bert-base 0.42 GB, bge-small 0.13 GB), plus 0.64 GB for the
  three further encoders of Part B. Collections: 14 MB in `data/`. Embedding cache for everything: 104 MB in `cache/` (23 MB for Part A alone).
* **Memory:** the LLM needs ≈3.5 GB of GPU memory (bf16) or ≈7 GB of RAM (fp32 on CPU). Everything else fits in 2 GB.
* **Other machines (e.g. Colab):** not tested. The steps are the same: clone the repository, `pip install -e .`, run the notebook.

---

## 9. Installation, data, reproduction

```bash
git clone <this repository> && cd IR_2026_bert
uv sync                                      # exact versions from uv.lock (or: pip install -e . with pyproject.toml)
uv run jupyter lab                           # open LLM_BERT_Information_Retrieval_Lab.ipynb and run all cells
# or execute headless, top to bottom, in a fresh kernel:
uv run jupyter nbconvert --to notebook --execute --inplace LLM_BERT_Information_Retrieval_Lab.ipynb
# force the CPU:
IRLAB_DEVICE=cpu uv run jupyter nbconvert --to notebook --execute --inplace LLM_BERT_Information_Retrieval_Lab.ipynb
```

**Environment variables.** None is required. Each is read once when `irlab` is imported:

| variable | default | meaning |
|---|---|---|
| `IRLAB_DEVICE` | `auto` | `auto` = GPU if PyTorch sees one, else CPU; `cpu`; `cuda` (error if no GPU is visible). The device used is printed as `device_used`. |
| `IRLAB_DATA_DIR` | `data/` | downloaded collections |
| `IRLAB_CACHE_DIR` | `cache/` | embedding cache |
| `IRLAB_RESULTS_DIR` | `results/` | CSV/JSON outputs |

Because they are read at import time, setting them inside a running notebook *after* `import irlab` has no effect. The notebook's configuration
cell passes `device=os.environ.get("IRLAB_DEVICE", "auto")` explicitly, so the device choice is visible there.

**What "reproducible" means here.** Data and models are pinned by commit hash. Seeds are fixed, all tie-breaking is deterministic, and the LLM
decodes greedily. Embeddings are cached, so a rerun on the same cache gives identical numbers: repeated GPU runs reproduced every table and all
LLM answers exactly. Recomputing embeddings on different hardware (CPU vs. GPU) changes the floating-point values slightly. In the CPU run (§8) this changed
none of the printed values. The canonical BEIR zip download is not used (unreachable during preparation). If you switch to it, the
integrity checks will tell you whether the counts agree.

---

## 10. Adapting the notebook to another collection

1. Add a `CollectionSpec` to `COLLECTIONS` in `irlab/config.py`: Hub repositories and **pinned revisions** for corpus and qrels, which split is
   dev and which is test, and the expected counts (documents, total queries, and per split the number of judged queries and judgment pairs, from
   the dataset's documentation). The loader refuses to run if they disagree.
2. If the collection has no dev split (e.g. BEIR ArguAna, SciDocs), do **not** tune on it. Either run it with the settings chosen elsewhere (a
   transfer experiment) or create a dev split by sampling queries with a fixed seed and *removing them from test*.
3. Check the collection's conventions: ArguAna and Quora require removing the query's own document from its results (BEIR does this). Graded
   labels may include 0 for "judged non-relevant", which the measures handle (label ≥ 1 is relevant).
4. Cost scales with corpus size: encoding time is roughly linear in the number of documents (see the per-document times in §8). Beyond about 10⁵
   documents, use an approximate nearest-neighbour index (e.g. FAISS) instead of the exhaustive matrix product in `dense.similarity_matrix`. For
   collections with long documents (full papers, web pages), the 512-token limit discards evidence, and documents should be split into passages before encoding.
5. For the LLM part, replace the `QUESTIONS` list in Section 7 with questions that have a `source_claim` (or `None`) and the expected faithful behaviour **before**
   running them, and report all of them.
6. For non-English collections, replace the tokenizer/stemmer in `irlab/lexical.py` (PyStemmer supports many languages) and use a multilingual
   encoder.

---

## 11. Limitations, inconclusive results, what was not run

**Methodological limitations.**
* Unjudged documents count as non-relevant. SciFact has about one judged document per claim, so systems that retrieve other useful abstracts are
  penalised. The U1 answer cited such an unjudged, partly relevant abstract.
* SciFact qrels mark *cited* abstracts, including ones without evidence. NFCorpus includes tag-level links that no content-based system can find
  (e.g. *Peoria*).
* 300/323 test queries give wide intervals (B8), and several differences discussed in the notebook are not statistically distinguishable.
* The query-group analysis (B2) uses an overlap feature computed from the qrels, so it is diagnostic only.
* The query variants (B3) inherit qrels by assumption.
* BGE/E5 training data may overlap with these documents (§4), so "zero-shot" is not strictly guaranteed for them.
* The LLM part has four questions (twelve generations in B7). It demonstrates verification and is not an evaluation of the model. Different
  hardware or library versions may produce different answers despite greedy decoding.
* The Hugging Face mirror is used instead of the canonical BEIR zip (§3.2).

**Inconclusive or negative results.**
* BM25 tuning (B5), RRF-k tuning and min-max fusion (B6) produced test changes of at most a few thousandths.
* The B3 predictions that abbreviations, synonyms and a misspelling would break BM25 did not materialise for these claims.

**Removed: chunking.** An earlier version split abstracts into 64- or 128-word windows and scored a document by its best window. With 64-word windows
dev nDCG@10 rose from 0.747 to 0.754, but test nDCG@10 was unchanged (0.7125 vs. 0.7127), at about 6× the stored vectors. SciFact abstracts are short
(9% exceed 512 tokens), so truncation loses little. The experiment and `irlab/chunking.py` were removed. Truncation at 512 tokens is the only
passage-length handling.

**Removed: cross-encoder reranking.** An earlier version reranked first-stage candidates with `cross-encoder/ms-marco-MiniLM-L6-v2`. It was
removed at the course owner's request, together with its code, cached scores and result files. For the record, that evaluation had shown that
reranking BM25's top 100 reproduced the BEIR paper's BM25+CE values (SciFact 0.688; NFCorpus 0.352 vs. 0.350 in BEIR), while reranking RRF lowered
nDCG@10 on both collections (SciFact 0.728 → 0.688). Those figures are not regenerated by the current code.

**Not run, and why.**
* Byte-level comparison with the canonical BEIR zips: the server timed out.
* Inspection of the original NFCorpus archive (to document the BEIR label mapping): the download URL returned an HTML page.
* Larger encoders and approximate nearest-neighbour indexes: outside the hardware budget chosen for this notebook, so nothing is claimed about them.
* A systematic answer-quality evaluation (many questions, a rubric, multiple annotators): outside the scope of this notebook.
* Fine-tuning any model: no SciFact or NFCorpus training data is used anywhere.

---

## 12. References and official documentation

**Collections**
* BEIR benchmark: repository <https://github.com/beir-cellar/beir> · paper: Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models*, NeurIPS Datasets & Benchmarks 2021, <https://arxiv.org/abs/2104.08663>
* BEIR mirrors used: <https://huggingface.co/datasets/BeIR/scifact>, <https://huggingface.co/datasets/BeIR/scifact-qrels>,
  <https://huggingface.co/datasets/BeIR/nfcorpus>, <https://huggingface.co/datasets/BeIR/nfcorpus-qrels>
* `ir_datasets` documentation of the same BEIR collections: <https://ir-datasets.com/beir.html>
* SciFact: <https://github.com/allenai/scifact> · licence: <https://github.com/allenai/scifact/blob/master/LICENSE.md> · paper: Wadden et al., *Fact
  or Fiction: Verifying Scientific Claims*, EMNLP 2020, <https://aclanthology.org/2020.emnlp-main.609/>
* NFCorpus: <https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/> · paper: Boteva et al., *A Full-Text Learning to Rank Dataset for Medical
  Information Retrieval*, ECIR 2016, <https://doi.org/10.1007/978-3-319-30671-1_58>

**Models**
* <https://huggingface.co/BAAI/bge-small-en-v1.5> · C-Pack/BGE paper <https://arxiv.org/abs/2309.07597>
* <https://huggingface.co/google-bert/bert-base-uncased> · BERT paper <https://arxiv.org/abs/1810.04805>
* <https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b> · TAS-B paper <https://arxiv.org/abs/2104.06967>
* <https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4> · MS MARCO <https://microsoft.github.io/msmarco/>
* <https://huggingface.co/intfloat/e5-small-v2> · E5 paper <https://arxiv.org/abs/2212.03533>
* <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>

**Libraries**
* Sentence Transformers: <https://sbert.net> · semantic search <https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html>
* Hugging Face Transformers (generation, chat templates): <https://huggingface.co/docs/transformers>
* bm25s: <https://github.com/xhluca/bm25s> · PyStemmer (Snowball): <https://github.com/snowballstem/pystemmer>
* ir_measures: <https://ir-measur.es/> · pytrec_eval: <https://github.com/cvangysel/pytrec_eval> · trec_eval: <https://github.com/usnistgov/trec_eval>

**Methods**
* BM25: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009, <https://doi.org/10.1561/1500000019>
* nDCG: Järvelin & Kekäläinen, *Cumulated gain-based evaluation of IR techniques*, TOIS 2002, <https://doi.org/10.1145/582415.582418>
* RRF: Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009,
  <https://doi.org/10.1145/1571941.1572114>
* Significance testing in IR: Smucker, Allan & Carterette, *A comparison of statistical significance tests for information retrieval
  evaluation*, CIKM 2007, <https://doi.org/10.1145/1321440.1321528>
* Anisotropy of contextual embeddings: Ethayarajh, *How Contextual are Contextualized Word Representations?*, EMNLP 2019,
  <https://aclanthology.org/D19-1006/>
