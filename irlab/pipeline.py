"""The ``Lab`` object: builds runs on demand and memoises them by their parameters.

Any notebook cell can ask for e.g. ``lab.dense_run("scifact", "test")`` and get the
same result whether or not an earlier cell computed it, because results are keyed by
every parameter that influences them (no hidden cell-order state).
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import dense, fusion, metrics
from .config import PROJECT_ROOT, LabConfig
from .data import Collection, load_collection
from .lexical import BM25Index, Preprocessing
from .metrics import Run


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _qhash(queries: dict[str, str]) -> str:
    return hashlib.sha1(json.dumps(sorted(queries.items())).encode()).hexdigest()[:12]


class Lab:
    def __init__(self, cfg: LabConfig):
        self.cfg = cfg
        set_seeds(cfg.seed)
        self.device = dense.resolve_device(cfg.device)
        self._collections: dict[str, Collection] = {}
        self._bm25: dict[tuple, BM25Index] = {}
        self._runs: dict[tuple, Run] = {}
        self.timings: list[dict] = []
        for d in (cfg.cache_dir, cfg.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utilities
    def _time(self, stage, component, collection, seconds, items, cached=False, note=""):
        self.timings.append({"stage": stage, "component": component, "collection": collection,
                             "seconds": round(seconds, 3), "items": items,
                             "ms/item": round(1000 * seconds / items, 2) if items else None,
                             "device": self.device, "from_cache": cached, "note": note})

    def timing_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.timings)

    def environment(self) -> dict:
        import bm25s, sentence_transformers, transformers
        return {"python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "device_used": self.device, "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__, "bm25s": bm25s.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}

    def collection(self, name: str) -> Collection:
        if name not in self._collections:
            t0 = time.perf_counter()
            self._collections[name] = load_collection(self.cfg.collections[name], self.cfg.data_dir)
            self._time("setup", "load collection", name, time.perf_counter() - t0, None)
        return self._collections[name]

    def queries(self, name: str, split: str) -> dict[str, str]:
        return self.collection(name).split_queries(split)

    def qrels(self, name: str, split: str):
        return self.collection(name).qrels[split]

    def split_name(self, name: str, role: str) -> str:
        spec = self.cfg.collections[name]
        return {"dev": spec.dev_split, "test": spec.test_split}[role]

    # ------------------------------------------------------------------ BM25
    def bm25_index(self, name: str, prep: Preprocessing | None = None, k1=None, b=None) -> BM25Index:
        prep = prep or Preprocessing(self.cfg.bm25_stopwords, self.cfg.bm25_stem)
        k1 = self.cfg.bm25_k1 if k1 is None else k1
        b = self.cfg.bm25_b if b is None else b
        key = (name, prep, k1, b)
        if key not in self._bm25:
            coll = self.collection(name)
            t0 = time.perf_counter()
            self._bm25[key] = BM25Index(coll.full_texts(), prep, k1, b)
            self._time("setup", f"BM25 index ({prep.label}, k1={k1}, b={b})", name,
                       time.perf_counter() - t0, len(coll))
        return self._bm25[key]

    def bm25_run(self, name: str, queries: dict[str, str], prep: Preprocessing | None = None,
                 k1=None, b=None, depth=None) -> Run:
        depth = depth or self.cfg.retrieval_depth
        index = self.bm25_index(name, prep, k1, b)
        key = ("bm25", name, index.prep, index.k1, index.b, depth, _qhash(queries))
        if key not in self._runs:
            coll = self.collection(name)
            t0 = time.perf_counter()
            run = {}
            for q, text in queries.items():
                pos, sc = index.search(text, depth)
                run[q] = [(coll.doc_ids[i], float(s)) for i, s in zip(pos, sc)]
            self._time("query", "BM25 search", name, time.perf_counter() - t0, len(queries))
            self._runs[key] = run
        return self._runs[key]

    # ------------------------------------------------------------------ dense
    def doc_embeddings(self, name: str, model_key: str | None = None, max_seq_length=None):
        spec = self.cfg.encoders[model_key or self.cfg.dense_model]
        coll = self.collection(name)
        emb, meta = dense.corpus_embeddings(spec, coll.full_texts(), coll.fingerprint(),
                                            self.cfg.cache_dir, self.device,
                                            self.cfg.dense_batch_size, max_seq_length)
        key = ("emb-logged", name, spec.key, meta["max_seq_length"])
        if key not in self._runs:  # log each embedding set once per session
            self._runs[key] = {}
            self._time("setup", f"encode corpus [{spec.key}, len {meta['max_seq_length']}]", name,
                       meta["encode_s"], meta["n"], cached=meta["loaded_from_cache"],
                       note=f"computed on {meta['device']}; model load {meta['model_load_s']} s")
        return emb, meta

    def dense_run(self, name: str, queries: dict[str, str], model_key: str | None = None,
                  similarity: str | None = None, max_seq_length=None, depth=None) -> Run:
        spec = self.cfg.encoders[model_key or self.cfg.dense_model]
        sim = similarity or spec.similarity
        depth = depth or self.cfg.retrieval_depth
        emb, meta = self.doc_embeddings(name, spec.key, max_seq_length)
        key = ("dense", name, spec.key, sim, meta["max_seq_length"], depth, _qhash(queries))
        if key not in self._runs:
            coll = self.collection(name)
            qids = list(queries)
            t0 = time.perf_counter()
            dense.load_encoder(spec.hf_id, spec.revision, self.device, meta["max_seq_length"])
            t_load = time.perf_counter() - t0
            if t_load > 0.05:  # first use in this process
                self._time("setup", f"load encoder [{spec.key}]", name, t_load, None)
            t0 = time.perf_counter()
            q_emb = dense.encode_texts(spec, [queries[q] for q in qids], "query", self.device,
                                       self.cfg.dense_batch_size, meta["max_seq_length"])
            s = dense.similarity_matrix(q_emb, emb, sim, self.device)
            pos, sc = dense.top_k(s, depth)
            self._time("query", f"dense encode+search [{spec.key}, {sim}]", name,
                       time.perf_counter() - t0, len(qids))
            self._runs[key] = {q: [(coll.doc_ids[i], float(v)) for i, v in zip(pos[r], sc[r])]
                               for r, q in enumerate(qids)}
        return self._runs[key]

    # ------------------------------------------------------------------ hybrid
    def hybrid_run(self, name: str, queries: dict[str, str], rrf_k=None, fusion_depth=None,
                   bm25_kwargs: dict | None = None, dense_kwargs: dict | None = None) -> Run:
        rrf_k = self.cfg.rrf_k if rrf_k is None else rrf_k
        fusion_depth = fusion_depth or self.cfg.retrieval_depth
        a = self.bm25_run(name, queries, depth=fusion_depth, **(bm25_kwargs or {}))
        b = self.dense_run(name, queries, depth=fusion_depth, **(dense_kwargs or {}))
        t0 = time.perf_counter()
        out = fusion.rrf([a, b], k=rrf_k, depth=fusion_depth, out_depth=self.cfg.retrieval_depth)
        self._time("query", f"RRF fusion (k={rrf_k})", name, time.perf_counter() - t0, len(queries))
        return out

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, name: str, split: str, runs: dict[str, Run],
                 measures=("nDCG@10", "MRR@10", "R@10", "R@100")) -> pd.DataFrame:
        qrels = self.qrels(name, split)
        rows = []
        for label, run in runs.items():
            missing = set(qrels) - set(run)
            if missing:
                raise ValueError(f"run {label!r} lacks {len(missing)} judged queries")
            row = {"collection": name, "split": split, "system": label, "queries": len(qrels)}
            row.update({m: round(v, 4) for m, v in metrics.evaluate(run, qrels, measures).items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def per_query(self, name: str, split: str, run: Run,
                  measures=("nDCG@10", "MRR@10", "R@10", "R@100")) -> pd.DataFrame:
        return metrics.per_query(run, self.qrels(name, split), measures)

    def save_table(self, df: pd.DataFrame, filename: str) -> None:
        path = Path(self.cfg.results_dir) / filename
        df.to_csv(path, index=False)
        try:
            shown = path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:  # results_dir outside the project (IRLAB_RESULTS_DIR)
            shown = path
        print(f"saved {shown}")

    def with_config(self, **changes) -> LabConfig:
        return replace(self.cfg, **changes)
