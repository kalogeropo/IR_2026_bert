"""Rank-based effectiveness measures, written out so students can read them.

Conventions (identical to trec_eval / pytrec_eval, which BEIR uses; the notebook
checks the numbers against ``ir_measures``):

* A run is ``{qid: [(doc_id, score), ...]}`` already sorted best-first.
* A document is *relevant* if its qrels label is >= 1. Unjudged documents count as
  non-relevant -- an assumption, not a fact (see the notebook discussion).
* nDCG uses the graded label as gain (gain = label, discount = log2(rank + 1)); the
  ideal ranking is built from *all* judged documents of the query.
* Every query in the qrels is averaged; a query missing from the run scores 0.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

Run = dict[str, list[tuple[str, float]]]


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r >= 1}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def rr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top k, else 0."""
    for pos, d in enumerate(ranked[:k], start=1):
        if rels.get(d, 0) >= 1:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(pos + 1) for pos, d in enumerate(ranked[:k], start=1))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def first_relevant_rank(ranked: list[str], rels: dict[str, int]) -> int | None:
    for pos, d in enumerate(ranked, start=1):
        if rels.get(d, 0) >= 1:
            return pos
    return None


MEASURES = {"nDCG": ndcg_at_k, "R": recall_at_k, "MRR": rr_at_k}


def per_query(run: Run, qrels: dict[str, dict[str, int]],
              measures: tuple[str, ...] = ("nDCG@10", "MRR@10", "R@10", "R@100")) -> pd.DataFrame:
    """One row per judged query, one column per measure (e.g. ``"nDCG@10"``)."""
    rows = {}
    for qid, rels in qrels.items():
        ranked = [d for d, _ in run.get(qid, [])]
        row = {}
        for m in measures:
            name, k = m.split("@")
            row[m] = MEASURES[name](ranked, rels, int(k))
        rows[qid] = row
    return pd.DataFrame.from_dict(rows, orient="index")[list(measures)]


def evaluate(run: Run, qrels, measures=("nDCG@10", "MRR@10", "R@10", "R@100")) -> dict[str, float]:
    return per_query(run, qrels, measures).mean().to_dict()


def paired_bootstrap(a: pd.Series, b: pd.Series, n: int, seed: int) -> dict[str, float]:
    """95% percentile interval for mean(b - a) under resampling of queries.

    Answers "how much could the average difference move if we had drawn a different
    set of queries of the same kind?" -- it does not correct for multiple comparisons.
    """
    if not a.index.equals(b.index):
        b = b.reindex(a.index)
    diff = (b - a).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean diff": float(diff.mean()), "95% CI low": float(lo), "95% CI high": float(hi),
            "wins": int((diff > 1e-12).sum()), "losses": int((diff < -1e-12).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum())}
