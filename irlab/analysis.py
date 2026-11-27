"""Per-query diagnostics used to choose examples *from the data* rather than by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Collection
from .lexical import BM25Index, tokenize
from .metrics import Run, first_relevant_rank


def query_features(coll: Collection, queries: dict[str, str], qrels, index: BM25Index) -> pd.DataFrame:
    """Measurable query properties.

    * ``query_terms``: index terms after the BM25 preprocessing.
    * ``overlap``: share of the query's distinct terms that appear in at least one of its
      relevant documents (uses qrels => a diagnostic, never available at search time).
    * ``min_df``: document frequency of the query's rarest term (rare entity / jargon).
    """
    rows = {}
    for q, text in queries.items():
        terms = set(tokenize(text, index.prep))
        rel_terms = set()
        for d, r in qrels[q].items():
            if r >= 1:
                rel_terms |= set(tokenize(coll.full_text(coll.doc_index[d]), index.prep))
        dfs = [index.doc_freq.get(t, 0) for t in terms]
        rows[q] = {"words": len(text.split()), "query_terms": len(terms),
                   "overlap": len(terms & rel_terms) / len(terms) if terms else 0.0,
                   "min_df": min(dfs) if dfs else 0,
                   "oov_terms": sum(d == 0 for d in dfs)}
    return pd.DataFrame.from_dict(rows, orient="index")


def first_ranks(runs: dict[str, Run], qrels, depth_label: int = 100) -> pd.DataFrame:
    """Rank of the first relevant document per query and system (NaN = not in list)."""
    out = {}
    for label, run in runs.items():
        out[label] = {q: first_relevant_rank([d for d, _ in run.get(q, [])], rels)
                      for q, rels in qrels.items()}
    return pd.DataFrame(out).astype(float)


def pick_disagreements(fr: pd.DataFrame, a: str, b: str, n: int = 3, cap: int = 101) -> dict[str, list[str]]:
    """Queries where system ``a`` puts a relevant document much higher than ``b`` and
    vice versa, measured by the reciprocal of the first relevant rank (missing = 1/cap).
    Deterministic: sorted by the size of the gap, then by query id."""
    ra = 1.0 / fr[a].fillna(cap)
    rb = 1.0 / fr[b].fillna(cap)
    gap = (ra - rb).rename("gap").to_frame()
    gap["qid"] = gap.index
    a_wins = gap[gap.gap > 0].sort_values(["gap", "qid"], ascending=[False, True]).index[:n]
    b_wins = gap[gap.gap < 0].sort_values(["gap", "qid"], ascending=[True, True]).index[:n]
    both_fail = fr[(fr[a].isna() | (fr[a] > 10)) & (fr[b].isna() | (fr[b] > 10))].index
    return {f"{a} better": list(a_wins), f"{b} better": list(b_wins),
            "both miss top-10": sorted(both_fail)[:n]}


def bucket_table(per_query: dict[str, pd.DataFrame], feature: pd.Series, bins, labels,
                 measure: str = "nDCG@10") -> pd.DataFrame:
    """Mean ``measure`` per system inside query buckets defined by ``feature``."""
    bucket = pd.cut(feature, bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for lab in labels:
        qids = bucket[bucket == lab].index
        row = {"bucket": lab, "queries": len(qids)}
        for name, pq in per_query.items():
            row[name] = round(float(pq.loc[qids, measure].mean()), 3) if len(qids) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
