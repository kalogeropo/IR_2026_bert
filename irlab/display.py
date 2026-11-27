"""Readable views of rankings next to the qrels."""
from __future__ import annotations

import textwrap

import pandas as pd

from .data import Collection
from .metrics import Run, first_relevant_rank


def label_of(rels: dict[str, int], doc_id: str) -> str:
    return str(rels[doc_id]) if doc_id in rels else "unjudged"


def ranking_table(coll: Collection, run: Run, qid: str, rels: dict[str, int], k: int = 5,
                  snippet_words: int = 22) -> pd.DataFrame:
    rows = []
    for rank, (d, s) in enumerate(run.get(qid, [])[:k], start=1):
        i = coll.doc_index[d]
        snippet = " ".join(coll.full_text(i).split()[:snippet_words])
        rows.append({"rank": rank, "doc_id": d, "score": round(s, 3), "qrel": label_of(rels, d),
                     "title + start of abstract": snippet + " ..."})
    return pd.DataFrame(rows)


def compare_systems(coll: Collection, runs: dict[str, Run], qid: str, query: str,
                    rels: dict[str, int], k: int = 5, snippet_words: int = 18,
                    show_relevant_text: bool = True) -> None:
    """Print the query, its judged-relevant documents, and the top-k of each system,
    with the rank at which each system first finds a relevant document."""
    print("=" * 100)
    print(f"QUERY {qid}: {query}")
    rel_sorted = sorted(rels.items(), key=lambda x: -x[1])
    print(f"judged relevant ({sum(r >= 1 for r in rels.values())}): "
          + ", ".join(f"{d} (label {r})" for d, r in rel_sorted[:8])
          + (" ..." if len(rels) > 8 else ""))
    if show_relevant_text:
        for d, _ in rel_sorted[:2]:
            i = coll.doc_index[d]
            print(textwrap.fill(f"  [{d}] " + " ".join(coll.full_text(i).split()[:45]) + " ...",
                                width=100, subsequent_indent="        "))
    for label, run in runs.items():
        ranked = [d for d, _ in run.get(qid, [])]
        fr = first_relevant_rank(ranked, rels)
        if fr:
            found = f"first relevant at rank {fr}"
        elif not ranked:
            found = "retrieved nothing (no query term occurs in the corpus)"
        else:
            found = f"no relevant doc in its top {len(ranked)}"
        print(f"--- {label}: {found}")
        for rank, (d, s) in enumerate(run.get(qid, [])[:k], start=1):
            i = coll.doc_index[d]
            mark = "*" if rels.get(d, 0) >= 1 else " "
            snippet = " ".join(coll.full_text(i).split()[:snippet_words])
            print(f" {mark}{rank:>2}. [{d}] ({label_of(rels, d)}) {s:8.3f}  {snippet} ...")


def rank_changes(runs: dict[str, Run], qid: str, rels: dict[str, int]) -> pd.DataFrame:
    """Rank of every judged-relevant document under each system (None = not retrieved)."""
    rows = []
    for d, r in sorted(rels.items(), key=lambda x: -x[1]):
        row = {"relevant doc": d, "label": r}
        for label, run in runs.items():
            ranked = [x for x, _ in run.get(qid, [])]
            row[label] = ranked.index(d) + 1 if d in ranked else None
        rows.append(row)
    return pd.DataFrame(rows)


def show_judged(coll: Collection, split: str, qid: str, words: int = 110,
                evidence: dict | None = None) -> None:
    """Print a query, its judged documents (with labels) and the start of each document."""
    if split not in coll.qrels or qid not in coll.qrels[split]:
        raise KeyError(f"query {qid!r} has no judgments in {coll.name}/{split}")
    rels = coll.qrels[split][qid]
    print(f"QUERY {qid}: {coll.queries[qid]}")
    for d, r in sorted(rels.items(), key=lambda x: -x[1]):
        i = coll.doc_index[d]
        extra = ""
        if evidence is not None and qid in evidence:
            extra = f", SciFact rationale label: {evidence[qid]['evidence'].get(d, 'none (NEI)')}"
        print(f"\n[{d}] qrels label {r}{extra}")
        print(textwrap.fill(" ".join(coll.full_text(i).split()[:words]) + " ...", width=100))
