"""Combining rankings: reciprocal rank fusion and (optional) normalised score fusion."""
from __future__ import annotations

from .metrics import Run


def rrf(runs: list[Run], k: int = 60, depth: int | None = None, weights: list[float] | None = None,
        out_depth: int = 100) -> Run:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    score(d) = sum_i w_i / (k + rank_i(d)), using only each run's top ``depth`` documents.
    A document absent from a run contributes nothing for that run. Only ranks are used,
    so BM25 scores (unbounded) and cosine scores ([-1, 1]) never need to be compared.
    Larger ``k`` flattens the contribution curve: rank 1 vs rank 10 matters less.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    weights = weights or [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError("one weight per run")
    qids = sorted(set().union(*runs))
    fused: Run = {}
    for q in qids:
        acc: dict[str, float] = {}
        for run, w in zip(runs, weights):
            for rank, (d, _) in enumerate(run.get(q, [])[:depth], start=1):
                acc[d] = acc.get(d, 0.0) + w / (k + rank)
        fused[q] = sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:out_depth]
    return fused


def minmax_fusion(a: Run, b: Run, alpha: float, out_depth: int = 100) -> Run:
    """alpha * norm(score_a) + (1 - alpha) * norm(score_b), min-max normalised per query
    over each run's own list; a document missing from a list gets 0 for that list.
    Unlike RRF this uses score *gaps*, and is sensitive to outliers and list depth."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    def norm(lst):
        if not lst:
            return {}
        s = [x for _, x in lst]
        lo, hi = min(s), max(s)
        return {d: (x - lo) / (hi - lo) if hi > lo else 1.0 for d, x in lst}

    out: Run = {}
    for q in sorted(set(a) | set(b)):
        na, nb = norm(a.get(q, [])), norm(b.get(q, []))
        docs = set(na) | set(nb)
        scored = {d: alpha * na.get(d, 0.0) + (1 - alpha) * nb.get(d, 0.0) for d in docs}
        out[q] = sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:out_depth]
    return out
