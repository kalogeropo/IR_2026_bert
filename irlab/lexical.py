"""BM25 with explicit, switchable preprocessing (backed by the ``bm25s`` library)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Preprocessing:
    """Lowercasing is always on. Tokens are maximal runs of [a-z0-9], so
    "IL-6" -> ["il", "6"] and "β-cells" -> ["cells"] (non-ASCII letters are dropped):
    a known weakness for biomedical text that students can observe."""

    stopwords: bool = True
    stem: bool = True

    @property
    def label(self) -> str:
        return f"{'stop' if self.stopwords else 'nostop'}+{'stem' if self.stem else 'nostem'}"


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    from bm25s.stopwords import STOPWORDS_EN  # 33 common English function words
    return frozenset(STOPWORDS_EN)


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer  # PyStemmer: Snowball English stemmer
    return Stemmer.Stemmer("english")


def tokenize(text: str, prep: Preprocessing) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if prep.stopwords:
        sw = _stopwords()
        tokens = [t for t in tokens if t not in sw]
    if prep.stem:
        tokens = _stemmer().stemWords(tokens)
    return tokens


class BM25Index:
    """Index a list of texts once; score queries against every document.

    ``search`` returns positions and scores of the top ``depth`` documents whose score is
    > 0 (a document sharing no term with the query is not retrieved, as in Lucene).
    Ties are broken by document position, so results are deterministic.
    """

    def __init__(self, texts: list[str], prep: Preprocessing, k1: float, b: float,
                 method: str = "lucene"):
        import bm25s

        if not (k1 >= 0 and 0 <= b <= 1):
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.prep, self.k1, self.b, self.method = prep, k1, b, method
        corpus_tokens = [tokenize(t, prep) for t in texts]
        self.n_docs = len(texts)
        self.model = bm25s.BM25(k1=k1, b=b, method=method)
        self.model.index(corpus_tokens, show_progress=False)
        self.vocab = self.model.vocab_dict
        self.doc_freq = self._doc_freq(corpus_tokens)

    @staticmethod
    def _doc_freq(corpus_tokens: list[list[str]]) -> dict[str, int]:
        df: dict[str, int] = {}
        for toks in corpus_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        return df

    def scores(self, query: str) -> np.ndarray:
        q = [t for t in tokenize(query, self.prep) if t in self.vocab]
        if not q:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.model.get_scores(q), dtype=np.float32)

    def search(self, query: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[: min(depth, self.n_docs)]
        order = order[s[order] > 0]
        return order, s[order]
