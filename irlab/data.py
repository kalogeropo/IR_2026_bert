"""Download (once) and load BEIR collections with integrity checks."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import CollectionSpec

Qrels = dict[str, dict[str, int]]  # qid -> {doc_id: graded relevance}


@dataclass
class Collection:
    """Corpus, queries and qrels of one BEIR collection.

    Documents are addressed by their position ``i`` (for matrices) and by their
    BEIR id ``doc_ids[i]`` (for qrels and display). ``doc_index`` maps id -> position.
    """

    name: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, Qrels]
    spec: CollectionSpec
    doc_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def full_text(self, i: int) -> str:
        """Title + abstract, joined as in BEIR's reference implementation."""
        return f"{self.titles[i]} {self.texts[i]}".strip()

    def full_texts(self) -> list[str]:
        return [self.full_text(i) for i in range(len(self))]

    def split_queries(self, split: str) -> dict[str, str]:
        """Queries that have judgments in ``split``, in a fixed (sorted) order."""
        if split not in self.qrels:
            raise KeyError(f"{self.name} has no split {split!r}; available: {sorted(self.qrels)}")
        return {q: self.queries[q] for q in sorted(self.qrels[split], key=_natural_key)}

    def fingerprint(self) -> str:
        """Hash of the document order and text: embeddings cached under another
        fingerprint are never reused for this corpus."""
        h = hashlib.sha256()
        for i in range(len(self)):
            h.update(self.doc_ids[i].encode())
            h.update(self.full_text(i).encode())
        return h.hexdigest()[:16]


def _natural_key(s: str):
    head, _, tail = s.rpartition("-")
    return (head, int(tail)) if tail.isdigit() else (s, 0)


def download(spec: CollectionSpec, data_dir: Path) -> Path:
    """Fetch the pinned revisions of the BEIR mirror on the Hugging Face Hub.

    The canonical BEIR zip (public.ukp.informatik.tu-darmstadt.de) is equivalent but
    was unreachable when this lab was prepared; the Hub copy is pinned by commit hash.
    """
    from huggingface_hub import snapshot_download

    root = Path(data_dir) / "beir" / spec.name
    snapshot_download(spec.corpus_repo, repo_type="dataset", revision=spec.corpus_revision,
                      local_dir=root / "corpus_repo")
    snapshot_download(spec.qrels_repo, repo_type="dataset", revision=spec.qrels_revision,
                      local_dir=root / "qrels_repo")
    return root


def _read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def load_collection(spec: CollectionSpec, data_dir: Path) -> Collection:
    root = download(spec, data_dir)
    corpus = pd.read_parquet(next((root / "corpus_repo" / "corpus").glob("*.parquet")))
    queries = pd.read_parquet(next((root / "corpus_repo" / "queries").glob("*.parquet")))
    corpus = corpus.fillna("")
    qrels = {p.stem: _read_qrels(p) for p in sorted((root / "qrels_repo").glob("*.tsv"))}

    coll = Collection(
        name=spec.name,
        doc_ids=corpus["_id"].astype(str).tolist(),
        titles=corpus["title"].astype(str).tolist(),
        texts=corpus["text"].astype(str).tolist(),
        queries=dict(zip(queries["_id"].astype(str), queries["text"].astype(str))),
        qrels=qrels,
        spec=spec,
    )
    _validate(coll)
    return coll


def _validate(c: Collection) -> None:
    e = c.spec.expected
    problems = []
    if len(c) != e["docs"]:
        problems.append(f"{len(c)} documents, expected {e['docs']}")
    if len(c.doc_index) != len(c):
        problems.append("duplicate document ids")
    if len(c.queries) != e["queries_total"]:
        problems.append(f"{len(c.queries)} queries, expected {e['queries_total']}")
    for split in (c.spec.dev_split, c.spec.test_split):
        n_q, n_pairs = e[split]
        got_q = len(c.qrels.get(split, {}))
        got_pairs = sum(len(v) for v in c.qrels.get(split, {}).values())
        if (got_q, got_pairs) != (n_q, n_pairs):
            problems.append(f"split {split}: {got_q} queries/{got_pairs} judgments, expected {n_q}/{n_pairs}")
    for split, qr in c.qrels.items():
        missing_q = [q for q in qr if q not in c.queries]
        missing_d = {d for v in qr.values() for d in v if d not in c.doc_index}
        if missing_q or missing_d:
            problems.append(f"split {split}: {len(missing_q)} unknown queries, {len(missing_d)} unknown docs")
    dev, test = set(c.qrels[c.spec.dev_split]), set(c.qrels[c.spec.test_split])
    if dev & test:
        problems.append(f"{len(dev & test)} queries occur in both dev and test")
    if problems:
        raise ValueError(f"{c.name}: " + "; ".join(problems))


def describe(c: Collection) -> pd.DataFrame:
    """One row per judged split: sizes, query length and relevance-label statistics."""
    doc_words = pd.Series([len(c.full_text(i).split()) for i in range(len(c))])
    rows = []
    for split, qr in sorted(c.qrels.items()):
        qlen = pd.Series([len(c.queries[q].split()) for q in qr])
        labels = pd.Series([r for v in qr.values() for r in v.values()])
        rows.append({
            "collection": c.name, "split": split, "role": (
                "dev (tuning)" if split == c.spec.dev_split else
                "test (final)" if split == c.spec.test_split else "unused"),
            "queries": len(qr), "judgments": len(labels),
            "judged/query (mean)": round(len(labels) / len(qr), 2),
            "judged/query (median)": float(pd.Series([len(v) for v in qr.values()]).median()),
            "label counts": dict(sorted(labels.value_counts().to_dict().items())),
            "query words (median)": float(qlen.median()),
            "docs": len(c), "doc words (median)": float(doc_words.median()),
        })
    return pd.DataFrame(rows)


SCIFACT_RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_RELEASE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"


def load_scifact_evidence(data_dir: Path) -> dict[str, dict]:
    """Original SciFact *dev* claims (= BEIR SciFact test queries) with their
    rationale labels. Returns ``{claim_id: {"claim": str, "evidence": {doc_id: label},
    "cited": [doc_id, ...]}}`` where label is SUPPORT or CONTRADICT; a cited document
    without an entry has no rationale (NEI). Used only to *check LLM answers*, never
    for retrieval evaluation (BEIR qrels = cited documents, whatever their label).
    """
    import tarfile
    import urllib.request

    root = Path(data_dir) / "scifact_release"
    archive = root / "data.tar.gz"
    if not archive.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SCIFACT_RELEASE_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SCIFACT_RELEASE_SHA256:
        raise ValueError(f"SciFact release changed: sha256 {digest} != {SCIFACT_RELEASE_SHA256}")
    member = "data/claims_dev.jsonl"
    with tarfile.open(archive) as tar:
        f = tar.extractfile(member)
        if f is None:
            raise ValueError(f"{member} missing from archive")
        lines = f.read().decode().splitlines()
    import json
    out = {}
    for line in lines:
        c = json.loads(line)
        ev = {str(d): sets[0]["label"] for d, sets in c["evidence"].items()}
        out[str(c["id"])] = {"claim": c["claim"], "evidence": ev,
                             "cited": [str(d) for d in c["cited_doc_ids"]]}
    return out
