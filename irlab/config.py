"""All experiment settings live here, so a notebook run is fully described by one object.

The notebook creates a single :class:`LabConfig` in its first code cell. Every other
function receives the values it needs explicitly; nothing reads global state.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CollectionSpec:
    """Where a BEIR collection comes from and what it must contain once loaded.

    ``expected`` holds the sizes published by BEIR; the loader refuses to continue if
    the downloaded files disagree, so a truncated or changed download cannot silently
    produce different numbers.
    """

    name: str
    corpus_repo: str
    corpus_revision: str
    qrels_repo: str
    qrels_revision: str
    dev_split: str
    test_split: str
    expected: dict
    homepage: str
    license_note: str


@dataclass(frozen=True)
class EncoderSpec:
    """A bi-encoder (or a plain encoder used as one) and how it must be called."""

    key: str
    hf_id: str
    revision: str
    similarity: str  # "cos" or "dot": the function the model was trained with
    query_prompt: str = ""
    doc_prompt: str = ""
    max_seq_length: int = 512
    note: str = ""


COLLECTIONS = {
    "scifact": CollectionSpec(
        name="scifact",
        corpus_repo="BeIR/scifact",
        corpus_revision="b3b5335604bf5ee3c4447671af975ea25143d4f5",
        qrels_repo="BeIR/scifact-qrels",
        qrels_revision="2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        dev_split="train",  # BEIR 'train' = original SciFact train claims; we use it only for tuning
        test_split="test",  # BEIR 'test' = original SciFact *dev* claims (SciFact test labels are hidden)
        expected={"docs": 5183, "queries_total": 1109, "train": (809, 919), "test": (300, 339)},
        homepage="https://github.com/allenai/scifact",
        license_note="Claims and annotations: CC BY 4.0; abstracts (from S2ORC): ODC-By 1.0.",
    ),
    "nfcorpus": CollectionSpec(
        name="nfcorpus",
        corpus_repo="BeIR/nfcorpus",
        corpus_revision="b5026a0e96e8a7ac4f95f482a596389289d46269",
        qrels_repo="BeIR/nfcorpus-qrels",
        qrels_revision="a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
        dev_split="dev",
        test_split="test",
        expected={"docs": 3633, "queries_total": 3237, "dev": (324, 11385), "test": (323, 12334)},
        homepage="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        license_note="Free for academic use (NFCorpus page); NutritionFacts.org content has its own terms.",
    ),
}

ENCODERS = {
    "bert-base-mean": EncoderSpec(
        key="bert-base-mean",
        hf_id="google-bert/bert-base-uncased",
        revision="86b5e0934494bd15c9632b12f734a8a67f723594",
        similarity="cos",
        note="Pretrained BERT, masked-LM + NSP only, mean pooling added by us. Negative control.",
    ),
    "distilbert-msmarco-v4": EncoderSpec(
        key="distilbert-msmarco-v4",
        hf_id="sentence-transformers/msmarco-distilbert-base-v4",
        revision="b2f66c95aba1481a880479165582020c2b9b64d7",
        similarity="cos",
        note="DistilBERT fine-tuned on MS MARCO for cosine similarity (model of the original notebook).",
    ),
    "distilbert-tas-b": EncoderSpec(
        key="distilbert-tas-b",
        hf_id="sentence-transformers/msmarco-distilbert-base-tas-b",
        revision="b12d9352e776979147078a8975a4885042984fd1",
        similarity="dot",
        note="DistilBERT trained on MS MARCO with TAS-Balanced distillation; dot product, CLS pooling.",
    ),
    "bge-small": EncoderSpec(
        key="bge-small",
        hf_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        similarity="cos",
        query_prompt="Represent this sentence for searching relevant passages: ",
        note="Small BERT-style encoder, contrastive pre-training + fine-tuning on large mixed data.",
    ),
    "e5-small": EncoderSpec(
        key="e5-small",
        hf_id="intfloat/e5-small-v2",
        revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        similarity="cos",
        query_prompt="query: ",
        doc_prompt="passage: ",
        note="Small BERT-style encoder; requires 'query: ' / 'passage: ' prefixes.",
    ),
}


@dataclass(frozen=True)
class LabConfig:
    seed: int = 13
    device: str = os.environ.get("IRLAB_DEVICE", "auto")  # "auto", "cpu", "cuda"
    data_dir: Path = Path(os.environ.get("IRLAB_DATA_DIR", PROJECT_ROOT / "data"))
    cache_dir: Path = Path(os.environ.get("IRLAB_CACHE_DIR", PROJECT_ROOT / "cache"))
    results_dir: Path = Path(os.environ.get("IRLAB_RESULTS_DIR", PROJECT_ROOT / "results"))
    # --- first stage -------------------------------------------------------
    retrieval_depth: int = 100        # documents kept per query by every first-stage run
    bm25_k1: float = 1.2              # a-priori textbook / Elasticsearch default (not tuned)
    bm25_b: float = 0.75
    bm25_stopwords: bool = True
    bm25_stem: bool = True
    dense_model: str = "bge-small"         # chosen on dev data (notebook section B4)
    dense_batch_size: int = 64
    # --- fusion -------------------------------------------------------------
    rrf_k: int = 60                   # Cormack et al. (2009) default
    # --- generation --------------------------------------------------------
    llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    llm_max_new_tokens: int = 220
    llm_n_sources: int = 3
    llm_passage_words: int = 300      # words of each abstract shown to the LLM (median abstract: ~205)
    # --- evaluation --------------------------------------------------------
    eval_cutoffs: tuple = (10, 100)
    bootstrap_samples: int = 10_000
    collections: dict = field(default_factory=lambda: dict(COLLECTIONS))
    encoders: dict = field(default_factory=lambda: dict(ENCODERS))

    def summary(self) -> dict:
        """Settings that matter for results (paths and registries omitted)."""
        d = asdict(self)
        for k in ("collections", "encoders", "data_dir", "cache_dir", "results_dir"):
            d.pop(k)
        return d
