"""Bi-encoder retrieval: encode the corpus once (cached on disk), encode each query, rank by similarity."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import EncoderSpec


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no CUDA GPU is visible")
    return device


@lru_cache(maxsize=4)
def load_encoder(hf_id: str, revision: str, device: str, max_seq_length: int):
    """Load a SentenceTransformer once per process. A plain HF checkpoint such as
    bert-base-uncased is wrapped automatically with *mean pooling* (a warning says so)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(hf_id, revision=revision, device=device)
    model.max_seq_length = max_seq_length
    model.eval()
    return model


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def encode_texts(spec: EncoderSpec, texts: list[str], role: str, device: str,
                 batch_size: int, max_seq_length: int | None = None) -> np.ndarray:
    """Encode texts as 'query' or 'document'. Normalises only for cosine models."""
    if role not in ("query", "document"):
        raise ValueError(role)
    msl = max_seq_length or spec.max_seq_length
    model = load_encoder(spec.hf_id, spec.revision, device, msl)
    prompt = spec.query_prompt if role == "query" else spec.doc_prompt
    with torch.inference_mode():
        emb = model.encode(list(texts), prompt=prompt or None, batch_size=batch_size,
                           convert_to_numpy=True, show_progress_bar=False,
                           normalize_embeddings=(spec.similarity == "cos"))
    return emb.astype(np.float32, copy=False)


def corpus_embeddings(spec: EncoderSpec, texts: list[str], fingerprint: str, cache_dir: Path,
                      device: str, batch_size: int, max_seq_length: int | None = None) -> tuple[np.ndarray, dict]:
    """Return document embeddings, computing them only if no valid cache exists.

    The cache key contains the model revision, max length, prompt and a hash of the
    corpus, so changing any of them forces recomputation instead of silently reusing
    stale vectors. Returns (embeddings, metadata) where metadata records how and on
    which device they were made and how long it took.
    """
    msl = max_seq_length or spec.max_seq_length
    name = f"{spec.key}__{spec.revision[:8]}__len{msl}__{fingerprint}"
    path = Path(cache_dir) / "embeddings" / f"{name}.npy"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        emb = np.load(path)
        meta = json.loads(meta_path.read_text())
        if emb.shape[0] == len(texts):
            meta["loaded_from_cache"] = True
            return emb, meta
    _sync(device)
    t0 = time.perf_counter()
    model = load_encoder(spec.hf_id, spec.revision, device, msl)
    _sync(device)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    emb = encode_texts(spec, texts, "document", device, batch_size, msl)
    _sync(device)
    t_enc = time.perf_counter() - t1
    meta = {"model": spec.hf_id, "revision": spec.revision, "max_seq_length": msl,
            "similarity": spec.similarity, "n": len(texts), "dim": int(emb.shape[1]),
            "device": device, "model_load_s": round(t_load, 2), "encode_s": round(t_enc, 2),
            "truncated_docs": _count_truncated(model, texts, msl, spec.doc_prompt)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    meta_path.write_text(json.dumps(meta, indent=1))
    meta["loaded_from_cache"] = False
    return emb, meta


def _count_truncated(model, texts, msl, prompt) -> int:
    tok = model.tokenizer
    lengths = [len(tok((prompt or "") + t, add_special_tokens=True)["input_ids"]) for t in texts]
    return int(sum(n > msl for n in lengths))


def similarity_matrix(q: np.ndarray, d: np.ndarray, similarity: str, device: str) -> np.ndarray:
    """(n_queries, n_docs) similarity. 'cos' assumes unit-length rows (we normalise
    at encoding time), so cosine == dot product of the normalised vectors."""
    if similarity == "cos":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
    elif similarity != "dot":
        raise ValueError(similarity)
    with torch.inference_mode():
        s = torch.from_numpy(q).to(device) @ torch.from_numpy(d).to(device).T
    return s.float().cpu().numpy()


def top_k(scores: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Best ``depth`` positions per row, ties broken by document position."""
    depth = min(depth, scores.shape[1])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :depth]
    return order, np.take_along_axis(scores, order, axis=1)
