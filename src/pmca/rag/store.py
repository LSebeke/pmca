from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

from pmca.rag.chunker import chunk_file
from pmca.rag.embedder import embed
from pmca.types import Chunk

_DIMS = 1536


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray = np.empty((0, _DIMS), dtype=np.float32)

    def build(self, files: list[Path], cache_dir: Path) -> None:
        all_chunks: list[Chunk] = []
        all_embeddings: list[np.ndarray] = []

        stale = [p for p in files if _load_cache(p, cache_dir) is None]
        if stale:
            print(f"[RAG] embedding {len(stale)} new/changed file(s)...", file=sys.stderr)

        for path in files:
            cached = _load_cache(path, cache_dir)
            if cached is not None:
                all_chunks.extend(cached["chunks"])
                all_embeddings.append(cached["embeddings"])
                continue

            chunks = chunk_file(path)
            if not chunks:
                continue

            embeddings = embed([c.content for c in chunks])
            _save_cache(path, chunks, embeddings, cache_dir)
            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)

        self._chunks = all_chunks
        self._embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, _DIMS), dtype=np.float32)

    def query(self, text: str, top_k: int) -> list[Chunk]:
        if not self._chunks:
            return []

        query_vec = embed([text])[0]
        similarities = _cosine_similarity(query_vec, self._embeddings)
        k = min(top_k, len(self._chunks))
        indices = np.argsort(similarities)[::-1][:k]
        return [self._chunks[i] for i in indices]


def _load_cache(path: Path, cache_dir: Path) -> dict | None:
    cache_file = _cache_path(path, cache_dir)
    if not cache_file.exists():
        return None
    data = pickle.loads(cache_file.read_bytes())
    if data["file_hash"] != hashlib.sha256(path.read_bytes()).hexdigest():
        return None
    return data


def _save_cache(path: Path, chunks: list[Chunk], embeddings: np.ndarray, cache_dir: Path) -> None:
    data = {
        "file_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chunks": chunks,
        "embeddings": embeddings,
    }
    _cache_path(path, cache_dir).write_bytes(pickle.dumps(data))


def _cache_path(path: Path, cache_dir: Path) -> Path:
    key = hashlib.sha256(str(path).encode()).hexdigest()
    return cache_dir / f"{key}.pkl"


def _cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    dots = embeddings @ query
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((query_norm == 0) | (emb_norms == 0), 0.0, dots / (emb_norms * query_norm))
