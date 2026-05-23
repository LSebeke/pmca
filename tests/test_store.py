import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from pmca.rag.store import VectorStore
from pmca.types import Chunk

DIMS = 1536


def _chunks(n: int, path: Path) -> list[Chunk]:
    return [Chunk(content=f"chunk {i}", source_file=path, label=f"label {i}") for i in range(n)]


def _embeddings(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, DIMS), dtype=np.float64).astype(np.float32)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_key(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _write_cache(cache_dir: Path, path: Path, chunks: list[Chunk],
                 embeddings: np.ndarray, file_hash: str) -> None:
    key = _cache_key(path)
    (cache_dir / f"{key}.pkl").write_bytes(pickle.dumps({
        "file_hash": file_hash,
        "chunks": chunks,
        "embeddings": embeddings,
    }))


# ---------------------------------------------------------------------------
# build() — cache miss
# ---------------------------------------------------------------------------

def test_build_embeds_when_no_cache(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    chunks = _chunks(1, f)
    embs = _embeddings(1)

    with patch("pmca.rag.store.chunk_file", return_value=chunks):
        with patch("pmca.rag.store.embed", return_value=embs) as mock_embed:
            VectorStore().build([f], cache_dir)

    mock_embed.assert_called_once()


def test_build_writes_cache_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    chunks = _chunks(2, f)
    embs = _embeddings(2)

    with patch("pmca.rag.store.chunk_file", return_value=chunks):
        with patch("pmca.rag.store.embed", return_value=embs):
            VectorStore().build([f], cache_dir)

    cache_file = cache_dir / f"{_cache_key(f)}.pkl"
    assert cache_file.exists()
    data = pickle.loads(cache_file.read_bytes())
    assert "file_hash" in data
    assert "chunks" in data
    assert "embeddings" in data


# ---------------------------------------------------------------------------
# build() — cache hit
# ---------------------------------------------------------------------------

def test_build_skips_embed_when_cache_matches(tmp_path):
    f = tmp_path / "code.py"
    content = b"x = 1"
    f.write_bytes(content)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    chunks = _chunks(1, f)
    embs = _embeddings(1)
    _write_cache(cache_dir, f, chunks, embs, _content_hash(content))

    with patch("pmca.rag.store.embed") as mock_embed:
        VectorStore().build([f], cache_dir)

    mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# build() — cache invalidation
# ---------------------------------------------------------------------------

def test_build_reembeds_when_content_changed(tmp_path):
    f = tmp_path / "code.py"
    f.write_bytes(b"x = 2")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    stale_chunks = _chunks(1, f)
    stale_embs = _embeddings(1)
    _write_cache(cache_dir, f, stale_chunks, stale_embs, _content_hash(b"x = 1"))

    new_chunks = _chunks(1, f)
    new_embs = _embeddings(1)

    with patch("pmca.rag.store.chunk_file", return_value=new_chunks):
        with patch("pmca.rag.store.embed", return_value=new_embs) as mock_embed:
            VectorStore().build([f], cache_dir)

    mock_embed.assert_called_once()


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

def _store_with(chunks: list[Chunk], embeddings: np.ndarray) -> VectorStore:
    store = VectorStore()
    store._chunks = chunks
    store._embeddings = embeddings
    return store


def test_query_returns_top_k(tmp_path):
    n = 5
    f = tmp_path / "f.py"
    chunks = _chunks(n, f)
    # unit vectors along each axis so chunk i is perfectly matched by a query
    # vector with a 1 at position i
    embeddings = np.zeros((n, DIMS), dtype=np.float32)
    for i in range(n):
        embeddings[i, i] = 1.0

    store = _store_with(chunks, embeddings)

    query_vec = np.zeros((1, DIMS), dtype=np.float32)
    query_vec[0, 2] = 1.0  # most similar to chunk 2

    with patch("pmca.rag.store.embed", return_value=query_vec):
        result = store.query("anything", top_k=3)

    assert len(result) == 3
    assert result[0] is chunks[2]  # highest cosine similarity


def test_query_returns_fewer_than_top_k_when_store_is_small(tmp_path):
    f = tmp_path / "f.py"
    chunks = _chunks(2, f)
    embeddings = _embeddings(2)
    store = _store_with(chunks, embeddings)

    query_vec = _embeddings(1)
    with patch("pmca.rag.store.embed", return_value=query_vec):
        result = store.query("anything", top_k=10)

    assert len(result) == 2


def test_query_results_sorted_by_descending_similarity(tmp_path):
    f = tmp_path / "f.py"
    n = 4
    chunks = _chunks(n, f)
    embeddings = np.zeros((n, DIMS), dtype=np.float32)
    # set different magnitudes along dim 0 so similarities are predictable
    for i in range(n):
        embeddings[i, 0] = float(i + 1)  # chunk 3 most similar to query

    store = _store_with(chunks, embeddings)

    query_vec = np.zeros((1, DIMS), dtype=np.float32)
    query_vec[0, 0] = 1.0

    with patch("pmca.rag.store.embed", return_value=query_vec):
        result = store.query("anything", top_k=n)

    assert result[0] is chunks[n - 1]  # highest dot product
    assert result[-1] is chunks[0]      # lowest
