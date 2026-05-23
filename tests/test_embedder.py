import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import openai as openai_module

from pmca.rag.embedder import EmbedError, embed

DIMS = 1536


def _mock_response(n: int) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.0] * DIMS) for _ in range(n)]
    return response


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_embed_returns_correct_shape():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = _mock_response(2)
        result = embed(["hello", "world"])
    assert result.shape == (2, DIMS)


def test_embed_returns_float32():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = _mock_response(1)
        result = embed(["hello"])
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_embed_single_api_call_for_small_input():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.embeddings.create.return_value = _mock_response(2)
        embed(["hello", "world"])
    mock_client.embeddings.create.assert_called_once()


def test_embed_two_api_calls_for_101_texts():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.embeddings.create.side_effect = [_mock_response(100), _mock_response(1)]
        result = embed(["x"] * 101)
    assert mock_client.embeddings.create.call_count == 2
    assert result.shape == (101, DIMS)


def test_embed_batch_boundary_exactly_100():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_client.embeddings.create.return_value = _mock_response(100)
        result = embed(["x"] * 100)
    mock_client.embeddings.create.assert_called_once()
    assert result.shape == (100, DIMS)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_embed_raises_embed_error_on_api_failure():
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.side_effect = openai_module.OpenAIError("boom")
        with pytest.raises(EmbedError, match="boom"):
            embed(["hello"])


def test_embed_error_wraps_original_exception():
    original = openai_module.OpenAIError("original")
    with patch("pmca.rag.embedder.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.side_effect = original
        with pytest.raises(EmbedError) as exc_info:
            embed(["hello"])
    assert exc_info.value.__cause__ is original
