from __future__ import annotations

import openai
import numpy as np

_BATCH_SIZE = 100
_MODEL = "text-embedding-3-small"
_DIMS = 1536


class EmbedError(Exception):
    pass


def embed(texts: list[str]) -> np.ndarray:
    client = openai.OpenAI()
    results: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        try:
            response = client.embeddings.create(model=_MODEL, input=batch)
        except openai.OpenAIError as exc:
            raise EmbedError(str(exc)) from exc
        results.extend(item.embedding for item in response.data)

    return np.array(results, dtype=np.float32)
