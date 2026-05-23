from __future__ import annotations

import time

import openai

from pmca.config import Config

_BACKOFF = [1, 2, 4]
_MAX_RETRIES = len(_BACKOFF)


class APIError(Exception):
    pass


class APITransientError(Exception):
    pass


def chat_completion(messages: list[dict], config: Config) -> str:
    client = openai.OpenAI()
    kwargs = _optional_params(config)
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except (openai.RateLimitError, openai.APIConnectionError) as exc:
            last_exc = exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
            else:
                raise APIError(str(exc)) from exc

        if attempt < _MAX_RETRIES:
            print(f"[retrying... attempt {attempt + 1}/{_MAX_RETRIES}]")
            time.sleep(_BACKOFF[attempt])

    raise APITransientError(str(last_exc)) from last_exc


def _is_transient(exc: openai.OpenAIError) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


def _optional_params(config: Config) -> dict:
    params = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
    }
    return {k: v for k, v in params.items() if v is not None}
