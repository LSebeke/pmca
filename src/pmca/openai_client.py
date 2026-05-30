from __future__ import annotations

import ast
import json
import time
from typing import TYPE_CHECKING

import openai

from pmca.config import Config
from pmca.types import ToolCallRequest

if TYPE_CHECKING:
    from pmca.logger import SessionLogger

_BACKOFF = [1, 2, 4]
_MAX_RETRIES = len(_BACKOFF)


class APIError(Exception):
    pass


class APITransientError(Exception):
    pass


class MalformedToolCallError(Exception):
    pass


def chat_completion(
    messages: list[dict],
    config: Config,
    tools: list[dict] | None = None,
    logger: SessionLogger | None = None,
) -> str | ToolCallRequest:
    client = openai.OpenAI()
    kwargs = _optional_params(config)
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = False
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                **kwargs,
            )
            duration = time.monotonic() - t0
            msg = response.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                if logger is not None:
                    logger.log_api_call(config.model, duration)
                    logger.log_api_payload(messages, f"[tool_call] {tc.function.name}({tc.function.arguments})")
                return ToolCallRequest(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_tool_arguments(tc.function.arguments),
                )
            if logger is not None:
                logger.log_api_call(config.model, duration)
                logger.log_api_payload(messages, msg.content or "")
            return msg.content
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


def _parse_tool_arguments(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise MalformedToolCallError(f"Cannot parse tool arguments: {raw!r}") from exc


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
