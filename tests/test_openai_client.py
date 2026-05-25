from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import openai
import pytest

from pmca.config import Config
from pmca.openai_client import APIError, APITransientError, chat_completion
from pmca.types import ToolCallRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> Config:
    defaults = dict(
        name="test",
        model="gpt-4o-mini",
        system_prompt="You are helpful.",
        rag_files=[],
        top_k_chunks=3,
        log_folder=Path("/tmp/logs"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_request(), content=b'{"error": "err"}')


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("rate limited", response=_response(429), body=None)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(message="timeout", request=_request())


def _server_error() -> openai.APIStatusError:
    return openai.InternalServerError("server error", response=_response(500), body=None)


def _auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_response(401), body=None)


def _bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError("bad request", response=_response(400), body=None)


def _mock_success(content: str = "Hello!") -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    return response


# ---------------------------------------------------------------------------
# Successful call
# ---------------------------------------------------------------------------

def test_returns_assistant_message_on_success():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _mock_success("Hi!")
        result = chat_completion([{"role": "user", "content": "Hello"}], _config())
    assert result == "Hi!"


# ---------------------------------------------------------------------------
# Optional config params
# ---------------------------------------------------------------------------

def test_optional_params_passed_when_set():
    cfg = _config(temperature=0.7, max_tokens=100, top_p=0.9,
                  frequency_penalty=0.1, presence_penalty=0.2)
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = _mock_success()
        chat_completion([], cfg)

    kwargs = mock_create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 100
    assert kwargs["top_p"] == 0.9
    assert kwargs["frequency_penalty"] == 0.1
    assert kwargs["presence_penalty"] == 0.2


def test_none_params_omitted():
    cfg = _config()  # all optional params are None
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = _mock_success()
        chat_completion([], cfg)

    kwargs = mock_create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "top_p" not in kwargs


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------

def test_retries_on_rate_limit_error():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = [_rate_limit_error(), _rate_limit_error(), _mock_success()]
            result = chat_completion([], _config())
    assert result == "Hello!"
    assert mock_create.call_count == 3


def test_retries_on_connection_error():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = [_connection_error(), _mock_success()]
            result = chat_completion([], _config())
    assert result == "Hello!"


def test_retries_on_server_error():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = [_server_error(), _mock_success()]
            result = chat_completion([], _config())
    assert result == "Hello!"


def test_prints_retry_notice_each_attempt(capsys):
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = [_rate_limit_error(), _rate_limit_error(), _mock_success()]
            chat_completion([], _config())

    out = capsys.readouterr().out
    assert "[retrying... attempt 1/3]" in out
    assert "[retrying... attempt 2/3]" in out


def test_backoff_delays_are_1_2_4():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep") as mock_sleep:
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = [
                _rate_limit_error(), _rate_limit_error(), _rate_limit_error(), _mock_success()
            ]
            chat_completion([], _config())

    assert mock_sleep.call_args_list == [call(1), call(2), call(4)]


# ---------------------------------------------------------------------------
# Exhausted retries → APITransientError
# ---------------------------------------------------------------------------

def test_raises_transient_error_after_3_retries():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = _rate_limit_error()
            with pytest.raises(APITransientError):
                chat_completion([], _config())

    assert mock_create.call_count == 4  # 1 initial + 3 retries


# ---------------------------------------------------------------------------
# Permanent errors → immediate APIError, no retry
# ---------------------------------------------------------------------------

def test_raises_api_error_on_auth_error():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep") as mock_sleep:
            MockClient.return_value.chat.completions.create.side_effect = _auth_error()
            with pytest.raises(APIError):
                chat_completion([], _config())
    mock_sleep.assert_not_called()


def test_raises_api_error_on_bad_request():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep") as mock_sleep:
            MockClient.return_value.chat.completions.create.side_effect = _bad_request_error()
            with pytest.raises(APIError):
                chat_completion([], _config())
    mock_sleep.assert_not_called()


def test_no_retry_on_permanent_error():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        with patch("pmca.openai_client.time.sleep"):
            mock_create = MockClient.return_value.chat.completions.create
            mock_create.side_effect = _auth_error()
            with pytest.raises(APIError):
                chat_completion([], _config())
    assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------

def _mock_tool_call_response(tool_call_id: str = "call_abc", name: str = "write_file", args_json: str = '{"path": "/tmp/f.py", "content": "x", "description": "test"}') -> MagicMock:
    import json
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = name
    tool_call.function.arguments = args_json
    response = MagicMock()
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call]
    return response


def test_returns_tool_call_request_when_model_issues_tool_call():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _mock_tool_call_response()
        result = chat_completion([], _config(), tools=[{"type": "function", "function": {"name": "write_file"}}])

    assert isinstance(result, ToolCallRequest)
    assert result.tool_call_id == "call_abc"
    assert result.name == "write_file"
    assert result.arguments["path"] == "/tmp/f.py"


def test_tools_passed_to_api_with_parallel_tool_calls_false():
    tools = [{"type": "function", "function": {"name": "write_file"}}]
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = _mock_success()
        chat_completion([], _config(), tools=tools)

    kwargs = mock_create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["parallel_tool_calls"] is False


def test_tools_none_does_not_pass_tools_kwarg():
    with patch("pmca.openai_client.openai.OpenAI") as MockClient:
        mock_create = MockClient.return_value.chat.completions.create
        mock_create.return_value = _mock_success()
        chat_completion([], _config(), tools=None)

    kwargs = mock_create.call_args.kwargs
    assert "tools" not in kwargs
    assert "parallel_tool_calls" not in kwargs
