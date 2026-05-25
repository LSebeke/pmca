"""
Integration smoke test — full pipeline with mocked OpenAI, real files.

Covers: config loading → chunking → embedding (mocked) → cache write →
        RAG query → attachment resolution → message assembly →
        chat completion (mocked) → JSONL log write → exit.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmca.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_mock():
    """Mock OpenAI client for both embeddings and chat."""
    client = MagicMock()

    def _fake_embed(**kwargs):
        n = len(kwargs["input"])
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
        return resp

    client.embeddings.create.side_effect = _fake_embed

    chat_resp = MagicMock()
    chat_resp.choices[0].message.content = "Here is my answer."
    chat_resp.choices[0].message.tool_calls = None
    client.chat.completions.create.return_value = chat_resp

    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    rag_file = tmp_path / "utils.py"
    rag_file.write_text(
        "def greet(name: str) -> str:\n"
        "    return f'Hello, {name}'\n"
    )

    attachment_file = tmp_path / "context.py"
    attachment_file.write_text("SECRET = 'do-not-share'\n")

    log_folder = tmp_path / "logs"
    cfg = tmp_path / "Test.yaml"
    cfg.write_text(f"""\
name: Test
model: gpt-4o-mini
system_prompt: "You are a helpful assistant."
rag_files:
  - {rag_file}
top_k_chunks: 1
log_folder: {log_folder}
""")
    return {"config": cfg, "attachment": attachment_file, "log_folder": log_folder}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_full_session_writes_correct_jsonl_log(project, monkeypatch):
    attachment = project["attachment"]
    log_folder = project["log_folder"]

    mock_prompt = MagicMock()
    mock_prompt.prompt.side_effect = [
        f"What does [[{attachment}]] do?",
        EOFError(),
    ]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("openai.OpenAI", return_value=_openai_mock()):
        with patch("pmca.repl.PromptSession", return_value=mock_prompt):
            with patch("sys.argv", ["pmca", str(project["config"]), "--unsafe"]):
                main()

    # Two log entries written
    jsonl_files = list(log_folder.glob("chat_*.jsonl"))
    assert len(jsonl_files) == 1, "Expected exactly one chat log file"

    all_entries = [
        json.loads(line)
        for line in jsonl_files[0].read_text().splitlines()
        if line.strip()
    ]

    # Verify session-start entries are present
    assert any(e.get("type") == "system_prompt" for e in all_entries)

    # Extract exchange entries for the rest of the assertions
    entries = [e for e in all_entries if e.get("type") == "exchange"]
    assert len(entries) == 2

    user_entry, asst_entry = entries

    # Roles
    assert user_entry["role"] == "user"
    assert asst_entry["role"] == "assistant"

    # Assistant response matches the mock
    assert asst_entry["content"] == "Here is my answer."

    # User message has had [[path]] replaced with CONTEXT_1
    assert "CONTEXT_1" in user_entry["content"]
    assert str(attachment) not in user_entry["content"]

    # RAG chunks present on user entry
    assert isinstance(user_entry["rag_chunks"], list)
    assert len(user_entry["rag_chunks"]) >= 1
    assert "greet" in user_entry["rag_chunks"][0]["label"]

    # Attachment metadata on user entry
    assert len(user_entry["attachments"]) == 1
    assert user_entry["attachments"][0]["identifier"] == "CONTEXT_1"
    assert user_entry["attachments"][0]["size_warning"] is False

    # Timestamps present on both entries
    assert user_entry["timestamp"]
    assert asst_entry["timestamp"]

    # Cache written for the RAG file
    cache_files = list((log_folder / "cache").glob("*.pkl"))
    assert len(cache_files) == 1


def test_second_run_uses_cache(project, monkeypatch):
    """Second invocation with unchanged RAG file skips re-embedding."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    mock_client = _openai_mock()
    mock_prompt = MagicMock()
    mock_prompt.prompt.side_effect = [EOFError()]

    # First run — embeds
    with patch("openai.OpenAI", return_value=mock_client):
        with patch("pmca.repl.PromptSession", return_value=mock_prompt):
            with patch("sys.argv", ["pmca", str(project["config"])]):
                main()

    first_embed_count = mock_client.embeddings.create.call_count

    # Second run — should load from cache, not re-embed
    mock_client2 = _openai_mock()
    mock_prompt2 = MagicMock()
    mock_prompt2.prompt.side_effect = [EOFError()]

    with patch("openai.OpenAI", return_value=mock_client2):
        with patch("pmca.repl.PromptSession", return_value=mock_prompt2):
            with patch("sys.argv", ["pmca", str(project["config"])]):
                main()

    assert mock_client2.embeddings.create.call_count == 0
