from pathlib import Path
from unittest.mock import patch

import pytest

from pmca.attachments import (
    AttachmentAborted,
    AttachmentError,
    parse_attachment_paths,
    resolve_attachments,
    substitute_identifiers,
)
from pmca.types import Attachment


# ---------------------------------------------------------------------------
# parse_attachment_paths
# ---------------------------------------------------------------------------

def test_parse_single_absolute_path(tmp_path):
    f = tmp_path / "file.py"
    paths = parse_attachment_paths(f"see [[{f}]] here")
    assert paths == [f]


def test_parse_multiple_paths(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.md"
    paths = parse_attachment_paths(f"[[{a}]] and [[{b}]]")
    assert paths == [a, b]


def test_parse_returns_empty_when_no_tokens():
    assert parse_attachment_paths("no attachments here") == []


def test_parse_strips_double_quotes_from_path(tmp_path):
    f = tmp_path / "file.py"
    paths = parse_attachment_paths(f'[["{f}"]]')
    assert paths == [f]


def test_parse_raises_for_relative_path():
    with pytest.raises(AttachmentError, match="absolute"):
        parse_attachment_paths("[[./relative.py]]")


def test_parse_raises_for_bare_filename():
    with pytest.raises(AttachmentError, match="absolute"):
        parse_attachment_paths("[[file.py]]")


# ---------------------------------------------------------------------------
# resolve_attachments — existence and size
# ---------------------------------------------------------------------------

def test_resolve_raises_when_file_missing(tmp_path):
    with pytest.raises(AttachmentError, match="not found|does not exist"):
        resolve_attachments([tmp_path / "missing.py"], max_attachment_kb=500, unsafe=True)


def test_resolve_assigns_context_identifiers_in_order(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1")
    b.write_text("y = 2")

    result = resolve_attachments([a, b], max_attachment_kb=500, unsafe=True)

    assert result[0].identifier == "CONTEXT_1"
    assert result[1].identifier == "CONTEXT_2"


def test_resolve_reads_file_content(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello(): pass")

    result = resolve_attachments([f], max_attachment_kb=500, unsafe=True)

    assert result[0].content == "def hello(): pass"


def test_resolve_size_warning_false_when_within_limit(tmp_path):
    f = tmp_path / "small.py"
    f.write_text("x = 1")

    result = resolve_attachments([f], max_attachment_kb=500, unsafe=True)

    assert result[0].size_warning is False


def test_resolve_size_warning_true_when_exceeds_limit(tmp_path):
    f = tmp_path / "big.py"
    f.write_bytes(b"x" * 600 * 1024)  # 600 KB

    result = resolve_attachments([f], max_attachment_kb=500, unsafe=True)

    assert result[0].size_warning is True


def test_resolve_size_warning_continues_after_warning(tmp_path):
    f = tmp_path / "big.py"
    f.write_bytes(b"x" * 600 * 1024)

    result = resolve_attachments([f], max_attachment_kb=500, unsafe=True)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# resolve_attachments — security prompt (safe mode)
# ---------------------------------------------------------------------------

def test_resolve_prompts_in_safe_mode(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")

    with patch("builtins.input", return_value="y") as mock_input:
        resolve_attachments([f], max_attachment_kb=500, unsafe=False)

    mock_input.assert_called_once()


def test_resolve_skips_prompt_in_unsafe_mode(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")

    with patch("builtins.input") as mock_input:
        resolve_attachments([f], max_attachment_kb=500, unsafe=True)

    mock_input.assert_not_called()


def test_resolve_raises_aborted_when_user_answers_n(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")

    with patch("builtins.input", return_value="n"):
        with pytest.raises(AttachmentAborted):
            resolve_attachments([f], max_attachment_kb=500, unsafe=False)


def test_resolve_raises_aborted_on_any_non_y_answer(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")

    with patch("builtins.input", return_value="maybe"):
        with pytest.raises(AttachmentAborted):
            resolve_attachments([f], max_attachment_kb=500, unsafe=False)


def test_resolve_proceeds_when_user_answers_y(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")

    with patch("builtins.input", return_value="y"):
        result = resolve_attachments([f], max_attachment_kb=500, unsafe=False)

    assert len(result) == 1


def test_resolve_prompt_mentions_path(tmp_path):
    f = tmp_path / "secret.py"
    f.write_text("x = 1")

    with patch("builtins.input", return_value="y") as mock_input:
        resolve_attachments([f], max_attachment_kb=500, unsafe=False)

    prompt_text = mock_input.call_args[0][0]
    assert str(f) in prompt_text


def test_resolve_aborted_on_first_rejection_skips_rest(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1")
    b.write_text("y = 2")

    with patch("builtins.input", return_value="n"):
        with pytest.raises(AttachmentAborted):
            resolve_attachments([a, b], max_attachment_kb=500, unsafe=False)


# ---------------------------------------------------------------------------
# substitute_identifiers
# ---------------------------------------------------------------------------

def test_substitute_replaces_path_with_identifier(tmp_path):
    f = tmp_path / "code.py"
    attachments = [Attachment(path=f, content="", identifier="CONTEXT_1", size_warning=False)]
    msg = f"look at [[{f}]] please"
    result = substitute_identifiers(msg, attachments)
    assert result == "look at CONTEXT_1 please"


def test_substitute_multiple(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    attachments = [
        Attachment(path=a, content="", identifier="CONTEXT_1", size_warning=False),
        Attachment(path=b, content="", identifier="CONTEXT_2", size_warning=False),
    ]
    msg = f"[[{a}]] and [[{b}]]"
    result = substitute_identifiers(msg, attachments)
    assert result == "CONTEXT_1 and CONTEXT_2"


def test_substitute_leaves_unmatched_tokens_unchanged(tmp_path):
    f = tmp_path / "code.py"
    other = tmp_path / "other.py"
    attachments = [Attachment(path=f, content="", identifier="CONTEXT_1", size_warning=False)]
    msg = f"[[{other}]] stays"
    result = substitute_identifiers(msg, attachments)
    assert result == f"[[{other}]] stays"


def test_substitute_matches_via_path_normalisation(tmp_path):
    f = tmp_path / "code.py"
    attachments = [Attachment(path=f, content="", identifier="CONTEXT_1", size_warning=False)]
    # Path normalises away the redundant "." so the lookup still matches
    dotted = str(tmp_path) + "/./code.py"
    result = substitute_identifiers(f"[[{dotted}]]", attachments)
    assert result == "CONTEXT_1"
