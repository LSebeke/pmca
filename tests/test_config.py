import os
import textwrap
from pathlib import Path

import pytest

from pmca.config import Config, ConfigError, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def minimal_yaml(rag_file: Path, log_folder: Path) -> str:
    return f"""\
name: TestConfig
model: gpt-4o-mini
system_prompt: "You are helpful."
rag_files:
  - {rag_file}
log_folder: {log_folder}
"""


# ---------------------------------------------------------------------------
# Load valid config
# ---------------------------------------------------------------------------

def test_load_valid_config_returns_config(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    cfg_path = write_yaml(tmp_path, "MyConfig.yaml", minimal_yaml(rag_file, log_folder))

    cfg = load_config(str(cfg_path))

    assert isinstance(cfg, Config)
    assert cfg.name == "TestConfig"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.system_prompt == "You are helpful."
    assert cfg.rag_files == [rag_file]
    assert cfg.log_folder == log_folder


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def test_bare_name_resolves_to_package_configs_dir(tmp_path, monkeypatch):
    """A name with no path separators and no .yaml suffix resolves to
    <pmca_package_dir>/configs/<name>.yaml."""
    import pmca.config as config_module

    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    fake_configs = tmp_path / "fake_configs"
    fake_configs.mkdir()
    write_yaml(fake_configs, "MyConfig.yaml", minimal_yaml(rag_file, log_folder))

    monkeypatch.setattr(config_module, "_CONFIGS_DIR", fake_configs)

    cfg = load_config("MyConfig")
    assert cfg.name == "TestConfig"


def test_path_like_with_slash_used_directly(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    cfg_path = write_yaml(tmp_path, "custom.yaml", minimal_yaml(rag_file, log_folder))

    cfg = load_config(str(cfg_path))
    assert cfg.name == "TestConfig"


def test_path_like_ending_in_yaml_used_directly(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    cfg_path = write_yaml(tmp_path, "myconfig.yaml", minimal_yaml(rag_file, log_folder))

    cfg = load_config(str(cfg_path))
    assert cfg.name == "TestConfig"


# ---------------------------------------------------------------------------
# File-not-found
# ---------------------------------------------------------------------------

def test_raises_config_error_when_file_does_not_exist(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nonexistent.yaml"))


def test_bare_name_not_found_raises_config_error(tmp_path, monkeypatch):
    import pmca.config as config_module

    fake_configs = tmp_path / "fake_configs"
    fake_configs.mkdir()

    monkeypatch.setattr(config_module, "_CONFIGS_DIR", fake_configs)

    with pytest.raises(ConfigError, match="not found"):
        load_config("NoSuchConfig")


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field,yaml_snippet", [
    ("name",         "model: gpt-4o\nsystem_prompt: x\nrag_files: []\nlog_folder: ~/logs\n"),
    ("model",        "name: x\nsystem_prompt: x\nrag_files: []\nlog_folder: ~/logs\n"),
    ("system_prompt","name: x\nmodel: gpt-4o\nrag_files: []\nlog_folder: ~/logs\n"),
    ("log_folder",   "name: x\nmodel: gpt-4o\nsystem_prompt: x\nrag_files: []\n"),
])
def test_missing_required_field_raises(tmp_path, missing_field, yaml_snippet):
    cfg_path = write_yaml(tmp_path, "bad.yaml", yaml_snippet)
    with pytest.raises(ConfigError, match=missing_field):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# rag_files validation
# ---------------------------------------------------------------------------

def test_raises_when_rag_file_path_not_absolute(tmp_path):
    log_folder = tmp_path / "logs"
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - relative/path/file.py
log_folder: {log_folder}
"""
    cfg_path = write_yaml(tmp_path, "bad.yaml", yaml_content)
    with pytest.raises(ConfigError, match="absolute"):
        load_config(str(cfg_path))


def test_raises_when_rag_file_does_not_exist(tmp_path):
    log_folder = tmp_path / "logs"
    missing = tmp_path / "no_such_file.py"
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - {missing}
log_folder: {log_folder}
"""
    cfg_path = write_yaml(tmp_path, "bad.yaml", yaml_content)
    with pytest.raises(ConfigError, match="not found|does not exist"):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# log_folder validation
# ---------------------------------------------------------------------------

def test_raises_when_log_folder_not_absolute(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - {rag_file}
log_folder: relative/logs
"""
    cfg_path = write_yaml(tmp_path, "bad.yaml", yaml_content)
    with pytest.raises(ConfigError, match="absolute"):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# Optional field defaults
# ---------------------------------------------------------------------------

def test_optional_fields_default_correctly(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    cfg_path = write_yaml(tmp_path, "MyConfig.yaml", minimal_yaml(rag_file, log_folder))
    cfg = load_config(str(cfg_path))

    assert cfg.max_attachment_kb == 500
    assert cfg.history_token_budget == 4000
    assert cfg.temperature is None
    assert cfg.max_tokens is None
    assert cfg.top_p is None
    assert cfg.frequency_penalty is None
    assert cfg.presence_penalty is None
    assert cfg.startup_docs == []
    assert cfg.write_allowed_dirs == []
    assert cfg.read_allowed_dirs == []


def test_write_allowed_dirs_loads_absolute_paths(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"
    allowed = tmp_path / "output"

    yaml_content = minimal_yaml(rag_file, log_folder) + f"write_allowed_dirs:\n  - {allowed}\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    cfg = load_config(str(cfg_path))

    assert cfg.write_allowed_dirs == [allowed]


def test_read_allowed_dirs_loads_absolute_paths(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"
    allowed = tmp_path / "src"

    yaml_content = minimal_yaml(rag_file, log_folder) + f"read_allowed_dirs:\n  - {allowed}\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    cfg = load_config(str(cfg_path))

    assert cfg.read_allowed_dirs == [allowed]


def test_read_allowed_dirs_raises_when_not_absolute(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + "read_allowed_dirs:\n  - relative/src\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)

    with pytest.raises(ConfigError, match="absolute"):
        load_config(str(cfg_path))


def test_write_allowed_dirs_raises_when_not_absolute(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + "write_allowed_dirs:\n  - relative/output\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)

    with pytest.raises(ConfigError, match="absolute"):
        load_config(str(cfg_path))


def test_optional_fields_override(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + textwrap.dedent("""\
        max_attachment_kb: 200
        history_token_budget: 2000
        temperature: 0.7
        max_tokens: 1000
        top_p: 0.9
        frequency_penalty: 0.1
        presence_penalty: 0.2
    """)
    cfg_path = write_yaml(tmp_path, "MyConfig.yaml", yaml_content)
    cfg = load_config(str(cfg_path))

    assert cfg.max_attachment_kb == 200
    assert cfg.history_token_budget == 2000
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.max_tokens == 1000
    assert cfg.top_p == pytest.approx(0.9)
    assert cfg.frequency_penalty == pytest.approx(0.1)
    assert cfg.presence_penalty == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# startup_docs
# ---------------------------------------------------------------------------

def test_startup_docs_loads_content_from_files(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    doc1 = tmp_path / "framework.md"
    doc1.write_text("# Framework docs")
    doc2 = tmp_path / "conventions.md"
    doc2.write_text("# Coding conventions")

    yaml_content = minimal_yaml(rag_file, log_folder) + f"startup_docs:\n  - {doc1}\n  - {doc2}\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    cfg = load_config(str(cfg_path))

    assert cfg.startup_docs == [(doc1, "# Framework docs"), (doc2, "# Coding conventions")]


def test_startup_docs_raises_when_path_not_absolute(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + "startup_docs:\n  - relative/doc.md\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)

    with pytest.raises(ConfigError, match="absolute"):
        load_config(str(cfg_path))


def test_startup_docs_raises_when_path_does_not_exist(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"
    missing_doc = tmp_path / "no_such_doc.md"

    yaml_content = minimal_yaml(rag_file, log_folder) + f"startup_docs:\n  - {missing_doc}\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)

    with pytest.raises(ConfigError, match="not found"):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# ~ expansion
# ---------------------------------------------------------------------------

def test_tilde_in_startup_docs_is_expanded(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - {rag_file}
log_folder: {log_folder}
startup_docs:
  - ~/nonexistent_pmca_tilde_doc_test.md
"""
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(cfg_path))


def test_tilde_in_rag_files_is_expanded(tmp_path):
    log_folder = tmp_path / "logs"
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - ~/nonexistent_pmca_tilde_rag_test.py
log_folder: {log_folder}
"""
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(cfg_path))


def test_tilde_in_log_folder_is_valid(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = f"""\
name: x
model: gpt-4o
system_prompt: x
rag_files:
  - {rag_file}
log_folder: ~/pmca_tilde_test_logs
"""
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.log_folder == Path("~/pmca_tilde_test_logs").expanduser()


# ---------------------------------------------------------------------------
# system_context_fields
# ---------------------------------------------------------------------------

def test_system_context_fields_defaults_to_empty_list(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    cfg_path = write_yaml(tmp_path, "cfg.yaml", minimal_yaml(rag_file, log_folder))
    cfg = load_config(str(cfg_path))

    assert cfg.system_context_fields == []


def test_system_context_fields_parsed_from_yaml(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + "system_context_fields:\n  - datetime\n  - os\n"
    cfg_path = write_yaml(tmp_path, "cfg.yaml", yaml_content)
    cfg = load_config(str(cfg_path))

    assert cfg.system_context_fields == ["datetime", "os"]


# ---------------------------------------------------------------------------
# Unknown keys ignored
# ---------------------------------------------------------------------------

def test_unknown_yaml_keys_are_ignored(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    log_folder = tmp_path / "logs"

    yaml_content = minimal_yaml(rag_file, log_folder) + "unknown_key: some_value\n"
    cfg_path = write_yaml(tmp_path, "MyConfig.yaml", yaml_content)

    cfg = load_config(str(cfg_path))
    assert cfg.name == "TestConfig"


# ---------------------------------------------------------------------------
# test_dir and test_timeout
# ---------------------------------------------------------------------------

def test_test_dir_defaults_to_none(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    cfg_path = write_yaml(tmp_path, "c.yaml", minimal_yaml(rag_file, tmp_path / "logs"))
    cfg = load_config(str(cfg_path))
    assert cfg.test_dir is None


def test_test_timeout_defaults_to_60(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    cfg_path = write_yaml(tmp_path, "c.yaml", minimal_yaml(rag_file, tmp_path / "logs"))
    cfg = load_config(str(cfg_path))
    assert cfg.test_timeout == 60


def test_test_dir_parsed_from_yaml(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = minimal_yaml(rag_file, tmp_path / "logs") + f"test_dir: {tmp_path}\n"
    cfg_path = write_yaml(tmp_path, "c.yaml", yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.test_dir == tmp_path


def test_test_timeout_parsed_from_yaml(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = minimal_yaml(rag_file, tmp_path / "logs") + f"test_dir: {tmp_path}\ntest_timeout: 120\n"
    cfg_path = write_yaml(tmp_path, "c.yaml", yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.test_timeout == 120


def test_test_dir_raises_when_non_absolute(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = minimal_yaml(rag_file, tmp_path / "logs") + "test_dir: relative/path\n"
    cfg_path = write_yaml(tmp_path, "c.yaml", yaml_content)
    with pytest.raises(ConfigError):
        load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# rag depth fields
# ---------------------------------------------------------------------------

def test_rag_depth_fields_default_to_3_7_15(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    cfg_path = write_yaml(tmp_path, "c.yaml", minimal_yaml(rag_file, tmp_path / "logs"))
    cfg = load_config(str(cfg_path))
    assert cfg.rag_shallow_k == 3
    assert cfg.rag_medium_k == 7
    assert cfg.rag_deep_k == 15


def test_rag_depth_fields_accept_custom_values(tmp_path):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = minimal_yaml(rag_file, tmp_path / "logs") + "rag_shallow_k: 5\nrag_medium_k: 10\nrag_deep_k: 20\n"
    cfg_path = write_yaml(tmp_path, "c.yaml", yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.rag_shallow_k == 5
    assert cfg.rag_medium_k == 10
    assert cfg.rag_deep_k == 20


@pytest.mark.parametrize("field", ["rag_shallow_k", "rag_medium_k", "rag_deep_k"])
def test_rag_depth_field_raises_when_non_positive(tmp_path, field):
    rag_file = tmp_path / "code.py"
    rag_file.write_text("x = 1")
    yaml_content = minimal_yaml(rag_file, tmp_path / "logs") + f"{field}: 0\n"
    cfg_path = write_yaml(tmp_path, "c.yaml", yaml_content)
    with pytest.raises(ConfigError, match=field):
        load_config(str(cfg_path))
