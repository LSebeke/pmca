# pmca — Poor Man's Coding Assistant

A terminal chat tool that wraps the OpenAI API with project-aware context. Point it at your codebase, and it can read files, write and edit code, run tests, and answer questions grounded in your actual project — all from a single interactive session.

---

## Features

- **RAG knowledge base** — index any set of local files at startup; the model queries them on demand via vector search rather than stuffing everything into every prompt
- **File tools** — `read_file`, `list_dir`, `search`, `find_files`, `get_definition` let the model navigate your codebase; `write_file`, `edit_file`, `insert_at_line`, `delete_file`, and `move_file` let it make changes
- **Git tools** — `git_status`, `git_log`, `git_diff`, `git_blame`, `git_show_file`, `git_branches`, `git_current_branch` give the model read-only access to your repo history; enabled by setting `git_root` in config
- **Test runner** — `run_tests` executes your test suite and feeds the output back to the model
- **File attachments** — paste `[[/absolute/path/to/file]]` into any message to inject a file verbatim into context
- **Skills** — inject reusable behaviour guides (`SKILL.md`) into the session on demand; the model can follow `read_file` links into the skill directory for supporting docs
- **Persistent scratchpad** — the model saves key excerpts from tool call returns across turns, so it doesn't lose what it read earlier in the session
- **Session resume** — pick up any previous session exactly where you left off
- **Tool progress output** — each tool call prints a one-line status before it executes (e.g. `[tool: edit_file /src/pmca/chat.py]`) and a result summary after (e.g. `[tool: edit_file /src/pmca/chat.py → ok]`), so you can follow long chains without guessing what the model is doing
- **Resilient tool call parsing** — tool call arguments are parsed with `json.loads` first, with an `ast.literal_eval` fallback for single-quoted Python dicts that some models emit; unparseable arguments raise `MalformedToolCallError` rather than a bare `JSONDecodeError`

### Safety

- **Write operations are gated by directory allowlists** — `write_allowed_dirs` and `read_allowed_dirs` in your config define the only locations the model can touch; requests outside those directories are rejected outright
- **Every write and edit requires explicit approval by default** — the tool prints the full path, byte count, reason, and a unified diff before asking `[y/N]`; the model cannot write anything without a keypress from you. Set `auto_approve_writes: true` in config (or `/set auto_approve_writes=true` at runtime) to skip these prompts while keeping the directory guard. Set `show_diff_on_auto_approve: true` (or `/set show_diff_on_auto_approve=true`) to still see the diff even when auto-approving — the write proceeds without a keypress but you can follow what changed
- **The model must read before it can edit or overwrite** — `edit_file`, `write_file` (on existing files), `insert_at_line`, `delete_file`, and `move_file` are all blocked if the model has not called `read_file` on that path earlier in the same turn; this prevents blind overwrites. After any successful write, the path is removed from the read set — the model must re-read before making a further edit to the same file.
- **Git tools are read-only by design** — git operations use GitPython's library API rather than a shell subprocess. Paths passed to `git_diff`, `git_blame`, and `git_show_file` are validated against `read_allowed_dirs`. No write or remote operations (push, fetch, commit) are exposed.
- **File attachments prompt for secrets review** — before injecting any `[[filepath]]` attachment, the tool asks whether you have reviewed the file for secrets (prompt can be disabled with `--unsafe` if you know you will be working with non-secret files)
- **All file paths must be absolute** — all file paths for read and write operations, including attachments, must be absolute to avoid accidental passing of files

---

## Quick start

Create a config file, e.g. `~/.pmca/myproject.yaml`:

```yaml
name: myproject
model: gpt-4.1
system_prompt: "You are a helpful Python coding assistant."
log_folder: ~/.pmca/logs

# Files to index into the RAG knowledge base
rag_files:
  - /home/user/myproject/src/core.py
  - /home/user/myproject/docs/architecture.md

# Allow the model to read anywhere inside the project
read_allowed_dirs:
  - /home/user/myproject

# Allow the model to write only inside src/ and tests/
write_allowed_dirs:
  - /home/user/myproject/src
  - /home/user/myproject/tests

# Enable git tools (read-only: status, log, diff, blame, show_file, branches)
git_root: /home/user/myproject

# Run tests with pixi run pytest
test_dir: /home/user/myproject
```

Start a session:

```
pixi run pmca ~/.pmca/myproject.yaml
```

Example exchange:

```
> Why does parse_config raise on missing log_folder?

[model calls query_knowledge_base, reads relevant chunks]

The function calls _validate_required() which iterates over REQUIRED_FIELDS = ("name",
"model", "system_prompt", "log_folder"). If log_folder is absent it raises ConfigError
with the field name. The check happens before any path expansion...

> Can you add a clearer error message that includes the config file path?

[tool: read_file /home/user/myproject/src/config.py]
[tool: edit_file /home/user/myproject/src/config.py]
[edit_file] /home/user/myproject/src/config.py
Reason: Improve ConfigError message to include config file path
--- /home/user/myproject/src/config.py
+++ /home/user/myproject/src/config.py
...diff...
Approve? [y/N] y

Done. I've updated the error message to include the config path...
```

---

## Installation

Install [pixi](https://prefix.dev/docs/pixi/overview), then from the repo root:

```
pixi install
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...        # Linux / macOS
set OPENAI_API_KEY=sk-...           # Windows CMD
$env:OPENAI_API_KEY="sk-..."        # Windows PowerShell
```

---

## Usage

```
pixi run pmca <config> [--unsafe] [--resume <path>]
```

| Argument | Description |
|---|---|
| `config` | Config name (looked up in `<package>/configs/<name>.yaml`) or a direct path to a `.yaml` file |
| `--unsafe` | Skip the secrets-review prompt for `[[filepath]]` attachments |
| `--resume <path>` | Resume a previous session from a `chat_<timestamp>.jsonl` log file |

### Resuming a session

```
pixi run pmca ~/.pmca/myproject.yaml --resume ~/.pmca/logs/chat_2025-05-14_15-32-10.jsonl
```

The tool replays the full session state — system prompt, all attachments, and conversation history — and prints the last assistant response. The existing log file is appended to rather than replaced.

---

## Config reference

### Annotated example

```yaml
name: myproject                        # identifier shown at startup
model: gpt-4.1                         # any OpenAI chat model
system_prompt: "You are helpful."      # injected as the first system message every turn

log_folder: ~/.pmca/logs               # where chat and debug logs are written; created if absent

# RAG knowledge base — indexed at startup, queried on demand
rag_files:
  - /abs/path/to/file.py
  - /abs/path/to/docs.md

# Injected verbatim as system messages every turn (e.g. API docs, project conventions)
startup_docs:
  - /abs/path/to/conventions.md

# Tool access — omit a section to disable those tools entirely
read_allowed_dirs:
  - /abs/path/to/project
write_allowed_dirs:
  - /abs/path/to/project/src
test_dir: /abs/path/to/project         # enables run_tests; uses pixi if pixi.toml is present

# Inject runtime context into the system prompt
system_context_fields:
  - datetime                           # current date/time
  - os                                 # operating system
  - shell                              # $SHELL / %COMSPEC%
```

### Field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | **required** | Config identifier |
| `model` | string | **required** | Chat model name |
| `system_prompt` | string | **required** | Base system message sent every turn |
| `log_folder` | path | **required** | Folder for `chat_*.jsonl` and `debug_*.log` files |
| `rag_files` | list of paths | `[]` | Files to index into the RAG knowledge base |
| `startup_docs` | list of paths | `[]` | Files injected verbatim as system messages every turn |
| `read_allowed_dirs` | list of paths | `[]` | Enables `read_file`, `list_dir`, `search`, `find_files`, `get_definition` |
| `write_allowed_dirs` | list of paths | `[]` | Enables `write_file`, `edit_file`, `insert_at_line`, `delete_file`, `move_file` |
| `auto_approve_writes` | bool | `false` | Skip per-op approval prompts for write ops; directory guard still applies |
| `show_diff_on_auto_approve` | bool | `false` | Print unified diff even when `auto_approve_writes: true`; write proceeds without a keypress |
| `git_root` | path | `null` | Enables all `git_*` tools (read-only; must exist) |
| `skills_dir` | path | `null` | Directory of skill subdirectories; enables `/skill` command |
| `test_dir` | path | `null` | Enables `run_tests`; uses `pixi run pytest` if `pixi.toml` present |
| `test_timeout` | int (seconds) | `60` | Timeout for `run_tests` |
| `rag_shallow_k` | int | `3` | Chunks returned at `depth="shallow"` |
| `rag_medium_k` | int | `7` | Chunks returned at `depth="medium"` |
| `rag_deep_k` | int | `15` | Chunks returned at `depth="deep"` |
| `max_attachment_kb` | int | `500` | Soft size warning threshold for `[[filepath]]` attachments |
| `history_token_budget` | int | `4000` | Approximate token budget for conversation history (1 token ≈ 4 chars); oldest turns dropped first |
| `max_scratchpad_entries` | int | `20` | Hard cap on persisted scratchpad entries |
| `system_context_fields` | list | `[]` | Runtime fields to inject: `datetime`, `os`, `shell` |
| `temperature` | float | API default | Passed through to OpenAI |
| `max_tokens` | int | API default | Passed through to OpenAI |
| `top_p` | float | API default | Passed through to OpenAI |
| `frequency_penalty` | float | API default | Passed through to OpenAI |
| `presence_penalty` | float | API default | Passed through to OpenAI |

All path fields accept `~` and are expanded at load time.

---

## Attachments

Paste a file path wrapped in double brackets anywhere in a message:

```
Here is the file I'm working on: [[/abs/path/to/main.py]]
Can you spot any issues?
```

The token is replaced with an identifier (`CONTEXT_1`, `CONTEXT_2`, …) in the message sent to the model, and the file content is injected as a system message that persists for the rest of the session. On Windows, paths copied from Explorer (which wraps them in quotes) are handled automatically.

The tool prompts for secrets review before injecting. Pass `--unsafe` on startup to skip this prompt.

---

## Skills

Skills are reusable behaviour guides you can inject into the session on demand. Each skill is a directory containing a `SKILL.md` entry point and optional supporting `.md` files.

Enable skills by pointing `skills_dir` at a directory of skill directories:

```yaml
skills_dir: /abs/path/to/skills   # each subdirectory must contain SKILL.md
```

During a session:

| Command | Effect |
|---|---|
| `/skill` | List available skills (`*` = active) |
| `/skill <name>` | Activate — injects `SKILL.md` as a system message and grants `read_file` access to the skill directory so the model can fetch supporting docs |
| `/skill remove <name>` | Deactivate — removes from context and revokes `read_file` access |

When a skill is active the model sees:

```
[SKILL: tdd]
Directory: /abs/path/to/skills/tdd
Supporting files in this directory are readable via read_file.
---
<SKILL.md content>
---
```

The model can call `read_file` on sibling files (e.g. `mocking.md`, `tests.md`) if the skill content references them.

---

## Commands

| Command | Effect |
|---|---|
| `/set history_token_budget=N` | Set history token budget for this session |
| `/set test_timeout=N` | Set test run timeout in seconds for this session |
| `/set max_attachment_kb=N` | Set max attachment size in KB for this session |
| `/set model=NAME` | Switch the model for this session |
| `/set temperature=F\|none` | Set sampling temperature (0.0–2.0), or `none` to clear |
| `/set max_tokens=N\|none` | Set max response tokens, or `none` to clear the limit |
| `/set auto_approve_writes=true\|false` | Skip (or restore) per-op write approval prompts for this session |
| `/set show_diff_on_auto_approve=true\|false` | Show unified diff even when auto-approving writes (write still proceeds without a keypress) |
| `/read add <path>` | Add a directory to `read_allowed_dirs` for this session (requires approval) |
| `/read remove <path>` | Remove a directory from `read_allowed_dirs` for this session (requires approval) |
| `/extract <path>` | Extract code blocks from the last response into `<path>`; fence language inferred from extension (`.py`, `.yaml`/`.yml`, `.json`, `.toml`, `.sh`, `.md`) |
| `/scratchpad` | Print all scratchpad entries (title + content); prints "Scratchpad is empty." if none |
| `/clear` | Clear conversation history and scratchpad; rotate to a new log file |
| `/help` | Print command reference and key bindings |
| `/exit` | End session (also: Ctrl+C) |

**Key bindings:** `↑` recalls previous input for editing; `Esc` clears current input.

---

## Session logs

Each session writes two files to `log_folder`:

- `chat_<timestamp>.jsonl` — full exchange log: system prompt, startup docs, every user and assistant message with attachments, and every tool call with its arguments, approval status, and result
- `debug_<timestamp>.log` — timestamped debug entries: session start, API call timings (`chat_completion: 3.4s, model=gpt-4.1`), full API payloads (messages + response), tool dispatch results, history trim events, and attachment resolution

The JSONL log is the authoritative record of the session and is used verbatim by `--resume`. Partial logs written before an unexpected exit are valid and resumable.
