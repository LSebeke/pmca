# Implementation Plan: `pmca`

Each phase follows red-green-refactor: write a failing test first, implement the minimum to pass it, then clean up. Phases are ordered bottom-up — each builds only on what's already tested.

---

## Phase 1 — `config.py`

**Why first:** No dependencies. All other components receive a `Config` object; getting validation right early prevents bad state from propagating.

### Red
- Config loads a valid YAML file and returns a populated `Config` dataclass
- A bare name (e.g. `"MyConfig"`) resolves to `<pmca_package_dir>/configs/MyConfig.yaml`
- A path-like argument (e.g. `"./custom.yaml"` or `"/abs/path/config.yaml"`) is used directly
- Raises `ConfigError` when the resolved file does not exist
- Raises `ConfigError` when a required field is missing
- Raises `ConfigError` when a `rag_files` path is not absolute
- Raises `ConfigError` when a `rag_files` path does not exist
- Raises `ConfigError` when `log_folder` is not absolute
- Optional fields default correctly (`max_attachment_kb=500`, `history_token_budget=4000`)
- Unknown YAML keys are ignored (no error)

### Green
Implement `load_config(config_name: str) -> Config`:
- If `config_name` contains a path separator (`/` or `\`) or ends in `.yaml`: treat as a direct file path (absolute, or relative to cwd)
- Otherwise: look up `<pmca_package_dir>/configs/<config_name>.yaml`
- Parse with `pyyaml`, validate all rules, instantiate `Config`

### Refactor
- Extract field validators into small pure functions for reuse

---

## Phase 2 — `rag/chunker.py`

**Why here:** Pure transformation — no I/O beyond reading a file. Tests are fast and deterministic.

### Red
- Chunking a `.py` file returns one `Chunk` per top-level function/class/method
- Each Python chunk label includes the node name and line range
- Module-level code outside any def/class becomes a single chunk labelled `"module-level"`
- Chunking a `.md` file splits on ATX headings; each chunk carries its heading as label
- Chunking a `.txt` file splits on double newlines
- A file with a single paragraph / no headings returns one chunk

### Green
Implement `chunk_file(path: Path) -> list[Chunk]` dispatching to `_chunk_python` and `_chunk_prose`.

### Refactor
- Deduplicate label-building logic between Python node types

---

## Phase 3 — `rag/embedder.py`

**Why here:** Thin wrapper — tests mock the OpenAI client. Establishes the interface before the store depends on it.

### Red
- `embed(["hello", "world"])` calls the OpenAI embeddings API once and returns an `np.ndarray` of shape `(2, 1536)`
- Batches correctly when input exceeds 100 texts (two API calls for 101 texts)
- Raises `EmbedError` when the API call fails

### Green
Implement `embed(texts: list[str]) -> np.ndarray` with batching. Mock `openai.embeddings.create` in tests.

### Refactor
- Ensure the function signature is the only public surface (no leaking OpenAI types)

---

## Phase 4 — `rag/store.py`

**Why here:** Depends on chunker and embedder (both tested). Cache logic is the most complex part of the RAG pipeline.

### Red
- `build()` calls `embed()` for each file with no existing cache
- `build()` loads from cache and skips `embed()` for an unchanged file
- `build()` re-embeds a file whose content has changed (hash mismatch)
- Cache file is written to `<cache_dir>/<hex>.pkl` after embedding
- `query()` returns exactly `top_k` chunks sorted by cosine similarity
- `query()` returns fewer than `top_k` chunks when the store has fewer chunks total

### Green
Implement `VectorStore.build()` and `VectorStore.query()`. Use `hashlib.sha256` for both path-based cache key and content-based invalidation.

### Refactor
- Isolate cache read/write into private helpers `_load_cache` / `_save_cache`

---

## Phase 5 — `logger.py`

**Why here:** Pure I/O with no dependencies on other pmca modules. Tests use a temp directory.

### Red
- `log_exchange()` appends two valid JSON lines (user + assistant) to the `.jsonl` file
- User log entry includes `rag_chunks` and `attachments` fields
- `log_debug()` appends a timestamped line to the `.log` file
- Both files are flushed immediately (data visible before `close()`)
- A second `SessionLogger` for the same timestamp writes to the same files (append, not overwrite)

### Green
Implement `SessionLogger.__init__`, `log_exchange`, `log_debug`, `close`.

### Refactor
- Extract JSON serialisation of `Chunk` and `Attachment` into small helpers

---

## Phase 6 — `attachments.py`

**Why here:** Pure parsing + file I/O. Depends on nothing but stdlib and the `Attachment` datatype.

### Red
- `parse_attachment_paths("see [[/abs/path/file.py]] here")` returns `[Path("/abs/path/file.py")]`
- Raises `AttachmentError` for a non-absolute path (e.g. `[[./relative.py]]`)
- `resolve_attachments()` raises `AttachmentError` when the file does not exist
- `resolve_attachments()` sets `size_warning=True` when file exceeds `max_attachment_kb`
- `resolve_attachments()` assigns identifiers `CONTEXT_1`, `CONTEXT_2`, … in order (default `start_n=1`)
- In safe mode (not unsafe), prompts for confirmation; raises `AttachmentAborted` when user answers `n` — the caller (`ChatSession.process()`) catches this and cancels the entire message send, printing a notice to the user
- In unsafe mode, skips the prompt entirely
- `substitute_identifiers()` replaces `[[/abs/path/file.py]]` with `CONTEXT_1` in the message string

### Green
Implement all three functions. Use `unittest.mock.patch` for `builtins.input` in prompt tests.

### Refactor
- Make the prompt text a named constant for easy editing

---

## Phase 7 — `openai_client.py`

**Why here:** Depends only on `Config`. Retry logic must be solid before `ChatSession` relies on it.

### Red
- Successful call returns the assistant message string
- Retries on `RateLimitError` up to 3 times; prints retry notice each time
- Retries on `APIConnectionError` up to 3 times
- Retries on `APIStatusError` with status >= 500 up to 3 times
- Does not retry on `AuthenticationError`
- Does not retry on `BadRequestError`
- Raises `APITransientError` after 3 failed retries
- Raises `APIError` immediately on permanent errors
- Backoff delays are 1s, 2s, 4s (mock `time.sleep` in tests)
- Optional config params (`temperature`, `max_tokens`, etc.) are passed through when set; omitted when `None`

### Green
Implement `chat_completion(messages, config)` with retry loop.

### Refactor
- Extract the "is this error transient?" predicate into a named function

---

## Phase 8 — `chat.py`

**Why here:** Integrates store, attachments, client, and logger. Tests mock all four. This is the core logic phase.

### Red
- `process()` calls `store.query()` with the current user message
- `process()` assembles messages in order: system prompt → RAG system message → attachment system messages → trimmed history → current user message
- `process()` appends user and assistant turns to `self.history` after a successful call
- `_trim_history()` drops the oldest user+assistant pairs when history exceeds `history_token_budget`
- `_trim_history()` returns the number of turns dropped (0 when no trimming needed)
- Token estimation uses `len(content) // 4`
- When `resolve_attachments()` raises `AttachmentAborted`, `process()` catches it, prints a cancellation notice, and returns without calling the API or appending to history
- When no RAG chunks are retrieved, the RAG system message is omitted
- When no attachments are present, attachment system messages are omitted
- `process()` stores retrieved chunks in `_last_rag_chunks` for `/rag`

### Green
Implement `ChatSession` with all the above behaviour. Use `unittest.mock` for all dependencies.

### Refactor
- Extract message-assembly into a private `_build_messages()` method for testability

---

## Phase 9 — `repl.py`

**Why here:** Thin shell over `ChatSession`. Tests focus on command parsing and output formatting; prompt_toolkit is mocked.

### Red
- `/set chunksize=5` updates `session.top_k` to 5
- `/set history_token_budget=2000` updates `session.history_token_budget` to 2000
- `/set chunksize=-1` prints an error and leaves `session.top_k` unchanged
- `/set unknown=1` prints an error
- `/rag` prints the chunks from `session._last_rag_chunks`
- `/rag` before any message prints a "no RAG data yet" notice
- `/help` prints all commands and key bindings
- `/exit` raises `SystemExit`
- A non-command input calls `session.process()` and prints the response
- When `process()` returns `turns_dropped > 0`, prints `[N earlier turn(s) omitted from context]`

### Green
Implement `run_repl()` and `handle_command()`. Use `prompt_toolkit`'s `PromptSession` with history enabled.

### Refactor
- Extract trim-notice formatting into a one-liner helper

---

## Phase 10 — `cli.py`

**Why here:** Top-level bootstrap. Integration tests spin up the full stack against a real temp directory; mocking is minimal.

### Red
- `pmca MyConfig` starts a session (mock REPL to exit immediately)
- `pmca NonExistent` prints a config error and exits with a non-zero code
- `pmca MyConfig --unsafe` sets `session.unsafe = True`
- Missing `OPENAI_API_KEY` in environment prints an error and exits
- `log_folder` and `cache_dir` are created if absent

### Green
Implement `main()` using `argparse`. Wire all components together.

### Refactor
- Ensure all startup error paths print to stderr and exit with code 1

---

## Phase 19 — Windows compatibility

**Why here:** All production code and tests are in place; this phase makes the repo usable on `win-64` without changing any existing behaviour on Linux.

### Changes

**`config.py`**
- In `_validate_log_folder`, `_validate_rag_files`, and `_validate_startup_docs`: call `Path(value).expanduser()` before the `is_absolute()` check
- In `load_config`: apply `expanduser()` when constructing `Path` objects for all three path fields

**`src/pmca/configs/*.yaml`** (all three premade configs)
- Change `log_folder: /tmp/pmca-logs` → `log_folder: ~/.pmca/logs`

**`attachments.py`**
- `parse_attachment_paths`: strip leading/trailing `"` from the captured regex group before passing to `Path()`
- `substitute_identifiers`: build the lookup dict with `a.path` (a `Path` object) as the key; look up via `Path(raw)` — handles mixed `/` and `\` on Windows

**Tests — `test_config.py`**
- Parametrize snippets: replace `log_folder: /tmp/logs` with `log_folder: ~/logs`
- `test_raises_when_rag_file_does_not_exist`: replace `/nonexistent/path/file.py` with `tmp_path / "no_such_file.py"`
- `test_startup_docs_raises_when_path_does_not_exist`: replace `/nonexistent/doc.md` with `tmp_path / "no_such_doc.md"`
- Add `test_tilde_expanded_in_log_folder`, `test_tilde_expanded_in_rag_files`, `test_tilde_expanded_in_startup_docs`

**Tests — `test_attachments.py`**
- `test_parse_single_absolute_path`, `test_parse_multiple_paths`: replace hardcoded `/abs/...` Unix paths with `tmp_path`-derived paths
- `test_substitute_leaves_unmatched_tokens_unchanged`: replace `/other/path.py` with `tmp_path`-derived path
- Add `test_parse_strips_double_quotes_from_path` (using `tmp_path`)
- Add `test_substitute_handles_mixed_slashes` (using `tmp_path`)

**Tests — `test_repl.py`**
- `_chunk` helper: replace `Path("/a.py")` with a `tmp_path`-independent string or parametrised fixture; update `test_rag_prints_chunk_source` assertion accordingly
- `test_clear_prints_new_session_path`: replace the mock `Path("/logs/...")` value with a `tmp_path`-based path

**`README.md`** (new section)
- Add a short "Windows setup" paragraph instructing users to run `pixi install` after cloning

### Red
Write the new tests listed above; all fail on unmodified code.

### Green
Apply the source changes listed above; all tests pass.

### Refactor
None needed — changes are localised.

---

## Phase 20 — Continuous attachment numbering across turns

**Why:** Testing revealed that `resolve_attachments` always starts identifiers at `CONTEXT_1`. In a multi-turn session, a file attached in turn 2 also gets `CONTEXT_1`, colliding with turn 1's attachment in the conversation history and confusing the model.

### Red

**`attachments.py`**
- `resolve_attachments([a, b], ..., start_n=3)` assigns `CONTEXT_3` and `CONTEXT_4`
- Default `start_n=1` behaviour is unchanged (existing tests continue to pass)

**`chat.py`**
- After turn 1 attaches 2 files, `_next_attachment_n` is 3
- After turn 2 attaches 1 file, that file gets `CONTEXT_3` and `_next_attachment_n` is 4
- `AttachmentAborted` or `AttachmentError` does not advance `_next_attachment_n`
- `/clear` resets `_next_attachment_n` to 1 (alongside history and resumed_context)

**`resume.py`**
- `load_resume` on a log with `CONTEXT_1` and `CONTEXT_2` returns `next_attachment_n=3`
- `load_resume` on a log with no attachments returns `next_attachment_n=1`

**`cli.py`**
- `--resume` initialises `session._next_attachment_n` from `ResumedSession.next_attachment_n`

### Green

**`attachments.py`**
- Add `*, start_n: int = 1` keyword-only parameter to `resolve_attachments`; use `enumerate(paths, start=start_n)`

**`chat.py`**
- Add `self._next_attachment_n: int = 1` to `ChatSession.__init__`
- In `process()`: pass `start_n=self._next_attachment_n` to `resolve_attachments`; after success advance `self._next_attachment_n += len(attachments)`
- In `rotate_logger()`: reset `self._next_attachment_n = 1`

**`resume.py`**
- Add `next_attachment_n: int` field to `ResumedSession`
- In `load_resume`: scan all `attachments[].identifier` fields, extract N from `CONTEXT_N` strings, set `next_attachment_n = max(Ns) + 1` (or 1 if none found)

**`cli.py`**
- After `load_resume`, set `session._next_attachment_n = resumed.next_attachment_n`

### Refactor
- None needed — changes are localised to four modules.

---

## Phase 21 — Persistent attachments and RAG chunks

**Why:** Currently attachments and RAG chunks are injected only for the turn they appear in — the model cannot reference prior attachment content in follow-up turns. This phase makes both accumulate across turns so the full session context is always present in every API call. Resumed sessions are reconstructed entirely from the log (no `[RESUMED_CONTEXT]` special path).

### Changes across modules

**`logger.py`**
- Add `log_session_start(system_prompt, startup_docs)` — writes `{"type": "system_prompt", ...}` and `{"type": "startup_doc", ...}` entries
- `log_exchange` entries gain `{"type": "exchange", ...}` field; `rag_chunks` and `attachments` fields remain on user turns

**`chat.py`**
- Add `system_prompt: str` and `startup_docs: list[tuple[Path, str]]` fields to `ChatSession` (replacing direct reads from `config`)
- Add `session_attachments: list[Attachment]` — accumulates across turns; reset on `/clear`
- Add `session_rag_chunks: list[Chunk]` — accumulates across turns, deduplicated by `(source_file, label)`; reset on `/clear`
- Remove `resumed_context: str | None`
- Add `_merge_rag_chunks(new_chunks)` — merges into `session_rag_chunks` skipping duplicates
- `process()`: append new attachments to `session_attachments`; merge new RAG chunks into `session_rag_chunks`
- `_build_messages()`: inject all `session_attachments` then all `session_rag_chunks` (replacing per-turn injection and `resumed_context` block)
- `rotate_logger()`: reset `session_attachments`, `session_rag_chunks`, `_next_attachment_n`; call `logger.log_session_start()`
- `cli.py` bootstrap: call `logger.log_session_start()` after creating the logger

**`resume.py`**
- `ResumedSession` gains: `system_prompt`, `startup_docs`, `session_attachments`, `session_rag_chunks`; drops `resumed_context`
- `load_resume()`: reads `system_prompt` from `{"type": "system_prompt"}` entry (error if absent); reads `startup_docs` from `{"type": "startup_doc"}` entries; reconstructs `session_attachments` and `session_rag_chunks` from all exchange entries

**`cli.py`**
- On resume: set `session.system_prompt` and `session.startup_docs` from `ResumedSession`; warn if they differ from config values
- Set `session.session_attachments` and `session.session_rag_chunks` from `ResumedSession`
- Logger used via `from_existing` — does NOT write session-start entries again

**`repl.py`**
- `/clear` handler: also reset `session.session_attachments` and `session.session_rag_chunks`; call `session.rotate_logger()` (which handles the rest)

### Red

**`logger.py`**
- `log_session_start` writes a `system_prompt` entry followed by one `startup_doc` entry per doc
- `log_exchange` entries have `type: "exchange"` field
- Existing behaviour (two entries per turn, rag_chunks/attachments on user entry) is unchanged

**`chat.py`**
- After turn 1 attaches a file, `session_attachments` has that attachment; it is present in `_build_messages` output for turn 2 even when turn 2 has no attachments
- After turn 1 retrieves RAG chunk A and turn 2 retrieves chunk B, `session_rag_chunks` has both; both appear in turn 2's API call
- If chunk A is retrieved again in turn 3, `session_rag_chunks` still has only one copy
- `_build_messages` order: system_prompt → startup_docs → session_attachments → session_rag_chunks → history → current user message
- `/clear` (via `rotate_logger`) resets `session_attachments` and `session_rag_chunks` to `[]`
- `resumed_context` field no longer exists on `ChatSession`

**`resume.py`**
- `load_resume` on a log with typed entries returns correct `system_prompt`, `startup_docs`, `session_attachments`, `session_rag_chunks`
- `load_resume` raises `ResumeError` if no `system_prompt` entry found
- Duplicate RAG chunks across turns are deduplicated in `session_rag_chunks`

**`cli.py`**
- On resume with differing system_prompt: warning printed, log version used
- On resume with differing startup_docs: warning printed, log version used
- `session.session_attachments` and `session.session_rag_chunks` initialised from `ResumedSession`

### Green
Apply all source changes listed above.

### Refactor
- None needed — changes are localised to five modules.

---

## Phase 22 — System context injection

**Why:** The LLM has no awareness of the current datetime or host environment. This phase optionally injects a static system message so the model can give environment-aware answers. All fields are opt-in — nothing is sent by default.

### What is injected

A single system message computed once in `ChatSession.__init__` and stored as `_system_context` (`None` when no fields are configured). Fields are controlled by `config.system_context_fields` (default: `[]`). Supported values:

| Field value | Content added |
|---|---|
| `"datetime"` | `Session started: 2026-05-25 14:03:11 +0200` |
| `"os"` | `OS: Linux` (from `platform.system()` only — no version string) |
| `"shell"` | `Shell: /usr/bin/zsh` (from `$SHELL` on Unix; falls back to `%COMSPEC%` on Windows) |

Unknown field values are silently ignored.

### Red

**`config.py`**
- `system_context_fields` defaults to `[]` when absent from YAML
- Unknown values in the list are accepted without error (validated at build time, not load time)

**`chat.py`**
- `_system_context` is `None` when `config.system_context_fields` is empty → no system context message injected
- `_system_context` contains only the lines for the configured fields, in the order: datetime, os, shell
- `_build_messages` inserts the context message immediately after the main system prompt, before startup docs — only when `_system_context` is not `None`

### Green

**`config.py`**
- Add `system_context_fields: list[str]` field (default `[]`)
- Parse from YAML `system_context_fields` key (optional)

**`chat.py`**
- Remove imports of `getpass` (no longer needed); keep `os` for shell detection
- Shell detection: `os.environ.get('SHELL') or os.environ.get('COMSPEC', 'unknown')` — handles both Unix and Windows
- In `__init__`: `self._system_context = _build_system_context(config.system_context_fields)`
- Add module-level `_build_system_context(fields: list[str]) -> str | None` that builds the string from the field list, returning `None` if no recognised fields produce output
- In `_build_messages`: only insert the context system message when `self._system_context is not None`

### Refactor
- None needed — changes are contained to `config.py` and `chat.py`.

---

## Phase 23 — write_file tool

**Why here:** Builds on the stable `config.py`, `chat.py`, `openai_client.py`, and `logger.py`. Adds opt-in file-writing capability gated by `write_allowed_dirs` in the config YAML.

### Changes across modules

**`config.py`**
- Add `write_allowed_dirs: list[Path]` field (default `[]`)
- Validate: all entries must be absolute paths; existence not required

**`types.py`**
- Add `ToolCallRequest` dataclass: `tool_call_id: str`, `name: str`, `arguments: dict`

**`openai_client.py`**
- Add `tools: list[dict] | None = None` parameter to `chat_completion()`
- When `tools` is provided, pass to API with `parallel_tool_calls=False`
- Return `ToolCallRequest` when response contains `tool_calls`; return `str` otherwise

**`tools.py`** (new module)
- `get_tools(config) -> list[dict] | None` — returns `None` if `write_allowed_dirs` is empty; otherwise returns a list with the `write_file` function schema, with allowed dirs listed in the description
- `execute_write_file(arguments, config) -> tuple[bool, str]` — validates path, prompts user, writes file

**`chat.py`**
- `process()` passes `get_tools(config)` to `chat_completion()`
- Tool loop: while response is `ToolCallRequest`, execute tool, log it, append messages, call API again
- `_build_messages` unchanged

**`logger.py`**
- Add `log_tool_call(tool_call_id, name, arguments, approved, result)` — appends a `{"type": "tool_call", ...}` entry

### Red

**`config.py`**
- `write_allowed_dirs` defaults to `[]` when absent from YAML
- Raises `ConfigError` when a `write_allowed_dirs` entry is non-absolute

**`types.py`**
- `ToolCallRequest` is a dataclass with `tool_call_id`, `name`, `arguments`

**`openai_client.py`**
- `chat_completion` with `tools=None` behaves identically to before (returns string)
- `chat_completion` with tools returns `ToolCallRequest` when model issues a tool call
- `chat_completion` with tools returns string when model returns text directly

**`tools.py`**
- `get_tools(config)` returns `None` when `write_allowed_dirs` is empty
- `get_tools(config)` returns a list with one schema dict when dirs are configured; the description contains each allowed dir
- `execute_write_file` returns error string when path is outside allowed dirs (no prompt)
- `execute_write_file` prints approval prompt in Option C format; returns denial string when user inputs anything other than `y`
- `execute_write_file` shows "File exists — will be overwritten." when path exists
- `execute_write_file` shows "File does not exist." when path is new
- `execute_write_file` creates parent dirs and writes file on approval; returns `"Written: /path (N bytes)"`
- `execute_write_file` returns error string on I/O error without crashing

**`chat.py`**
- `process()` passes tools to `chat_completion` and loops on `ToolCallRequest`
- Tool result messages are appended to the message list in the correct OpenAI format
- Final text response is used as the assistant reply
- Tool calls are logged via `logger.log_tool_call`

**`logger.py`**
- `log_tool_call` appends a well-formed `{"type": "tool_call", ...}` JSON line

### Green
Apply all source changes listed above.

### Refactor
- None needed — changes are localised to five modules.

---

## Phase 24 — Codebase exploration tools (read_file, list_dir, search, get_definition)

**Why here:** Builds on the stable `config.py`, `tools.py`, `chat.py`, and `repl.py`. Adds opt-in read-only codebase exploration gated by `read_allowed_dirs` in the config YAML, with mid-session control via `/read add` and `/read remove`.

### Changes across modules

**`config.py`**
- Add `read_allowed_dirs: list[Path]` field (default `[]`)
- Validate: all entries must be absolute paths; existence not required

**`tools.py`**
- Add schemas and executors for `read_file`, `list_dir`, `search`, `get_definition`
- `get_tools()`: include read tools when `read_allowed_dirs` is non-empty; combine with write tools if both are configured
- `execute_read_file(arguments, config)`: validate path, read and return UTF-8 content; silent (no user prompt)
- `execute_list_dir(arguments, config)`: validate path, list immediate children or full tree based on `recursive` bool; return newline-separated paths
- `execute_search(arguments, config)`: validate path, grep-style regex search over file or directory tree; return `file:lineno: line` matches with `context_lines` lines of surrounding context; separate match groups with `--`
- `execute_get_definition(arguments, config)`: validate path (must be `.py`), use `ast.parse()` to locate top-level or nested symbol (`"MyClass"` or `"MyClass.my_method"`), extract full source including decorators via line numbers from AST nodes

**`chat.py`**
- `process()` tool loop: dispatch `read_file`, `list_dir`, `search`, `get_definition` to their executors alongside existing `write_file` dispatch
- Read tool results are logged via `logger.log_tool_call` (same as write_file; `approved=True` always)

**`repl.py`**
- Add `/read add <path>` and `/read remove <path>` commands to `handle_command()`
- Each prompts: `Add/Remove <resolved_path> from read_allowed_dirs? [y/N]`; on approval mutates `session.config.read_allowed_dirs`; changes are session-only

### Red

**`config.py`**
- `read_allowed_dirs` defaults to `[]` when absent from YAML
- Raises `ConfigError` when a `read_allowed_dirs` entry is non-absolute

**`tools.py`**
- `get_tools` returns read tool schemas when `read_allowed_dirs` is non-empty
- `execute_read_file` returns error when path is outside allowed dirs
- `execute_read_file` returns file content on success
- `execute_list_dir` returns immediate children when `recursive=False`
- `execute_list_dir` returns full tree when `recursive=True`
- `execute_list_dir` returns error when path is outside allowed dirs or not a directory
- `execute_search` returns error when path is outside allowed dirs
- `execute_search` returns formatted matches with context for a file path
- `execute_search` searches recursively when given a directory path
- `execute_search` returns `"No matches found."` when pattern matches nothing
- `execute_search` returns error on invalid regex
- `execute_get_definition` returns error when path is outside allowed dirs
- `execute_get_definition` returns full source of a top-level function/class
- `execute_get_definition` returns full source of a method via `"Class.method"` syntax
- `execute_get_definition` returns error when symbol not found
- `execute_get_definition` returns error when file is not a `.py` file

**`chat.py`**
- Tool loop dispatches to all four read executors correctly

**`repl.py`**
- `/read add <path>` prompts and on `y` appends resolved path to `session.config.read_allowed_dirs`
- `/read add <path>` on denial leaves `read_allowed_dirs` unchanged
- `/read remove <path>` prompts and on `y` removes the path from `session.config.read_allowed_dirs`
- `/read remove <path>` on unknown path prints an informative message
- `/help` output includes `/read add` and `/read remove`

### Green
Apply all source changes listed above.

### Refactor
- Extract the `read_allowed_dirs` path-validation check into a shared helper used by all four read executors (mirrors `_is_allowed` already used by `execute_write_file`)

---

## Phase 25 — `run_tests` tool

**Why here:** Builds on stable `config.py`, `tools.py`, `chat.py`, and `repl.py`. Adds opt-in test execution gated by `test_dir` in the config YAML, completing the tools needed for a red-green-refactor loop.

### Changes across modules

**`config.py`**
- Add `test_dir: Path | None = None` field — absolute path; `None` → `run_tests` tool not registered
- Add `test_timeout: int = 60` field — seconds before the subprocess is killed
- Validate: if `test_dir` is set, it must be an absolute path; existence not required at load time

**`tools.py`**
- Add `_RUN_TESTS_SCHEMA` — schema with a single optional `filter` string parameter
- `get_tools()`: include `run_tests` when `config.test_dir` is not `None`
- `execute_run_tests(arguments, config)`: auto-detect command (`pixi run pytest` if `pixi.toml` exists in `test_dir`, else `pytest`); append filter tokens if provided; print `[run_tests] <command>` to stdout; run subprocess with combined stdout+stderr, `cwd=config.test_dir`, `timeout=config.test_timeout`; return `(True, output)` on success or pytest failure; return `(False, "Error: ...")` on timeout or `OSError`

**`chat.py`**
- `_dispatch_tool`: add `run_tests` case → `execute_run_tests(args, config)`

**`repl.py`**
- `/set test_timeout=N` handler: validate N > 0; set `session.config.test_timeout = N`; print confirmation
- `/help` output: add `/set test_timeout=N`

### Red

**`config.py`**
- `test_dir` defaults to `None` when absent from YAML
- `test_timeout` defaults to `60` when absent from YAML
- Raises `ConfigError` when `test_dir` is set but non-absolute

**`tools.py`**
- `get_tools` includes `run_tests` schema when `test_dir` is configured; absent when `test_dir` is `None`
- `execute_run_tests` uses `pixi run pytest` when `pixi.toml` is present in `test_dir`
- `execute_run_tests` uses `pytest` when no `pixi.toml` is present
- `execute_run_tests` appends filter tokens when `filter` argument is provided
- `execute_run_tests` prints `[run_tests] <command>` to stdout before running
- `execute_run_tests` returns `(True, output)` when pytest exits 0 (all pass)
- `execute_run_tests` returns `(True, output)` when pytest exits non-zero (failures) — output contains the failure details
- `execute_run_tests` returns `(False, "Error: run_tests timed out after N seconds")` on timeout
- `execute_run_tests` returns `(False, "Error: ...")` on `OSError`

**`chat.py`**
- Tool loop dispatches `run_tests` to `execute_run_tests`

**`repl.py`**
- `/set test_timeout=60` sets `session.config.test_timeout` to 60
- `/set test_timeout=0` prints an error and leaves `test_timeout` unchanged
- `/set test_timeout=-1` prints an error and leaves `test_timeout` unchanged
- `/help` includes `/set test_timeout=N`

### Green
Apply all source changes listed above.

### Refactor
- None needed — changes are localised to four modules.

---

## Phase 11 — Integration smoke test

One end-to-end test with real files, mocked OpenAI API, and a real temp log directory:

- Start session → send one message with a `[[/abs/path]]` attachment → assert JSONL log written correctly → exit

This validates the full pipeline without hitting the network.

---

## Suggested development order summary

```
Phase 1   config.py
Phase 2   rag/chunker.py
Phase 3   rag/embedder.py
Phase 4   rag/store.py
Phase 5   logger.py
Phase 6   attachments.py
Phase 7   openai_client.py
Phase 8   chat.py
Phase 9   repl.py
Phase 10  cli.py
Phase 11  integration smoke test
```
