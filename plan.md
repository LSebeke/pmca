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
- `resolve_attachments()` assigns identifiers `CONTEXT_1`, `CONTEXT_2`, … in order
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
