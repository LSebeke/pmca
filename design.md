# Design Document: `pmca` (Poor Man's Coding Assistant)

---

## 1. Overview

`pmca` is a CLI chat tool that wraps the OpenAI API with project-aware context via RAG. It runs as a single interactive session, initiated by the user, and terminates on `/exit` or Ctrl+C.

---

## 2. Module Structure

```
pmca/
├── cli.py              # Entry point, argument parsing, session bootstrap
├── config.py           # YAML loading and validation
├── repl.py             # prompt_toolkit REPL, command dispatch
├── chat.py             # Conversation state, message assembly, history trimming
├── attachments.py      # [[filepath]] parsing, validation, security prompt
├── openai_client.py    # OpenAI API calls with retry logic
├── logger.py           # JSONL chat log + debug log writer
├── resume.py           # JSONL resume: parse log, validate, extract history + context
├── types.py            # Shared dataclasses: Chunk, Attachment, ScratchpadEntry, ToolCallRequest
└── rag/
    ├── chunker.py      # File → Chunk list (semantic / AST-based)
    ├── embedder.py     # Thin embed() interface over OpenAI embeddings
    └── store.py        # In-memory vector store + disk cache
```

---

## 3. Data Structures

### 3.1 Config

```python
@dataclass
class Config:
    name: str
    model: str
    system_prompt: str
    rag_files: list[Path]        # absolute paths only, validated at load time
    log_folder: Path
    startup_docs: list[tuple[Path, str]] = field(default_factory=list)  # (path, content) pairs; loaded at config parse time
    write_allowed_dirs: list[Path] = field(default_factory=list)  # absolute paths; empty → write_file tool not registered
    read_allowed_dirs: list[Path] = field(default_factory=list)   # absolute paths; empty → read_file/list_dir/search tools not registered
    system_context_fields: list[str] = field(default_factory=list)  # empty by default → no system context injected
    rag_shallow_k: int = 3       # chunks returned for depth="shallow"
    rag_medium_k: int = 7        # chunks returned for depth="medium"
    rag_deep_k: int = 15         # chunks returned for depth="deep"
    max_scratchpad_entries: int = 20  # hard cap on scratchpad entries; configurable
    test_dir: Path | None = None          # absolute path; None → run_tests tool not registered
    test_timeout: int = 60                # seconds before run_tests is killed
    max_attachment_kb: int = 500
    history_token_budget: int = 4000
    # OpenAI optional params (passed through as-is if set)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
```

### 3.2 Chunk

```python
@dataclass
class Chunk:
    content: str
    source_file: Path
    label: str          # e.g. "function `parse_config` (lines 15–32)"
                        # or "Section: Configuration Loading"
```

### 3.3 Attachment

```python
@dataclass
class Attachment:
    path: Path
    content: str
    identifier: str     # e.g. "CONTEXT_1"
    size_warning: bool  # True if file exceeded max_attachment_kb
```

### 3.4 ScratchpadEntry

```python
@dataclass
class ScratchpadEntry:
    title: str    # short label that makes the origin of the information clear (e.g. "read_file: src/pmca/config.py — load_config body")
    content: str  # arbitrary excerpt chosen by the LLM from a tool call return
```

Persists in `ChatSession._scratchpad` across turns and is injected as `[SCRATCHPAD_i]` system messages before history. The LLM must only save information that would otherwise be lost (tool call returns are not stored in history).

---

### 3.5 ToolCallRequest

```python
@dataclass
class ToolCallRequest:
    tool_call_id: str   # opaque ID from the OpenAI response, echoed back in the tool result message
    name: str           # tool name, e.g. "write_file"
    arguments: dict     # parsed JSON arguments, e.g. {"path": "...", "content": "...", "description": "..."}
```

Returned by `chat_completion()` when the model issues a tool call instead of a text response. `ChatSession.process()` inspects this, executes the tool, sends the result back, and loops until a text response is received.

---

### 3.6 LogEntry (JSONL)

Three entry types, distinguished by a mandatory `type` field:

```python
# Written once at session start — system prompt
{"type": "system_prompt", "content": "..."}

# Written once per startup doc at session start
{"type": "startup_doc", "path": "/abs/path/doc.md", "content": "..."}

# Written twice per user turn (user then assistant)
{
    "type": "exchange",
    "timestamp": "2025-05-14T15:32:10Z",
    "role": "user" | "assistant",
    "content": "...",
    "attachments": [                        # present on user turns only
        {"identifier": "CONTEXT_1", "path": "...", "content": "...", "size_warning": false}
    ]
}
```

```python
# Written once per tool call within a turn
{
    "type": "tool_call",
    "timestamp": "2025-05-14T15:32:10Z",
    "tool_call_id": "call_abc123",
    "name": "write_file",
    "arguments": {"path": "/abs/path/file.py", "content": "...", "description": "..."},
    "approved": true,        # false if user denied
    "result": "Written: /abs/path/file.py (1 842 bytes)"  # or denial/error message
}
```

`attachments[].content` stores verbatim content so that a resumed session can reconstruct full context without re-reading original files. The log is the authoritative source of truth for session state on resume.

---

## 4. Component Design

### 4.1 `config.py`

**Responsibilities:** Load and validate YAML config. Fail fast with a clear error message if any required field is missing, any path is non-absolute or unreadable, or the API key is absent from the environment.

```python
def load_config(config_name: str) -> Config:
    """
    Searches for <config_name>.yaml in the current directory and <package_dir>/configs/.
    Raises ConfigError with a descriptive message on any validation failure.
    """
```

Validation rules:
- All `rag_files` paths must be absolute, exist, and be readable
- `log_folder` must be absolute (created if absent)
- `OPENAI_API_KEY` must be present in environment (checked in `cli.py`)
- `rag_shallow_k`, `rag_medium_k`, `rag_deep_k`, `max_scratchpad_entries`, `max_attachment_kb`, and `history_token_budget` must be positive integers if provided
- All `write_allowed_dirs` paths must be absolute (existence not required — dirs may be created later)
- All `read_allowed_dirs` paths must be absolute (existence not required)

Path expansion: before the absolute-path check, all path fields (`log_folder`, `rag_files`, `startup_docs`) are expanded via `Path.expanduser()`. This allows cross-platform configs to use `~` (e.g. `log_folder: ~/.pmca/logs`) without hardcoding OS-specific absolute paths.

---

### 4.2 `rag/chunker.py`

**Responsibilities:** Split a file into `Chunk` objects. Strategy is determined by file extension.

```python
def chunk_file(path: Path) -> list[Chunk]:
    """Dispatch to _chunk_python or _chunk_prose based on suffix."""

def _chunk_python(path: Path) -> list[Chunk]:
    """
    Use ast.parse() to extract top-level functions, classes, and methods.
    Each node becomes one Chunk with label "function `name` (lines X–Y)".
    Module-level code outside any def/class becomes a single "module-level" chunk.
    """

def _chunk_prose(path: Path) -> list[Chunk]:
    """
    Split on double newlines (paragraph boundaries). For .md files,
    also split on ATX headings (lines starting with #).
    Each chunk retains its heading as a label if one precedes it.
    """
```

---

### 4.3 `rag/embedder.py`

**Responsibilities:** Provide a single `embed()` function over `text-embedding-3-small`. Isolated here so the provider can be swapped without touching other modules.

```python
def embed(texts: list[str]) -> np.ndarray:
    """
    Returns float32 array of shape (len(texts), 1536).
    Calls OpenAI embeddings API. Raises EmbedError on failure.
    Batches requests if len(texts) > 100.
    """
```

---

### 4.4 `rag/store.py`

**Responsibilities:** Build the in-memory vector store from `Config.rag_files`, using a disk cache to avoid re-embedding unchanged files.

#### Cache layout

```
<log_folder>/cache/<hex(sha256(str(absolute_path)))>.pkl
```

Each `.pkl` file contains:
```python
{
    "file_hash": str,        # SHA-256 hex digest of file content
    "chunks": list[Chunk],
    "embeddings": np.ndarray # shape (n_chunks, 1536), float32
}
```

#### Interface

```python
class VectorStore:
    def build(self, files: list[Path], cache_dir: Path) -> None:
        """
        Pre-passes over files to count stale/missing cache entries, prints
        "[RAG] embedding N new/changed file(s)..." to stderr before embedding begins.
        For each file: loads cache entry if present and file hash matches;
        otherwise chunks + embeds + writes cache.
        Assembles all chunks and embeddings into contiguous numpy arrays.
        """

    def query(self, text: str, top_k: int) -> list[Chunk]:
        """
        Embed text, compute cosine similarity against all stored embeddings,
        return top_k chunks sorted by descending similarity.
        """
```

---

### 4.5 `attachments.py`

**Responsibilities:** Parse `[[filepath]]` tokens from a user message, validate and load each file, run the security prompt.

```python
def parse_attachment_paths(message: str) -> list[Path]:
    """
    Extract all [[...]] tokens. Strip leading/trailing double-quotes from the
    captured path string (Windows Explorer wraps copied paths in quotes, e.g.
    [["C:\\temp\\file.py"]]). Raise AttachmentError if any path is non-absolute.
    """

def resolve_attachments(
    paths: list[Path],
    max_attachment_kb: int,
    unsafe: bool,
    *,
    start_n: int = 1,
) -> list[Attachment]:
    """
    For each path:
      1. Raise AttachmentError if file does not exist.
      2. If size > max_attachment_kb: print warning, continue.
      3. If not unsafe: prompt "You are about to attach <path>. Have you
         reviewed it for secrets? (y/n)". If 'n': raise AttachmentAborted.
      4. Read content, assign identifier CONTEXT_<n> starting from start_n.
    Returns list of Attachment objects.
    """

def substitute_identifiers(message: str, attachments: list[Attachment]) -> str:
    """
    Replace each [[filepath]] token with its CONTEXT_<n> identifier.
    Lookup is keyed on Path objects (not strings) so that mixed forward/backward
    slash styles on Windows still match (e.g. C:/temp/f.py == C:\\temp\\f.py).
    """
```

---

### 4.6 `chat.py`

**Responsibilities:** Maintain conversation history, assemble the full message list sent to the API, manage history trimming.

```python
class ChatSession:
    config: Config
    store: VectorStore
    unsafe: bool
    system_prompt: str           # active system prompt (from config or log on resume)
    startup_docs: list[tuple[Path, str]]  # active startup docs (from config or log on resume)
    history: list[dict]          # {"role": "user"|"assistant", "content": str}
    history_token_budget: int    # mutable via /set
    session_attachments: list[Attachment]  # all attachments accumulated this session
    _next_attachment_n: int      # session-global counter for CONTEXT_<n> identifiers
    _system_context: str | None  # computed once at __init__ from config.system_context_fields; None if list is empty
    _turn_seen_chunks: set[tuple[Path, str]]  # (source_file, label) pairs already returned this turn; reset at start of each process()
    _turn_read_files: set[Path]  # resolved paths read via read_file this turn; reset at start of each process(); gates edit_file and write_file on existing files
    _scratchpad: list[ScratchpadEntry]  # entries the LLM has saved from tool call returns; injected as [SCRATCHPAD_i] system messages each turn

    def process(self, user_input: str) -> str:
        """
        Full pipeline for one user turn:
          1. Reset _turn_seen_chunks and _turn_read_files to empty sets.
          2. Parse and resolve attachments (passing _next_attachment_n as start_n);
             substitute identifiers in message.
             Advance _next_attachment_n by len(attachments) on success only.
             Append new attachments to session_attachments.
          3. Trim history to fit history_token_budget; note turns dropped.
          4. Assemble message list (see Section 5).
          5. Call openai_client.chat_completion() with tools list.
          6. Tool loop: while response is a ToolCallRequest:
               a. Execute the tool via _dispatch_tool().
               b. Log the tool call via SessionLogger.
               c. Append assistant tool-call message + tool result message to messages list.
               d. Call chat_completion() again.
          7. Append user + assistant turns to self.history.
          8. Log exchange via SessionLogger.
          9. Return (assistant_response, turns_dropped).
        """

    def _dispatch_tool(self, response: ToolCallRequest) -> tuple[bool, str]:
        """
        Route a tool call to its executor. Has access to self.store and self._turn_seen_chunks
        for the query_knowledge_base tool.
        """

    def rotate_logger(self) -> Path:
        """
        Close the current logger, open a new one with a fresh timestamp in
        config.log_folder, write session-start entries (system_prompt, startup_docs),
        assign it to self.logger, and return the path of the new JSONL file.
        Also resets _next_attachment_n to 1, session_attachments to [], and _scratchpad to [].
        """

    def _trim_history(self) -> int:
        """
        Drop oldest turns (user+assistant pairs) until history fits within
        history_token_budget. Token count estimated as len(content) // 4.
        Returns number of turns dropped.
        """
```

---

### 4.7 `openai_client.py`

**Responsibilities:** Send chat completion requests with retry logic.

```python
def chat_completion(
    messages: list[dict],
    config: Config,
    tools: list[dict] | None = None,
) -> str | ToolCallRequest:
    """
    Calls openai.chat.completions.create() with config model + optional params.
    If tools is not None, passes them as the `tools` argument (parallel_tool_calls=False).
    Returns a ToolCallRequest if the model issues a tool call, otherwise returns the
    assistant message string.
    Retries up to 3 times on transient errors (429, network timeout, 5xx)
    with exponential backoff (1s, 2s, 4s).
    Prints "[retrying... attempt N/3]" before each retry.
    Raises APIError (permanent) or APITransientError (exhausted retries).
    """
```

Transient errors: `RateLimitError`, `APIConnectionError`, `APIStatusError` with status >= 500.
Permanent errors: `AuthenticationError`, `BadRequestError`, other `APIStatusError`.

---

### 4.8 `tools.py`

**Responsibilities:** Define tool schemas and implement execution for `query_knowledge_base`, `save_to_scratchpad`, `write_file`, `edit_file`, `read_file`, `list_dir`, `search`, `get_definition`, and `run_tests`. All read tools are gated by `read_allowed_dirs`; writes are gated by `write_allowed_dirs`; test execution is gated by `test_dir`; RAG tools are gated by store content. Reads and test runs execute without user approval; writes require per-call approval.

```python
def get_tools(config: Config, store: VectorStore) -> list[dict] | None:
    """
    Returns the full tools list to pass to chat_completion, or None if no tools
    are enabled.
    query_knowledge_base is included when store has indexed content (store._chunks is non-empty).
    save_to_scratchpad is included whenever any other tool is registered.
    write_file and edit_file are included when write_allowed_dirs is non-empty.
    read_file, list_dir, search, and get_definition are included when read_allowed_dirs is non-empty.
    run_tests is included when test_dir is not None.
    Tool descriptions include the relevant allowed directories / depth levels.
    """

def execute_save_to_scratchpad(
    arguments: dict,
    config: Config,
    scratchpad: list[ScratchpadEntry],
) -> str:
    """
    Upsert and/or delete scratchpad entries across turns.
    arguments: {
        "entries": [{"title": str, "content": str}, ...],   # optional — upsert by title
        "delete":  [str, ...]                                # optional — titles to delete
    }
    Processing order: deletes first, then upserts (so a same-call title replacement stays within the cap).
    Delete: remove entries whose title matches; unknown titles are silently ignored.
    Upsert: for each entry, overwrite if title already exists, otherwise add new.
            If adding all new titles would exceed config.max_scratchpad_entries, return an error
            string listing the cap and how many slots are free — do not partially apply.
            Entries whose titles already exist do not count against the cap (they are overwrites).
    Constraint enforced by description (not code): titles must make the origin clear
    (e.g. "read_file: src/pmca/config.py — validation logic"), and content must be
    information that would be lost otherwise (i.e. from tool call returns).
    Mutates scratchpad in-place.
    Returns a summary string, e.g.:
      "Deleted 1 entry. Saved 2 entries. [Scratchpad: 3 entries]"
    Returns an error string if the cap would be exceeded (after deletes): e.g.
      "Error: cap is 20; 18 slots used, 1 free — cannot add 3 new entries. Delete some first."
    """

def execute_rag_query(
    arguments: dict,
    config: Config,
    store: VectorStore,
    turn_seen: set[tuple[Path, str]],
) -> str:
    """
    Query the vector store and return only chunks not already in turn_seen.
    arguments: {"query": str, "depth": "shallow" | "medium" | "deep"}
    depth maps to config.rag_shallow_k / rag_medium_k / rag_deep_k (defaults 3/7/15).
    Adds newly returned chunks to turn_seen.
    Returns formatted chunks as:
      [RAG_1]
      File: /abs/path
      Chunk: <label>
      ---
      <content>
      ---
      [RAG_2] ...
    Returns "No results found." when store is empty or all top-k results were already seen.
    """

def execute_write_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    """
    Validate path against config.write_allowed_dirs, prompt user for approval,
    and write the file if approved.

    Approval prompt format:
      [write_file] /full/resolved/path (N bytes)
      Reason: <description>
      File exists — will be overwritten. Approve? [y/N]   ← if path exists
      File does not exist. Approve? [y/N]                 ← if new

    Path validation:
      - Resolve to absolute path (Path.resolve())
      - Must be within one of config.write_allowed_dirs (use Path.is_relative_to())
      - If invalid: return (False, "Error: path ... is outside allowed directories: ...")

    Read-before-write enforcement (existing files only):
      - If the file already exists and its resolved path is not in turn_read_files:
        return (False, "Error: <path> has not been read this turn. Call read_file first.")
      - New files (path does not yet exist) are exempt from this check.

    On approval:
      - Create parent directories (mkdir -p)
      - Write content (UTF-8)
      - Return (True, "Written: /full/path (N bytes)")

    On denial:
      - Return (False, "Write denied by user. Path: /full/path")
    """

def execute_edit_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    """
    Validate path against config.write_allowed_dirs, find old_string in the file,
    and replace it with new_string after user approval.

    Path validation:
      - Resolve to absolute path (Path.resolve())
      - Must be within one of config.write_allowed_dirs
      - If invalid: return (False, "Error: path ... is outside allowed directories: ...")
      - File must exist: if not, return (False, "Error: file not found: ...")

    Read-before-edit enforcement:
      - If the resolved path is not in turn_read_files:
        return (False, "Error: <path> has not been read this turn. Call read_file first.")

    String matching:
      - Count occurrences of old_string in file content
      - If 0: return (False, "Error: old_string not found in <path>")
      - If > 1: return (False, "Error: old_string is ambiguous (N occurrences) in <path>; provide more context")
      - If exactly 1: proceed to approval prompt

    Approval prompt format:
      [edit_file] /full/resolved/path
      Reason: <description>
      --- remove ---
      <old_string>
      --- insert ---
      <new_string>
      ---
      Approve? [y/N]

    On approval:
      - Replace the single occurrence of old_string with new_string
      - Write back (UTF-8)
      - Return (True, "Edited: /full/path")

    On denial:
      - Return (False, "Edit denied by user. Path: /full/path")
    """

def execute_read_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> str:
    """
    Validate each path in arguments["paths"] against config.read_allowed_dirs, read and
    return the file contents. No user prompt — reads are silent.
    arguments: {"paths": list[str]}
    Returns all results concatenated, each preceded by a header line:
        === /abs/path/to/file.py ===
        <content or error>
    On success for each path, adds the resolved path to turn_read_files (enabling
    subsequent edit_file or write_file calls on the same path within this turn).
    Failed paths (outside allowed dirs, file not found, I/O error) include an error
    message in place of content and are not added to turn_read_files.
    """

def execute_list_dir(arguments: dict, config: Config) -> str:
    """
    Validate path against config.read_allowed_dirs, list directory contents.
    arguments: {"path": str, "recursive": bool}
    If recursive=False: returns immediate children only.
    If recursive=True: returns full directory tree.
    Returns a newline-separated list of paths, or an error string.
    """

def execute_search(arguments: dict, config: Config) -> str:
    """
    Validate path against config.read_allowed_dirs, search for pattern.
    arguments: {"path": str, "pattern": str, "context_lines": int}
    path may be a file or a directory; if a directory, searches recursively.
    pattern is a regex string (re module); match is case-sensitive by default.
    context_lines controls how many lines before/after each match to include (default 3).
    Returns formatted results: for each match, "file:lineno: <line>" with
    context_lines lines of context above/below, separated by "--" between matches.
    Returns "No matches found." if pattern matches nothing.
    Returns an error string if path is outside allowed dirs or pattern is invalid.
    """

def execute_get_definition(arguments: dict, config: Config) -> str:
    """
    Validate path against config.read_allowed_dirs, then return the full source
    of the named Python function or class (including decorators and docstring).
    arguments: {"path": str, "symbol": str}
    path must be a .py file. symbol is a top-level or method name
    (e.g. "MyClass" or "MyClass.my_method").
    Uses ast.parse() to locate the node; extracts source lines via
    ast.get_source_segment() or equivalent.
    Returns the full source text on success.
    Returns an error string if path is outside allowed dirs, not a .py file,
    file not found, symbol not found, or parse error.
    """

def execute_run_tests(arguments: dict, config: Config) -> tuple[bool, str]:
    """
    Run the test suite in config.test_dir and return the full output.
    arguments: {"filter": str}  — filter is optional; omit to run the full suite.

    Command selection (checked once at call time):
      - If pixi.toml exists in config.test_dir: ["pixi", "run", "pytest"]
      - Otherwise: ["pytest"]
    The optional filter string is appended as-is, split on whitespace
    (e.g. "tests/test_tools.py -k my_test" → two extra argv tokens).

    Before running, prints to stdout:
      [run_tests] <full command string>
    (e.g. "[run_tests] pixi run pytest tests/test_tools.py")

    Runs subprocess with cwd=config.test_dir, captures stdout+stderr (combined),
    timeout=config.test_timeout seconds.

    Returns (True, combined_output) regardless of exit code — test failures are
    not errors from the tool's perspective, the LLM reads the output.
    Returns (False, "Error: run_tests timed out after N seconds") on timeout.
    Returns (False, "Error: <msg>") on OSError (e.g. command not found).
    """
```

---

### 4.9 `logger.py`

**Responsibilities:** Write per-session JSONL chat log and plaintext debug log. Both files are opened at session start and flushed after every write (no buffering).

```python
class SessionLogger:
    def __init__(self, log_folder: Path, timestamp: str) -> None:
        """
        Opens <log_folder>/chat_<timestamp>.jsonl and
               <log_folder>/debug_<timestamp>.log.
        """

    def log_session_start(
        self,
        system_prompt: str,
        startup_docs: list[tuple[Path, str]],
    ) -> None:
        """
        Writes one {"type": "system_prompt", ...} entry followed by one
        {"type": "startup_doc", ...} entry per startup doc. Called once at
        session start (and again after /clear rotates the logger).
        """

    def log_exchange(
        self,
        user_message: str,
        assistant_message: str,
        attachments: list[Attachment],
    ) -> None:
        """Appends two {"type": "exchange", ...} lines: one user entry, one assistant entry."""

    def log_tool_call(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict,
        approved: bool,
        result: str,
    ) -> None:
        """Appends one {"type": "tool_call", ...} line per tool call within a turn."""

    def log_debug(self, message: str) -> None:
        """Appends timestamped line to debug log."""

    def close(self) -> None: ...

    @classmethod
    def from_existing(cls, jsonl_path: Path) -> "SessionLogger":
        """
        Open an existing session's JSONL and debug log in append mode.
        The debug log path is inferred by replacing the 'chat_' prefix with 'debug_'
        and the '.jsonl' suffix with '.log' in the same directory.
        Does NOT write session-start entries — the log already has them.
        """
```

---

### 4.10 `resume.py`

**Responsibilities:** Parse a `chat_<timestamp>.jsonl` file, validate it strictly, and return the data needed to bootstrap a resumed session. The log is the authoritative source of truth — no `[RESUMED_CONTEXT]` wrapper is used; the reconstructed fields are injected into `ChatSession` the same way a fresh session would set them.

```python
@dataclass
class ResumedSession:
    system_prompt: str                      # from "system_prompt" log entry
    startup_docs: list[tuple[Path, str]]    # from "startup_doc" log entries (in order)
    history: list[dict]                     # {"role": "user"|"assistant", "content": str} pairs
    session_attachments: list[Attachment]   # all attachments seen, unique by identifier, in order
    last_assistant_message: str             # for printing at startup
    jsonl_path: Path                        # original path (for logger append)
    next_attachment_n: int                  # max(N for CONTEXT_N in log) + 1; new attachments start here

def load_resume(path: Path) -> ResumedSession:
    """
    Parse the JSONL at path. Raises ResumeError with a descriptive message on:
      - File not found
      - Any line that is not valid JSON or is missing required fields (reports line numbers)
      - Zero user/assistant exchange turns found after parsing
      - No "system_prompt" entry found

    On success:
      - Reads system_prompt from the first "system_prompt" entry.
      - Reads startup_docs from all "startup_doc" entries (preserving order).
      - Builds history from all "exchange" role lines (role + content only).
      - Collects session_attachments: unique by identifier, in first-seen order.
      - Returns the last assistant exchange message for startup display.
      - Computes next_attachment_n as max(N for all CONTEXT_N identifiers found) + 1,
        or 1 if no attachments were present.
    """
```

---

### 4.11 `repl.py`

**Responsibilities:** Run the interactive input loop using `prompt_toolkit`. Dispatch slash commands. Print chat output.

```python
def run_repl(session: ChatSession) -> None:
    """
    Main loop:
      - Read input via prompt_toolkit (history enabled for ↑ recall; Esc clears input).
      - If input starts with '/': dispatch to handle_command().
      - Otherwise: call session.process(); print response; print trim notice if needed.
    """

def handle_command(cmd: str, session: ChatSession) -> None:
    """
    /set <param>=<value>  — update session.history_token_budget or session.config.test_timeout
    /extract <path>       — write code blocks from last response to <path> (fence language inferred from extension)
    /scratchpad           — print all scratchpad entries (title + content for each); print
                            "Scratchpad is empty." when none exist
    /clear                — reset session.history;
                            call session.rotate_logger() (which resets _next_attachment_n,
                            session_attachments, and _scratchpad);
                            print "Conversation history cleared. New session: <path>"
    /help                 — print command reference
    /exit                 — raise SystemExit
    """

_EXT_TO_FENCE: dict[str, str] = {
    ".py": "python",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
}

def _extract(cmd: str, session: ChatSession) -> None:
    """
    Parse an absolute file path from cmd. Infer the fence language from the
    file extension using _EXT_TO_FENCE. Find all fenced blocks of that language
    in the last assistant message (session.history[-1]). Write them to the path
    separated by blank lines. Print an error if: no argument given, extension is
    unknown (list supported extensions), no history, or no matching blocks found.
    """
```

---

### 4.12 `cli.py`

**Responsibilities:** Parse CLI args, bootstrap all components, start REPL.

```python
# Usage: pmca <config_name> [--unsafe] [--resume <path>]
def main() -> None:
    # 1. Parse args (argparse)
    # 2. load_config(config_name)  — exits with error on failure
    # 3. Validate OPENAI_API_KEY in environment
    # 4. Create log_folder and cache_dir
    # 5. If --resume: call load_resume(path) — exits with error on failure
    #    Else: instantiate SessionLogger with fresh timestamp
    # 6. Build VectorStore (prints progress)
    # 7. Instantiate ChatSession; if resuming, set session.history, resumed_context,
    #    and _next_attachment_n from ResumedSession.next_attachment_n
    # 8. If resuming: print "Resumed N turns from <path>" and "[last response]\n<msg>"
    # 9. run_repl(session)
    # 10. On exit: session.logger.close()
```

---

## 5. Message Assembly (API call)

Order sent to OpenAI on each turn:

```
[system]  <session.system_prompt>

[system]  <system_context>           ← only present when config.system_context_fields is non-empty;
                                     supported fields: "datetime", "os", "shell" (shell uses $SHELL on Unix, %COMSPEC% on Windows)

[system]  <startup_doc 1>            ← one entry per startup doc, if any
[system]  <startup_doc 2> ...

[system]  [CONTEXT_1]                ← ALL session_attachments accumulated so far
          File: /absolute/path/to/main.py
          Type: py
          ---
          <verbatim file content>
          ---
          [CONTEXT_2] ...

[system]  [SCRATCHPAD_1]            ← ALL _scratchpad entries saved by the LLM (if any)
          Title: read_file: src/pmca/config.py — load_config body
          ---
          <content excerpt>
          ---
          [SCRATCHPAD_2] ...

[user]    <prior user message>
[assistant] <prior response>
[user]    ...  ← trimmed history (oldest dropped first)
[user]    <current message>
```

All `session_attachments` are injected on every turn. `_scratchpad` entries are injected as `[SCRATCHPAD_i]` system messages after attachments, before history — only when the list is non-empty. The tag distinguishes saved context ("the LLM chose to keep this from a tool call return") from transient tool results. Because tool call returns are not stored in `history`, the scratchpad is the only mechanism for the LLM to preserve information it finds across turns. RAG chunks are retrieved on demand via the `query_knowledge_base` tool and appear inline as tool result messages; the LLM can save relevant excerpts to the scratchpad if they are load-bearing. Only user/assistant exchanges are stored in history. There is no special `[RESUMED_CONTEXT]` block — resumed sessions use the same assembly path as live sessions.

---

## 6. Storage Layout

```
<log_folder>/
├── chat_2025-05-14_15-32-10.jsonl
├── debug_2025-05-14_15-32-10.log
└── cache/
    ├── a3f8c1...pkl    # cached chunks+embeddings for one RAG file
    └── 9d2b47...pkl
```

Cache filename: `sha256(str(absolute_file_path)).hexdigest() + ".pkl"`

Cache invalidation: on load, compare stored `file_hash` (SHA-256 of file *content*) against current file. Recompute if mismatch.

---

## 7. CLI Interface

```
pmca <config_name> [--unsafe] [--resume <path>]
```

| Argument | Required | Description |
|---|---|---|
| `config_name` | Yes | Name of config to load (`<name>.yaml`) |
| `--unsafe` | No | Skip the file-attachment security prompt |
| `--resume <path>` | No | Path to a `chat_<timestamp>.jsonl` file to resume |

Config resolution:
- If the argument contains a path separator (`/` or `\`) or ends in `.yaml`: treated as a direct file path (absolute, or relative to cwd)
- Otherwise: looked up as `<pmca_package_dir>/configs/<name>.yaml`

### Resume behaviour (`--resume`)

When `--resume <path>` is provided:

1. The JSONL file at `<path>` is parsed. Hard errors (exit before REPL) on:
   - File not found
   - Zero valid user/assistant exchange turns found
   - No `system_prompt` entry found
   - Any malformed (non-JSON or schema-invalid) lines — print offending line numbers
2. `session.system_prompt` and `session.startup_docs` are set from the log entries.
   - If the loaded config's `system_prompt` differs from the log's, print a warning:
     `Warning: config system_prompt differs from log — using log version`
   - If the loaded config's `startup_docs` differ from the log's, print a warning:
     `Warning: config startup_docs differ from log — using log version`
3. `session.history` is populated from all `exchange` entries.
4. `session.session_attachments` is populated from all unique attachments across the log (unique by identifier, first-seen order).
5. `session._next_attachment_n` is set from `ResumedSession.next_attachment_n`.
7. The logger appends to the original JSONL file and its matching debug log — it does NOT write new session-start entries (they are already in the log).
8. At startup the REPL prints:
   - `Resumed N turns from <path>`
   - `[last response]` followed by the final assistant message from the log

History trimming is lazy: the first `session.process()` call runs `_trim_history` as normal. Message assembly uses the identical `_build_messages` path as a live session.

---

## 8. In-Session Commands

| Command | Effect |
|---|---|
| `/read add <path>` | Add a directory to `read_allowed_dirs` for this session (requires user approval) |
| `/read remove <path>` | Remove a directory from `read_allowed_dirs` for this session (requires user approval) |
| `/set history_token_budget=N` | Set history token budget for this session |
| `/set test_timeout=N` | Set test run timeout (seconds) for this session |
| `/extract <path>` | Extract code blocks from the last response into `<path>`; fence language inferred from extension (`.py`, `.yaml`/`.yml`, `.json`, `.toml`, `.sh`) |
| `/scratchpad` | Print all scratchpad entries (title + content); prints "Scratchpad is empty." if none exist |
| `/clear` | Clear conversation history, session_attachments, and _scratchpad; rotate to a new log file (writes fresh system_prompt + startup_doc entries); print new log path |
| `/help` | Print command reference and key bindings |
| `/exit` | End session (also: Ctrl+C) |

Key bindings:
- `↑` — recall previous input for editing
- `Esc` — clear current input

---

## 9. Error Handling Summary

| Situation | Behaviour |
|---|---|
| Invalid/missing YAML config | Print error, exit before session starts |
| Non-absolute path in config or attachment | Print error, exit (config) or reject message (attachment) |
| RAG file unreadable at startup | Print error, exit before session starts |
| Missing `OPENAI_API_KEY` | Print error, exit before session starts |
| Embedding API failure at startup | Print error, exit before session starts |
| Attachment file not found | Print `file not found: <path>`; message not sent; user can recall via ↑ |
| Attachment size exceeds `max_attachment_kb` | Print warning; continue to security prompt |
| User rejects security prompt | Print notice; message not sent; user can recall via ↑ |
| Transient API error (chat) | Retry 3× with backoff (1s/2s/4s); print `[retrying... attempt N/3]` |
| Permanent API error (chat) | Print error to chat; log to debug log; session continues |
| History trimmed | Print `[N earlier turn(s) omitted from context]` in chat UI |
| Unexpected exit / crash | Partial JSONL log is valid; no finalisation step needed |
| write_file path outside allowed dirs | Tool returns error string to model; user is not prompted |
| User denies write_file | Tool returns `"Write denied by user. Path: ..."` to model; session continues |
| write_file I/O error (e.g. permission denied) | Tool returns error string to model; session continues |
| edit_file path outside allowed dirs | Tool returns error string to model; user is not prompted |
| edit_file file not found | Tool returns error string to model; user is not prompted |
| edit_file old_string not found | Tool returns error string to model; user is not prompted |
| edit_file old_string appears multiple times | Tool returns error string (with count) to model; user is not prompted |
| User denies edit_file | Tool returns `"Edit denied by user. Path: ..."` to model; session continues |
| edit_file I/O error | Tool returns error string to model; session continues |
| read_file/list_dir/search/get_definition — path outside read_allowed_dirs | Tool returns error string to model; no user prompt |
| read_file file not found or I/O error | Tool returns error string to model; session continues |
| get_definition file not found, not a .py file, or parse error | Tool returns error string to model; session continues |
| search invalid regex pattern | Tool returns error string to model; session continues |
| get_definition symbol not found | Tool returns error string to model; session continues |
| /read add/remove — user denies | read_allowed_dirs unchanged; message printed to user |
| run_tests — test_dir not configured | Tool returns error string to model |
| run_tests — timeout exceeded | Tool returns `"Error: run_tests timed out after N seconds"` to model |
| run_tests — subprocess error (e.g. command not found) | Tool returns error string to model; session continues |
| /set test_timeout — non-positive value | Print error; leave test_timeout unchanged |
| `--resume` file not found | Print error, exit before session starts |
| `--resume` file has zero valid turns | Print error, exit before session starts |
| `--resume` file has malformed lines | Print error with line numbers, exit before session starts |

---

## 10. Windows Compatibility

`pmca` targets `linux-64` and `win-64` (see `pixi.toml`). The following design decisions ensure cross-platform operation:

| Area | Decision |
|---|---|
| Path fields in config | Expanded via `Path.expanduser()` before the absolute-path check; premade configs use `~/.pmca/logs` |
| Attachment path parsing | Leading/trailing `"` stripped from `[[...]]` tokens (Windows Explorer quote-wraps copied paths) |
| Attachment identifier substitution | Keyed on `Path` objects so mixed `/` and `\` styles match on Windows |
| REPL and I/O | `prompt_toolkit` provides cross-platform terminal support; all file I/O uses `pathlib` and explicit `encoding="utf-8"` |

### First-time setup on Windows

After cloning, run:

```
pixi install
```

This regenerates `pixi.lock` with `win-64` package resolutions if not already present.

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `openai` | Chat completions + embeddings API |
| `numpy` | Embedding vectors, cosine similarity |
| `pyyaml` | YAML config parsing |
| `prompt_toolkit` | REPL input, history, key bindings |
| `pickle` (stdlib) | Embedding disk cache serialisation |
| `ast` (stdlib) | Python file chunking |
| `hashlib` (stdlib) | SHA-256 for cache keys and file invalidation |
| `pathlib` (stdlib) | All path handling |
| `argparse` (stdlib) | CLI argument parsing |
