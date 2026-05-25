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
    top_k_chunks: int
    log_folder: Path
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

### 3.4 LogEntry (JSONL)

```python
{
    "timestamp": "2025-05-14T15:32:10Z",
    "role": "user" | "assistant" | "system",
    "content": "...",
    "rag_chunks": [                         # present on user turns only
        {"label": "...", "source": "...", "content": "..."}
    ],
    "attachments": [                        # present on user turns only
        {"identifier": "CONTEXT_1", "path": "...", "content": "...", "size_warning": false}
    ]
}
```

`attachments[].content` stores the verbatim file content so that a resumed session can reconstruct attachment context without re-reading the original files (which may have moved or changed).

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
- `OPENAI_API_KEY` must be present in environment
- `top_k_chunks` must be a positive integer
- `max_attachment_kb` and `history_token_budget` must be positive integers if provided

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
        For each file: load cache entry if present and file hash matches;
        otherwise chunk + embed + write cache. Prints progress to stderr
        (e.g. "[RAG] embedding 3 new/changed files...").
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
) -> list[Attachment]:
    """
    For each path:
      1. Raise AttachmentError if file does not exist.
      2. If size > max_attachment_kb: print warning, continue.
      3. If not unsafe: prompt "You are about to attach <path>. Have you
         reviewed it for secrets? (y/n)". If 'n': raise AttachmentAborted.
      4. Read content, assign identifier CONTEXT_<n>.
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
    history: list[dict]         # {"role": "user"|"assistant", "content": str}
    top_k: int                  # mutable via /set
    history_token_budget: int   # mutable via /set
    _last_rag_chunks: list[Chunk]  # stored for /rag command
    resumed_context: str | None  # [RESUMED_CONTEXT] system message body, set at resume time

    def process(self, user_input: str) -> str:
        """
        Full pipeline for one user turn:
          1. Parse and resolve attachments; substitute identifiers in message.
          2. Trim history to fit history_token_budget; note turns dropped.
          3. Query RAG store for top_k chunks.
          4. Assemble message list (see Section 5).
          5. Call openai_client.chat_completion().
          6. Append user + assistant turns to self.history.
          7. Log exchange via SessionLogger.
          8. Return (assistant_response, turns_dropped, last_rag_chunks).
        """

    def rotate_logger(self) -> Path:
        """
        Close the current logger, open a new one with a fresh timestamp in
        config.log_folder, assign it to self.logger, and return the path of
        the new JSONL file.
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
def chat_completion(messages: list[dict], config: Config) -> str:
    """
    Calls openai.chat.completions.create() with config model + optional params.
    Retries up to 3 times on transient errors (429, network timeout, 5xx)
    with exponential backoff (1s, 2s, 4s).
    Prints "[retrying... attempt N/3]" before each retry.
    Raises APIError (permanent) or APITransientError (exhausted retries).
    """
```

Transient errors: `RateLimitError`, `APIConnectionError`, `APIStatusError` with status >= 500.
Permanent errors: `AuthenticationError`, `BadRequestError`, other `APIStatusError`.

---

### 4.8 `logger.py`

**Responsibilities:** Write per-session JSONL chat log and plaintext debug log. Both files are opened at session start and flushed after every write (no buffering).

```python
class SessionLogger:
    def __init__(self, log_folder: Path, timestamp: str) -> None:
        """
        Opens <log_folder>/chat_<timestamp>.jsonl and
               <log_folder>/debug_<timestamp>.log.
        """

    def log_exchange(
        self,
        user_message: str,
        assistant_message: str,
        rag_chunks: list[Chunk],
        attachments: list[Attachment],
    ) -> None:
        """Appends two JSONL lines: one user entry, one assistant entry."""

    def log_debug(self, message: str) -> None:
        """Appends timestamped line to debug log."""

    def close(self) -> None: ...

    @classmethod
    def from_existing(cls, jsonl_path: Path) -> "SessionLogger":
        """
        Open an existing session's JSONL and debug log in append mode.
        The debug log path is inferred by replacing the 'chat_' prefix with 'debug_'
        and the '.jsonl' suffix with '.log' in the same directory.
        """
```

---

### 4.9 `resume.py`

**Responsibilities:** Parse a `chat_<timestamp>.jsonl` file, validate it strictly, and return the data needed to bootstrap a resumed session.

```python
@dataclass
class ResumedSession:
    history: list[dict]             # {"role": "user"|"assistant", "content": str} pairs
    resumed_context: str            # formatted [RESUMED_CONTEXT] system message body
    last_assistant_message: str     # for printing at startup
    jsonl_path: Path                # original path (for logger)

def load_resume(path: Path) -> ResumedSession:
    """
    Parse the JSONL at path. Raises ResumeError with a descriptive message on:
      - File not found
      - Any line that is not valid JSON or is missing required fields (reports line numbers)
      - Zero user/assistant turns found after parsing

    On success:
      - Builds history from all "user"/"assistant" role lines (role + content only).
      - Collects all unique attachments (by identifier) and RAG chunks across all turns.
      - Formats them into a single [RESUMED_CONTEXT] system message string.
      - Returns the last assistant message for startup display.
    """
```

The `[RESUMED_CONTEXT]` block format:

```
[RESUMED_CONTEXT]
The following attachments and RAG chunks were present in the resumed session.

[CONTEXT_1]
File: /path/to/file.py
Type: py
---
<verbatim content>
---

[RAG_1]
File: /path/to/source.py
Chunk: function `foo` (lines 1–10)
---
<chunk content>
---
```

---

### 4.10 `repl.py`

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
    /set <param>=<value>  — update session.top_k or session.history_token_budget
    /rag                  — print session._last_rag_chunks
    /extract <path>       — write code blocks from last response to <path> (fence language inferred from extension)
    /clear                — reset session.history, session._last_rag_chunks, and session.resumed_context;
                            call session.rotate_logger(); print "Conversation history cleared. New session: <path>"
    /help                 — print command reference
    /exit                 — raise SystemExit
    """

_EXT_TO_FENCE: dict[str, str] = {
    ".py": "python",
    ".yaml": "yaml", ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
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

### 4.10 `cli.py`

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
    # 7. Instantiate ChatSession; if resuming, set session.history and resumed_context
    # 8. If resuming: print "Resumed N turns from <path>" and "[last response]\n<msg>"
    # 9. run_repl(session)
    # 10. On exit: session.logger.close()
```

---

## 5. Message Assembly (API call)

Order sent to OpenAI on each turn:

```
[system]  <config.system_prompt>

[system]  [RESUMED_CONTEXT]          ← only present when session was resumed
          <all prior attachments and RAG chunks from loaded log>

[system]  <startup_docs, if any>

[system]  [RAG_1]                    ← fresh RAG for the current turn
          File: /absolute/path/to/file.py
          Chunk: function `parse_config` (lines 15–32)
          ---
          <chunk content>
          ---
          [RAG_2] ...

[system]  [CONTEXT_1]                ← attachments for the current turn
          File: /absolute/path/to/main.py
          Type: py
          ---
          <verbatim file content>
          ---

[user]    <message with [[filepath]] replaced by CONTEXT_1>
[assistant] <prior response>
[user]    ...  ← trimmed history (oldest dropped first)
[user]    <current message>
```

RAG and attachment system messages are injected fresh each turn; only user/assistant exchanges are stored in history. The `[RESUMED_CONTEXT]` block is injected once per session when `session.resumed_context` is set.

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
   - Zero valid user/assistant turns found
   - Any malformed (non-JSON or schema-invalid) lines — print offending line numbers
2. All prior `user`/`assistant` turns are loaded into `session.history` as `{"role": ..., "content": ...}`.
3. All unique attachments and RAG chunks from the JSONL are collected and injected as a single `[RESUMED_CONTEXT]` system message immediately after `config.system_prompt` in `_build_messages`. This gives the model visibility into prior attachment content referenced in the resumed history.
4. The logger appends to the original JSONL file (passed path) and to its matching debug log (same directory, filename with `chat_` → `debug_` and `.jsonl` → `.log`).
5. At startup the REPL prints:
   - `Resumed N turns from <path>`
   - `[last response]` followed by the final assistant message from the log

History trimming is lazy: the first `session.process()` call runs `_trim_history` as normal.

---

## 8. In-Session Commands

| Command | Effect |
|---|---|
| `/set chunksize=N` | Set top-k RAG retrieval count for this session |
| `/set history_token_budget=N` | Set history token budget for this session |
| `/rag` | Print RAG chunks retrieved for the last query |
| `/extract <path>` | Extract code blocks from the last response into `<path>`; fence language inferred from extension (`.py`, `.yaml`/`.yml`, `.json`, `.toml`, `.sh`) |
| `/clear` | Clear conversation history, last RAG chunks, and resumed context; rotate to a new log file; print new log path |
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
