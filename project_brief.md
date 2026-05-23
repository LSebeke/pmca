

# Project Brief: `pmca` (Poor Man's Coding Assistant)

 

---

 

## 1. Problem Statement

 

**What business/operational need does this solve?**

 

A personal productivity tool to accelerate Python development workflows by providing a conversational LLM interface with:

- Project-aware context via RAG (Retrieval-Augmented Generation) from local code/documentation files

- On-demand file attachment for ad-hoc troubleshooting

- Session logging for review and audit

- Safety controls for file sharing

 

This tool addresses the friction of context-switching between code editors, documentation, and generic LLM chat interfaces by embedding project-specific knowledge directly into the assistant.

 

---

 

## 2. User & Audience

 

**Who is this for?**

 

- **Primary user:** You (experienced Python developer, self-taught, ~10 years experience)

- **Use case:** Assistance with Python projects — debugging, design discussions, code review, exploratory questions

- **Technical proficiency:** High — comfortable with CLI tools, YAML config, Python scripting

- **Future scope (out of scope for v1):** Potential use by technical colleagues

 

---

 

## 3. Data Sources

 

### 3.1 RAG Knowledge Base

- **Origin:** Local filesystem files specified in YAML config

- **Path format:** Absolute paths only (e.g., `/home/user/projects/myapp/src/`)

- **File types:** Text files (`.txt`, `.md`, `.py`, etc.) — prose and Python code

- **Volume:** Personal project scale — tens to low hundreds of files expected

- **Access pattern:** One-time ingestion at tool startup; no live file watching or updates during session

- **Chunking strategy:**

  - **Prose files (`.txt`, `.md`):** Semantic chunking (paragraph/section boundaries)

  - **Python files (`.py`):** Function/class/method-level chunking with metadata labels indicating origin (module, class, function name)

 

### 3.2 Manual File Attachments (During Session)

- **Origin:** User-specified files via `[[filepath]]` syntax in chat messages

- **Path format:** Absolute or relative paths (relative resolved from working directory where `pmca` was invoked)

- **File types:** Same as above — treated as plain text, injected verbatim

- **Access pattern:** Ad-hoc, user-initiated during conversation

 

### 3.3 OpenAI API

- **Models:** Configurable per YAML config (e.g., `gpt-4o`, `gpt-4o-mini`)

- **Embeddings:** `text-embedding-3-small` for RAG pipeline

- **API key:** Assumed available in environment or config (validation at startup)

 

---

 

## 4. Data Destinations

 

### 4.1 Chat Session Logs

- **Format:** JSONL (JSON Lines) — one JSON object per message/event

- **Metadata per line:**

  - Timestamp

  - Role (`user`, `assistant`, `system`)

  - Message content

  - RAG chunks retrieved (if applicable)

  - File attachments used (if applicable)

- **File naming:** `chat_<timestamp>.jsonl` (e.g., `chat_2025-05-14_15-32-10.jsonl`)

- **Storage location:** Folder path specified in YAML config

- **Scope:** User/assistant exchanges + RAG context injected (visible in log, not in chat UI)

 

### 4.2 Debug/Error Logs

- **Format:** Standard `.log` file (text-based, timestamped entries)

- **Content:** Errors, warnings, debug info (e.g., API errors, file read failures, invalid commands)

- **File naming:** `debug_<timestamp>.log` (same timestamp as corresponding chat session)

- **Storage location:** Same folder as chat logs

 

---

 

## 5. Transformation Summary (High-Level)

 

### 5.1 RAG Pipeline (Startup)

1. Read file paths for RAG from YAML config

2. Load and chunk files according to type (semantic for prose, AST-based for Python)

3. Embed chunks using OpenAI `text-embedding-3-small`

4. Store embeddings + chunk metadata in-memory (NumPy arrays for brute-force cosine similarity)

 

### 5.2 Message Processing (Runtime)

1. User types a message (may contain `[[filepath]]` attachments)

2. If file attachment present:

   - Validate file existence

   - If file size exceeds `max_attachment_kb` (default 500 KB): print a warning but allow user to proceed

   - Prompt: *"You are about to attach the file <filepath>. Have you reviewed it and confirmed that it does not contain secret information? (y/n)"*

   - Read file content verbatim

   - Inject as system message with identifier (e.g., `[CONTEXT_1]`)

   - Replace `[[filepath]]` in user message with identifier

3. Trim conversation history to fit `history_token_budget`; if turns are dropped, print `[N earlier turn(s) omitted from context]` in the chat UI

4. Retrieve top-k RAG chunks via cosine similarity against user query embedding

5. Inject RAG chunks as system message before user message

6. Send to OpenAI API (system messages + conversation history + user message)

7. Log full exchange (including RAG context and file attachments) to JSONL

8. Display assistant response to user

 

### 5.3 Commands

- **`/set <param>=<value>`** — Adjust session parameters mid-session. Supported params:
  - `chunksize=X` — top-k RAG retrieval count
  - `history_token_budget=X` — conversation history token budget

- **`/rag`** — Print the RAG chunks retrieved for the last query to the chat UI (for debugging retrieval quality)

- **`/help`** — Print all available commands and key bindings to the chat UI

- **`↑` (up arrow)** — Recall last user input for editing (e.g., after file-not-found error)

- **`Esc`** - Clear current user chat input to start composing a new message
 

---

 

## 6. Trigger / Schedule

 

**When and how does the code run?**

 

- **Trigger:** User-initiated via CLI command: `pmca <config_name> [--unsafe]`

  - Example: `pmca PythonAssistant --unsafe`

  - `--unsafe`: skips the "have you reviewed this file?" security prompt for `[[filepath]]` attachments; all other validation (file existence, size warning, absolute path requirement) still applies

- **Execution environment:** Local machine, terminal/shell session

- **Session model:** Interactive, stateful — user sends messages, receives responses, until manually exiting (e.g., `/exit`, Ctrl+C)

 

---

 

## 7. Success Criteria

 

**What does correct output look like? How will you know it worked?**

 

A successful session means:

1. Tool starts without errors when YAML config is valid

2. User can send messages and receive relevant, contextual responses

3. RAG retrieval injects appropriate chunks based on query (visible in log, not chat UI)

4. File attachments work seamlessly via `[[filepath]]` syntax

5. Session log (JSONL) and debug log are written correctly to configured folder

6. **No crashes** during normal operation (API errors, file-not-found, invalid commands are handled gracefully)

 

---

 

## 8. Error Expectations

 

**What should happen when things go wrong, at a business level?**

 

### Startup Errors (Tool Refuses to Start)

- Malformed YAML config

- Missing required fields (config name, model, log folder, etc.)

- RAG file paths that don't exist or aren't readable

- Invalid OpenAI API key or model name

- Embedding API call fails during startup

- **Behavior:** Print clear error message, do not start session

 

### Runtime Errors (Session Continues)

- **OpenAI API errors — transient** (rate limit 429, network timeout, 5xx):

  → Retry up to 3 times with exponential backoff (1s, 2s, 4s); print `[retrying... attempt N/3]` during wait; after 3 failures, print error message to chat, log to `.log`, user can retry manually

- **OpenAI API errors — permanent** (invalid request, auth failure):

  → Print error message to chat immediately, log to `.log`, no retry

 

- **File not found on `[[filepath]]` attachment**: 

  → Message not sent; print *"file not found: <filepath>"*; user can recall input via `/` or `↑` key to correct

 

- **Invalid `/set` command** (e.g., `/set chunksize=-5`): 

  → Print error message, log to `.log`, ignore command

 

- **User rejects file attachment prompt in safe mode**: 

  → Message not sent; user can edit and retry

 

---

 

## 9. Scope Boundaries

 

**What is explicitly out of scope?**

 

### Out of Scope for v1

- Multi-user support or shared configs

- File watching / live updates to RAG knowledge base during session

- Web search or external API integrations (beyond OpenAI)

- GUI or web interface (CLI only)

- Streaming responses from OpenAI API (batch response only)

- Autonomous RAG queries by the assistant (user query drives retrieval only)

- Support for non-text file types (images, PDFs, etc.)

- Advanced chunking strategies beyond semantic/AST-based (e.g., sliding windows, hybrid)

 

### Future Considerations (Not Committed)

- Full persistent vector store (ChromaDB, FAISS on disk)

- File watching for live RAG updates

- Assistant-initiated RAG queries

- Team/collaborative use

 

---

 

## 10. Constraints

 

### Technical Constraints

- **Dependencies:** Python 3.11, NumPy, OpenAI Python SDK, YAML parser

- **Embedding model:** OpenAI `text-embedding-3-small` (requires API access and billing)

- **Vector store:** In-memory NumPy arrays. Embeddings are cached to disk (`<log_folder>/cache/`) and invalidated per file via SHA-256 hash — regenerated only when source files change.

- **File I/O:** Local filesystem only (no network folders, databases, or cloud storage)

 

### Operational Constraints

- **Cost:** OpenAI API usage (embedding + chat) — minimal for personal use, but not zero

- **Performance:** Startup time scales with number/size of RAG files (embedding is API-bound); retrieval is O(n) brute-force cosine similarity

- **Platform:** Windows powershell environment (command syntax, path resolution)
 

### Time Constraints

- **Deadline:** None specified — personal project, iterative development expected

 

---

 

## 11. YAML Configuration Structure

 

**Required fields:**

- `name` (string) — Config identifier, used at startup (e.g., `PythonAssistant`)

- `model` (string) — OpenAI model name (e.g., `gpt-4o`)

- `system_prompt` (string) — System message sent with every API call

- `rag_files` (list of strings) — Absolute paths to files for RAG ingestion

- `top_k_chunks` (integer) — Default number of RAG chunks to retrieve per query

- `log_folder` (string) — Absolute or relative path to folder for chat/debug logs

 

**Optional fields:**

- `max_attachment_kb` (integer, default 500) — soft warning threshold for file attachments

- `history_token_budget` (integer, default 4000) — approximate token budget for conversation history sent per message; oldest turns dropped first when exceeded; estimated via character count (1 token ≈ 4 chars)

- `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, etc. (OpenAI API parameters)

- If omitted, OpenAI API defaults are used

 

**Example YAML:**

```yaml

name: PythonAssistant

model: gpt-4o-mini

system_prompt: "You are a helpful Python coding assistant."

rag_files:

  - /home/user/projects/myapp/src/main.py

  - /home/user/projects/myapp/docs/architecture.md

top_k_chunks: 3

log_folder: /home/user/pmca_logs

temperature: 0.7

max_tokens: 1500

```

 

---

 

## 12. Message Structure (Sent to OpenAI API)

**Message ordering:** (1) base system prompt, (2) RAG chunks, (3) file attachments, (4) user message. Most-specific context is closest to the user message.

### When RAG + File Attachment Both Present

 

**System message (RAG chunks):**

```

[RAG_1]

File: /home/user/projects/myapp/src/utils.py

Chunk: function `parse_config` (lines 15-32)

---

<chunk content>

---

 

[RAG_2]

File: /home/user/projects/myapp/docs/architecture.md

Chunk: Section "Configuration Loading"

---

<chunk content>

---

```

 

**System message (file attachment):**

```

[CONTEXT_1]

File: ./src/main.py

Type: py

---

<verbatim file content>

---

```

 

**User message:**

```

I'm seeing an issue in [CONTEXT_1] around line 42. How does this interact with the config loader?

```

 

---

 

## 13. Assumptions & Open Questions

 

### Assumptions

- User has OpenAI API key configured and valid

- Python environment with NumPy and OpenAI SDK available

- Files specified in YAML are readable and text-based

- All file paths (RAG files and attachments) must be absolute; no relative path resolution

 

### Flagged for Future Clarification

- None outstanding.


---

 

## 14. Handoff Notes for Stage 2 (Design & Implementation)

 

- **RAG chunking for Python files:** Recommend `ast` module for parsing (function/class extraction); attach metadata as docstring or comment prefix

- **Embedding interface:** Isolate embedding calls behind a thin `embed(texts) -> np.ndarray` function from the start, even though only OpenAI is supported in v1 — makes provider swaps a one-file change later

- **Command parsing:** Consider `cmd` or `prompt_toolkit` library for slash commands and input history (up-arrow recall)

- **Security prompt:** Simple `input()` prompt with y/n validation; state tracked per session (don't re-prompt for same file)

- **JSONL logging:** Use `json.dumps()` per message, append to file; rotate/timestamp file per session

- **Error handling:** Wrap API calls in try/except; distinguish transient (retry-able) vs. fatal errors

 

---
