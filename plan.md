# pmca improvement roadmap

Four pain points addressed in order of implementation dependency.

---

## Phase 1 — Tool progress output

**Goal:** user sees what the LLM is doing during tool call chains.

**Change:** in `chat.py` `process()`, print one line before each `_dispatch_tool` call:
```
[tool: edit_file /src/pmca/chat.py]
```
Extract the "key arg" per tool name (path for file tools, query for rag/search, ref for git tools).

**Files:** `src/pmca/chat.py`

**No config needed.** Always on.

---

## Phase 2 — JSON repair via `ast.literal_eval` fallback

**Goal:** stop crashing on single-quoted tool call arguments emitted by the LLM.

**Change:** in `openai_client.py`, wrap `json.loads(tc.function.arguments)` in a try/except; on `JSONDecodeError` fall back to `ast.literal_eval`. If both fail, raise a new `MalformedToolCallError` with the raw string included, so the caller can return a useful error to the model.

**Files:** `src/pmca/openai_client.py`, `src/pmca/types.py` (new exception)

**No new dependencies.**

---

## Phase 3 — Unified diffs at approval time

**Goal:** show exactly what will change before the user approves a write.

**Changes:**
- `execute_edit_file`: replace the current `--- remove / insert ---` block with `difflib.unified_diff` of old vs new full file content.
- `execute_write_file` (existing file): read current content, compute unified diff against incoming content, print before prompt.
- `execute_insert_at_line`: compute unified diff of lines before/after insertion, print before prompt.
- When `auto_approve_writes=True`: no diff printed (existing behaviour preserved).
- New config field `show_diff_on_approve: bool = True`; new `/set show_diff_on_approve=true|false` command.

**Files:** `src/pmca/tools.py`, `src/pmca/config.py`, `src/pmca/repl.py`

---

## Phase 4 — Logging improvements

**Goal:** make the debug log actually useful.

**Four sub-changes (all independently testable):**

### 4a — API call timing
Log duration of each `chat_completion` call to `debug_*.log`:
```
[2025-01-01T12:00:01Z] chat_completion: 3.4s, model=gpt-4.1
```
**Files:** `src/pmca/openai_client.py`, `src/pmca/logger.py` (new `log_api_call` method)

### 4b — Raw API payloads
Log full `messages` list sent and raw response received to `debug_*.log` (not `.jsonl`).
**Files:** `src/pmca/openai_client.py`, `src/pmca/logger.py` (new `log_api_payload` method)  
Requires passing `logger` into `chat_completion` — thread it through from `chat.py`.

### 4c — Tool result summaries in terminal
After each tool call, print a one-line result summary to stdout:
```
[tool: edit_file /src/pmca/chat.py → ok]
[tool: run_tests → FAILED (3 errors)]
[tool: read_file /src/pmca/chat.py → 142 lines]
```
Condense result string per tool type (line count for reads, pass/fail for tests, ok/error for writes).
**Files:** `src/pmca/chat.py`

### 4d — `log_debug` actually used
Wire `log_debug` calls into: session start, each tool dispatch (name + truncated result), history trim events, attachment resolution.
**Files:** `src/pmca/chat.py`, `src/pmca/logger.py`

---

## Implementation order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4a → 4b → 4c → 4d
```

Phases 1 and 2 are independent and can be done in either order.
Phase 3 depends on nothing but touches the most code in tools.py.
Phase 4 sub-changes are all independent of each other.
