# Security Review: Accidental Data Inclusion in LLM API Calls

**Date:** 2026-05-25
**Scope:** All code paths where data reaches the OpenAI API without explicit user action
**Threat model:** The user does not actively try to pass secrets to the LLM. Findings cover accidental inclusion only.

---

## Finding 1 — HIGH: System metadata auto-injected into every request ✅ RESOLVED

**Location:** `src/pmca/chat.py:39, 169–177` (as of the review; see commit `f733e6f` for the fix)

`_build_system_context()` previously transmitted unconditionally: datetime, full OS version string (`platform.version()`), hostname (`platform.node()`), username (`getpass.getuser()`), and shell.

**Resolution (`f733e6f` — Phase 22 revised):** The function now takes a `fields: list[str]` argument driven by a new config key `system_context_fields` (default: `[]`). When the list is empty — which it is unless the user explicitly opts in — **nothing is injected**. Hostname and username have been removed entirely as recognised fields; even an opt-in user cannot accidentally send them. The reduced field set is:

| Field value | Content injected |
|---|---|
| `"datetime"` | `Session started: 2026-05-25 14:03:11 +0200` |
| `"os"` | `OS: Linux` (`platform.system()` only — no version string or hostname) |
| `"shell"` | `Shell: /usr/bin/zsh` (`$SHELL` on Unix, `%COMSPEC%` on Windows) |

The sample configs document the field via a commented-out line so users are aware the option exists without enabling it by default.

---

## Finding 2 — HIGH: RAG chunks accumulate silently and grow without bound ✅ RESOLVED

**Location:** `src/pmca/chat.py:58–60, 142–148`

`_merge_rag_chunks` only ever appends to `session_rag_chunks` — it never removes or caps entries. Every distinct chunk retrieved by any query during a session is permanently added to the context and **re-transmitted on every subsequent API call**. There is no eviction policy, no size cap, and no user-visible feedback about how many chunks are currently in context.

**Why it is accidental:** The user's mental model is "RAG retrieves relevant bits per question." The actual behaviour is "RAG builds an ever-growing set of file excerpts that ride along with every subsequent prompt." Over a wide-ranging session, a significant portion of the indexed files can end up permanently in the context window.

**Concrete scenario:** A user indexes a repo with `rag_files`. An early question causes retrieval of a chunk from `config.py` that contains a commented-out credential (`# SECRET_KEY = "prod-abc123"`). That chunk then accompanies every remaining turn for the rest of the session, including turns asking about completely unrelated topics.

**Recommendation:**
- Show a chunk count after each response, e.g. `[RAG context: 7 chunks from 3 files]`.
- Add a hard cap on `session_rag_chunks` (configurable, e.g. `max_rag_context_chunks`).
- Add a `/rag-clear` command that evicts accumulated chunks without clearing conversation history (currently `/clear` does both, so users avoid it).

**Resolution (Phase 27 — RAG converted to LLM-callable tool):** The `session_rag_chunks` accumulation mechanism was removed entirely. RAG is no longer auto-fired; the LLM calls `query_knowledge_base` explicitly as a tool. Chunk results are returned as tool-result messages scoped to the local `messages` list for the current turn and are never written to `self.history`. Cross-turn accumulation is therefore structurally impossible. Within-turn persistence of tool results is a necessary consequence of the OpenAI tool-use protocol and is a conscious design decision. The chunk-count feedback recommendation is retained as a minor UX item.

---

## Finding 3 — MEDIUM: Every user message is sent to the embeddings API

**Location:** `src/pmca/rag/store.py:54`, `src/pmca/rag/embedder.py:15–27`

`session.process()` calls `self.store.query(user_input, self.top_k)` unconditionally whenever `rag_files` is configured. `query()` calls `embed([text])`, which sends the raw user message to `text-embedding-3-small` via a **separate API endpoint** from the chat completions endpoint.

The early-return guard (`if not self._chunks: return []`) only protects the case where no RAG files were configured. Any session with `rag_files` sends every user message to the embeddings endpoint in addition to the chat completions endpoint.

**Why it is accidental:** The user configured `rag_files` to index their codebase, not to share their conversational queries with a second API endpoint. The embeddings endpoint may have different data-retention policies than the chat completions endpoint. There is no disclosure of this behaviour in the CLI output or documentation.

**Recommendation:** Document that user queries are embedded when `rag_files` is set. Consider a config flag (`rag_embed_queries: false`) that disables per-query retrieval, relying instead on startup docs and explicit attachments for context injection.

---

## Finding 4 — HIGH: Pickle deserialization of cache files enables RCE

**Location:** `src/pmca/rag/store.py:65`

```python
data = pickle.loads(cache_file.read_bytes())
```

The cache directory (`config.log_folder / "cache"`) is created with default filesystem permissions. Any process or user with write access to that directory can place a crafted `.pkl` file that executes arbitrary code the next time `pmca` starts — before any user interaction occurs.

**Why it matters for data leakage:** An attacker exploiting this can exfiltrate any file the process can read, including files that were intentionally never passed to the LLM. This entirely bypasses the access controls the rest of the codebase enforces.

**Concrete scenario:** `log_folder` is set to a shared project directory (e.g. `/srv/project/logs`). An attacker with write access drops a crafted `.pkl` in `cache/`. The next team member to run `pmca` executes the payload with their privileges.

**Recommendation:** Replace pickle with a safe format. Embeddings can be stored with `numpy.save` / `numpy.load`; chunk metadata can be stored as JSON. Pickle should not be used for data that persists to disk across process boundaries.

---

## Finding 5 — LOW: Resume reloads startup-doc content from the log, not from disk

**Location:** `src/pmca/resume.py:53–57`

When `--resume` is used, `startup_docs` content is taken verbatim from the JSONL log file rather than being re-read from the original paths on disk. If the log file has been modified after the original session (by a post-processing script, manual edit, or accident), the modified content is injected into the LLM context without the user noticing. The warning at `cli.py:62–65` only compares paths, not file contents.

**Recommendation:** On resume, re-read startup docs from their original paths on disk (consistent with how a fresh session behaves). If exact log replay is intentional, add a visible diff or hash check between the logged content and the current file so the user is aware of any divergence.

---

## Summary

| # | Severity | Location | Issue |
|---|---|---|---|
| 1 | HIGH ✅ | `chat.py:169–177` | Hostname, username, OS version auto-injected into every API call |
| 2 | HIGH ✅ | `chat.py:142–148` | RAG chunks accumulate without bound across a session |
| 3 | MEDIUM | `store.py:54` | User queries sent to embeddings API without disclosure |
| 4 | HIGH | `store.py:65` | Pickle deserialization of cache enables RCE |
| 5 | LOW | `resume.py:53–57` | Resume injects stale startup-doc content from log |
