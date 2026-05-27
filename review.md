# Code Review Notes

## Medium

- [ ] **Config search path doesn't match design** (`config.py:8`)
  Design says search cwd then `~/.pmca/`; implementation only looks in `<package_dir>/configs/`. Bare name `pmca myconfig` won't find a config in cwd.

- [ ] **Dead code** (`openai_client.py:49–54`)
  `_is_transient()` is defined but never called — transient logic is already inlined in the `except` clauses. Delete it.

- [ ] **`openai.OpenAI()` client instantiated per call** (`embedder.py:16`, `openai_client.py:22`)
  New client (and HTTP connection pool) created on every chat turn and every embed call. Make them module-level singletons.

- [ ] **Progress message appears after embedding** (`store.py:44–45`)
  `"[RAG] embedding N new/changed file(s)..."` is printed after the work is done. Move it before the loop so the user sees feedback during the wait.

## Minor

- [ ] **Pickle cache has no error recovery** (`store.py:65`)
  `pickle.loads()` on a corrupted or version-incompatible cache raises an opaque error and crashes startup. Wrap the load in try/except and treat failure as a cache miss.

- [ ] **No streaming**
  The tool waits silently for full responses. The OpenAI SDK supports `stream=True`; even basic character-by-character output would improve perceived responsiveness.

- [ ] **Attachment paths must be absolute**
  `[[/absolute/path]]` is verbose. Relative-to-cwd paths would be more ergonomic; the security prompt still covers the risk.

- [ ] **`repl.py` accesses `session._last_rag_chunks` directly** (`repl.py:72`)
  Cross-module access to a private attribute. Add a `last_rag_chunks` property to `ChatSession`.

- [ ] **No `/clear` command**
  No way to reset history mid-session. Common need when context pivots to a new topic.

- [ ] **`print()` in library code** (`chat.py`, `attachments.py`)
  Direct `print()` calls in library modules make unit testing awkward. Return messages or accept a writer callback instead.

## Future consideration

- RAG chunks are injected fresh as system messages every turn, which rules out prompt caching at the API level. If caching support matters later, the injection strategy will need rethinking.
