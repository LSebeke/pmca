# pmca — Poor Man's Coding Assistant

A CLI chat tool that wraps the OpenAI API with project-aware context via RAG.

## Setup

Install [pixi](https://prefix.dev/docs/pixi/overview), then:

```
pixi install
```

This resolves all dependencies for your platform (Linux or Windows) and writes them into the lock file if not already present.

Set your API key:

```
export OPENAI_API_KEY=sk-...   # Linux / macOS
set OPENAI_API_KEY=sk-...      # Windows CMD
$env:OPENAI_API_KEY="sk-..."   # Windows PowerShell
```

## Usage

```
pixi run pmca <config_name> [--unsafe] [--resume <path>]
```

`<config_name>` is either the name of a built-in config (`grill-me`, `review`, `tdd`) or a path to a custom YAML file.

### Windows

After cloning, run `pixi install` to generate the `win-64` package resolutions in `pixi.lock`. No other platform-specific setup is required — all paths in config files support `~` (e.g. `log_folder: ~/.pmca/logs`) and attachment tokens accept both `/` and `\` separators.

## Config format

```yaml
name: my-assistant
model: gpt-4.1
system_prompt: "You are a helpful assistant."
rag_files:
  - ~/project/src/main.py
top_k_chunks: 5
log_folder: ~/.pmca/logs
```

All path fields (`log_folder`, `rag_files`, `startup_docs`) accept `~` and are expanded at load time.
