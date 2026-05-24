from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from pmca.chat import ChatSession
from pmca.config import ConfigError, load_config
from pmca.logger import SessionLogger
from pmca.rag.store import VectorStore
from pmca.repl import run_repl
from pmca.resume import ResumeError, load_resume


def main() -> None:
    parser = argparse.ArgumentParser(prog="pmca", description="Poor Man's Coding Assistant")
    parser.add_argument("config_name", help="Config name or path to YAML file")
    parser.add_argument("--unsafe", action="store_true", help="Skip file-attachment security prompt")
    parser.add_argument("--resume", metavar="PATH", help="Path to a chat JSONL log to resume")
    args = parser.parse_args()

    try:
        config = load_config(args.config_name)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set in the environment", file=sys.stderr)
        sys.exit(1)

    resumed = None
    if args.resume:
        try:
            resumed = load_resume(Path(args.resume))
        except ResumeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    config.log_folder.mkdir(parents=True, exist_ok=True)
    cache_dir = config.log_folder / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if resumed:
        logger = SessionLogger.from_existing(resumed.jsonl_path)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = SessionLogger(config.log_folder, timestamp)

    try:
        store = VectorStore()
        store.build(config.rag_files, cache_dir)

        session = ChatSession(config=config, store=store, logger=logger, unsafe=args.unsafe)

        if resumed:
            session.history = resumed.history
            session.resumed_context = resumed.resumed_context or None
            turn_count = len(resumed.history) // 2
            print(f"Resumed {turn_count} turn(s) from {resumed.jsonl_path}")
            print(f"[last response]\n{resumed.last_assistant_message}")

        run_repl(session, logger)
    finally:
        logger.close()
