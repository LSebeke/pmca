from __future__ import annotations

from pathlib import Path

import git as gitlib

from pmca.config import Config
from pmca.tools.fs import _is_allowed


class SafeGitOps:
    def __init__(self, repo_path: Path, read_allowed_dirs: list[Path]) -> None:
        self._repo = gitlib.Repo(repo_path)
        self._root = Path(self._repo.working_dir)
        self._read_allowed_dirs = read_allowed_dirs

    def _validate_path(self, path: str) -> Path | str:
        resolved = (self._root / path).resolve()
        if not _is_allowed(resolved, self._read_allowed_dirs):
            dirs_str = ", ".join(str(d) for d in self._read_allowed_dirs)
            return f"Error: path {resolved} is outside allowed directories: {dirs_str}"
        return resolved

    def _resolve_ref(self, ref: str):
        try:
            return self._repo.commit(ref)
        except (gitlib.BadName, gitlib.BadObject, ValueError):
            return None

    def status(self) -> dict:
        return {
            "dirty": self._repo.is_dirty(),
            "untracked": self._repo.untracked_files,
            "staged": [d.a_path for d in self._repo.index.diff("HEAD")],
            "unstaged": [d.a_path for d in self._repo.index.diff(None)],
        }

    def log(self, max_count: int = 20) -> list[dict]:
        commits = list(self._repo.iter_commits(max_count=max_count))
        return [
            {
                "sha": c.hexsha[:8],
                "message": c.message.strip(),
                "author": str(c.author),
                "date": c.committed_datetime.isoformat(),
            }
            for c in commits
        ]

    def diff(self, ref: str = "HEAD", path: str | None = None, staged: bool = False) -> str:
        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        if path is not None:
            result = self._validate_path(path)
            if isinstance(result, str):
                return result
            path_posix = Path(path).as_posix()
        else:
            path_posix = None

        try:
            if staged:
                diff_args = [commit.hexsha]
                if path_posix:
                    diff_args += ["--", path_posix]
                return self._repo.git.diff("--cached", *diff_args)
            else:
                diff_args = [commit.hexsha]
                if path_posix:
                    diff_args += ["--", path_posix]
                return self._repo.git.diff(*diff_args)
        except gitlib.GitCommandError as e:
            return f"Error: {e}"

    def blame(self, ref: str = "HEAD", path: str = "") -> str:
        result = self._validate_path(path)
        if isinstance(result, str):
            return result

        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        try:
            blame = self._repo.blame(commit.hexsha, path)
        except gitlib.GitCommandError as e:
            return f"Error: {e}"

        lines = []
        for entry_commit, entry_lines in blame:
            sha = entry_commit.hexsha[:8]
            author = str(entry_commit.author)
            for line in entry_lines:
                text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                lines.append(f"{sha} {author}: {text}")
        return "".join(lines)

    def show_file(self, ref: str, path: str) -> str:
        result = self._validate_path(path)
        if isinstance(result, str):
            return result

        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        try:
            blob = commit.tree / Path(path).as_posix()
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except (KeyError, gitlib.GitCommandError) as e:
            return f"Error: {e}"

    def branches(self) -> list[str]:
        return [b.name for b in self._repo.branches]

    def current_branch(self) -> str:
        try:
            return self._repo.active_branch.name
        except TypeError:
            return "(detached HEAD)"


def execute_git_status(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    status = ops.status()
    lines = [f"dirty: {status['dirty']}"]
    if status["staged"]:
        lines.append("staged: " + ", ".join(status["staged"]))
    if status["unstaged"]:
        lines.append("unstaged: " + ", ".join(status["unstaged"]))
    if status["untracked"]:
        lines.append("untracked: " + ", ".join(status["untracked"]))
    return "\n".join(lines)


def execute_git_log(arguments: dict, config: Config) -> str:
    max_count = int(arguments.get("max_count", 20))
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    commits = ops.log(max_count=max_count)
    lines = [f"{c['sha']} {c['date'][:10]} {c['author']}: {c['message']}" for c in commits]
    return "\n".join(lines) if lines else "No commits found."


def execute_git_diff(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.diff(
        ref=arguments.get("ref", "HEAD"),
        path=arguments.get("path"),
        staged=bool(arguments.get("staged", False)),
    )


def execute_git_blame(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.blame(ref=arguments.get("ref", "HEAD"), path=arguments["path"])


def execute_git_show_file(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.show_file(ref=arguments["ref"], path=arguments["path"])


def execute_git_branches(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return "\n".join(ops.branches())


def execute_git_current_branch(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.current_branch()
