from __future__ import annotations

import subprocess

from pmca.config import Config


def execute_run_tests(arguments: dict, config: Config) -> tuple[bool, str]:
    test_dir = config.test_dir
    use_pixi = (test_dir / "pixi.toml").exists()
    cmd = ["pixi", "run", "pytest"] if use_pixi else ["pytest"]

    raw_filter = arguments.get("filter", "").strip()
    if raw_filter:
        cmd.extend(raw_filter.split())

    print(f"[run_tests] {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=test_dir,
            capture_output=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.test_timeout,
        )
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, f"Error: run_tests timed out after {config.test_timeout} seconds"
    except OSError as e:
        return False, f"Error: {e}"
