"""Subprocess wrappers around the `olares-cli` binary."""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

log = logging.getLogger("llm_bench")

# Set by set_cli_path() at startup; defaults to whatever the OS resolves
# from PATH. `--cli-path` / config `cli_path` can override it.
_CLI_PATH = "olares-cli"


def set_cli_path(path: str) -> None:
    global _CLI_PATH
    if path:
        _CLI_PATH = path
        log.info("using olares-cli at %s", _CLI_PATH)


def cli() -> str:
    return _CLI_PATH


def run(cmd: list, *, timeout: int, capture: bool = False,
        check: bool = True) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, timeout=timeout, capture_output=capture,
                          text=True, check=check)


def cli_json(args: list, *, timeout: int = 60) -> Any:
    """Run `olares-cli ... -o json` and decode stdout."""
    proc = run([cli(), *args, "-o", "json"], timeout=timeout, capture=True)
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def apply_user_envs(user_envs: dict) -> None:
    """Set USER-level env vars once before the install loop. Per-USER (not
    per-app) so we write all keys in one `env user set --var KEY=VAL` call.
    """
    if not user_envs:
        return
    cmd = [cli(), "settings", "advanced", "env", "user", "set"]
    for key, value in user_envs.items():
        cmd.extend(["--var", f"{key}={value}"])
    log.info("applying %d user-level env var(s): %s",
             len(user_envs), ", ".join(user_envs.keys()))
    run(cmd, timeout=120)
