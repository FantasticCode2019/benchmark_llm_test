"""Subprocess wrappers around the `olares-cli` binary."""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

from llm_bench.constants import (
    CLI_JSON_DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_OLARES_CLI,
    LOG_NAMESPACE,
)

log = logging.getLogger(LOG_NAMESPACE)

# Set by set_cli_path() at startup; defaults to PATH-resolved `olares-cli`.
# `--cli-path` / config `cli_path` override at boot time.
_CLI_PATH = DEFAULT_OLARES_CLI


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


def cli_json(args: list, *, timeout: int = CLI_JSON_DEFAULT_TIMEOUT_SECONDS) -> Any:
    """Run `olares-cli ... -o json` and decode stdout."""
    proc = run([cli(), *args, "-o", "json"], timeout=timeout, capture=True)
    out = proc.stdout.strip()
    return json.loads(out) if out else None
