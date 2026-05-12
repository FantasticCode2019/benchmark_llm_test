"""Config file IO + log setup.

Thin glue around :class:`AppConfig.from_dict` — this module owns the
*filesystem* + *logging-setup* concerns, the structured validation lives
in :mod:`llm_bench.domain.config`.

Errors raised here (file not found, JSON parse error, structural
mismatch) all inherit from :class:`ConfigError` so the CLI can render a
friendly error without leaking a stack trace; the CLI translates them
to a non-zero exit code.
"""
from __future__ import annotations

import json
import logging
import os
import sys

from llm_bench.domain import AppConfig
from llm_bench.exceptions import ConfigError, ConfigValidationError


def setup_logging(log_path: str | None) -> None:
    """Configure the root logger with stderr + optional file output.

    Idempotent in practice: ``basicConfig`` no-ops once a handler is
    attached, so successive calls (e.g. in tests) don't stack
    handlers. Format keeps the timestamp + level prefix so cron-piped
    output remains greppable.
    """
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers)


def load_config(path: str) -> AppConfig:
    """Read + parse + validate the JSON config at ``path``.

    Returns a fully-typed :class:`AppConfig`. Raises:

      * :class:`ConfigError` when the file cannot be opened or the JSON
        is malformed.
      * :class:`ConfigValidationError` when the JSON parses but doesn't
        match the schema (missing required section, wrong types, empty
        ``models[]`` / ``questions[]``).

    Both translate to a clean CLI error in :func:`llm_bench.cli.main`.
    """
    try:
        with open(path, encoding="utf-8") as fp:
            raw = json.load(fp)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"config file {path}: invalid JSON: {exc.msg} "
            f"(line {exc.lineno}, col {exc.colno})") from exc
    except OSError as exc:
        raise ConfigError(
            f"config file {path}: cannot read ({exc.__class__.__name__}: "
            f"{exc})") from exc

    try:
        return AppConfig.from_dict(raw)
    except ConfigValidationError:
        # Already specific enough — bubble unchanged so the CLI's
        # except-ConfigError handler renders the exact message.
        raise
