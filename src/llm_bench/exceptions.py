"""Project-wide exception hierarchy.

This module declares the **types** the rest of the codebase should raise
for known failure modes. Phase 1 intentionally only *introduces* the
hierarchy — call-sites that today raise bare ``RuntimeError`` are NOT
migrated yet so behavior is byte-for-byte identical. Later phases can
convert them one module at a time without breaking ``except RuntimeError``
catches in user scripts (every leaf class below inherits from
``BenchmarkError`` which itself inherits from ``RuntimeError``).

Hierarchy::

    BenchmarkError                        (root; subclass of RuntimeError)
    ├── ConfigError                       (config file missing / malformed)
    │   └── ConfigValidationError         (well-formed JSON, bad shape)
    ├── CliError                          (olares-cli non-zero exit)
    ├── EntranceError                     (entrance discovery / flip failed)
    │   └── EntranceFlipTimeout
    ├── ReadinessError                    (bundle / probe failures)
    │   ├── BundleConfigError             (/cfg unusable)
    │   └── BundleProgressError           (/progress reported `error`)
    └── BenchmarkRunError                 (per-prompt benchmark POST failed)

Note: there is no longer a ``ModelPullError`` — the script no longer
issues its own ``ollama /api/pull``. Every supported olares-market
``ollama*`` chart pulls in its launcher container, and readiness is
established by polling ``/api/tags``.

Note: there is no longer a ``ReadinessTimeout`` — the readiness pollers
run indefinitely (no outer deadline). Cap wall-clock time at the
scheduler level if needed.

OpenAI-compatible HTTP failures keep their own ``OpenAIHTTPError`` (in
``llm_bench.clients.openai_errors``) because callers introspect its
``.status`` / ``.body`` attributes — it is NOT folded into this tree.
"""
from __future__ import annotations


class BenchmarkError(RuntimeError):
    """Root of all benchmark-specific exceptions.

    Inherits from ``RuntimeError`` so any pre-existing
    ``except RuntimeError:`` block in caller code continues to catch us
    while migrations proceed.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(BenchmarkError):
    """Top-level config problem (file missing, JSON parse error, etc.)."""


class ConfigValidationError(ConfigError):
    """Config parsed OK but failed structural validation (missing
    required section, wrong type, empty `models[]`, ...).
    """


# ---------------------------------------------------------------------------
# olares-cli subprocess
# ---------------------------------------------------------------------------


class CliError(BenchmarkError):
    """`olares-cli` exited non-zero. Carries the rendered command and
    captured stderr/stdout snippet for the report.
    """

    def __init__(self, message: str, *,
                 cmd: list[str] | None = None,
                 returncode: int | None = None,
                 stderr: str = "",
                 stdout: str = ""):
        super().__init__(message)
        self.cmd = list(cmd) if cmd else []
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


# ---------------------------------------------------------------------------
# Entrance discovery / auth flip
# ---------------------------------------------------------------------------


class EntranceError(BenchmarkError):
    """No usable entrance, or auth flip failed."""


class EntranceFlipTimeout(EntranceError):
    """`apps auth-level set …` was accepted but the controller didn't
    propagate `public` to the ingress within the verify window.
    """


# ---------------------------------------------------------------------------
# Bundle readiness (/cfg, /progress, /health, /ping)
# ---------------------------------------------------------------------------


class ReadinessError(BenchmarkError):
    """Bundle never reached a usable terminal state."""


class BundleConfigError(ReadinessError):
    """`/cfg` body unusable: HTTP error, non-JSON, missing jobId, or no
    task matched the configured `model_name`.
    """


class BundleProgressError(ReadinessError):
    """`/progress` reported the terminal `error` / `unavailable` status."""


# ---------------------------------------------------------------------------
# Per-model operations
# ---------------------------------------------------------------------------


class BenchmarkRunError(BenchmarkError):
    """The actual /api/generate or /v1/chat/completions request failed in
    a way that wasn't already covered by ``OpenAIHTTPError``.
    """


__all__ = [
    "BenchmarkError",
    "BenchmarkRunError",
    "BundleConfigError",
    "BundleProgressError",
    "CliError",
    "ConfigError",
    "ConfigValidationError",
    "EntranceError",
    "EntranceFlipTimeout",
    "ReadinessError",
]
