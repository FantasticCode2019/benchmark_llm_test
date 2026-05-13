"""Centralized constants — single source of truth for magic numbers and
text fragments that previously lived as private `_FOO` definitions inside
individual modules.

Rules of the road:

* Values here are FACTS about the protocols / wire formats we talk to, or
  hard-coded operational knobs that have been stable over the project's
  history. They are NOT user configuration — anything a config file can
  override belongs in ``config.example.json`` / ``GlobalDefaults``, not
  here.
* Names are UPPER_SNAKE_CASE per PEP 8.
* Group with section comments; sort within group by domain (HTTP →
  Ollama → vLLM → SMTP → CLI) so newcomers can locate things visually.
* Re-exports from modules that historically owned a constant (e.g.
  ``readiness._FAILURE_RETRY_SECONDS``) keep the old private name as a
  thin alias so we don't ripple a rename through the whole codebase.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Logger name — every module does `logging.getLogger("llm_bench")` so a
# single attach-handler at startup captures the whole project's logs.
# Keeping this as a constant lets future code do
# `logging.getLogger(LOG_NAMESPACE)` instead of repeating the literal.
# ---------------------------------------------------------------------------

LOG_NAMESPACE: Final[str] = "llm_bench"

# ---------------------------------------------------------------------------
# Generic HTTP / byte-math
# ---------------------------------------------------------------------------

#: 1 GiB in bytes. Used by ``utils.http._bytes_to_gb`` and any future code
#: that needs the same byte → GiB rounding the bash one-liner used.
GIB_BYTES: Final[int] = 1024 ** 3

#: Default timeout (seconds) for one-shot readiness HTTP probes
#: (``http_get_status``, ``http_get_json``). Long enough to absorb the
#: TLS handshake on a fresh ingress, short enough that a hung peer is
#: caught quickly — readiness pollers re-issue requests until success
#: rather than relying on a wall-clock deadline.
HTTP_DEFAULT_TIMEOUT_SECONDS: Final[int] = 10

# ---------------------------------------------------------------------------
# Readiness pollers (Ollama + vLLM bundle protocols)
# ---------------------------------------------------------------------------

#: Backoff between RETRY-mode probes (transport error, non-2xx, non-JSON,
#: non-SSE). Decoupled from the happy-path interval so a slow / flapping
#: chart launcher doesn't slow the steady-state poll cadence. The
#: happy-path interval is now config-driven via
#: ``readiness_probe_interval_seconds`` (see GlobalDefaults).
READINESS_FAILURE_RETRY_SECONDS: Final[int] = 5

# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

#: Default binary name when neither ``--cli-path`` nor ``cfg.cli_path`` is
#: set. Resolved against ``$PATH`` at execve time.
DEFAULT_OLARES_CLI: Final[str] = "olares-cli"

#: Default timeout (seconds) for ``cli_json`` invocations — JSON queries
#: should be near-instant; longer ones go through ``cli.run`` with an
#: explicit per-call timeout.
CLI_JSON_DEFAULT_TIMEOUT_SECONDS: Final[int] = 60

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

#: Default email subject template (substitution tokens documented in
#: ``data.mailer._render_subject``). Falls through when the config omits
#: ``email.subject``.
DEFAULT_EMAIL_SUBJECT: Final[str] = "Olares LLM benchmark {date}"
