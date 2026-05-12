"""Tiny time helpers.

Centralizes the project's UTC-clock access so we can replace the
deprecated ``datetime.utcnow()`` (3.12+ emits a ``DeprecationWarning``)
without forking the output format of every JSON timestamp / log line
that previously relied on its NAIVE-UTC return value.
"""
from __future__ import annotations

from datetime import UTC, datetime


def utc_now_naive() -> datetime:
    """Drop-in replacement for ``datetime.utcnow()``.

    Returns a NAIVE UTC ``datetime`` (no ``tzinfo``), so existing string
    rendering downstream — e.g. ``.isoformat() + "Z"`` in
    ``ModelResult.started_at`` and ``finished_at`` — keeps emitting
    byte-identical output across Python 3.11 / 3.12 / 3.13. Using
    ``datetime.now(timezone.utc).isoformat()`` directly would attach a
    ``+00:00`` suffix and break consumers of the JSON report.
    """
    return datetime.now(UTC).replace(tzinfo=None)
