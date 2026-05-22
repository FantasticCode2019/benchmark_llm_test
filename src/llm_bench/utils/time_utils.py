"""Tiny time helpers — Beijing time (UTC+8 / CST).

Centralizes the project's wall-clock access so every JSON timestamp,
file-name stamp and HTML report header is rendered in Beijing time
instead of UTC. We deliberately return a NAIVE ``datetime`` so existing
``strftime`` / ``isoformat`` call sites keep their byte layout; callers
that emit ISO 8601 should append :data:`BEIJING_ISO_SUFFIX` (``+08:00``)
in place of the old ``Z`` (UTC) suffix.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

BEIJING_TZ = timezone(timedelta(hours=8))
BEIJING_ISO_SUFFIX = "+08:00"


def beijing_now_naive() -> datetime:
    """Return current Beijing time (UTC+8) as a NAIVE ``datetime``.

    Equivalent to ``datetime.now(BEIJING_TZ).replace(tzinfo=None)``.
    The tzinfo is stripped so downstream code can append a fixed
    ``+08:00`` offset (see :data:`BEIJING_ISO_SUFFIX`) without producing
    a double suffix like ``+08:00+08:00``, mirroring how the previous
    UTC-based helper appended ``Z`` to a naive UTC value.
    """
    return datetime.now(BEIJING_TZ).replace(tzinfo=None)
