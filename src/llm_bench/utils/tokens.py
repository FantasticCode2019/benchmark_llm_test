"""Token-count + ms→s helpers for the OpenAI-shape benchmark."""
from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def rough_token_count(text: str) -> int:
    """Fallback token estimate when the server didn't include `usage`.
    Counts CJK chars individually and groups latin/digit/symbol runs as one.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin_only = _CJK_RE.sub(" ", text)
    return cjk + len(_LATIN_RE.findall(latin_only))


def ms_to_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
