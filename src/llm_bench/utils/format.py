"""Tiny formatting helpers — kept dependency-free for cross-cutting reuse."""
from __future__ import annotations


def human_bytes(n: int | None) -> str:
    """`9876543` -> `9.4 MiB`."""
    if n is None:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024
    return f"{f:.1f} {units[-1]}"


def fmt_duration(seconds: float) -> str:
    """`123.4` -> `2m 03s`, `5.4` -> `5.4s`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(round(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
