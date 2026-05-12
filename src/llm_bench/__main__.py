"""Enables `python -m llm_bench …` as an alternative to the `llm-bench`
console script. Both routes call the same `main()` so behavior is identical.
"""
from __future__ import annotations

from llm_bench.cli import main

if __name__ == "__main__":  # pragma: no cover - manual invocation surface
    raise SystemExit(main())
