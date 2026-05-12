#!/usr/bin/env python3
"""Backward-compatibility shim for `python3 llm_bench.py -c config.json`.

Existing cron jobs that invoke the script directly (see the cron example in
readme.md) continue to work without `pip install`. The recommended entry
points after the src/ migration are:

    llm-bench …              (console script, registered by pyproject.toml)
    python -m llm_bench …    (equivalent module invocation)

Both resolve to `llm_bench.cli:main`.
"""
from __future__ import annotations

import os
import sys

# Prepend src/ so the in-tree package wins even without `pip install -e .`.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_bench.cli import main  # noqa: E402 — must follow sys.path tweak

if __name__ == "__main__":
    raise SystemExit(main())
