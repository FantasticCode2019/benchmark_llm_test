"""Backward-compatibility shim for the historical import path.

The dataclasses now live in ``llm_bench.domain``. This module re-exports
them so existing user scripts (and the README's embed example from v0.1)
continue to work::

    from llm_bench.models import ModelResult, QuestionResult, OpenAIConfig

New code should prefer the canonical paths::

    from llm_bench.domain import ModelResult, QuestionResult, OpenAIConfig

This shim has no logic of its own; the source of truth is the ``domain``
package.
"""
from __future__ import annotations

from llm_bench.domain import ModelResult, OpenAIConfig, QuestionResult

__all__ = ["ModelResult", "OpenAIConfig", "QuestionResult"]
