"""Domain layer — pure data classes + enums shared across the codebase.

The split mirrors the read/write direction:

  * :mod:`llm_bench.domain.enums`    -- typed values used by everything
    below (api_type / install_decision).
  * :mod:`llm_bench.domain.results`  -- benchmark OUTPUTS (per-prompt
    + per-model). Serialized to the JSON report.
  * :mod:`llm_bench.domain.config`   -- benchmark INPUTS (parsed JSON
    config tree + the resolver that produces per-model
    :class:`ResolvedOptions`).

Modules outside this package should depend on ``llm_bench.domain.*``
ONLY — they must not reach back into ``llm_bench.models`` (kept solely
as a back-compat shim for external scripts).
"""
from __future__ import annotations

from llm_bench.domain.config import (
    AppConfig,
    EmailConfig,
    GlobalDefaults,
    ModelSpec,
    OpenAIConfig,
    ResolvedOptions,
)
from llm_bench.domain.enums import ApiType, InstallDecision
from llm_bench.domain.results import ModelResult, QuestionResult

__all__ = [
    "ApiType",
    "AppConfig",
    "EmailConfig",
    "GlobalDefaults",
    "InstallDecision",
    "ModelResult",
    "ModelSpec",
    "OpenAIConfig",
    "QuestionResult",
    "ResolvedOptions",
]
