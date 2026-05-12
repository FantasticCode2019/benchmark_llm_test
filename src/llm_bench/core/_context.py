"""Private mutable context used by the orchestrator step functions.

Lives next to :mod:`llm_bench.core.orchestrator` (sibling module, leading
underscore) because it has no value outside that single use site. We
keep it OUT of :mod:`llm_bench.domain` so the domain layer stays
strictly pure / immutable / read-only data.

Each ``_step_*`` function in :mod:`llm_bench.core.orchestrator` takes a
:class:`BenchmarkContext` and mutates the relevant fields in place; the
top-level :func:`bench_model` reads ``ctx.result`` at the end. This
avoids threading 8+ positional arguments through the step boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_bench.domain import (
    AppConfig,
    ModelResult,
    ModelSpec,
    OpenAIConfig,
    ResolvedOptions,
)


@dataclass
class BenchmarkContext:
    """One model's worth of state shared between orchestrator steps.

    Layered roughly as:
      * **immutable inputs**  -- ``spec`` / ``cfg`` / ``opts`` / ``openai``
        (resolved once at construction time)
      * **side-effect output** -- ``result`` (mutated in place; serialized
        at the end of the run by ``write_reports``)
      * **runtime state**     -- ``model_name`` / ``already_existed`` /
        entrance fields, written by early steps and read by later ones

    The class is intentionally *not* frozen: steps need to write into
    it. It is also intentionally NOT exposed in :mod:`llm_bench.domain`
    because it carries mid-flight state that has no place in a serialised
    report.
    """
    spec: ModelSpec
    cfg: AppConfig
    opts: ResolvedOptions
    openai: OpenAIConfig
    result: ModelResult
    # Runtime state populated as steps execute.
    model_name: str = ""           # mutates after wait_until_api_ready discovery
    already_existed: bool = False  # set by ensure_installed
    entrance: str = ""             # entrance name (for logging)
    entrance_url: str = ""         # post-discovery base URL
    auth_level: str = ""           # raw authLevel string from olares-cli
    archived_logs: bool = False    # set when archive step already ran in
                                   # the try-block so the finally-block
                                   # doesn't double-archive on uninstall
                                   # failure
    prompts: list[str] = field(default_factory=list)

    @property
    def app(self) -> str:
        """Shortcut for ``spec.app_name`` — the orchestrator threads it
        through 20+ log lines, so the abbreviation is worth its keep.
        """
        return self.spec.app_name

    def set_error(self, message: str) -> None:
        """Record an error on the result if one isn't already recorded.

        The orchestrator chains multiple try/finally blocks (benchmark
        body, uninstall in finally) — both can fail, and the FIRST
        failure is the one we want to surface in the report because
        every subsequent failure usually cascades from it.
        """
        if not self.result.error:
            self.result.error = message

    @property
    def any_prompt_ok(self) -> bool:
        """True iff at least one prompt produced an ``ok=True``
        :class:`QuestionResult`. Drives the "should we archive pod
        logs?" decision in the finally-block.
        """
        return any(q.ok for q in self.result.questions)


__all__ = ["BenchmarkContext"]
