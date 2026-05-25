"""Per-prompt and per-model result dataclasses.

These are the **outputs** the benchmark produces. They get serialized to
the JSON report via ``dataclasses.asdict`` and rendered into the HTML
email table. The wire format (field names and value types) is part of
the project's public contract — DO NOT rename or retype existing
fields without coordinating with consumers.

``api_type`` and ``install_decision`` are ``StrEnum`` values; because
``StrEnum`` inherits from ``str``, ``json.dumps`` emits them as the same
plain strings the pre-enum code used (``"ollama"``, ``"fresh"`` etc.),
so the JSON output is byte-identical with v0.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_bench.domain.enums import ApiType, InstallDecision


@dataclass
class QuestionResult:
    """Per-prompt timing record. Field semantics differ between backends:

    Canonical metric definitions (consistent across BOTH backends so
    rows can be read without knowing which API served them):
      - wall_seconds : client wall clock from request → final chunk.
                       For Ollama this is `urlopen` to the `done:true`
                       chunk; for OpenAI-compatible this is `urlopen`
                       to the last streamed chunk (stream=true).
      - eval_count   : number of tokens in the response.
                       Ollama: server-reported `eval_count`.
                       OpenAI: `usage.completion_tokens` from the
                       final usage-only chunk (vLLM / llama.cpp emit
                       this when `stream_options.include_usage=true`),
                       falling back to `rough_token_count(answer)`
                       with `tokens_estimated=True`.
      - eval_seconds : decode duration.
                       Ollama: server-reported `eval_duration`.
                       OpenAI: client-observed
                       `last_content_delta_at - first_content_delta_at`
                       (falls back to `wall - ttft` for single-chunk
                       responses).
      - tps          : `eval_count / eval_seconds` — decode-only
                       throughput. Excludes prefill and streaming I/O.
      - client_tps   : `eval_count / wall_seconds` — wall-clock
                       throughput. Kept for diagnostics so operators
                       can see the prefill + network overhead delta.

      ollama (/api/generate, stream=true — see core/benchmark/ollama.py)
        - ttft_seconds        : client wall-clock to the FIRST
                                non-empty visible response chunk
                                (whitespace-only chunks filtered).
        - thinking_ttft_seconds: client wall-clock to the FIRST
                                non-empty thinking chunk; populated
                                only when the runtime probe said the
                                model supports thinking AND the model
                                actually emitted a thinking chunk.
        - load_seconds /
          prompt_eval_seconds : server-side aggregate timings from the
                                final done:true chunk. Diagnostic.
        - total_server_seconds: server's `total_duration`.

      openai-compatible (/v1/chat/completions, stream=true)
        - ttft_seconds        : client wall-clock to the FIRST non-empty
                                `delta.content` chunk. Real TTFT.
        - thinking_ttft_seconds: client wall-clock to the FIRST
                                non-empty `delta.reasoning` /
                                `delta.reasoning_content` chunk.
                                Populated only when the per-model
                                `spec.thinking=true` flag is set AND the
                                model actually emitted a reasoning delta.
        - prompt_eval_seconds : llama.cpp's `timings.prompt_ms` when
                                the server emits a `timings` block on
                                the stream; 0 for vLLM. Diagnostic.
        - total_server_seconds: equals wall_seconds.
        - prompt_tokens / total_tokens / server_tps_reported are
          populated from `usage` (final chunk) and `timings` (final
          chunk). `server_tps_reported` is the server's self-reported
          decode tokens/s — diagnostic only; the headline `tps` always
          uses the client-observed eval_seconds denominator above.

    Thinking (DeepSeek-R1 / Qwen3 / GPT-OSS / o1-style):
      - For ollama, both `ttft_seconds` and `thinking_ttft_seconds`
        share the same `time.perf_counter()` epoch (right before
        urlopen) so they're directly comparable. For a model that
        thinks-then-answers the expected order is
        `thinking_ttft_seconds < ttft_seconds`.
      - For vLLM, `thinking_ttft_seconds` is the wall-clock to the
        first reasoning chunk (`delta.reasoning` /
        `delta.reasoning_content`); `ttft_seconds` is the existing
        max_tokens=1 round-trip approximation. They're NOT in the same
        coordinate system — kept that way to preserve the openai
        backend's existing wire format.
      - `has_thinking`: for ollama, ECHOED FROM the runtime
        `ollama_supports_thinking` probe; for openai/vLLM, ECHOED FROM
        `spec.thinking` config. Set on every prompt of the run so the
        JSON attachment still carries the per-row signal even though
        it's uniform per model.
    """
    prompt: str
    ok: bool = False
    error: str | None = None
    response_chars: int = 0
    wall_seconds: float = 0.0
    ttft_seconds: float = 0.0
    thinking_ttft_seconds: float = 0.0
    has_thinking: bool = False
    load_seconds: float = 0.0
    prompt_eval_seconds: float = 0.0
    eval_count: int = 0
    eval_seconds: float = 0.0
    tps: float = 0.0
    total_server_seconds: float = 0.0
    # OpenAI-only extras (zero for ollama rows)
    prompt_tokens: int = 0
    total_tokens: int = 0
    client_tps: float = 0.0
    server_tps_reported: float = 0.0
    tokens_estimated: bool = False
    note: str = ""


@dataclass
class ModelResult:
    """Aggregated record for one model run. Holds the install / uninstall
    timing plus the per-prompt list. Serialized to JSON via ``asdict``;
    enum fields render as their string values (see module docstring).
    """
    app_name: str
    model: str
    api_type: ApiType = ApiType.OLLAMA
    started_at: str = ""
    finished_at: str = ""
    install_decision: InstallDecision = InstallDecision.UNKNOWN
    install_ok: bool = False
    install_seconds: float = 0.0
    uninstall_skipped: bool = False
    uninstall_ok: bool = False
    uninstall_seconds: float = 0.0
    endpoint: str = ""
    error: str | None = None
    # Set ONLY when the model run failed (install / readiness / uninstall
    # error or no prompt succeeded) and `save_pod_logs_on_failure=true`
    # successfully tar.gz'd the pod log directories. Path is local to
    # the host that ran the benchmark.
    pod_logs_archive: str | None = None
    # Ollama-only runtime metadata. Populated by the orchestrator AFTER
    # readiness succeeds:
    #   * `ollama_supports_thinking` — result of probing /api/show's
    #     `capabilities` array (independent of the configured
    #     `spec.thinking` flag). None means "not probed" (non-Ollama
    #     backend, or the probe itself failed).
    #   * `ollama_descriptor` — full /api/ps + /api/tags + /api/show
    #     descriptor dict as returned by `ollama_describe_model`.
    #     None for non-Ollama runs; empty descriptor when the daemon
    #     was reachable but had no models loaded yet.
    # Both feed the per-run Excel attachment; they are intentionally
    # absent from the HTML email body (kept lean).
    ollama_supports_thinking: bool | None = None
    ollama_descriptor: dict | None = None
    questions: list[QuestionResult] = field(default_factory=list)

    def avg(self, attr: str) -> float:
        """Mean of ``attr`` across all successful prompts; 0.0 when
        nothing succeeded. Used by the HTML renderer to fill the
        summary columns.
        """
        ok_values = [getattr(q, attr) for q in self.questions if q.ok]
        return (sum(ok_values) / len(ok_values)) if ok_values else 0.0

    def has_thinking_label(self) -> str:
        """``"Yes"`` / ``"No"`` for the email column, derived from the
        per-prompt ``has_thinking`` flag (which is just the echo of
        ``spec.thinking`` from the config). We pick the first ok row's
        flag — they all agree because the value comes straight from
        the same per-model config knob.
        """
        for q in self.questions:
            if q.ok:
                return "Yes" if q.has_thinking else "No"
        return "No"


__all__ = ["ModelResult", "QuestionResult"]
