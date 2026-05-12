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

      ollama (/api/generate, stream=false)
        - ttft_seconds        : load + prompt_eval (PRECISE; from server)
        - eval_seconds        : decode duration (server)
        - tps                 : decode tokens / decode seconds (server)
        - total_server_seconds: total_duration (server)

      openai-compatible (/v1/chat/completions, stream=false)
        - ttft_seconds        : APPROX. round-trip of a separate
                                max_tokens=1 request; 0 if disabled
        - eval_seconds        : llama.cpp's `timings.predicted_ms` if
                                returned by the server, else 0
        - tps                 : server tps if `timings` available,
                                else client_tps (eval_count / wall)
        - total_server_seconds: equals wall_seconds
        - prompt_tokens / total_tokens / client_tps / server_tps_reported
          are populated from `usage` and `timings` blocks

    Thinking (DeepSeek-R1 / Qwen3 / GPT-OSS / o1-style):
      - `ttft_seconds` reflects the model's NATURAL behavior — for a
        thinking model this is the time to the FIRST ANSWER token
        (after the reasoning trace ends).
      - `thinking_ttft_seconds` is the time the model takes to emit its
        FIRST reasoning/thinking token (Ollama `message.thinking`,
        vLLM `delta.reasoning` / `delta.reasoning_content`). Only
        measured when `spec.thinking=true` in the config — we trust
        the user to know whether the model has a thinking phase. For
        non-thinking models (or when the streaming probe failed) it
        is left at 0.0 (rendered as `—` in the email).
      - `has_thinking`: ECHOED FROM `spec.thinking` in the config.
        Set on every prompt of the run so the JSON attachment still
        carries the per-row signal even though it's now uniform per
        model.
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
