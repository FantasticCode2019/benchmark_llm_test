"""Dataclasses shared across the project. Stdlib-only on purpose."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAIConfig:
    """Per-model knobs for the openai-shape benchmark; all fields fall
    back to the global `openai_defaults` block in the config.
    """
    api_key: str = "EMPTY"
    endpoint: str = "chat"            # "chat" -> /v1/chat/completions
                                      # "completion" -> /v1/completions
    extra_headers: dict = field(default_factory=dict)
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: Optional[float] = None
    extra_body: dict = field(default_factory=dict)
    measure_ttft_approx: bool = True


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
        thinking model in its default mode this may include the entire
        reasoning phase if the server doesn't surface reasoning tokens
        as regular content (vLLM with reasoning_parser, OpenAI o1, ...).
      - `ttft_no_think_seconds` is an OPTIONAL second probe with
        thinking explicitly disabled (Ollama: `think:false`; vLLM/Qwen3:
        `chat_template_kwargs.enable_thinking:false`). 0.0 means "not
        measured" (model has no thinking, or `spec.thinking` was off).
    """
    prompt: str
    ok: bool = False
    error: Optional[str] = None
    response_chars: int = 0
    wall_seconds: float = 0.0
    ttft_seconds: float = 0.0
    ttft_no_think_seconds: float = 0.0
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
    app_name: str
    model: str
    api_type: str = "ollama"
    started_at: str = ""
    finished_at: str = ""
    install_decision: str = ""        # "fresh" / "reused" / "recovered" / ""
    install_ok: bool = False
    install_seconds: float = 0.0
    uninstall_skipped: bool = False
    uninstall_ok: bool = False
    uninstall_seconds: float = 0.0
    endpoint: str = ""
    error: Optional[str] = None
    questions: list = field(default_factory=list)  # list[QuestionResult]

    def avg(self, attr: str) -> float:
        ok = [getattr(q, attr) for q in self.questions if q.ok]
        return (sum(ok) / len(ok)) if ok else 0.0
