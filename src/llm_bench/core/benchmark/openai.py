"""OpenAI-compatible benchmark (vLLM / llama.cpp / oai-compat).

ONE streaming /v1/chat/completions (or /v1/completions) request per
prompt — same shape as the Ollama benchmark, so all per-prompt metrics
live in the same coordinate system:

  * ``wall_seconds`` — client-observed wall clock from
    ``urlopen`` to the final chunk. End-to-end "how long the caller
    waited for the full response".

  * ``ttft_seconds`` — client-observed wall clock to the first
    non-empty ``delta.content`` chunk. Real TTFT, not an
    approximation — captured for free as part of the same stream.

  * ``thinking_ttft_seconds`` — client-observed wall clock to the
    first non-empty ``delta.reasoning`` /
    ``delta.reasoning_content`` chunk. Populated only when the
    per-model ``spec.thinking=true`` flag is set AND the model
    actually emits a reasoning delta. vLLM / Qwen3 / DeepSeek-R1
    chat templates honor ``extra_body={"chat_template_kwargs":
    {"thinking": True}}`` to opt the render into emitting these
    deltas.

  * ``eval_count`` — preferred from the final usage-only chunk
    (``chunk.usage.completion_tokens``) emitted by servers that
    honor ``stream_options={"include_usage": True}`` (vLLM,
    llama.cpp). Falls back to ``rough_token_count(answer)`` with
    ``tokens_estimated=True`` when the server skips the usage
    chunk.

  * ``eval_seconds`` — **decode duration** observed client-side as
    ``last_content_delta_at - first_content_delta_at``. This is the
    time the model spent emitting answer tokens, excluding prefill
    and post-DONE network latency. For single-token responses
    where this collapses to ~0, falls back to ``wall - ttft``.

  * ``tps`` — **decode-only** generated-tokens-per-second
    (``eval_count / eval_seconds``). Pure "how fast the model emits
    tokens once it starts" reading, directly comparable with the
    Ollama backend. Wall-clock throughput
    (``eval_count / wall_seconds``) is preserved separately as
    ``client_tps`` for diagnostics.

The ``measure_ttft_approx`` config knob is preserved for backward
compatibility but is now a NO-OP — streaming gives us real TTFT
unconditionally, so there is no extra round-trip to opt out of.
"""
from __future__ import annotations

import logging
import time

from openai import OpenAI

from llm_bench.clients.openai_errors import auth_hint
from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import AppConfig, ModelSpec, OpenAIConfig, QuestionResult
from llm_bench.utils.tokens import ms_to_seconds, rough_token_count, to_float

log = logging.getLogger(LOG_NAMESPACE)


def openai_config_from(spec: ModelSpec, cfg: AppConfig) -> OpenAIConfig:
    """Merge per-model openai overrides on top of global ``openai_defaults``.

    Layered resolution: ``spec.openai_overrides`` > ``cfg.openai_defaults``
    > hard-coded :class:`OpenAIConfig` defaults. Each layer is a raw
    dict so callers can sprinkle ad-hoc keys via ``extra_body`` without
    having to update a typed schema.
    """
    g = cfg.openai_defaults
    s = spec.openai_overrides

    def pick(key: str, default):
        if key in s:
            return s[key]
        if key in g:
            return g[key]
        return default

    return OpenAIConfig(
        api_key=str(pick("api_key", "EMPTY")),
        endpoint=str(pick("endpoint", "chat")).lower(),
        extra_headers=dict(pick("extra_headers", {}) or {}),
        max_tokens=int(pick("max_tokens", 256)),
        temperature=float(pick("temperature", 0.0)),
        top_p=pick("top_p", None),
        extra_body=dict(pick("extra_body", {}) or {}),
        measure_ttft_approx=bool(pick("measure_ttft_approx", True)),
    )


def _openai_base_url(url: str) -> str:
    """`OpenAI(base_url=...)` expects the path to end at /v1; the rest of
    this module uses the raw entrance URL (which may or may not end at
    /v1). Normalize to "<entrance>/v1".
    """
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _make_openai_client(url: str, conf: OpenAIConfig,
                        *, timeout: int) -> OpenAI:
    """Construct a per-request OpenAI client. We disable retries because
    a benchmark is supposed to surface the FIRST observed failure, not
    paper over it with silent retry latency that would distort the
    timing readings.
    """
    api_key = conf.api_key
    if not api_key or api_key.strip().upper() == "EMPTY":
        # The SDK refuses to be constructed without a string here, but
        # vLLM / llama-server in their default no-auth mode treat any
        # placeholder as fine. "EMPTY" is the canonical placeholder the
        # vLLM docs recommend.
        api_key = "EMPTY"
    return OpenAI(
        base_url=_openai_base_url(url),
        api_key=api_key,
        timeout=float(timeout),
        max_retries=0,
        default_headers=(dict(conf.extra_headers) if conf.extra_headers
                         else None),
    )


def _merge_thinking_extra_body(base: dict | None) -> dict:
    """Inject `chat_template_kwargs.thinking=True` into `conf.extra_body`
    one level deep so user-supplied siblings (e.g. `enable_reasoning`)
    survive. vLLM / Qwen3 / DeepSeek-R1 templates honor this key to opt
    the render into emitting `<think>` blocks (the reasoning_parser
    then surfaces them as `delta.reasoning`).
    """
    out = dict(base or {})
    user_ck = out.get("chat_template_kwargs") or {}
    if not isinstance(user_ck, dict):
        user_ck = {}
    out["chat_template_kwargs"] = {**user_ck, "thinking": True}
    return out


def _build_stream_kwargs(model: str, prompt: str, conf: OpenAIConfig,
                        *, thinking: bool) -> tuple[dict, bool]:
    """Assemble kwargs for the streaming SDK call. Returns ``(kwargs,
    is_chat)`` — ``is_chat=False`` means use the legacy /v1/completions
    surface (``client.completions.create``) which carries ``text`` on
    each choice instead of ``delta.content``.

    ``stream_options={"include_usage": True}`` asks the server to emit
    a final usage-only chunk. vLLM, llama.cpp and most oai-compat
    backends honor this; servers that don't simply omit the usage
    chunk and we fall back to a char-based token estimate.
    """
    extra_body = (_merge_thinking_extra_body(conf.extra_body)
                  if thinking else dict(conf.extra_body or {}))

    kwargs: dict = {
        "model": model,
        "stream": True,
        "max_tokens": conf.max_tokens,
        "temperature": conf.temperature,
        "extra_body": extra_body,
        "stream_options": {"include_usage": True},
    }
    if conf.top_p is not None:
        kwargs["top_p"] = conf.top_p

    if conf.endpoint == "completion":
        kwargs["prompt"] = prompt
        return kwargs, False

    kwargs["messages"] = [{"role": "user", "content": prompt}]
    return kwargs, True


def _extract_chat_deltas(chunk) -> tuple[str | None, str | None]:
    """Pull (reasoning, content) text out of one chat-completions chunk.
    Either side can be None when the chunk doesn't carry that delta.
    """
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None, None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None, None
    reasoning = (getattr(delta, "reasoning", None)
                 or getattr(delta, "reasoning_content", None))
    content = getattr(delta, "content", None)
    return reasoning, content


def _extract_completion_text(chunk) -> str | None:
    """Pull the streamed `text` payload out of a legacy /v1/completions
    chunk. Returns None when the chunk is empty.
    """
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "text", None)


def _extract_server_timings(chunk) -> dict:
    """llama.cpp emits its `timings` block inline on each streamed chunk
    (most reliably on the final one). vLLM does not. Read it whenever
    present so we can populate `prompt_eval_seconds` /
    `server_tps_reported` diagnostics for free.
    """
    timings = (getattr(chunk, "timings", None)
               or getattr(chunk, "timing", None))
    if not isinstance(timings, dict):
        return {}
    return {
        "server_prompt_eval_seconds": ms_to_seconds(
            timings.get("prompt_ms") or timings.get("prompt_eval_ms")),
        "server_generation_seconds": ms_to_seconds(
            timings.get("predicted_ms") or timings.get("generation_ms")
            or timings.get("eval_ms")),
        "server_tps": to_float(
            timings.get("predicted_per_second")
            or timings.get("tokens_per_second")),
    }


def _run_openai_stream(url: str, model: str, prompt: str,
                       conf: OpenAIConfig,
                       *, timeout: int, thinking: bool) -> dict:
    """Open ONE streaming chat/completions request, drain every chunk,
    and return all the per-prompt metrics the benchmark needs.

    Returned dict carries:
      - wall, ttft, thinking_ttft (None when not observed)
      - eval_dur (client-side decode duration proxy)
      - eval_count, prompt_tokens, total_tokens, tokens_estimated
      - answer (joined `delta.content` text)
      - server_timings (dict; empty when no `timings` block was emitted)
    """
    client = _make_openai_client(url, conf, timeout=timeout)
    kwargs, is_chat = _build_stream_kwargs(model, prompt, conf,
                                          thinking=thinking)

    started = time.perf_counter()
    first_thinking_at: float | None = None
    first_content_at: float | None = None
    last_content_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    server_timings: dict = {}

    api = (client.chat.completions.create if is_chat
           else client.completions.create)
    stream = api(**kwargs)
    try:
        for chunk in stream:
            now = time.perf_counter() - started

            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

            timings = _extract_server_timings(chunk)
            if timings:
                server_timings = timings

            if is_chat:
                reasoning, content = _extract_chat_deltas(chunk)
            else:
                reasoning = None
                content = _extract_completion_text(chunk)

            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                if first_thinking_at is None:
                    first_thinking_at = now
                reasoning_parts.append(reasoning)
            if content and isinstance(content, str) and content.strip():
                if first_content_at is None:
                    first_content_at = now
                last_content_at = now
                content_parts.append(content)
            elif content:
                # Non-empty but whitespace-only — accumulate text but do
                # NOT advance the TTFT clock (matches the Ollama behavior
                # of filtering whitespace-only leading chunks).
                content_parts.append(content)
    finally:
        try:
            stream.close()
        except Exception:
            pass

    wall = time.perf_counter() - started
    answer = "".join(content_parts)

    if (first_content_at is not None and last_content_at is not None
            and last_content_at > first_content_at):
        eval_dur = last_content_at - first_content_at
    elif first_content_at is not None and wall > first_content_at:
        # Single content chunk (or all tokens emitted in one batch):
        # fall back to "wall minus prefill" so eval_dur stays positive.
        eval_dur = wall - first_content_at
    else:
        eval_dur = 0.0

    eval_count: int | None = None
    prompt_tokens = 0
    total_tokens: int | None = None
    if usage is not None:
        eval_count = getattr(usage, "completion_tokens", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        total_tokens = getattr(usage, "total_tokens", None)

    estimated = False
    if eval_count is None:
        eval_count = rough_token_count(answer)
        estimated = True
    eval_count = int(eval_count or 0)
    if total_tokens is None and prompt_tokens and eval_count:
        total_tokens = prompt_tokens + eval_count

    return {
        "wall": wall,
        "ttft": first_content_at,
        "thinking_ttft": first_thinking_at,
        "eval_dur": eval_dur,
        "eval_count": eval_count,
        "prompt_tokens": prompt_tokens,
        "total_tokens": int(total_tokens or 0),
        "answer": answer,
        "estimated": estimated,
        "server_timings": server_timings,
    }


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int,
                            thinking: bool = False,
                            ) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / oai-compat
    backends. ONE streaming request per prompt — see this module's
    docstring for the metric coordinate system.

    `thinking` is echoed onto `QuestionResult.has_thinking`. When True,
    the request carries `extra_body={"chat_template_kwargs":
    {"thinking": True}}` so vLLM / Qwen3 / DeepSeek-R1 chat templates
    emit a reasoning delta we can time.
    """
    try:
        result = _run_openai_stream(
            url, model, prompt, conf,
            timeout=request_timeout, thinking=thinking)
    except Exception as exc:
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        return QuestionResult(
            prompt=prompt,
            ok=False,
            error=msg,
            wall_seconds=0.0,
            has_thinking=thinking,
        )

    wall = result["wall"]
    eval_dur = result["eval_dur"]
    eval_count = result["eval_count"]
    answer = result["answer"]

    ttft_val = (round(result["ttft"], 3)
                if result["ttft"] is not None else 0.0)
    think_val = (round(result["thinking_ttft"], 3)
                 if result["thinking_ttft"] is not None else 0.0)

    decode_tps = (eval_count / eval_dur) if eval_dur > 0 else 0.0
    client_tps = (eval_count / wall) if wall > 0 and eval_count else 0.0

    server_timings = result["server_timings"]
    server_gen = float(server_timings.get("server_generation_seconds") or 0.0)
    server_prompt_eval = float(
        server_timings.get("server_prompt_eval_seconds") or 0.0)
    server_tps_reported = float(server_timings.get("server_tps") or 0.0)

    notes: list[str] = []
    notes.append("ttft / thinking_ttft from first delta of the streaming "
                 "main request (no separate probe).")
    notes.append("tps = eval_count / eval_seconds where eval_seconds is "
                 "the client-observed decode window "
                 "(last_content_delta - first_content_delta). "
                 "client_tps = eval_count / wall is kept for diagnostics.")
    if not server_timings:
        notes.append("server `timings` block absent (vLLM); "
                     "prompt_eval_seconds / server_tps_reported = 0.")
    if result["estimated"]:
        notes.append("server skipped the final usage chunk; eval_count is "
                     "a char-count estimate (rough_token_count).")
    if (result["ttft"] is None
            and result["thinking_ttft"] is None
            and eval_count == 0):
        notes.append("stream completed but emitted no content / reasoning "
                     "deltas — server may have returned an empty answer.")

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(answer),
        wall_seconds=round(wall, 3),
        ttft_seconds=ttft_val,
        thinking_ttft_seconds=think_val,
        has_thinking=thinking,
        load_seconds=0.0,
        prompt_eval_seconds=round(server_prompt_eval, 3),
        eval_count=eval_count,
        eval_seconds=round(eval_dur, 3),
        tps=round(decode_tps, 2),
        total_server_seconds=round(wall, 3),
        prompt_tokens=result["prompt_tokens"],
        total_tokens=result["total_tokens"],
        client_tps=round(client_tps, 2),
        server_tps_reported=round(server_tps_reported, 2),
        tokens_estimated=result["estimated"],
        note="; ".join(notes),
    )
