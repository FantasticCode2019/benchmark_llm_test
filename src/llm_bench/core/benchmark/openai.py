"""OpenAI-compatible benchmark (vLLM / llama.cpp / oai-compat).

Mirrors scripts/llm_api_benchmark.py so numbers are comparable:
  - per-call max_tokens / temperature / top_p / extra_body
  - optional Bearer auth header (llama-server `--api-key` mode)
  - TTFT approximated via a separate max_tokens=1 round-trip
  - llama.cpp `timings` block is decoded when present so we surface
    real server-side decode tokens/s, not just the wall-clock estimate
  - per-model `spec.thinking=true` triggers ONE extra streaming probe
    that asks vLLM to render with
        extra_body={"chat_template_kwargs": {"thinking": True}}
    via the official `openai` SDK and reads the first reasoning
    delta into `thinking_ttft_seconds`. Whether the model has a
    thinking phase is taken straight from config — no auto-detection.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from openai import OpenAI

from models import OpenAIConfig, QuestionResult
from utils.http import auth_hint, post_openai
from utils.tokens import ms_to_seconds, rough_token_count, to_float

log = logging.getLogger("llm_bench")


def openai_config_from(spec: dict, cfg: dict) -> OpenAIConfig:
    """Merge per-model openai overrides on top of global config defaults."""
    g = (cfg.get("openai_defaults") or {})
    s = (spec.get("openai") or {})

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


def openai_url(base: str, endpoint: str) -> str:
    base = base.rstrip("/")
    suffix = "/completions" if endpoint == "completion" else "/chat/completions"
    return base + suffix if base.endswith("/v1") else base + "/v1" + suffix


def openai_headers(api_key: str, extra: dict) -> dict:
    """Mirrors `curl` semantics: only sends `Authorization: Bearer ...`
    when the user actually set a key — `EMPTY` (or empty string) is
    treated as "no auth". `extra_headers` from config can override.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key and api_key.strip().upper() not in {"", "EMPTY"}:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def build_openai_payload(model: str, prompt: str, conf: OpenAIConfig,
                         *, max_tokens_override: Optional[int] = None) -> dict:
    mt = (max_tokens_override
          if max_tokens_override is not None else conf.max_tokens)
    # IMPORTANT: stream=False is REQUIRED. The whole script parses the
    # response as a single JSON object; with stream=True the server replies
    # with SSE chunks and json.loads() blows up.
    if conf.endpoint == "completion":
        payload: dict = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": mt,
            "temperature": conf.temperature,
        }
    else:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": mt,
            "temperature": conf.temperature,
        }
    if conf.top_p is not None:
        payload["top_p"] = conf.top_p
    if conf.extra_body:
        payload.update(conf.extra_body)
    return payload


def _extract_openai_response(body: dict) -> dict:
    """Pull answer + token + (optional) server timings out of the response.
    Handles both Chat Completion and the legacy Completion shape.
    llama-server's OpenAI endpoint additionally exposes a `timings` block
    with millisecond precision; vLLM does not.
    """
    choices = body.get("choices") or []
    answer = ""
    if choices:
        first = choices[0] or {}
        msg = first.get("message")
        if isinstance(msg, dict):
            answer = msg.get("content") or ""
        else:
            answer = first.get("text") or ""

    usage = body.get("usage") or {}
    completion = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    total = usage.get("total_tokens")

    timings = body.get("timings") or body.get("timing") or {}
    server_prompt = ms_to_seconds(
        timings.get("prompt_ms") or timings.get("prompt_eval_ms"))
    server_gen = ms_to_seconds(
        timings.get("predicted_ms") or timings.get("generation_ms")
        or timings.get("eval_ms"))
    server_tps = to_float(
        timings.get("predicted_per_second") or timings.get("tokens_per_second"))

    return {
        "answer": answer,
        "completion_tokens": int(completion) if completion is not None else None,
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "total_tokens": int(total) if total is not None else None,
        "server_prompt_eval_seconds": server_prompt,
        "server_generation_seconds": server_gen,
        "server_tps": server_tps,
    }


def _measure_openai_ttft(url: str, model: str, prompt: str,
                         conf: OpenAIConfig,
                         *, timeout: int) -> Optional[float]:
    """Approximate TTFT via a max_tokens=1 round-trip. Returns None on
    failure — caller treats that as "no measurement".
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)
    payload = build_openai_payload(model, prompt, conf, max_tokens_override=1)
    try:
        wall, _ = post_openai(full, payload, headers, timeout=timeout)
        return wall
    except Exception as exc:  # noqa: BLE001
        log.debug("ttft probe failed: %s", exc)
        return None


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
    a probe is supposed to surface the FIRST observed failure, not paper
    over it with silent retry latency that would distort TTFT readings.
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


def _merge_thinking_extra_body(base: Optional[dict]) -> dict:
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


def _measure_openai_streaming_ttfts(url: str, model: str, prompt: str,
                                    conf: OpenAIConfig,
                                    *, timeout: int,
                                    ) -> Tuple[Optional[float],
                                               Optional[float]]:
    """Open ONE streaming /v1/chat/completions request via the `openai`
    SDK with `extra_body={"chat_template_kwargs": {"thinking": True}}`
    and capture both the first reasoning delta and the first content
    delta. Returns `(thinking_ttft, answer_ttft)`; either field is
    None when not observed (probe failure, no reasoning emitted, etc.).

    Per the user-supplied vLLM snippet (cankao.md):

        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"chat_template_kwargs": {"thinking": True}},
        )

    Only invoked when `spec.thinking=true` AND
    `conf.measure_ttft_approx=true`. Replaces the max_tokens=1 ping
    for those models so we don't pay for an extra round-trip.
    """
    client = _make_openai_client(url, conf, timeout=timeout)
    extra_body = _merge_thinking_extra_body(conf.extra_body)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": conf.max_tokens,
        "temperature": conf.temperature,
        "extra_body": extra_body,
    }
    if conf.top_p is not None:
        kwargs["top_p"] = conf.top_p

    thinking_ttft: Optional[float] = None
    answer_ttft: Optional[float] = None
    stream = None
    started = time.perf_counter()
    try:
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning = (getattr(delta, "reasoning", None)
                         or getattr(delta, "reasoning_content", None))
            content = getattr(delta, "content", None)

            if reasoning and thinking_ttft is None:
                thinking_ttft = time.perf_counter() - started
            if content and answer_ttft is None:
                answer_ttft = time.perf_counter() - started

            if (thinking_ttft is not None
                    and answer_ttft is not None):
                break
    except Exception as exc:  # noqa: BLE001
        log.warning("openai streaming ttft probe failed: %s",
                    str(exc)[:240])
        return None, None
    finally:
        # Closing the response releases the underlying httpx connection
        # so the server stops generating tokens we're never going to read.
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    return thinking_ttft, answer_ttft


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int,
                            thinking: bool = False,
                            ) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / other oai-compat
    backends. See QuestionResult docstring for stream=false field semantics.

    `thinking` is the per-model `spec.thinking` config flag echoed straight
    onto `QuestionResult.has_thinking`. When True, an extra streaming probe
    runs with `extra_body={"chat_template_kwargs": {"thinking": True}}` and
    fills `thinking_ttft_seconds` from the first reasoning delta (the same
    probe also gives a real `ttft_seconds` from the first content delta,
    so it REPLACES the max_tokens=1 ping for thinking models). When False,
    only the classic max_tokens=1 probe runs and `thinking_ttft_seconds`
    is left at 0 (rendered as `—` in the email).
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)

    ttft_approx: Optional[float] = None
    thinking_ttft: Optional[float] = None

    if thinking and conf.measure_ttft_approx:
        # One streaming probe captures BOTH ttft and thinking_ttft.
        thinking_ttft, ttft_approx = _measure_openai_streaming_ttfts(
            url, model, prompt, conf, timeout=request_timeout)
        # Fall back to max_tokens=1 if the stream gave us nothing for
        # ttft (e.g. the entrance dropped SSE).
        if ttft_approx is None:
            ttft_approx = _measure_openai_ttft(url, model, prompt, conf,
                                               timeout=request_timeout)
    elif conf.measure_ttft_approx:
        ttft_approx = _measure_openai_ttft(url, model, prompt, conf,
                                           timeout=request_timeout)

    payload = build_openai_payload(model, prompt, conf)
    started = time.perf_counter()
    try:
        wall, body = post_openai(full, payload, headers,
                                 timeout=request_timeout)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        ttft_val = round(ttft_approx, 3) if ttft_approx else 0.0
        think_val = (round(thinking_ttft, 3)
                     if thinking_ttft is not None else 0.0)
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            ttft_seconds=ttft_val,
            thinking_ttft_seconds=think_val,
            has_thinking=thinking,
        )

    parsed = _extract_openai_response(body)
    answer = parsed["answer"]
    completion = parsed["completion_tokens"]
    estimated = False
    if completion is None:
        completion = rough_token_count(answer)
        estimated = True
    prompt_tokens = parsed["prompt_tokens"] or 0
    total_tokens = parsed["total_tokens"]
    if total_tokens is None and prompt_tokens and completion:
        total_tokens = prompt_tokens + completion

    server_gen = parsed["server_generation_seconds"]
    server_tps_reported = parsed["server_tps"]
    # Prefer the real server TPS when llama.cpp gives it to us, else fall
    # back to client end-to-end (the only honest number vLLM exposes
    # under stream=false).
    client_tps = (completion / wall) if wall > 0 and completion else 0.0
    if server_tps_reported and server_tps_reported > 0:
        chosen_tps = server_tps_reported
    elif server_gen and server_gen > 0 and completion:
        chosen_tps = completion / server_gen
    else:
        chosen_tps = client_tps

    notes: list = []
    if ttft_approx is None and conf.measure_ttft_approx:
        notes.append("ttft probe failed; ttft_seconds=0")
    elif thinking and thinking_ttft is not None:
        notes.append("ttft_seconds and thinking_ttft_seconds captured via "
                     "stream=true with chat_template_kwargs.thinking=true "
                     "(first content delta / first reasoning delta)")
    elif thinking and thinking_ttft is None:
        notes.append("spec.thinking=true but no reasoning delta observed; "
                     "thinking_ttft_seconds=0")
    elif ttft_approx is not None:
        notes.append("ttft_seconds is APPROX (max_tokens=1 round-trip; "
                     "TRUE TTFT needs stream=true)")
    if not (server_tps_reported and server_tps_reported > 0) and \
       not (server_gen and server_gen > 0):
        notes.append("no server-side `timings` block; tps = client end-to-end "
                     "(includes prefill+decode+network)")
    if estimated:
        notes.append("usage missing; eval_count is a char-count estimate")

    ttft_val = round(ttft_approx, 3) if ttft_approx else 0.0
    think_val = (round(thinking_ttft, 3)
                 if thinking_ttft is not None else 0.0)

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(answer),
        wall_seconds=round(wall, 3),
        ttft_seconds=ttft_val,
        thinking_ttft_seconds=think_val,
        has_thinking=thinking,
        load_seconds=0.0,
        prompt_eval_seconds=round(parsed["server_prompt_eval_seconds"], 3),
        eval_count=int(completion or 0),
        eval_seconds=round(server_gen, 3),
        tps=round(chosen_tps, 2),
        total_server_seconds=round(wall, 3),
        prompt_tokens=prompt_tokens,
        total_tokens=int(total_tokens or 0),
        client_tps=round(client_tps, 2),
        server_tps_reported=round(server_tps_reported, 2)
            if server_tps_reported else 0.0,
        tokens_estimated=estimated,
        note="; ".join(notes),
    )
