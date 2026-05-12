"""OpenAI-compatible benchmark (vLLM / llama.cpp / oai-compat).

Mirrors scripts/llm_api_benchmark.py so numbers are comparable:
  - per-call max_tokens / temperature / top_p / extra_body
  - optional Bearer auth header (llama-server `--api-key` mode)
  - TTFT approximated via a separate max_tokens=1 round-trip
  - llama.cpp `timings` block is decoded when present so we surface
    real server-side decode tokens/s, not just the wall-clock estimate
"""
from __future__ import annotations

import logging
import time
from typing import Optional

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


def _merge_extra_body(base: Optional[dict],
                      override: Optional[dict]) -> dict:
    """Shallow merge with one special case: nested `chat_template_kwargs`
    is merged one level deeper, because that's the typical shape Qwen3 /
    DeepSeek / etc use to carry thinking / reasoning toggles and we don't
    want a default like `{enable_thinking:false}` to wipe out a
    user-supplied `{some_other_key:...}`.
    """
    out = dict(base or {})
    for k, v in (override or {}).items():
        if (k == "chat_template_kwargs" and isinstance(v, dict)
                and isinstance(out.get(k), dict)):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _measure_openai_ttft(url: str, model: str, prompt: str,
                         conf: OpenAIConfig,
                         *, timeout: int,
                         extra_body_override: Optional[dict] = None,
                         ) -> Optional[float]:
    """Approximate TTFT via a max_tokens=1 round-trip. Returns None on
    failure — caller treats that as "no measurement".

    `extra_body_override` is merged on top of `conf.extra_body` ONLY for
    this probe (used by the no-think TTFT measurement to inject the
    backend-specific "disable thinking" knob without polluting the main
    benchmark call).
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)
    if extra_body_override:
        from dataclasses import replace
        merged = _merge_extra_body(conf.extra_body, extra_body_override)
        conf = replace(conf, extra_body=merged)
    payload = build_openai_payload(model, prompt, conf, max_tokens_override=1)
    try:
        wall, _ = post_openai(full, payload, headers, timeout=timeout)
        return wall
    except Exception as exc:  # noqa: BLE001
        log.debug("ttft probe failed: %s", exc)
        return None


# Default knob to disable thinking for an OpenAI-compatible backend that
# ships a chat template with `enable_thinking` (Qwen3, DeepSeek-R1 chat
# template, ...). Override per-model with `thinking_disable_extra_body`.
DEFAULT_OPENAI_THINKING_DISABLE = {
    "chat_template_kwargs": {"enable_thinking": False},
}


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int,
                            thinking: bool = False,
                            thinking_disable_extra_body:
                                Optional[dict] = None,
                            ) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / other oai-compat
    backends. See QuestionResult docstring for stream=false field semantics.

    When `thinking=True`, run an EXTRA max_tokens=1 probe with thinking
    explicitly disabled (default `chat_template_kwargs.enable_thinking
    =false`) and store its wall-time in `ttft_no_think_seconds`. Useful
    for backends where the default TTFT measurement gets inflated by an
    invisible reasoning phase (vLLM with reasoning_parser, OpenAI o1, ...).
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)

    ttft_approx: Optional[float] = None
    if conf.measure_ttft_approx:
        ttft_approx = _measure_openai_ttft(url, model, prompt, conf,
                                           timeout=request_timeout)

    ttft_no_think: Optional[float] = None
    if thinking:
        ttft_no_think = _measure_openai_ttft(
            url, model, prompt, conf,
            timeout=request_timeout,
            extra_body_override=(thinking_disable_extra_body
                                 or DEFAULT_OPENAI_THINKING_DISABLE),
        )

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
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            ttft_seconds=round(ttft_approx, 3) if ttft_approx else 0.0,
            ttft_no_think_seconds=(round(ttft_no_think, 3)
                                   if ttft_no_think else 0.0),
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
    elif ttft_approx is not None:
        notes.append("ttft_seconds is APPROX (max_tokens=1 round-trip; "
                     "TRUE TTFT needs stream=true)")
    if thinking and ttft_no_think is None:
        notes.append("ttft_no_think probe failed; ttft_no_think_seconds=0")
    elif thinking and ttft_no_think is not None:
        notes.append("ttft_no_think_seconds is APPROX (max_tokens=1 with "
                     "thinking disabled)")
    if not (server_tps_reported and server_tps_reported > 0) and \
       not (server_gen and server_gen > 0):
        notes.append("no server-side `timings` block; tps = client end-to-end "
                     "(includes prefill+decode+network)")
    if estimated:
        notes.append("usage missing; eval_count is a char-count estimate")

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(answer),
        wall_seconds=round(wall, 3),
        ttft_seconds=round(ttft_approx, 3) if ttft_approx else 0.0,
        ttft_no_think_seconds=(round(ttft_no_think, 3)
                               if ttft_no_think else 0.0),
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
