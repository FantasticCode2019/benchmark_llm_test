"""OpenAI-compatible benchmark (vLLM / llama.cpp / oai-compat).

Mirrors scripts/llm_api_benchmark.py so numbers are comparable:
  - per-call max_tokens / temperature / top_p / extra_body
  - optional Bearer auth header (llama-server `--api-key` mode)
  - TTFT approximated via a separate max_tokens=1 round-trip
  - llama.cpp `timings` block is decoded when present so we surface
    real server-side decode tokens/s, not just the wall-clock estimate
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

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


def _measure_openai_streaming_ttfts(url: str, model: str, prompt: str,
                                    conf: OpenAIConfig,
                                    *, timeout: int,
                                    ) -> Tuple[Optional[float],
                                               Optional[float]]:
    """Open ONE streaming /v1/chat/completions request and capture both
    the first reasoning token and the first content token times in a
    single round-trip. Returns (thinking_ttft, answer_ttft); either may
    be None if the corresponding field never appeared.

    Per cankao.md: vLLM exposes thinking via `delta.reasoning` (new) or
    `delta.reasoning_content` (legacy). Plain content tokens land in
    `delta.content`. We close the connection as soon as both are seen
    so the server stops generating early and the probe doesn't pay for
    a full decode.
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)
    headers["Accept"] = "text/event-stream"

    payload = build_openai_payload(model, prompt, conf)
    payload["stream"] = True

    req = urllib.request.Request(
        full,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    thinking_ttft: Optional[float] = None
    answer_ttft: Optional[float] = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    if payload_str == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                first = choices[0] or {}
                # Streaming uses `delta`; some legacy servers reuse `message`.
                delta = first.get("delta") or first.get("message") or {}
                if not isinstance(delta, dict):
                    continue
                reasoning = (delta.get("reasoning")
                             or delta.get("reasoning_content"))
                content = delta.get("content")

                if reasoning and thinking_ttft is None:
                    thinking_ttft = time.perf_counter() - started
                if content and answer_ttft is None:
                    answer_ttft = time.perf_counter() - started

                if (thinking_ttft is not None
                        and answer_ttft is not None):
                    break
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        log.warning("streaming ttft probe failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("streaming ttft probe raised: %s", exc)

    return thinking_ttft, answer_ttft


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int,
                            thinking: bool = False,
                            ) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / other oai-compat
    backends. See QuestionResult docstring for stream=false field semantics.

    When `thinking=True`, open ONE streaming probe and read both the first
    reasoning delta and the first content delta. The streaming probe
    REPLACES the max_tokens=1 ttft probe (saves a request) and gives a
    real `ttft_seconds` plus a new `thinking_ttft_seconds` in one shot.
    For non-thinking models, `thinking_ttft_seconds` mirrors `ttft_seconds`.
    """
    full = openai_url(url, conf.endpoint)
    headers = openai_headers(conf.api_key, conf.extra_headers)

    ttft_approx: Optional[float] = None
    thinking_ttft: Optional[float] = None

    if thinking:
        # Single streaming probe gives us both metrics in one round-trip;
        # avoids a second max_tokens=1 ping for the same model.
        thinking_ttft, ttft_approx = _measure_openai_streaming_ttfts(
            url, model, prompt, conf, timeout=request_timeout)
        # Fall back to the classic max_tokens=1 probe if streaming
        # didn't yield anything useful (e.g. the entrance dropped SSE).
        if (ttft_approx is None and conf.measure_ttft_approx):
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
        # For non-thinking runs the new column mirrors TTFT (per request:
        # "若模型不带thinking能力，那么设置这一列和现有的TTFT值相同即可").
        if thinking_ttft is not None:
            think_val = round(thinking_ttft, 3)
        else:
            think_val = ttft_val
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            ttft_seconds=ttft_val,
            thinking_ttft_seconds=think_val,
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
                     "stream=true (first content delta / first reasoning "
                     "delta)")
    elif ttft_approx is not None:
        notes.append("ttft_seconds is APPROX (max_tokens=1 round-trip; "
                     "TRUE TTFT needs stream=true)")
    if thinking and thinking_ttft is None:
        notes.append("thinking_ttft_seconds mirrors ttft_seconds "
                     "(streaming probe saw no reasoning delta — server "
                     "may merge thinking into content)")
    if not (server_tps_reported and server_tps_reported > 0) and \
       not (server_gen and server_gen > 0):
        notes.append("no server-side `timings` block; tps = client end-to-end "
                     "(includes prefill+decode+network)")
    if estimated:
        notes.append("usage missing; eval_count is a char-count estimate")

    ttft_val = round(ttft_approx, 3) if ttft_approx else 0.0
    if thinking_ttft is not None:
        think_val = round(thinking_ttft, 3)
    else:
        # Non-thinking model (or thinking probe never saw a reasoning
        # delta because the server merged thinking into `content`):
        # mirror the answer-side TTFT so the new column is always set.
        think_val = ttft_val

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(answer),
        wall_seconds=round(wall, 3),
        ttft_seconds=ttft_val,
        thinking_ttft_seconds=think_val,
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
