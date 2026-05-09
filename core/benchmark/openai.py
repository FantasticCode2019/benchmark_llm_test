"""OpenAI-compatible benchmark (vLLM / llama.cpp / oai-compat).

Uses the official `openai` SDK (>=1.30) so we don't have to maintain our
own URL/header/payload assembly. Mirrors scripts/llm_api_benchmark.py so
numbers stay comparable:
  - per-call max_tokens / temperature / top_p / extra_body
  - optional Bearer auth header (llama-server `--api-key` mode)
  - TTFT approximated via a separate max_tokens=1 round-trip
  - llama.cpp `timings` block decoded from the raw response (it's a
    non-standard extension, so we read it via `with_raw_response`)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from openai import OpenAI
from openai import APIStatusError

from models import OpenAIConfig, QuestionResult
from utils.http import auth_hint
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


def make_openai_client(url: str, conf: OpenAIConfig,
                       *, timeout: int) -> OpenAI:
    """Build an `openai.OpenAI` pointed at the entrance URL.

    base_url MUST end with /v1; the SDK appends /chat/completions etc.
    after that. We force max_retries=0 because `warmup_until_ok` already
    retries at a higher level — a second retry layer just multiplies the
    wait time and obscures the real failure.
    """
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return OpenAI(
        api_key=conf.api_key or "EMPTY",
        base_url=base,
        default_headers=conf.extra_headers or None,
        timeout=timeout,
        max_retries=0,
    )


def _create_chat(client: OpenAI, model: str, prompt: str,
                 conf: OpenAIConfig, *, max_tokens: int):
    """Chat-completions create with our knobs.
    Returns the `LegacyAPIResponse` from with_raw_response so callers can
    BOTH parse() typed AND read .json() to grab llama.cpp's `timings`.
    """
    return client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=conf.temperature,
        top_p=conf.top_p,
        extra_body=conf.extra_body or None,
        stream=False,
    )


def _create_completion(client: OpenAI, model: str, prompt: str,
                       conf: OpenAIConfig, *, max_tokens: int):
    return client.completions.with_raw_response.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=conf.temperature,
        top_p=conf.top_p,
        extra_body=conf.extra_body or None,
        stream=False,
    )


def openai_warmup_call(client: OpenAI, model: str, conf: OpenAIConfig,
                       *, max_tokens: int = 4) -> None:
    """One inference call used by warmup_until_ok. Raises on any failure
    so the warmup loop can see it.
    """
    if conf.endpoint == "completion":
        _create_completion(client, model, "ping", conf, max_tokens=max_tokens)
    else:
        _create_chat(client, model, "ping", conf, max_tokens=max_tokens)


def _extract_response(body: dict) -> dict:
    """Pull answer + token + (optional) server timings out of the raw
    response dict. Handles both Chat Completion and the legacy Completion
    shape. llama-server's OpenAI endpoint additionally exposes a
    `timings` block with millisecond precision; vLLM does not.
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


def _format_api_error(exc: Exception) -> str:
    """Compact one-line summary of an OpenAI SDK exception, with our
    auth_hint() appended when applicable.
    """
    msg = str(exc)
    if isinstance(exc, APIStatusError):
        body_snippet = ""
        try:
            body_snippet = (exc.response.text or "").strip()[:300].replace("\n", " ")
        except Exception:  # noqa: BLE001
            pass
        # llama-server / vLLM happily 200 + HTML-login-page if Authelia
        # intercepts; the SDK then raises APIResponseValidationError or
        # JSONDecodeError. By the time we catch APIStatusError it's
        # usually a real 4xx/5xx — but body_snippet still helps debug.
        msg = f"HTTP {exc.status_code} from {exc.request.url}: {body_snippet or exc.message}"
    hint = auth_hint(exc)
    return f"{msg} ({hint})" if hint else msg


def _measure_ttft(client: OpenAI, model: str, prompt: str,
                  conf: OpenAIConfig) -> Optional[float]:
    """max_tokens=1 round-trip approximation. Returns None on failure."""
    try:
        t = time.perf_counter()
        if conf.endpoint == "completion":
            _create_completion(client, model, prompt, conf, max_tokens=1)
        else:
            _create_chat(client, model, prompt, conf, max_tokens=1)
        return time.perf_counter() - t
    except Exception as exc:  # noqa: BLE001
        log.debug("ttft probe failed: %s", exc)
        return None


def _to_dict(raw_response: Any) -> dict:
    """Decode a `with_raw_response` result into a plain dict.
    Tolerates both the legacy `.http_response` accessor (older SDKs) and
    the current `.parse()` + `.json()` shape.
    """
    # Newer SDKs expose `.http_response` (httpx.Response) on the wrapper.
    http_resp = getattr(raw_response, "http_response", None)
    if http_resp is not None:
        return http_resp.json()
    # Fall back to .text + json — works on every supported SDK version.
    import json as _json
    return _json.loads(raw_response.text)


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / other oai-compat
    backends. See QuestionResult docstring for stream=false field semantics.
    """
    client = make_openai_client(url, conf, timeout=request_timeout)

    ttft_approx: Optional[float] = None
    if conf.measure_ttft_approx:
        ttft_approx = _measure_ttft(client, model, prompt, conf)

    started = time.perf_counter()
    try:
        if conf.endpoint == "completion":
            raw = _create_completion(client, model, prompt, conf,
                                     max_tokens=conf.max_tokens)
        else:
            raw = _create_chat(client, model, prompt, conf,
                               max_tokens=conf.max_tokens)
    except Exception as exc:  # noqa: BLE001
        return QuestionResult(
            prompt=prompt, ok=False, error=_format_api_error(exc),
            wall_seconds=round(time.perf_counter() - started, 3),
            ttft_seconds=round(ttft_approx, 3) if ttft_approx else 0.0,
        )
    wall = time.perf_counter() - started
    body = _to_dict(raw)

    parsed = _extract_response(body)
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
    # Prefer server-reported TPS when llama.cpp gives it to us, else fall
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
