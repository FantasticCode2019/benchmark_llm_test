"""
Ollama-specific helpers: /api/generate benchmark with precise
server-side timing fields, plus a one-shot streaming probe used to
fill ``thinking_ttft_seconds`` when the model exposes a reasoning
phase.

Note on /api/pull: every supported olares-market ``ollama*`` chart
ships a launcher container that runs ``ollama pull <model>`` at
startup, so this module no longer issues its own pull. Readiness is
established by ``wait_until_api_ready`` watching ``/api/tags`` —
identical to the launcher's own "Waiting for Ollama" → "Ready to
chat" gate.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from llm_bench.clients.openai_errors import auth_hint
from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import QuestionResult
from llm_bench.utils.http import http_post_json

log = logging.getLogger(LOG_NAMESPACE)


def _measure_ollama_streaming_ttfts(url: str, model: str, prompt: str,
                                    *, request_timeout: int,
                                    ) -> tuple[float | None,
                                               float | None]:
    """Open ONE streaming /api/generate call with `think:true` and read
    the first non-empty `thinking` and `response` chunks. Returns
    `(thinking_ttft, answer_ttft)`; either field is None when not
    observed (probe failure, model didn't actually emit reasoning, ...).

    Per cankao.md: ollama 0.10+ exposes thinking via the `thinking`
    field on each NDJSON chunk; the actual answer arrives in `response`.
    We close the connection as soon as both are seen so the server stops
    decoding early and the probe doesn't pay for a full generation.

    Only invoked when the runtime ``/api/show`` capability probe
    reported ``thinking`` support — we trust that signal, so probe
    failures (including the occasional Ollama-0.10
    ``HTTP 400 "does not support thinking"`` for half-converted
    weights) just degrade ``thinking_ttft_seconds`` to 0; they don't
    change ``has_thinking``.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "think": True,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "application/x-ndjson"},
        method="POST",
    )

    thinking_ttft: float | None = None
    answer_ttft: float | None = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(
                        raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                # /api/generate returns thinking at the top level;
                # /api/chat would nest it under message.thinking instead.
                thinking = chunk.get("thinking")
                response = chunk.get("response")

                if thinking and thinking_ttft is None:
                    thinking_ttft = time.perf_counter() - started
                if response and answer_ttft is None:
                    answer_ttft = time.perf_counter() - started

                if (thinking_ttft is not None
                        and answer_ttft is not None):
                    break
                if chunk.get("done"):
                    break
    except Exception as exc:
        log.warning("ollama streaming ttft probe failed: %s", exc)
        return None, None

    return thinking_ttft, answer_ttft


def benchmark_prompt_ollama(url: str, model: str, prompt: str,
                            *,
                            request_timeout: int,
                            thinking: bool = False,
                            ) -> QuestionResult:
    """One non-streaming /api/generate call. Decode server-reported
    durations (nanoseconds) into the unified QuestionResult shape.

    `thinking` reflects the runtime ``/api/show`` capability probe
    (see ``_step_probe_ollama_thinking``), echoed onto
    ``QuestionResult.has_thinking``. When True, an extra streaming
    probe runs with `think:true` and fills `thinking_ttft_seconds`
    from the first reasoning chunk. When False, no extra probe runs
    and `thinking_ttft_seconds` is left at 0 (rendered as `—` in the
    email).
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    started = time.perf_counter()
    try:
        body = http_post_json(f"{url}/api/generate", payload,
                              timeout=request_timeout)
    except Exception as exc:
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            has_thinking=thinking,
        )
    wall = time.perf_counter() - started

    # Ollama reports durations in nanoseconds.
    load = body.get("load_duration", 0) / 1e9
    prompt_eval = body.get("prompt_eval_duration", 0) / 1e9
    eval_count = int(body.get("eval_count", 0))
    eval_dur = body.get("eval_duration", 0) / 1e9
    total = body.get("total_duration", 0) / 1e9

    ttft_seconds = round(load + prompt_eval, 3)
    thinking_ttft_seconds = 0.0
    if thinking:
        think_ttft, _ = _measure_ollama_streaming_ttfts(
            url, model, prompt, request_timeout=request_timeout)
        if think_ttft is not None:
            thinking_ttft_seconds = round(think_ttft, 3)

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(body.get("response", "")),
        wall_seconds=round(wall, 3),
        ttft_seconds=ttft_seconds,
        thinking_ttft_seconds=thinking_ttft_seconds,
        has_thinking=thinking,
        load_seconds=round(load, 3),
        prompt_eval_seconds=round(prompt_eval, 3),
        eval_count=eval_count,
        eval_seconds=round(eval_dur, 3),
        tps=round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0,
        total_server_seconds=round(total, 3),
    )
