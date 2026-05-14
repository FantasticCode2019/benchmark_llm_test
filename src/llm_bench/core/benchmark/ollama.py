"""
Ollama-specific helper: ONE streaming /api/generate call that captures
the client-observed time-to-first-chunk for two distinct phases
(hidden thinking trace + visible response) plus the server's aggregate
stats. No second probe required.

Metrics emitted:

  * ``ttft_seconds`` — client-observed wall-clock time from request
    start to the first non-empty **visible response** chunk. Always
    measured regardless of the ``thinking`` flag. For a non-thinking
    model this IS the TTFT; for a thinking model this is the
    user-perceived "first answer token after the reasoning trace
    ends".

  * ``thinking_ttft_seconds`` — client-observed wall-clock time from
    request start to the first non-empty **thinking** chunk.
    Populated only when the caller passed ``thinking=True`` AND the
    model actually emitted a thinking chunk; left at ``0.0`` otherwise.

  * Server aggregate stats — ``load_seconds``,
    ``prompt_eval_seconds``, ``eval_count``, ``eval_seconds``,
    ``tps``, ``total_server_seconds`` — are decoded from the final
    ``done:true`` chunk and stored on the :class:`QuestionResult` for
    diagnostics. They are NOT used as TTFT: for a thinking model
    ``load_duration + prompt_eval_duration`` would understate the
    user-visible TTFT because it ignores the time the model spends
    emitting the hidden reasoning trace.

Both TTFTs share the same ``time.perf_counter()`` epoch (captured
immediately BEFORE ``urllib.request.urlopen``), so they're directly
comparable in the same coordinate system: ``thinking_ttft_seconds``
marks the start of the thinking phase, ``ttft_seconds`` marks the
start of the visible-response phase. For models that emit reasoning
before the answer the expected order is
``thinking_ttft_seconds < ttft_seconds``.

Whitespace-only chunks do NOT count as the first chunk for either
metric (``.strip()`` filter), because Ollama occasionally emits an
empty leading chunk before the real first token.

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

log = logging.getLogger(LOG_NAMESPACE)


def benchmark_prompt_ollama(
    url: str,
    model: str,
    prompt: str,
    *,
    request_timeout: int,
    thinking: bool = False,
) -> QuestionResult:
    """One streaming /api/generate call. Returns client-observed TTFT
    plus aggregate server timings in a single request.

    Metric semantics:

      * thinking=True
          - Send ``think:true``.
          - ``thinking_ttft_seconds`` is the client-observed wall-clock time
            from request start to the first non-empty thinking chunk.
          - ``ttft_seconds`` is the client-observed wall-clock time from
            request start to the first non-empty visible response chunk.
          - Server aggregate stats are still taken from the final done chunk.

      * thinking=False
          - Send ``think:false``.
          - ``thinking_ttft_seconds`` stays ``0.0``.
          - ``ttft_seconds`` is still the client-observed wall-clock time to
            the first non-empty visible response chunk.

    Notes:

      ``load_duration`` and ``prompt_eval_duration`` from Ollama are server-side
      aggregate timings. They are useful for diagnostics, but they should not be
      used as user-visible TTFT for thinking models, because the model may spend
      additional time generating hidden reasoning before emitting visible output.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": bool(thinking),
    }

    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    started = time.perf_counter()
    first_thinking_at: float | None = None
    first_response_at: float | None = None
    response_parts: list[str] = []
    final_chunk: dict = {}

    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue

                try:
                    chunk = json.loads(
                        raw_line.decode("utf-8", errors="replace")
                    )
                except json.JSONDecodeError:
                    continue

                if not isinstance(chunk, dict):
                    continue

                # /api/generate exposes thinking/response at the top level;
                # /api/chat nests them under message.{thinking, content}.
                # Accept both shapes to avoid silently losing TTFT if a future
                # caller switches from generate to chat.
                message = chunk.get("message") or {}
                thinking_text = (
                    chunk.get("thinking")
                    or message.get("thinking")
                    or ""
                )
                response_text = (
                    chunk.get("response")
                    or message.get("content")
                    or ""
                )

                now = time.perf_counter() - started

                # First hidden reasoning token/chunk observed by the client.
                # Use strip() so whitespace-only chunks do not falsely become
                # the first thinking token.
                if (
                    thinking
                    and first_thinking_at is None
                    and isinstance(thinking_text, str)
                    and thinking_text.strip()
                ):
                    first_thinking_at = now

                # First visible response token/chunk observed by the client.
                # This is the user-perceived TTFT.
                if (
                    first_response_at is None
                    and isinstance(response_text, str)
                    and response_text.strip()
                ):
                    first_response_at = now

                if response_text:
                    response_parts.append(response_text)

                if chunk.get("done"):
                    final_chunk = chunk
                    break

    except Exception as exc:
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"

        return QuestionResult(
            prompt=prompt,
            ok=False,
            error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            has_thinking=thinking,
        )

    wall = time.perf_counter() - started

    # Aggregate stats are reported on the final done:true chunk.
    # These are server-side timings and should be interpreted separately from
    # client-observed TTFT.
    load = final_chunk.get("load_duration", 0) / 1e9
    prompt_eval = final_chunk.get("prompt_eval_duration", 0) / 1e9
    eval_count = int(final_chunk.get("eval_count", 0))
    eval_dur = final_chunk.get("eval_duration", 0) / 1e9
    total = final_chunk.get("total_duration", 0) / 1e9

    server_prefill_seconds = round(load + prompt_eval, 3)

    log.info(
        "ollama benchmark_prompt_ollama report durations: "
        "load=%.3fs prompt_eval=%.3fs server_prefill=%.3fs "
        "eval_count=%d eval_dur=%.3fs total=%.3fs",
        load,
        prompt_eval,
        server_prefill_seconds,
        eval_count,
        eval_dur,
        total,
    )

    # Client-observed visible TTFT. For thinking models, this includes the time
    # spent before the model emits its first visible response chunk.
    ttft_seconds = (
        round(first_response_at, 3)
        if first_response_at is not None
        else 0.0
    )

    # Client-observed hidden-thinking TTFT. We do NOT add load_duration
    # here because first_thinking_at already measures wall-clock from
    # right before urlopen() to the first thinking chunk arrival, which
    # naturally includes any disk->VRAM load the server performed.
    thinking_ttft_seconds = 0.0
    if thinking and first_thinking_at is not None:
        thinking_ttft_seconds = round(first_thinking_at, 3)
        log.info(
            "ollama thinking ttft: first_thinking_chunk=%.3fs "
            "(server load=%.3fs reported separately)",
            first_thinking_at,
            load,
        )
    elif thinking:
        log.info(
            "ollama thinking ttft: model accepted think:true but emitted "
            "no non-empty thinking chunk; leaving thinking_ttft=0"
        )

    if first_response_at is not None:
        log.info(
            "ollama visible ttft: first_response_chunk=%.3fs "
            "server_prefill=%.3fs ttft=%.3fs",
            first_response_at,
            server_prefill_seconds,
            ttft_seconds,
        )
    else:
        log.info(
            "ollama visible ttft: no non-empty response chunk emitted; "
            "leaving ttft=0"
        )

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=sum(len(p) for p in response_parts),
        wall_seconds=round(wall, 3),

        # User-visible, client-observed TTFT.
        ttft_seconds=ttft_seconds,

        # Hidden-thinking, client-observed TTFT.
        thinking_ttft_seconds=thinking_ttft_seconds,
        has_thinking=thinking,

        # Server-side aggregate timings from Ollama final chunk.
        load_seconds=round(load, 3),
        prompt_eval_seconds=round(prompt_eval, 3),
        eval_count=eval_count,
        eval_seconds=round(eval_dur, 3),
        tps=round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0,
        total_server_seconds=round(total, 3),
    )
