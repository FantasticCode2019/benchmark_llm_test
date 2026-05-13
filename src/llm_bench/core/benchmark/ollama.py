"""
Ollama-specific helper: ONE streaming /api/generate call that captures
both server-side aggregate timings and the client-side wall-clock time
to the first non-empty `thinking` chunk — no second probe required.

Metrics emitted (per cankao2.md):

  * ``ttft_seconds = load_duration + prompt_eval_duration`` — non-streaming
    aggregate approximation of "model load + prefill" time, read from
    the server's stats on the final NDJSON chunk (``done:true``).
  * ``thinking_ttft_seconds = first_thinking_chunk_wall_time + load_duration``
    — wall-clock time from POST send to the first non-empty
    ``thinking`` chunk, *plus* the server-reported ``load_duration``
    so the metric reflects cold-start behaviour even when the daemon
    has the model pre-warmed in VRAM. Populated only when the caller
    passed ``thinking=True`` (i.e. /api/show's capabilities[] reported
    thinking support); left at ``0.0`` otherwise. We do NOT fall back
    to first-response-chunk timing — that's the answer TTFT, a
    different metric.

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


def benchmark_prompt_ollama(url: str, model: str, prompt: str,
                            *,
                            request_timeout: int,
                            thinking: bool = False,
                            ) -> QuestionResult:
    """One streaming /api/generate call. Returns TTFT + aggregate
    timings in a single request.

    Behaviour by ``thinking`` flag:

      * ``thinking=True`` — send ``think:true``. Record the wall-clock
        time to the first non-empty ``thinking`` chunk and report
        ``thinking_ttft_seconds = first_thinking_at + load_duration``.
        Aggregate stats from the final chunk feed ``ttft_seconds``,
        ``eval_seconds``, ``tps``, ``total_server_seconds``.
      * ``thinking=False`` — send ``think:false``. Skip the first-thinking
        bookkeeping; ``thinking_ttft_seconds`` stays at ``0.0``.
        Same aggregate decoding as above.

    The caller is expected to drive ``thinking`` from the runtime
    probe (``ctx.result.ollama_supports_thinking``) — see
    ``_step_probe_ollama_thinking`` in the orchestrator.
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
        headers={"Content-Type": "application/json",
                 "Accept": "application/x-ndjson"},
        method="POST",
    )

    started = time.perf_counter()
    first_thinking_at: float | None = None
    response_parts: list[str] = []
    final_chunk: dict = {}

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

                # /api/generate exposes thinking/response at the top
                # level; /api/chat nests them under message.{thinking,
                # content}. Accept both shapes so a future caller
                # switching to chat doesn't silently lose TTFT.
                message = chunk.get("message") or {}
                thinking_text = (chunk.get("thinking")
                                 or message.get("thinking") or "")
                response_text = (chunk.get("response")
                                 or message.get("content") or "")

                if (thinking and thinking_text
                        and first_thinking_at is None):
                    first_thinking_at = time.perf_counter() - started

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
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            has_thinking=thinking,
        )

    wall = time.perf_counter() - started

    # Aggregate stats are reported on the final chunk (done:true).
    # When the stream ended without a done chunk (unlikely, e.g. peer
    # close mid-stream) final_chunk is {} and every duration is 0 —
    # callers see ok=True with zeroed timings instead of a misleading
    # short ttft.
    load = final_chunk.get("load_duration", 0) / 1e9
    prompt_eval = final_chunk.get("prompt_eval_duration", 0) / 1e9
    eval_count = int(final_chunk.get("eval_count", 0))
    eval_dur = final_chunk.get("eval_duration", 0) / 1e9
    total = final_chunk.get("total_duration", 0) / 1e9
    log.info("ollama benchmark_prompt_ollama report durations: "
             "load=%.3fs prompt_eval=%.3fs eval_count=%d "
             "eval_dur=%.3fs total=%.3fs",
             load, prompt_eval, eval_count, eval_dur, total)

    ttft_seconds = round(load + prompt_eval, 3)
    thinking_ttft_seconds = 0.0
    if thinking and first_thinking_at is not None:
        # cankao2.md: thinking_ttft = wall_clock_to_first_thinking_chunk
        # + model load time. The wall clock starts when we send POST,
        # so when the chart launcher has already warmed the model the
        # load is invisible from the client; explicitly adding the
        # server-reported load_duration makes the metric reflect
        # cold-start behaviour on a hot daemon.
        thinking_ttft_seconds = round(first_thinking_at + load, 3)
        log.info("ollama thinking ttft: first_chunk=%.3fs + load=%.3fs "
                 "= thinking_ttft=%.3fs",
                 first_thinking_at, load, thinking_ttft_seconds)
    elif thinking:
        log.info("ollama thinking ttft: model accepted think:true but "
                 "emitted no thinking chunk; leaving thinking_ttft=0")

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=sum(len(p) for p in response_parts),
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
