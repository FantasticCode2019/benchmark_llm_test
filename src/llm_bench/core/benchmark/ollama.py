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
    ``total_server_seconds`` — are decoded from the final
    ``done:true`` chunk and stored on the :class:`QuestionResult` for
    diagnostics. They are NOT used as TTFT: for a thinking model
    ``load_duration + prompt_eval_duration`` would understate the
    user-visible TTFT because it ignores the time the model spends
    emitting the hidden reasoning trace.

  * ``tps`` — **decode-only** generated-tokens-per-second
    (``eval_count / eval_duration``). This isolates raw decode
    throughput from prefill and streaming I/O so the number is a
    pure "how fast the model emits tokens once it starts" reading.
    The server's ``eval_duration`` is authoritative — it counts
    nanoseconds spent in the decode loop on the GPU. Wall-clock
    throughput (``eval_count / wall_seconds``) is exposed
    separately on the per-prompt log line as ``client_tps`` for
    diagnostics, but is NOT the headline ``tps``.

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

import errno
import http.client
import json
import logging
import socket
import ssl
import time
import urllib.error
import urllib.request

from llm_bench.clients.openai_errors import auth_hint
from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import QuestionResult

log = logging.getLogger(LOG_NAMESPACE)

# Default retry policy for transient network / transport errors. Three total
# attempts (one initial + two retries) is enough to ride out a brief network
# blip, an upstream cold-start, or a single TLS connection that the load
# balancer tore down mid-stream — without dragging a benchmark run out for
# minutes when the endpoint is genuinely down. Each retry starts a fresh
# timer epoch inside `_attempt_once_ollama`, so wall_seconds / ttft_seconds
# reflect ONLY the successful attempt's timings — partial work from the
# failed attempt is discarded.
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 5.0

# errno codes that signal a transient/recoverable transport condition.
# Anything in this set means "the connection broke for an environmental
# reason, not because our request was wrong" — retrying with a fresh
# socket is the right move.
_RETRYABLE_ERRNOS = frozenset({
    errno.ETIMEDOUT,        # [Errno 110] Connection timed out (Linux)
    errno.ECONNRESET,       # [Errno 104] Connection reset by peer
    errno.ECONNREFUSED,     # [Errno 111] briefly-down upstream
    errno.ECONNABORTED,     # local connection aborted
    errno.EPIPE,             # Broken pipe — server closed mid-write
    errno.ENETUNREACH,
    errno.EHOSTUNREACH,
    errno.ENETRESET,
    errno.ENETDOWN,
})

# String fragments seen in the wild that indicate a retryable transport
# problem when the underlying exception type is lost (e.g. wrapped by a
# logging / proxy / TLS middleware that only preserved str(exc)).
_RETRYABLE_SUBSTRINGS = (
    "timed out",
    "connection reset",
    "connection refused",
    "connection aborted",
    "broken pipe",
    "remote end closed",
    "unexpected_eof",
    "eof occurred",  # ssl.SSLEOFError stringifies as "EOF occurred in ..."
)


def _is_retryable_network_error(exc: BaseException) -> bool:
    """True if ``exc`` is a transient network/transport failure worth retrying.

    Covers the shapes we see in practice when talking to ollama through a
    load balancer / ingress:

      * Python-level read timeout — ``socket.timeout`` / ``TimeoutError``
        raised by ``urlopen(timeout=...)`` when no bytes arrive in time.
      * Kernel-level transport errors — ``OSError`` with one of the
        ``errno`` codes in :data:`_RETRYABLE_ERRNOS` (ETIMEDOUT for the
        Linux ``[Errno 110] Connection timed out`` form, ECONNRESET for
        the "peer killed our socket" form, etc.).
      * Python's :class:`ConnectionError` family — ``ConnectionResetError``,
        ``ConnectionAbortedError``, ``BrokenPipeError``, ...
      * TLS-layer failures — :class:`ssl.SSLError` and friends, including
        ``SSLEOFError`` ("``[SSL: UNEXPECTED_EOF_WHILE_READING] EOF
        occurred in violation of protocol``"), which happens when an
        upstream / ingress drops the TLS connection mid-stream.
      * HTTP-layer transport breakages — :class:`http.client.HTTPException`
        and subclasses: ``RemoteDisconnected``, ``IncompleteRead``,
        ``BadStatusLine``. Symptoms of a transport that died between
        request and response.
      * :class:`urllib.error.URLError` wrapping any of the above on its
        ``reason`` attribute, which is how the streaming loop typically
        surfaces these.

    A defensive string fallback (:data:`_RETRYABLE_SUBSTRINGS`) catches
    the same wording in environments where the underlying exception type
    has been lost.

    NOTE: :class:`urllib.error.HTTPError` (4xx / 5xx with a body) is a
    subclass of ``URLError`` but is NOT considered retryable here — those
    are application-level responses (auth, bad request, ...) and
    retrying them just delays the failure.
    """
    # 4xx / 5xx are application-level responses; never retry.
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, http.client.HTTPException):
        return True
    if isinstance(exc, OSError) and exc.errno in _RETRYABLE_ERRNOS:
        return True
    reason = getattr(exc, "reason", None)
    if (
        reason is not None
        and reason is not exc
        and isinstance(reason, BaseException)
    ):
        return _is_retryable_network_error(reason)
    s = str(exc).lower()
    return any(frag in s for frag in _RETRYABLE_SUBSTRINGS)


def benchmark_prompt_ollama(
    url: str,
    model: str,
    prompt: str,
    *,
    request_timeout: int,
    thinking: bool = False,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
) -> QuestionResult:
    """One streaming /api/generate call. Returns client-observed TTFT
    plus aggregate server timings in a single request.

    On transient network / transport errors (read timeouts, kernel-level
    connect timeouts, connection resets, TLS ``SSLEOFError`` /
    ``UNEXPECTED_EOF_WHILE_READING`` mid-stream, ``RemoteDisconnected``,
    ...) the request is retried up to ``max_attempts`` times (default
    3 = one initial attempt plus two retries). Each retry runs a fresh
    :func:`_attempt_once_ollama`, which resets the
    ``time.perf_counter()`` epoch and all chunk-tracking state, so the
    returned ``wall_seconds`` / ``ttft_seconds`` /
    ``thinking_ttft_seconds`` describe ONLY the attempt that finally
    succeeded — partial timings from a failed attempt are intentionally
    discarded. Permanent errors (auth failures, HTTP 4xx, JSON decode
    errors, ...) are returned immediately without burning retry budget.

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
    attempts = max(1, int(max_attempts))
    last_msg = ""
    last_wall = 0.0

    for attempt in range(1, attempts + 1):
        try:
            return _attempt_once_ollama(
                url,
                model,
                prompt,
                request_timeout=request_timeout,
                thinking=thinking,
            )
        except _OllamaRetryable as exc:
            last_msg = exc.msg
            last_wall = exc.wall_seconds
            if attempt < attempts:
                log.warning(
                    "ollama request failed (attempt %d/%d, "
                    "wasted=%.3fs): %s -- retrying in %.1fs "
                    "(timers will be reset)",
                    attempt,
                    attempts,
                    exc.wall_seconds,
                    exc.msg,
                    retry_backoff_seconds,
                )
                if retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds)
            else:
                log.warning(
                    "ollama request failed after %d attempts "
                    "(last wasted=%.3fs): %s",
                    attempts,
                    exc.wall_seconds,
                    exc.msg,
                )

    return QuestionResult(
        prompt=prompt,
        ok=False,
        error=(
            f"transient failure after {attempts} attempts: {last_msg}"
            if last_msg
            else f"transient failure after {attempts} attempts"
        ),
        wall_seconds=round(last_wall, 3),
        has_thinking=thinking,
    )


class _OllamaRetryable(Exception):
    """Internal signal: this attempt failed with a transient network /
    transport error and the outer loop should retry with a fresh timer
    epoch. Carries ``wall_seconds`` so the retry log line can show how
    long the failed attempt actually spent waiting, and so the final
    "all attempts exhausted" QuestionResult can report the last
    attempt's wall time instead of a hard-coded zero.
    """

    def __init__(self, msg: str, wall_seconds: float) -> None:
        super().__init__(msg)
        self.msg = msg
        self.wall_seconds = wall_seconds


def _attempt_once_ollama(
    url: str,
    model: str,
    prompt: str,
    *,
    request_timeout: int,
    thinking: bool,
) -> QuestionResult:
    """Single streaming /api/generate attempt. Caller is responsible for
    retry policy.

    Raises :class:`_OllamaRetryable` on transient network/transport
    errors so the outer ``benchmark_prompt_ollama`` can retry with a
    fresh timer epoch. Permanent failures are still returned as
    ``ok=False`` :class:`QuestionResult` instances (auth errors,
    HTTP 4xx, etc. should not consume retry budget).
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
        wall_at_failure = time.perf_counter() - started

        # Bubble transient transport errors up so the outer retry loop
        # can restart with a fresh timer epoch. Everything else (auth,
        # JSON decode, HTTP 4xx, ...) is terminal for this prompt —
        # return a failed QuestionResult immediately.
        if _is_retryable_network_error(exc):
            raise _OllamaRetryable(str(exc), wall_at_failure) from exc

        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"

        return QuestionResult(
            prompt=prompt,
            ok=False,
            error=msg,
            wall_seconds=round(wall_at_failure, 3),
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

    client_tps = (eval_count / wall) if wall > 0 and eval_count else 0.0

    log.info(
        "ollama benchmark_prompt_ollama report durations: "
        "load=%.3fs prompt_eval=%.3fs server_prefill=%.3fs "
        "eval_count=%d eval_dur=%.3fs total=%.3fs "
        "wall=%.3fs client_tps=%.2f",
        load,
        prompt_eval,
        server_prefill_seconds,
        eval_count,
        eval_dur,
        total,
        wall,
        client_tps,
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
        # Decode-only throughput: tokens emitted by the server divided
        # by the server's reported decode duration (`eval_duration`).
        # Isolates raw model speed from prefill + streaming I/O.
        tps=round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0,
        client_tps=round(client_tps, 2),
        total_server_seconds=round(total, 3),
    )
