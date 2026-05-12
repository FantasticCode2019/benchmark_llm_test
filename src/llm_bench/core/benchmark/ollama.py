"""
Ollama-specific helpers: /api/pull progress streaming + /api/generate
benchmark with precise server-side timing fields.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from llm_bench.clients.ollama_client import ollama_model_names, ollama_tags
from llm_bench.clients.openai_errors import auth_hint
from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import QuestionResult
from llm_bench.utils.format import human_bytes
from llm_bench.utils.http import http_post_json

log = logging.getLogger(LOG_NAMESPACE)


def pull_model_ollama(url: str, model: str, *,
                      timeout: int,
                      max_attempts: int = 5,
                      retry_sleep_seconds: int = 30,
                      log_every_seconds: int = 30) -> None:
    """Trigger an Ollama model pull. Streams progress; idempotent + resumable.

    OPT-IN escape hatch — disabled by default in `bench_model`. The Ollama
    charts in olares-market all ship a launcher container that runs
    `ollama pull <model>` at startup, so the canonical flow is:

        chart launcher pulls -> wait_until_api_ready watches /api/tags

    Set `pull_model: true` per-model only when the chart does NOT auto-pull
    or when you want streaming progress in the SCRIPT's stdout.

    Implementation notes:
      1. `stream:true` so Ollama emits an event every few seconds; the
         connection stays warm AND we get a real progress log instead of
         a silent block past most ingresses' 60s idle timeout.
      2. Transport failures retry up to `max_attempts` times. Ollama's
         pull is idempotent + resumable, so a reconnect resumes from
         where the last attempt died. Between attempts we cheaply
         re-check /api/tags; if the model is already present (e.g. the
         chart's launcher beat us to it), we exit cleanly.

    Returns when Ollama emits `{"status":"success"}` OR when /api/tags
    starts listing the target model. Raises if all attempts fail.
    """
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        log.info("ollama pull %s (attempt %d/%d, stream=true, "
                 "per-attempt timeout=%ds)",
                 model, attempt, max_attempts, timeout)
        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/api/pull",
                data=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/x-ndjson"},
                method="POST",
            )
            last_log = 0.0
            saw_success = False
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:  # NDJSON: one event per line
                    if not raw_line:
                        continue
                    try:
                        evt = json.loads(
                            raw_line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(evt, dict):
                        continue
                    err = evt.get("error")
                    if err:
                        # Ollama signals fatal pull errors inline (auth
                        # failure, manifest not found, OOM-on-disk, ...).
                        # Don't retry — the next attempt would just hit
                        # the same wall.
                        raise RuntimeError(f"ollama pull error: {err}")
                    status_text = evt.get("status") or ""
                    completed = evt.get("completed")
                    total = evt.get("total")
                    now = time.monotonic()
                    if (now - last_log >= log_every_seconds
                            or status_text == "success"):
                        if total:
                            pct = (float(completed or 0) / float(total)) * 100
                            log.info("ollama pull %s: %s "
                                     "(%s / %s, %.1f%%)",
                                     model, status_text,
                                     human_bytes(completed),
                                     human_bytes(total), pct)
                        elif status_text:
                            log.info("ollama pull %s: %s",
                                     model, status_text)
                        last_log = now
                    if status_text == "success":
                        saw_success = True
                        log.info("ollama pull %s: server reported success",
                                 model)
                        return
            if not saw_success:
                # Stream ended without a "success" sentinel — could be a
                # mid-pull EOF (proxy reset). Fall through to retry.
                last_exc = RuntimeError("stream ended without success event")
                log.warning("ollama pull %s attempt %d/%d: %s",
                            model, attempt, max_attempts, last_exc)
        except RuntimeError:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("ollama pull %s attempt %d/%d transport failed: %s",
                        model, attempt, max_attempts, exc)
        # Maybe the chart launcher already pulled it — check before sleeping.
        models, _, _ = ollama_tags(url)
        if models is not None and model in ollama_model_names(models):
            log.info("ollama pull %s: target appears in /api/tags after "
                     "attempt %d — treating as success", model, attempt)
            return
        if attempt < max_attempts:
            log.info("ollama pull %s: sleeping %ds before retry",
                     model, retry_sleep_seconds)
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        f"ollama pull {model!r} failed after {max_attempts} attempts; "
        f"last error: {last_exc}"
    )


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

    Only invoked when `spec.thinking=true` — we trust the config that
    the model can think, so probe failures (including the famous
    Ollama-0.10 `HTTP 400 "does not support thinking"`) just degrade
    `thinking_ttft_seconds` to 0; they don't change `has_thinking`.
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

    `thinking` is the per-model `spec.thinking` config flag, echoed
    onto `QuestionResult.has_thinking`. When True, an extra streaming
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
