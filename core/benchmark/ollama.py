"""Ollama-specific helpers: /api/pull progress streaming + /api/generate
benchmark with precise server-side timing fields.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

from models import QuestionResult
from utils.format import human_bytes
from utils.http import (
    auth_hint,
    http_post_json,
    ollama_model_names,
    ollama_tags,
)

log = logging.getLogger("llm_bench")


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
    last_exc: Optional[BaseException] = None
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
        except Exception as exc:  # noqa: BLE001
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


def _ttft_no_think_probe(url: str, model: str, prompt: str,
                         disable_payload: Optional[dict],
                         *, request_timeout: int) -> Optional[float]:
    """Send one short /api/generate call with thinking disabled and read
    `load_duration + prompt_eval_duration` from the response.

    Caps generation to 1 token via `options.num_predict=1` so the probe
    isn't a full inference (we only need the prefill timing).
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_predict": 1},
    }
    if disable_payload:
        for k, v in disable_payload.items():
            if k == "options" and isinstance(v, dict) \
                    and isinstance(payload.get("options"), dict):
                payload["options"].update(v)
            else:
                payload[k] = v
    try:
        body = http_post_json(f"{url}/api/generate", payload,
                              timeout=request_timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("ttft_no_think probe failed: %s", exc)
        return None
    load = body.get("load_duration", 0) / 1e9
    prompt_eval = body.get("prompt_eval_duration", 0) / 1e9
    return load + prompt_eval


def benchmark_prompt_ollama(url: str, model: str, prompt: str,
                            *,
                            request_timeout: int,
                            thinking: bool = False,
                            thinking_disable_payload: Optional[dict] = None,
                            ) -> QuestionResult:
    """One non-streaming /api/generate call. Decode server-reported
    durations (nanoseconds) into the unified QuestionResult shape.

    When `thinking=True`, send an additional short probe with thinking
    explicitly disabled (default `{"think": false}`) so we can compare
    the ttft you'd actually see if reasoning were turned off.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    started = time.perf_counter()
    try:
        body = http_post_json(f"{url}/api/generate", payload,
                              timeout=request_timeout)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
        )
    wall = time.perf_counter() - started

    # Ollama reports durations in nanoseconds.
    load = body.get("load_duration", 0) / 1e9
    prompt_eval = body.get("prompt_eval_duration", 0) / 1e9
    eval_count = int(body.get("eval_count", 0))
    eval_dur = body.get("eval_duration", 0) / 1e9
    total = body.get("total_duration", 0) / 1e9

    ttft_no_think = 0.0
    if thinking:
        probed = _ttft_no_think_probe(url, model, prompt,
                                      thinking_disable_payload,
                                      request_timeout=request_timeout)
        if probed is not None:
            ttft_no_think = round(probed, 3)

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(body.get("response", "")),
        wall_seconds=round(wall, 3),
        ttft_seconds=round(load + prompt_eval, 3),
        ttft_no_think_seconds=ttft_no_think,
        load_seconds=round(load, 3),
        prompt_eval_seconds=round(prompt_eval, 3),
        eval_count=eval_count,
        eval_seconds=round(eval_dur, 3),
        tps=round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0,
        total_server_seconds=round(total, 3),
    )
