"""Two-level readiness gate for the LLM backend.

Level 1: poll the listing endpoint until the target model is registered.
Level 2: send one tiny inference call (warmup) to flush any lazy init.

Both gates tolerate transport / 5xx / non-JSON failures and just retry.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from models import OpenAIConfig
from utils.format import human_bytes
from utils.http import (
    http_get_status,
    http_post_json,
    ollama_tags,
)

log = logging.getLogger("llm_bench")


def wait_until_api_ready(
    url: str, api_type: str, model: str,
    *,
    max_wait_minutes: int = 60,
    probe_interval_seconds: int = 30,
) -> None:
    """Block until the backend reports a model as fully loaded.

    `market install --watch` only signals chart=running; the LLM server
    inside the container may still be pulling weights or initializing.
    No chart-level signal exists, so we poll the LLM API itself:

      ollama  -> GET /api/tags, `models[]` non-empty
                 (since each chart hosts exactly one model and we
                 uninstall between models, any entry == the target)
      openai  -> GET /v1/models, `data[]` non-empty (vLLM / llama-server
                 only register this endpoint AFTER weights finish loading)
    """
    deadline = time.monotonic() + max_wait_minutes * 60
    base = url.rstrip("/")
    if api_type == "ollama":
        log.info("waiting for ollama /api/tags models[] non-empty "
                 "(max %dm, probe every %ds)",
                 max_wait_minutes, probe_interval_seconds)
    else:
        log.info("waiting for vllm /v1/models data[] non-empty "
                 "(max %dm, probe every %ds)",
                 max_wait_minutes, probe_interval_seconds)
    attempt = 0
    last_msg = ""
    while True:
        attempt += 1
        ready = False
        if api_type == "ollama":
            models, _, err = ollama_tags(base)
            if models is None:
                last_msg = f"daemon not reachable ({err})"
            elif models:
                sizes = ", ".join(
                    f"{(m or {}).get('name', '?')}="
                    f"{human_bytes((m or {}).get('size'))}"
                    for m in models
                )
                last_msg = f"daemon up, {len(models)} model(s): {sizes}"
                ready = True
            else:
                last_msg = ("daemon up but /api/tags is empty; "
                            "waiting for the chart to pull")
        else:
            status, body = http_get_status(f"{base}/v1/models", timeout=10)
            if not (200 <= status < 300):
                last_msg = f"HTTP {status}"
            else:
                try:
                    items = json.loads(body or "{}").get("data") or []
                except json.JSONDecodeError:
                    items = []
                    last_msg = "200 but body not JSON (login page?)"
                else:
                    ids = [i for i in
                           ((it or {}).get("id") for it in items) if i]
                    if ids:
                        if model and model not in ids:
                            last_msg = (
                                f"served ids={ids}, {model!r} not in list "
                                "(treating as ready; chart probably set "
                                "served-model-name)")
                        else:
                            last_msg = f"served ids={ids}"
                        ready = True
                    else:
                        last_msg = ("HTTP 200 but data=[] "
                                    "(server up, weights still loading)")
        if ready:
            log.info("API ready after %d probe(s) (%s)", attempt, last_msg)
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"API not ready after {max_wait_minutes}m: last -> {last_msg}"
            )
        log.info("api not ready (%s); sleeping %ds", last_msg,
                 probe_interval_seconds)
        time.sleep(probe_interval_seconds)


def warmup_until_ok(url: str, model: str, api_type: str,
                    openai_conf: Optional[OpenAIConfig],
                    *, request_timeout: int,
                    retries: int = 10,
                    sleep_seconds: int = 30) -> None:
    """Send one tiny inference, retrying as needed (level-2 readiness gate).

    Catches the small window where wait_until_api_ready already saw the
    listing endpoint go live but the first real inference still 503s
    (KV cache init, lazy CUDA compile, Modelfile re-quant, vLLM scheduler
    warmup, ...). Defaults: 10×30s ≈ 5 min. On final failure we just log
    and let the real benchmark capture the error in QuestionResult, so a
    flaky model doesn't take down the whole multi-model run.
    """
    # Lazy import to avoid pulling in `openai` when callers are benchmarking
    # ollama only.
    from core.benchmark.openai import make_openai_client, openai_warmup_call

    client = None
    if api_type == "openai":
        assert openai_conf is not None
        client = make_openai_client(url, openai_conf, timeout=request_timeout)

    last_exc: Optional[BaseException] = None
    for i in range(1, retries + 1):
        try:
            if api_type == "openai":
                openai_warmup_call(client, model, openai_conf)
            else:
                http_post_json(
                    f"{url}/api/generate",
                    {"model": model, "prompt": "ping", "stream": False},
                    timeout=request_timeout,
                )
            log.info("warmup ok (attempt %d/%d)", i, retries)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("warmup attempt %d/%d failed: %s", i, retries, exc)
            if i < retries:
                time.sleep(sleep_seconds)
    log.warning("warmup exhausted retries (last error: %s); "
                "continuing into the benchmark anyway", last_exc)
