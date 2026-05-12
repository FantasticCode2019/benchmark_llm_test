"""vLLM bundle HTTP client.

Wraps the two vLLM-bundle endpoints that ``core.readiness`` poll-loops
care about:

* ``GET /cfg``                  the chart launcher's job descriptor
* ``GET /progress?id=<jobId>``  per-job SSE progress stream

Both helpers are best-effort: transport errors, non-2xx, non-JSON / non-
SSE bodies collapse to ``(None, status, raw_body)`` so the readiness
poller can log and retry without sprinkling ``try``/``except``.
"""
from __future__ import annotations

import json
import urllib.parse

from llm_bench.constants import HTTP_DEFAULT_TIMEOUT_SECONDS
from llm_bench.utils.http import http_get_status


def bundle_cfg(url: str, *,
               timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
               ) -> tuple[dict | None, int, str]:
    """GET <base>/cfg. Returns (parsed_dict_or_None, http_status, raw_body).

    The bundle-aware chart exposes /cfg with the running download-job
    descriptor (typically a top-level {jobId, probeUrl, probeIntervalMs,
    modelName, repo, ref, appUrl, ...}; legacy multi-task charts may
    instead expose {tasks:[{jobId,repo,file,...}], jobIds, probeUrl,
    probeIntervalMs}). The raw body is returned alongside so callers can
    log exactly what came back when parsing fails.
    """
    status, body = http_get_status(f"{url.rstrip('/')}/cfg", timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, body)
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (None, status, body)
    return ((data if isinstance(data, dict) else None), status, body)


def vllm_progress(url: str, job_id: str, *,
                  timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                  ) -> tuple[dict | None, int, str]:
    """GET <base>/progress?id=<job_id> for **vllm** bundles.

    The vllm-side chart serves /progress as Server-Sent Events:
        data: {"id":"...","status":"running","downloaded":...}\\n
        \\n
        data: {"id":"...","status":"done","downloaded":...}\\n

    We GET once, scan for `data: {json}` events, and return the LAST
    parsed event (= latest known state). Plain-JSON bodies are also
    accepted as a single snapshot. Returns
    (last_event_or_None, http_status, raw_body); non-2xx / nothing
    parseable returns (None, status, raw).
    """
    qid = urllib.parse.quote(job_id, safe="")
    status, body = http_get_status(
        f"{url.rstrip('/')}/progress?id={qid}", timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, body)

    # Snapshot mode: whole body is plain JSON (no SSE framing).
    try:
        parsed = json.loads(body or "")
        if isinstance(parsed, dict) and parsed:
            return (parsed, status, body)
    except json.JSONDecodeError:
        pass

    # SSE mode: one event = one or more contiguous `data:` lines,
    # terminated by a blank line. We accumulate `data:` payloads and
    # parse them on the blank-line boundary (or at EOF), keeping the
    # LAST successfully-parsed event.
    last: dict | None = None
    buf: list = []

    def _flush() -> None:
        nonlocal last
        if not buf:
            return
        blob = "".join(buf)
        buf.clear()
        try:
            evt = json.loads(blob)
        except json.JSONDecodeError:
            return
        if isinstance(evt, dict):
            last = evt

    for line in (body or "").splitlines():
        stripped = line.rstrip("\r")
        if not stripped.strip():
            _flush()
            continue
        if stripped.startswith("data:"):
            buf.append(stripped[5:].lstrip())
    _flush()

    return (last, status, body)


__all__ = ["bundle_cfg", "vllm_progress"]
